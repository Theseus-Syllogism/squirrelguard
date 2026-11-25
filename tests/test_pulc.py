"""PULC attributes: ONNX wrapper (metadata, preprocessing), visit aggregation, the full-body gate, and the weak
matching evidence in the bridge's explain.py.

No network and no real model: a tiny ONNX graph with the export's metadata returns fixed probabilities.
Run:  .venv/bin/python -m pytest tests/test_pulc.py      (pytest optional)
      .venv/bin/python tests/test_pulc.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard import pulc  # noqa: E402

spec = importlib.util.spec_from_file_location("explain", ROOT / "deploy" / "recorder" / "bridge" / "explain.py")
explain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(explain)

PERSON_LABELS = ["Hat", "Glasses", "ShortSleeve", "LongSleeve", "UpperStride", "UpperLogo", "UpperPlaid",
                 "UpperSplice", "LowerStripe", "LowerPattern", "LongCoat", "Trousers", "Shorts", "Skirt&Dress",
                 "Boots", "HandBag", "ShoulderBag", "Backpack", "HoldObjectsInFront", "AgeLess18", "Age18-60",
                 "AgeOver60", "Female", "Front", "Side", "Back"]


def fake_model(path: Path, probs: list[float], labels: list[str], kind: str, wh=(192, 256)) -> None:
    """x (N,3,H,W) -> probs broadcast to (N, L): constant + 0 * mean(x) so the batch dimension follows the input."""
    import onnx
    from onnx import TensorProto, helper
    L = len(labels)
    const = helper.make_tensor("c", TensorProto.FLOAT, [1, L], np.asarray(probs, np.float32))
    nodes = [helper.make_node("ReduceMean", ["x"], ["m"], axes=[1, 2, 3], keepdims=0),   # (N,)
             helper.make_node("Unsqueeze", ["m", "ax"], ["m1"]),                         # (N,1)
             helper.make_node("Mul", ["m1", "zero"], ["z"]),
             helper.make_node("Add", ["z", "c"], ["y"])]
    g = helper.make_graph(nodes, "fake", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, wh[1], wh[0]])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", L])],
                          [const, helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0]),
                           helper.make_tensor("ax", TensorProto.INT64, [1], [1])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    for k, v in {"sg_kind": kind, "sg_labels": labels, "sg_input_wh": list(wh), "sg_mean": [0.485, 0.456, 0.406],
                 "sg_std": [0.229, 0.224, 0.225], "sg_thresholds": {"default": 0.5, "Glasses": 0.3}}.items():
        p = m.metadata_props.add()
        p.key, p.value = k, v if isinstance(v, str) else json.dumps(v)
    onnx.save(m, str(path))


def person_probs(**kw) -> dict:
    d = {lab: 0.05 for lab in PERSON_LABELS}
    d.update(kw)
    return d


def test_wrapper_reads_metadata_and_batches():
    probs = person_probs(LongSleeve=0.9, ShortSleeve=0.1, Trousers=0.95, Backpack=0.8, Glasses=0.35, Back=0.9,
                         Front=0.05, Side=0.1, Female=0.2, **{"Age18-60": 0.95})
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pa.onnx"
        fake_model(p, [probs[lab] for lab in PERSON_LABELS], PERSON_LABELS, "pulc_person_attribute")
        m = pulc.PersonAttributes(str(p), threads=1, batch=2)
        assert m.labels == PERSON_LABELS and m.wh == (192, 256)
        crops = [np.full((h, h // 2, 3), 128, np.uint8) for h in (100, 200, 300)]
        assert m.prep(crops[0]).shape == (3, 256, 192)
        out = m.probs(crops)                           # 3 crops in batches of 2
        assert len(out) == 3 and abs(out[2]["Backpack"] - 0.8) < 1e-6
        v = m.visit(crops, weights=[1, 3, 2], max_crops=2)
        assert v["crops"] == 2
        assert v["tags"][0] == "long_sleeves" and {"trousers", "backpack", "glasses"} <= set(v["tags"])
        assert v["orientation"]["back"] > 0.8 and v["views"] == {"front": 0, "side": 0, "back": 2}
        assert "Female" not in v["attributes"] and "Front" not in v["attributes"]
        assert abs(v["apparent"]["gender_female"] - 0.2) < 1e-3 and v["apparent"]["age"]["18_60"] > 0.8
        try:   # a vehicle model must not load as a person model
            pulc.VehicleAttributes(str(p))
            raise AssertionError("kind check missing")
        except ValueError:
            pass


def test_aggregate_person_orientation_and_spread():
    a = person_probs(Front=0.9, Side=0.1, Back=0.0, Backpack=0.9)
    b = person_probs(Front=0.0, Side=0.2, Back=0.8, Backpack=0.1)
    v = pulc.aggregate_person([a, b, b])
    o = v["orientation"]
    assert abs(sum(o.values()) - 1) < 1e-3 and o["back"] > o["front"] > o["side"]
    assert v["views"] == {"front": 1, "side": 0, "back": 2}
    assert abs(v["attributes"]["Backpack"] - (0.9 + 0.1 + 0.1) / 3) < 1e-3 and "backpack" not in v["tags"]
    assert v["spread"] > 0.3
    assert pulc.aggregate_person([]) is None
    assert "mostly back view" in pulc.describe_person(v)


def test_aggregate_vehicle_threshold():
    labels = pulc.VEHICLE_COLOURS + pulc.VEHICLE_TYPES
    d = {lab: 0.01 for lab in labels}
    v = pulc.aggregate_vehicle([dict(d, blue=0.9, suv=0.4), dict(d, blue=0.8, suv=0.45)])
    assert v["color"] == "blue" and v["type"] is None and list(v["types"])[0] == "suv"


def test_full_body_gate():
    from squirrel_guard.profiler import full_body, pulc_person_visit
    shp = (720, 1280, 3)
    assert full_body((100, 100, 160, 300), shp)            # upright, inside the frame
    assert not full_body((100, 400, 400, 719), shp)         # close-up cut by the bottom edge
    assert not full_body((100, 100, 300, 300), shp)         # square: seated / partial
    note = pulc_person_visit(type("M", (), {"kind": "pulc_person_attribute"})(), [(1.0, None, (100, 400, 400, 719), shp)])
    assert note == {"model": "pulc_person_attribute", "crops": 0, "skipped": "no full-body view", "considered": 1}


def visit(**attrs):
    base = {"LongSleeve": 0.5, "ShortSleeve": 0.5, "Backpack": 0.05, "Hat": 0.05, "Shorts": 0.05, "LongCoat": 0.05,
            "Skirt&Dress": 0.05}
    base.update(attrs)
    return {"crops": 3, "attributes": base, "apparent": {"gender_female": 0.1, "age": {"under_18": 0.02}}}


def test_explain_consistent_raises_odds_slightly():
    cur = visit(Backpack=0.9, LongSleeve=0.9)
    hist = [(visit(Backpack=0.8, LongSleeve=0.85), 2.0), (visit(Backpack=0.7), 30.0)]
    ev = explain.pulc_evidence(cur, hist)
    assert ev["reasons"] == ["PULC_CONSISTENT"] and set(ev["agree"]) == {"backpack", "long sleeves"}
    assert 1.0 < ev["factor"] <= explain.PULC_CLAMP[1]


def test_explain_contradiction_lowers_odds():
    cur = visit(Shorts=0.9, LongCoat=0.01)
    hist = [(visit(Shorts=0.02, LongCoat=0.9), 1.0)]           # same day: long coat, no shorts
    ev = explain.pulc_evidence(cur, hist)
    assert ev["reasons"][0] == "PULC_CONTRADICTS" and ev["factor"] < 1.0
    assert "shorts" in ev["differ"] and "no long coat" in ev["differ"]
    old = explain.pulc_evidence(cur, [(visit(Shorts=0.02, LongCoat=0.9), 48.0)])   # clothing from 2 days ago: ignored
    assert old["factor"] == 1.0 and not old["reasons"]


def test_explain_apparent_gender_age_and_clamp():
    cur = visit(Shorts=0.9, Hat=0.9, LongCoat=0.01)
    cur["apparent"] = {"gender_female": 0.95, "age": {"under_18": 0.9}}
    hist = [(visit(Shorts=0.01, Hat=0.01, LongCoat=0.9), 1.0), (visit(Shorts=0.01, Hat=0.01, LongCoat=0.9), 3.0)]
    ev = explain.pulc_evidence(cur, hist)
    assert "PULC_APPARENT_GENDER_DIFFERS" in ev["reasons"] and "PULC_APPARENT_AGE_DIFFERS" in ev["reasons"]
    assert ev["factor"] == explain.PULC_CLAMP[0]                   # many contradictions, still only a weak push


def test_explain_nothing_to_compare():
    assert explain.pulc_evidence(None, [])["factor"] == 1.0
    assert explain.pulc_evidence({"crops": 0, "skipped": "no full-body view"}, [(visit(), 1.0)])["factor"] == 1.0
    assert explain.pulc_evidence(visit(Backpack=0.9), [(visit(Backpack=0.9), 100.0)])["factor"] == 1.0  # 1 cross-day visit
    for code in ("PULC_CONSISTENT", "PULC_CONTRADICTS", "PULC_APPARENT_GENDER_DIFFERS", "PULC_APPARENT_AGE_DIFFERS"):
        assert code in explain.REASON_TEXT


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

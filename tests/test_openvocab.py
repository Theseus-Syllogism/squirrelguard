"""Open-vocabulary objects: decoding, label mapping, distractors, person association and clip aggregation.

No GPU and no real model needed: a tiny ONNX graph that returns a fixed YOLO-style output stands in for the detector.
Run:  .venv/bin/python -m pytest tests/test_openvocab.py      (pytest optional)
      .venv/bin/python tests/test_openvocab.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squirrel_guard.openvocab import ObjectAggregator, OpenVocabDetector, describe, relation  # noqa: E402
from squirrel_guard.pose import Pose  # noqa: E402

SIZE = 64
PROMPTS = ["ladder", "crowbar", "pry bar", "coat"]
LABELS = ["ladder", "pry bar", "pry bar", None]          # coat = distractor


def fake_model(path: Path, rows: list[list[float]]) -> None:
    """ONNX model: images (1,3,64,64) -> constant (1, 4 + 4 prompts, anchors) + 0 * mean(images)."""
    import onnx
    from onnx import TensorProto, helper
    out = np.array(rows, np.float32).T[None]                 # (1, 8, anchors)
    const = helper.make_tensor("c", TensorProto.FLOAT, out.shape, out.ravel())
    zero = helper.make_tensor("z", TensorProto.FLOAT, [], [0.0])
    nodes = [helper.make_node("ReduceMean", ["images"], ["m"], keepdims=0),
             helper.make_node("Mul", ["m", "z"], ["m0"]),
             helper.make_node("Add", ["c", "m0"], ["output0"])]
    g = helper.make_graph(nodes, "fake", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
                          [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(out.shape))], [const, zero])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    for k, v in (("names", str(dict(enumerate(PROMPTS)))), ("sg_labels", json.dumps(LABELS))):
        m.metadata_props.add(key=k, value=v)
    onnx.save(m, str(path))


def row(cx, cy, w, h, scores):
    return [cx, cy, w, h, *scores]


def detector(rows, **kw) -> OpenVocabDetector:
    td = tempfile.mkdtemp()
    p = Path(td) / "fake.onnx"
    fake_model(p, rows)
    return OpenVocabDetector(str(p), threads=1, **kw)


def test_detect_labels_distractors_nms():
    rows = [row(20, 20, 10, 20, [0.6, 0.0, 0.0, 0.1]),    # ladder
            row(21, 20, 10, 20, [0.5, 0.0, 0.0, 0.1]),    # duplicate ladder -> NMS
            row(40, 40, 8, 8, [0.0, 0.35, 0.30, 0.0]),    # crowbar prompt -> label pry bar
            row(41, 40, 8, 8, [0.0, 0.0, 0.33, 0.0]),     # pry bar prompt, same box -> merged with the crowbar one
            row(10, 50, 8, 8, [0.0, 0.3, 0.0, 0.5]),      # coat wins -> dropped
            row(50, 10, 8, 8, [0.1, 0.0, 0.0, 0.0])]      # below min_score
    det = detector(rows, min_score=0.2)
    assert det.vocabulary == ["ladder", "pry bar"]
    frame = np.zeros((128, 128, 3), np.uint8)             # letterbox scale 0.5: boxes come back x2
    d = det.detect(frame)
    assert [x["label"] for x in d] == ["ladder", "pry bar"], d
    assert d[1]["prompt"] == "crowbar"
    assert np.allclose(d[0]["box"], (30, 20, 50, 60), atol=0.6), d[0]["box"]
    agg = ObjectAggregator(min_frames=1)
    agg.add(d, (20, 10, 60, 80), t=0.0, size=(128, 128))
    json.dumps(agg.summary())                                # numpy scalars would break the MQTT message
    d2 = detector(rows, min_score=0.2, classes=["ladder"]).detect(frame)
    assert [x["label"] for x in d2] == ["ladder"]
    try:
        detector(rows, classes=["unicorn"])
        raise AssertionError("unknown class accepted")
    except ValueError:
        pass


def test_detect_around_offsets():
    det = detector([row(32, 32, 16, 16, [0.9, 0, 0, 0])], min_score=0.2)
    frame = np.zeros((400, 600, 3), np.uint8)
    d = det.detect_around(frame, (250, 150, 300, 250), grow=2.0, min_side=64)   # window 200x200 at (175, 100)
    assert len(d) == 1
    x1, y1, x2, y2 = d[0]["box"]
    assert abs((x1 + x2) / 2 - 275) < 2 and abs((y1 + y2) / 2 - 200) < 2, d[0]["box"]


def pose_at(box, wrist=None, conf=0.9):
    kp = np.zeros((17, 3), np.float32)
    if wrist is not None:
        kp[10] = (*wrist, conf)
    return Pose(box, 0.9, kp)


def test_relation():
    person = pose_at((100, 100, 200, 400), wrist=(210, 260))
    assert relation((205, 150, 260, 380), person) == "carried"              # wrist on the ladder
    assert relation((205, 150, 260, 380), pose_at((100, 100, 200, 400))) == "near"   # no wrist on it
    assert relation((120, 150, 180, 250), pose_at((100, 100, 200, 400))) == "carried"  # backpack on the body
    assert relation((120, 150, 180, 250), (100, 100, 200, 400)) == "carried"           # plain box works too
    assert relation((600, 100, 700, 200), person) == "scene"
    assert relation((205, 150, 260, 380), pose_at((100, 100, 200, 400), wrist=(210, 260), conf=0.1)) == "near"
    assert relation((205, 150, 260, 380), None) == "scene"
    # a big object behind the person (ladder on the wall, 3x the person) is not "carried" by overlap alone
    assert relation((50, 50, 350, 450), pose_at((100, 100, 200, 400))) != "carried"


def test_aggregator_min_frames_and_carried():
    person = pose_at((100, 100, 200, 400), wrist=(210, 260))
    agg = ObjectAggregator(min_frames=2)
    lad = {"label": "ladder", "score": 0.4, "box": (205, 150, 260, 380), "prompt": "ladder"}
    knife = {"label": "knife", "score": 0.9, "box": (600, 100, 650, 150), "prompt": "knife"}
    agg.add([lad, dict(lad, score=0.3)], person, t=0.0, size=(1000, 500))   # two boxes of one label: one frame
    agg.add([knife], person, t=0.25)                                       # single frame -> suppressed
    agg.add([dict(lad, score=0.5)], person, t=0.5, size=(1000, 500))
    agg.add([], None, t=0.75)
    agg.add([dict(lad, box=(600, 150, 700, 380))], person, t=1.0)          # same label far away: seen, not carried
    s = agg.summary()
    assert [o["label"] for o in s] == ["ladder"], s
    o = s[0]
    assert o["frames"] == 3 and o["sampled"] == 5 and o["carried_frames"] == 2 and o["carried"], o
    assert o["score"] == 0.5 and o["best_t"] == 0.5 and o["box"] == [0.205, 0.3, 0.26, 0.76], o
    assert describe(s) == "carrying ladder (0.50)"
    json.dumps(s)                                                          # goes into the MQTT message


def test_static_fixture_not_carried():
    """A person walks past a chair (hand touches it on every frame); a second person walks carrying a box."""
    agg_fixture, agg_carry = ObjectAggregator(min_frames=2), ObjectAggregator(min_frames=2)
    for k in range(6):
        x = 100 + 80 * k                                    # person moves 400 px, body 300 px tall
        person = pose_at((x, 100, x + 100, 400), wrist=(x + 100, 260))
        chair = {"label": "ladder", "score": 0.5, "box": (300, 200, 360, 320), "prompt": "ladder"}
        box = {"label": "package", "score": 0.5, "box": (x + 90, 220, x + 150, 300), "prompt": "package"}
        agg_fixture.add([chair], person)                    # the wrist is on the chair on frames 1 and 2
        agg_carry.add([box], person)
    f, c = agg_fixture.summary(), agg_carry.summary()
    assert f and f[0]["static"] and not f[0]["carried"], f
    assert c and not c[0]["static"] and c[0]["carried"] and c[0]["carried_fraction"] == 1.0, c


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

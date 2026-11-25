"""Bridge face-template decision (deploy/recorder/bridge/facematch.py): enrolled-only naming, liveness/quality gates.

Run:  .venv/bin/python tests/test_facematch.py      (or pytest). numpy only, like the recorder host bridge.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy/recorder/bridge"))
import facematch as fm  # noqa: E402

RNG = np.random.default_rng(3)


def unit():
    v = RNG.normal(size=512)
    return v / np.linalg.norm(v)


def toward(u, cos):
    """A unit vector with the given cosine to u."""
    r = unit()
    r -= (r @ u) * u
    r /= np.linalg.norm(r)
    return cos * u + np.sqrt(1 - cos ** 2) * r


def meta(**k):
    m = {"n_faces": 12, "quality": 0.4, "size_px": 70, "consistency": 0.9, "liveness": {"ok": True, "score": 0.9},
         "model": "Megatron"}
    m.update(k)
    return m


A, B = unit(), unit()
ENROLLED = [
    {"template_id": 1, "identity_id": 10, "label": "Robin", "kind": "household", "vec": A, "model": "Megatron"},
    {"template_id": 2, "identity_id": 11, "label": "Neighbour", "kind": "known", "vec": B, "model": "Megatron"},
]


def test_match_enrolled_household():
    d = fm.decide(toward(A, 0.55), meta(), ENROLLED)
    assert d["status"] == "match" and d["identity"] == 10 and d["reasons"] == ["FACE_TEMPLATE_MATCH"]
    assert abs(d["sim"] - 0.55) < 1e-3 and d["template"] == 1
    assert 0.9 <= fm.face_prob(d["sim"]) <= 0.99


def test_liveness_fail_never_names():
    d = fm.decide(toward(A, 0.7), meta(liveness={"ok": False, "score": 0.1}), ENROLLED)
    assert d["status"] == "liveness_fail" and d["identity"] is None and d["candidate"] == 10
    assert "FACE_LIVENESS_FAIL" in d["reasons"]


def test_weak_template_gates():
    d = fm.decide(toward(A, 0.7), meta(n_faces=1), ENROLLED)
    assert d["status"] == "weak" and d["identity"] is None and "FACE_TOO_FEW" in d["reasons"]
    d = fm.decide(toward(A, 0.7), meta(quality=0.02), ENROLLED)
    assert d["status"] == "weak" and "FACE_LOW_QUALITY" in d["reasons"]
    d = fm.decide(toward(A, 0.7), meta(consistency=0.1), ENROLLED)
    assert d["status"] == "weak" and "FACE_INCONSISTENT" in d["reasons"]


def test_no_match_for_strangers():
    d = fm.decide(unit(), meta(), ENROLLED)              # a stranger: cosine ~0 to everyone
    assert d["status"] == "no_match" and d["identity"] is None and d["reasons"] == ["FACE_NO_MATCH"]
    d = fm.decide(toward(A, 0.30), meta(), ENROLLED)     # highest impostor template cosine measured was 0.29
    assert d["status"] == "no_match"


def test_ambiguous_between_two_enrolled():
    v = A + B
    v /= np.linalg.norm(v)                                # equally like both (cos ~0.71 each)
    d = fm.decide(v, meta(), ENROLLED)
    assert d["status"] == "no_match" and d["identity"] is None


def test_only_household_known_and_same_model():
    unknown = [{**ENROLLED[0], "kind": "unknown"}, {**ENROLLED[1], "kind": "watch"}]
    assert fm.decide(toward(A, 0.8), meta(), unknown)["status"] == "no_match"
    other = [{**ENROLLED[0], "model": "Pikachu"}]
    assert fm.decide(toward(A, 0.8), meta(), other)["status"] == "no_match"


def test_no_template_reasons():
    d = fm.decide(None, {"reason": "too_small", "max_size_px": 14}, ENROLLED)
    assert d["status"] == "none" and d["reasons"] == ["FACE_TOO_SMALL"]
    d = fm.decide(None, {"reason": "no_face", "max_size_px": 0}, ENROLLED)
    assert d["status"] == "none" and d["reasons"] == []


def test_best_of_several_templates_per_identity():
    more = ENROLLED + [{"template_id": 3, "identity_id": 10, "label": "Robin", "kind": "household",
                        "vec": toward(A, 0.6), "model": "Megatron"}]
    probe = toward(A, 0.5)
    d = fm.decide(probe, meta(), more)
    assert d["identity"] == 10 and d["status"] == "match"


def test_better_template():
    assert fm.better({"vec": [1], "weight_sum": 2.0, "n_faces": 10}, {"weight_sum": 1.0, "n_faces": 10})
    assert not fm.better({"vec": [1], "weight_sum": 0.5, "n_faces": 5}, {"weight_sum": 1.0, "n_faces": 5})
    assert fm.better({"vec": [1], "weight_sum": 0.5}, None)
    assert not fm.better({"vec": None}, None)


# -- unnamed-face auto-tracking (decide_unknown) ---------------------------------------------------------------------
ADA = "adaface_ir101_webface12m"


def fid_of(v, v2=None, **k):
    f = {"vec": list(v), **meta(**k)}
    if v2 is not None:
        f["v2"] = {"model": ADA, "vec": list(v2)}
    return f


def test_unknown_match_uses_adaface_threshold_and_never_names():
    U, W = unit(), unit()
    gal = [{"identity_id": 21, "vec": U, "model": ADA}, {"identity_id": 22, "vec": W, "model": ADA},
           {"identity_id": 21, "vec": unit(), "model": "Megatron"}]
    d = fm.decide_unknown(fid_of(unit(), toward(U, fm.FACE_UNKNOWN_SIM_V2 + 0.05)), gal)
    assert d["status"] == "match" and d["identity"] == 21 and d["engine"] == ADA
    assert "label" not in d and d["reasons"] == ["FACE_UNKNOWN_MATCH"]
    d = fm.decide_unknown(fid_of(unit(), toward(U, fm.FACE_UNKNOWN_SIM_V2 - 0.05)), gal)
    assert d["status"] == "new" and d["identity"] is None
    # the highest visit-vs-visit impostor score measured (CelebA, 197,820 pairs) was 0.323: never a merge
    assert fm.decide_unknown(fid_of(unit(), toward(U, 0.33)), gal)["status"] == "new"


def test_unknown_margin_between_two_unnamed():
    U = unit()
    W = toward(U, 0.9)                     # two unnamed identities that look alike
    gal = [{"identity_id": 1, "vec": U, "model": ADA}, {"identity_id": 2, "vec": W, "model": ADA}]
    d = fm.decide_unknown(fid_of(unit(), toward(U, 0.7)), gal)
    assert d["status"] == "new" and d["reasons"] == ["FACE_UNKNOWN_AMBIGUOUS"]


def test_unknown_same_gates_as_enrolled():
    U = unit()
    gal = [{"identity_id": 5, "vec": U, "model": ADA}]
    for bad in ({"liveness": {"ok": False}}, {"n_faces": 1}, {"quality": 0.01}, {"size_px": 20}, {"consistency": 0.1}):
        d = fm.decide_unknown(fid_of(unit(), toward(U, 0.9), **bad), gal)
        assert d["status"] == "unusable" and d["identity"] is None, bad
    assert not fm.usable(fid_of(unit(), liveness={"ok": False}))
    assert fm.usable(fid_of(unit()))


def test_unknown_isf_fallback_threshold():
    U = unit()
    gal = [{"identity_id": 9, "vec": U, "model": "Megatron"}]          # no AdaFace templates in the gallery
    assert fm.decide_unknown(fid_of(toward(U, 0.45), unit()), gal)["status"] == "new"   # 0.45 < 0.50 (isf)
    d = fm.decide_unknown(fid_of(toward(U, 0.55), unit()), gal)
    assert d["status"] == "match" and d["engine"] == "Megatron"


def test_public_strips_scores_and_candidates():
    U = unit()
    d = fm.decide_unknown(fid_of(unit(), toward(U, 0.1)), [{"identity_id": 3, "vec": U, "model": ADA}])
    p = fm.public(d)
    assert "sims" not in p and "identity" not in p and p["status"] == "new"
    w = fm.decide(toward(A, 0.7), meta(n_faces=1), ENROLLED)            # weak: carries a candidate
    p = fm.public(w, named=True)
    assert "candidate" not in p and "identity" not in p and "label" not in p
    m = fm.decide(toward(A, 0.7), meta(), ENROLLED)
    assert fm.public(m, named=True)["label"] == "Robin" and "label" not in fm.public(m)


def test_face_disagrees_split_rule():
    U = unit()
    gal = [{"identity_id": 4, "vec": U, "model": ADA}]
    assert fm.face_disagrees(fm.decide_unknown(fid_of(unit(), toward(U, 0.05)), gal), 4)
    assert not fm.face_disagrees(fm.decide_unknown(fid_of(unit(), toward(U, 0.3)), gal), 4)   # ambiguous: keep re-ID
    assert not fm.face_disagrees(fm.decide_unknown(fid_of(unit(), toward(U, 0.05)), gal), 99)  # no faces there


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            print("ok", k)

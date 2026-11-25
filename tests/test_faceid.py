"""Face template aggregation (squirrel_guard/faceid.py) on synthetic embeddings; InspireFace is mocked.

Run:  .venv/bin/python tests/test_faceid.py      (or pytest). No network, no GPU, no model files.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squirrel_guard import faceid as fi  # noqa: E402

RNG = np.random.default_rng(7)
D = fi.EMB_DIM


def unit(rng=RNG):
    return fi.l2n(rng.normal(size=D))


def noisy(u, sigma, rng=RNG):
    """Embedding of the identity u seen through noise: cosine to u drops as sigma grows."""
    return fi.l2n(u + sigma * rng.normal(size=D) / np.sqrt(D))


def obs(v, q=0.5, size=80, yaw=0.0, live=0.9, expr=0, t=0.0, crop=b"jpg"):
    return fi.FaceObs(vec=v, quality=q, size=size, yaw=yaw, pitch=0.0, det=0.8, sharp=150.0, liveness=live,
                      expression=expr, t=t, crop=crop)


def test_quality_weighting_beats_plain_mean():
    u = unit()
    agg = fi.FaceAggregator()
    vs = []
    for i in range(6):                                # good faces: big, frontal, sharp, high quality, little noise
        v = noisy(u, 0.6)
        vs.append(v)
        agg.add(obs(v, q=0.7, size=100, t=i))
    for i in range(12):                               # many poor ones: small, turned, low quality, very noisy
        v = noisy(u, 3.0)
        vs.append(v)
        agg.add(obs(v, q=0.05, size=28, yaw=60, t=10 + i))
    r = agg.result()
    plain = fi.l2n(np.mean(vs, 0))
    assert float(np.dot(r["vec"], u)) > float(np.dot(plain, u)) + 0.03, (np.dot(r["vec"], u), np.dot(plain, u))


def test_outlier_second_person_rejected():
    a, b = unit(), unit()
    agg = fi.FaceAggregator()
    for i in range(10):
        agg.add(obs(noisy(a, 0.8), q=0.5, t=i))
    for i in range(3):                                # someone else walks past, even with good quality
        agg.add(obs(noisy(b, 0.8), q=0.6, t=20 + i))
    r = agg.result()
    assert r["n_rejected"] == 3 and r["n_faces"] == 10, r
    assert float(np.dot(r["vec"], a)) > 0.85 and float(np.dot(r["vec"], b)) < 0.2
    assert r["consistency"] > 0.6
    assert all(not f["inlier"] for f in r["faces"] if f["t"] >= 20)


def test_pool_is_bounded_and_keeps_few_crops():
    u = unit()
    p = fi.AggParams(max_faces=8, keep_crops=3)
    agg = fi.FaceAggregator(p)
    for i in range(100):
        agg.add(obs(noisy(u, 1.0), q=float(RNG.uniform(0.05, 0.9)), t=i))
    assert len(agg.pool) == 8
    assert sum(f.crop is not None for _, f in agg.pool) <= 3
    ws = [w for w, _ in agg.pool]
    assert ws == sorted(ws, reverse=True)
    r = agg.result()
    assert 1 <= len(r["crops_b64"]) <= 3 and r["best_crop_b64"] == r["crops_b64"][0]
    assert agg.seen == 100


def test_no_face_and_too_small():
    r = fi.FaceAggregator().result()
    assert r["vec"] is None and r["reason"] == "no_face" and r["n_faces"] == 0
    agg = fi.FaceAggregator()
    agg.add(obs(unit(), size=12))
    r = agg.result()
    assert r["vec"] is None and r["reason"] == "too_small" and r["max_size_px"] == 12


def test_liveness_summary():
    u = unit()
    agg = fi.FaceAggregator()
    for i in range(5):
        agg.add(obs(noisy(u, 0.5), live=0.95, t=i))
    assert agg.result()["liveness"]["ok"] is True
    agg = fi.FaceAggregator()
    for i in range(5):                                # a printed photo: liveness low on most frames
        agg.add(obs(noisy(u, 0.5), live=0.1 if i else 0.9, t=i))
    lv = agg.result()["liveness"]
    assert lv["ok"] is False and lv["pass_frac"] < 0.6
    agg = fi.FaceAggregator()
    agg.add(obs(u, live=None))
    assert agg.result()["liveness"]["ok"] is False   # not measured = not passed


def test_apparent_is_description_only_and_has_no_race():
    u = unit()
    agg = fi.FaceAggregator()
    for i in range(4):
        o = obs(noisy(u, 0.5), expr=1 if i < 3 else 0, t=i)
        o.age_bracket, o.gender = 3, 1
        agg.add(o)
    ap = agg.result()["apparent"]
    assert ap["expression"] == "happy" and "apparent expression" in ap["expression_note"]
    assert ap["age"]["low"] == 20 and ap["gender"]["value"] == "male"
    flat = repr(agg.result()).lower()
    assert "race" not in flat and "ethnic" not in flat


def test_pairwise_combiner_interface():
    vs = np.stack([unit() for _ in range(5)])
    plain = fi.PairwiseCombiner()(vs)
    pm = fi.l2n(np.mean([fi.l2n(vs[0] + vs[1]), fi.l2n(vs[2] + vs[3]), vs[4]], 0))   # each pair output is L2N
    assert np.allclose(plain, pm, atol=1e-5)
    zero = fi.PairwiseCombiner(delta=lambda pair: np.zeros(D, np.float32))(vs)
    assert np.allclose(zero, plain, atol=1e-5)
    u = unit()
    agg = fi.FaceAggregator(combiner=fi.PairwiseCombiner())
    for i in range(6):
        agg.add(obs(noisy(u, 0.5), t=i))
    assert abs(np.linalg.norm(agg.result()["vec"]) - 1) < 1e-3


def test_face_weight_orders_sensibly():
    p = fi.AggParams()
    v = unit()
    good = fi.face_weight(obs(v, q=0.6, size=100, yaw=5), p)
    turned = fi.face_weight(obs(v, q=0.6, size=100, yaw=70), p)
    small = fi.face_weight(obs(v, q=0.6, size=26, yaw=5), p)
    assert good > turned and good > small


def test_head_window_and_yaw():
    kp = np.zeros((17, 3), np.float32)
    kp[:5] = [[500, 200, .9], [490, 190, .9], [510, 190, .9], [480, 195, .9], [520, 195, .9]]
    w = fi.head_window((440, 170, 560, 600), kp, (1080, 1920, 3))
    assert w[0] < 480 and w[2] > 520 and w[1] < 190 and w[3] > 200
    w2 = fi.head_window((0, 0, 100, 400), None, (1080, 1920, 3))
    assert w2[0] == 0 and w2[1] == 0
    pts = np.array([[40, 50], [70, 50], [55, 65], [45, 80], [65, 80]], np.float32)
    assert abs(fi.landmark_yaw(pts)) < 1
    pts[2, 0] = 68
    assert fi.landmark_yaw(pts) > 20


class FakeEngine:
    """Stands in for FaceEngine: one face per call, same identity."""
    pack = "Fake"

    def __init__(self, u):
        self.u, self.calls = u, 0

    def faces(self, img, offset=(0, 0), scale=1.0, t=0.0, want_crop=None, frame=None):
        self.calls += 1
        o = obs(noisy(self.u, 0.5), t=t, crop=None)
        if want_crop is None or want_crop(o):
            o.crop = b"jpg"
        return [o]


class P:
    def __init__(self, box, kp=None):
        self.box, self.kp = box, kp


def test_sampler_budget_and_skips():
    u = unit()
    eng = FakeEngine(u)
    s = fi.FaceSampler(eng, every=2, max_attempts=5, max_cpu_s=100)
    fr = np.zeros((720, 1280, 3), np.uint8)
    for i in range(40):
        s.on_frame(fr, P((100, 100, 300, 700)) if i != 3 else None)
    assert eng.calls == 5 and s.attempts == 5
    r = s.result()
    assert r["n_faces"] == 5 and r["model"] == "Fake" and float(np.dot(r["vec"], u)) > 0.8
    s2 = fi.FaceSampler(FakeEngine(u))
    s2.on_frame(fr, P((100, 100, 120, 160)))       # tiny distant person: no detector call
    assert s2.attempts == 0 and s2.result()["reason"] == "too_small"


def test_engine_discards_race_with_mocked_inspireface():
    """FaceEngine against a fake `inspireface` module: race is never read into FaceObs, age/gender only when
    attributes are switched on."""
    calls = {}

    class Ext:
        rgb_liveness_confidence, quality_confidence, emotion, race, gender, age_bracket = 0.97, 0.6, 1, 2, 0, 4

    class Face:
        location, yaw, pitch, roll, detection_confidence = (10, 10, 70, 80), 3.0, 1.0, 0.0, 0.9

    class Sess:
        def __init__(self, opt, *a, **k):
            calls["opt"] = opt

        def set_detection_confidence_threshold(self, v): pass
        def set_filter_minimum_face_pixel_size(self, v): pass
        def face_detection(self, img): return [Face()]
        def face_pipeline(self, img, fs, opt):
            calls["ext"] = [Ext() for _ in fs]
            return calls["ext"]
        def face_feature_extract(self, img, f): return np.ones(D, np.float32)

    fake = types.SimpleNamespace(
        HF_ENABLE_FACE_RECOGNITION=1, HF_ENABLE_QUALITY=2, HF_ENABLE_FACE_POSE=4, HF_ENABLE_LIVENESS=8,
        HF_ENABLE_FACE_EMOTION=16, HF_ENABLE_FACE_ATTRIBUTE=32, HF_DETECT_MODE_ALWAYS_DETECT=0, HF_LOG_ERROR=0,
        set_logging_level=lambda l: None, launch=lambda *a, **k: True, InspireFaceSession=Sess)
    old = sys.modules.get("inspireface")
    sys.modules["inspireface"] = fake
    fi._LAUNCHED = None
    try:
        eng = fi.FaceEngine("Fake")
        assert not calls["opt"] & 32                  # attribute network off by default
        img = np.full((100, 100, 3), 128, np.uint8)
        o = eng.faces(img, offset=(200, 300), frame=np.zeros((720, 1280, 3), np.uint8))[0]
        assert o.gender is None and o.age_bracket is None and o.expression == 1
        assert not hasattr(o, "race") and calls["ext"][0].race == -1
        assert o.box == (210, 310, 270, 380) and o.size == 60
        assert abs(np.linalg.norm(o.vec) - np.sqrt(D)) < 1e-3   # raw from SDK; the aggregator normalises
        fi._LAUNCHED = None
        eng = fi.FaceEngine("Fake", attributes=True)
        o = eng.faces(img)[0]
        assert o.gender == 0 and o.age_bracket == 4 and calls["ext"][0].race == -1
    finally:
        fi._LAUNCHED = None
        if old is None:
            sys.modules.pop("inspireface", None)
        else:
            sys.modules["inspireface"] = old


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            print("ok", k)

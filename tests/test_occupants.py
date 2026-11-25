"""Vehicle occupant faces (squirrel_guard/occupants.py) with a mocked face engine: cabin tiling, per-occupant
grouping inside one visit, pedestrian exclusion, what the template carries.

Run:  .venv/bin/python tests/test_occupants.py      (or pytest). No model files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squirrel_guard import faceid as fi  # noqa: E402
from squirrel_guard import occupants as oc  # noqa: E402

RNG = np.random.default_rng(17)


def unit():
    return fi.l2n(RNG.normal(size=fi.EMB_DIM))


def noisy(u, s=0.5):
    return fi.l2n(u + s * RNG.normal(size=fi.EMB_DIM) / np.sqrt(fi.EMB_DIM))


class FakeEngine:
    """Returns the faces placed in the frame (frame-px boxes) that fall inside the tile it is given."""
    pack = "Megatron"

    def __init__(self):
        self.faces_in_frame = []        # (box, identity vector)
        self.calls = []

    def faces(self, img, offset=(0, 0), scale=1.0, t=0.0, want_crop=None, frame=None, largest_only=True):
        self.calls.append((offset, img.shape, scale))
        X1, Y1 = offset
        X2, Y2 = X1 + img.shape[1] / scale, Y1 + img.shape[0] / scale
        out = []
        for box, u in self.faces_in_frame:
            if box[0] >= X1 and box[1] >= Y1 and box[2] <= X2 and box[3] <= Y2:
                out.append(fi.FaceObs(vec=noisy(u), quality=0.4, size=float(min(box[2] - box[0], box[3] - box[1])),
                                      det=0.8, sharp=120.0, liveness=0.9, expression=1, t=t, box=tuple(box),
                                      crop=b"jpg"))
        return out


def test_cabin_region_and_tiles():
    p = oc.OccParams()
    assert oc.cabin_region((0, 0, 50, 50), (1440, 2560), p) is None            # too small to resolve occupants
    r = oc.cabin_region((100, 300, 1300, 900), (1440, 2560), p)
    assert r[1] < 300 and r[3] == int(300 + 0.7 * 600)                           # cabin = upper part of the box
    ts = oc.tiles(r, p)
    assert 1 < len(ts) <= p.max_tiles
    xs = sorted({t[0] for t in ts}); ys = sorted({t[1] for t in ts})
    assert xs[0] == r[0] and max(t[2] for t in ts) == r[2] and ys[0] == r[1] and max(t[3] for t in ts) == r[3]
    big = oc.tiles((0, 0, 2560, 1440), p)                                        # huge region: tiles grow, count capped
    assert len(big) <= p.max_tiles
    assert oc.tiles((10, 10, 200, 150), p) == [(10, 10, 200, 150)]


def test_two_occupants_grouped_and_upscaled():
    eng = FakeEngine()
    driver, passenger = unit(), unit()
    vbox = (400, 400, 1600, 1000)
    eng.faces_in_frame = [((600, 450, 650, 500), driver), ((1300, 460, 1345, 505), passenger)]
    s = oc.OccupantSampler(eng, fi.AggParams(min_px=24), oc.OccParams())
    for t in range(6):
        s.on_frame(np.zeros((1440, 2560, 3), np.uint8), vbox, t=t)
    r = s.result()
    assert r["occupants_detected"] == 2 and len(r["occupant_faces"]) == 2
    assert all(c[2] >= 2.0 - 1e-6 for c in eng.calls)                           # 320 px tiles -> 640 detector input
    sims = sorted(float(np.dot(o["vec"], driver)) for o in r["occupant_faces"])
    assert sims[-1] > 0.8 and sims[0] < 0.3                                      # one template per occupant
    for o in r["occupant_faces"]:
        assert o["n_faces"] == 6 and "faces" not in o and "crops_b64" not in o and o["apparent"] == {}
        assert o["model"] == "Megatron" and o["best_crop_b64"]                   # one crop, for an enrolled match only
    d = max(r["occupant_faces"], key=lambda o: float(np.dot(o["vec"], driver)))
    assert d["rel_x"] < 0.3                                                      # left of the vehicle box


def test_pedestrian_next_to_vehicle_is_not_an_occupant():
    eng = FakeEngine()
    walker = unit()
    vbox = (400, 400, 1600, 1000)
    eng.faces_in_frame = [((700, 380, 760, 440), walker)]
    person = (650, 350, 820, 1100)                                               # standing: reaches the ground
    s = oc.OccupantSampler(eng, fi.AggParams(min_px=24), oc.OccParams())
    for t in range(4):
        s.on_frame(np.zeros((1440, 2560, 3), np.uint8), vbox, t=t, people=[person])
    r = s.result()
    assert r["occupants_detected"] == 0 and r["pedestrian_faces"] == 4 and r["occupant_faces"] == []
    seated = (650, 350, 820, 700)                                                 # a person box ending in the cabin
    s = oc.OccupantSampler(eng, fi.AggParams(min_px=24), oc.OccParams())
    s.on_frame(np.zeros((1440, 2560, 3), np.uint8), vbox, people=[seated])
    assert s.result()["occupants_detected"] == 1


def test_tiny_faces_and_budget():
    eng = FakeEngine()
    eng.faces_in_frame = [((600, 450, 612, 462), unit())]                         # 12 px: below min_px
    s = oc.OccupantSampler(eng, fi.AggParams(min_px=24), oc.OccParams())
    s.on_frame(np.zeros((1440, 2560, 3), np.uint8), (400, 400, 1600, 1000))
    r = s.result()
    assert r["occupants_detected"] == 0 and r["faces_seen"] == 1 and r["reason"] == "too_small"
    s = oc.OccupantSampler(eng, fi.AggParams(), oc.OccParams(max_cpu_s=0.0))
    s.on_frame(np.zeros((1440, 2560, 3), np.uint8), (400, 400, 1600, 1000))
    assert s.result()["tiles"] == 0


def test_dedupe_overlapping_tiles():
    a = fi.FaceObs(vec=unit(), quality=0.5, size=40, det=0.9, box=(10, 10, 50, 50))
    b = fi.FaceObs(vec=unit(), quality=0.5, size=38, det=0.8, box=(12, 11, 50, 49))
    c = fi.FaceObs(vec=unit(), quality=0.5, size=40, det=0.9, box=(200, 10, 240, 50))
    assert oc.dedupe([b, a, c]) == [a, c]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            print("ok", k)

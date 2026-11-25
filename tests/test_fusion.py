"""squirrel_guard.fusion on synthetic data: a known walker (tests/test_rig.py) seen by three calibrated cameras whose
clips start at different moments -> the time offsets are recovered (to a fraction of a frame), a camera that moved
after its calibration or saw nothing is dropped, one rig over all views recovers the stature, and the live planning
(which Frigate events overlap, which window, recording holes) behaves.

Run:  .venv/bin/python tests/test_fusion.py      (or pytest)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squirrel_guard import fusion, rig  # noqa: E402
from squirrel_guard.geometry import Camera  # noqa: E402
from tests.test_rig import _cam  # noqa: E402

FPS = 12.0
CAMS = {"a": [0.0, -1.0, 2.4], "b": [3.0, 0.5, 2.0], "c": [-2.5, 1.0, 1.2]}


def _walker(n, dt, H=1.78, t0=0.0):
    import torch
    t = t0 + np.arange(n) * dt
    root = np.c_[0.9 * (t - t0) - 2.0, 4.0 + 0.1 * np.sin(t), np.full(n, 0.53 * H)]
    yaw = np.full(n, -math.pi / 2)
    ang = np.zeros((n, len(rig.ANGLES)))
    ph = 2 * math.pi * t / 1.1
    for s, off in (("L", 0.0), ("R", math.pi)):
        ang[:, rig.ANG_IDX[("hip_flex", s)]] = np.radians(25) * np.sin(ph + off)
        ang[:, rig.ANG_IDX[("knee", s)]] = np.radians(30) * (1 + np.sin(ph + off - 1.2))
        ang[:, rig.ANG_IDX[("elbow", s)]] = np.radians(15)
    seg = {k: torch.tensor(rig.RATIO[k] * H, dtype=torch.float64) for k in rig.SEGS}
    X = rig.forward(torch.tensor(root), torch.tensor(yaw), torch.tensor(ang), seg, torch.tensor(H, dtype=torch.float64))
    return X.numpy()


def _view(name, cam, X, shift, n, rng, noise=1.5, visible=None):
    """Keypoints of camera `name` whose clip runs `shift` frames AHEAD: its frame i + shift = truth frame i, i.e.
    its frame j shows truth frame j - shift. X covers truth frames -pad .. n + pad."""
    import torch
    pad = (len(X) - n) // 2
    kp = np.full((n, 133, 3), np.nan)
    box = np.full((n, 4), np.nan)
    for j in range(n):
        ti = j - shift + pad
        if not (0 <= ti < len(X)) or (visible is not None and not visible[j]):
            continue
        uv, _ = rig.project(torch.tensor(X[ti]), cam, cam.size)
        uv = uv.numpy() + rng.normal(0, noise, (23, 2))
        kp[j, :23, :2] = uv
        kp[j, :23, 2] = np.where(rng.random(23) < 0.1, 0.0, 0.9)
        box[j] = [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
    return {"name": name, "cam": cam, "size": cam.size, "kp": kp, "box": box, "picks": [None] * n}


def _scene(shifts, n=72, seed=0, moved=None, empty=None):
    rng = np.random.default_rng(seed)
    pad = 40
    X = _walker(n + 2 * pad, 1 / FPS, t0=-pad / FPS)
    views = []
    for name, C in CAMS.items():
        cam = _cam(C, [0.0, 4.0, 0.9])
        cam_used = cam
        if moved == name:                      # the camera turned 8 deg after its calibration: keypoints from the
            yaw = math.radians(8)              # real (turned) camera, fused with the stale calibration
            Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            cam_used = Camera(cam.K, cam.D, Rz @ cam.R, cam.C, cam.size)
        v = _view(name, cam_used, X, shifts.get(name, 0), n, rng,
                  visible=np.zeros(n, bool) if empty == name else None)
        v["cam"] = cam                         # what the fusion believes
        views.append(v)
    return views, X[pad:pad + n]


def test_closest_meets_on_intersecting_rays():
    C1, C2 = np.array([0.0, 0, 0]), np.array([2.0, 0, 0])
    P = np.array([[1.0, 3.0, 1.0]])
    d1 = (P - C1) / np.linalg.norm(P - C1)
    d2 = (P - C2) / np.linalg.norm(P - C2)
    X, g = fusion.closest(C1, d1, C2, d2)
    assert np.allclose(X, P, atol=1e-9) and g[0] < 1e-9
    _, g = fusion.closest(C1, -d1, C2, d2)        # behind camera 1
    assert np.isnan(g[0])


def test_shift_series_fractional():
    a = np.arange(10, dtype=float)
    s = fusion.shift_series(a, 2.5)
    assert abs(s[0] - 2.5) < 1e-9 and abs(s[6] - 8.5) < 1e-9 and np.isnan(s[9])
    s = fusion.shift_series(a, -1)
    assert np.isnan(s[0]) and s[1] == 0


def test_synchronise_recovers_offsets():
    views, _ = _scene({"b": 7, "c": -11})
    shifts, rep = fusion.synchronise(views, FPS, max_offset_s=3.0)
    assert set(shifts) == {"a", "b", "c"}, rep
    rel = {k: v - shifts["a"] for k, v in shifts.items()}       # shifts come on the anchor camera's time base
    assert abs(rel["b"] - 7) < 0.35, rel
    assert abs(rel["c"] + 11) < 0.35, rel
    anchors = [k for k, r in rep.items() if r["role"] == "anchor"]
    assert len(anchors) == 2 and rep[anchors[1]]["median_ray_gap_m"] < 0.03, rep
    third = next(k for k, r in rep.items() if r["role"] == "secondary")
    assert rep[third]["reproj_err_box"] < 0.02, rep


def test_synchronise_fractional_offset():
    """A secondary 2.5 frames ahead (clip cut mid-frame relative to the primary): parabolic refinement."""
    import torch
    rng = np.random.default_rng(1)
    n, pad, sub = 72, 40, 2                          # truth at 24 fps, cameras sample every 2nd -> 12 fps
    X = _walker((n + 2 * pad) * sub, 1 / (FPS * sub), t0=-pad / FPS)
    views = []
    for name, off in (("a", 0), ("b", 5)):           # b starts 5 truth frames = 2.5 camera frames later
        cam = _cam(CAMS[name], [0.0, 4.0, 0.9])
        kp = np.full((n, 133, 3), np.nan)
        box = np.full((n, 4), np.nan)
        for j in range(n):
            ti = (j + pad) * sub + off
            uv, _ = rig.project(torch.tensor(X[ti]), cam, cam.size)
            uv = uv.numpy() + rng.normal(0, 1.0, (23, 2))
            kp[j, :23, :2], kp[j, :23, 2] = uv, 0.9
            box[j] = [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
        views.append({"name": name, "cam": cam, "size": cam.size, "kp": kp, "box": box})
    shifts, _ = fusion.synchronise(views, FPS)
    assert abs((shifts["b"] - shifts["a"]) + 2.5) < 0.4, shifts     # b's frame j shows truth j + 2.5 -> b[i - 2.5] ~ a[i]


def test_moved_camera_is_dropped():
    """An 8 deg turn: paired with ONE other camera a wrong time shift still gives an 11 cm ray gap while the person
    walks, but it cannot reproject onto the other two cameras' triangulation."""
    for moved in ("b", "c"):
        views, _ = _scene({"b": 3}, moved=moved)
        shifts, rep = fusion.synchronise(views, FPS, max_gap_m=0.15)
        assert moved not in shifts and rep[moved]["role"] == "dropped" and "moved" in rep[moved]["reason"], rep
        assert len(shifts) == 2


def test_two_cameras_far_apart_are_not_fused():
    views, _ = _scene({}, moved="b")
    two = [v for v in views if v["name"] in ("a", "b")]
    for v in two:                               # a gross move (as on 2026-11-08: v4 tilted from the ceiling to level)
        if v["name"] == "b":
            yaw = math.radians(35)
            Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            v["cam"] = Camera(v["cam"].K, v["cam"].D, Rz @ v["cam"].R, v["cam"].C, v["cam"].size)
    fz = fusion.fuse_views(two, FPS, iters=(50, 50))
    assert not fz["ok"] and all(r["role"] == "dropped" for r in fz["report"].values()), fz["report"]


def test_empty_camera_is_dropped_and_fusion_fits():
    views, X = _scene({"b": 4}, empty="c")
    fz = fusion.fuse_views(views, FPS)
    assert fz["ok"], fz.get("reason")
    assert fz["report"]["c"]["role"] == "dropped"
    assert sorted(v["name"] for v in fz["aligned"]) == ["a", "b"]
    assert abs(fz["shifts"]["b"] - fz["shifts"]["a"] - 4) < 0.35, fz["shifts"]
    assert abs(fz["res"]["stature_m"] - 1.78) / 1.78 < 0.05, fz["res"]["stature_m"]
    err = np.linalg.norm(fz["res"]["markers"][:, :17] - X[:, :17], axis=-1)
    assert np.nanmedian(err) < 0.08, np.nanmedian(err)


def test_heldout_camera_error_small_for_good_fit():
    views, _ = _scene({"b": 0, "c": 0})
    fz = fusion.fuse_views(views[:2], FPS, iters=(150, 400))
    held = fusion.align(views[2], 0.0, len(views[2]["kp"]))
    e = fusion.view_error(fz["res"], held)
    assert e["median_err_box"] is not None and e["median_err_box"] < 0.03, e


def test_plan_overlap_wait_and_window():
    trig = {"id": "t", "camera": "v4", "start_time": 100.0, "end_time": 130.0, "label": "person"}
    evs = [trig,
           {"id": "f1", "camera": "frontyard", "start_time": 120.0, "end_time": 140.0, "label": "person"},
           {"id": "x", "camera": "garden", "start_time": 101.0, "end_time": 120.0, "label": "person"},   # other frame
           {"id": "old", "camera": "testcam", "start_time": 10.0, "end_time": 20.0, "label": "person"}]
    p = fusion.plan(trig, evs, {"v4", "frontyard", "testcam"}, pad_s=1.0, max_s=20.0)
    assert p["status"] == "fuse" and {e["id"] for e in p["members"]} == {"t", "f1"}
    w0, w1 = p["window"]
    assert abs((w1 - w0) - 20.0) < 1e-6                        # union 99..141 capped at 20 s ...
    assert w0 <= 120.0 and w1 >= 130.0                         # ... around the part both cameras saw (120..130)
    evs.append({"id": "live", "camera": "testcam", "start_time": 125.0, "end_time": None, "label": "person"})
    assert fusion.plan(trig, evs, {"v4", "frontyard", "testcam"})["status"] == "wait"
    lone = fusion.plan(trig, [trig], {"v4", "frontyard"})
    assert lone["status"] == "none"


def test_coverage_skips_holes():
    segs = [{"start_time": 90, "end_time": 100}, {"start_time": 100, "end_time": 110},
            {"start_time": 115, "end_time": 140}]
    assert fusion.coverage(segs, 95, 135) == (115, 135)        # longest contiguous run inside the window
    assert fusion.coverage([], 0, 10) is None


def test_frame_id():
    cam = _cam([0, 0, 2], [0, 3, 1])
    assert fusion.frame_id(cam) is None
    cam.meta = {"method": "multicam-vggt", "date": "2026-11-07"}
    assert fusion.frame_id(cam) == "multicam-vggt:2026-11-07"
    cam.meta["frame"] = "basement-3cam-0244"
    assert fusion.frame_id(cam) == "basement-3cam-0244"
    cam.meta = {"method": "walk", "date": "2026-11-07"}
    assert fusion.frame_id(cam) is None


def test_live_fusion_candidates():
    cams = {n: _cam(C, [0.0, 4.0, 0.9]) for n, C in CAMS.items()}
    for c in cams.values():
        c.meta = {"method": "multicam-vggt", "date": "2026-11-07"}
    evs = [{"id": "e1", "camera": "a", "start_time": 100.0, "end_time": 110.0},
           {"id": "e2", "camera": "b", "start_time": 104.0, "end_time": 112.0}]
    lf = fusion.LiveFusion(lambda path: evs if path.startswith("/api/events") else [], lambda p: None,
                           cams.get, list(cams))
    p = lf.candidates(evs[0])
    assert p["status"] == "fuse" and {e["id"] for e in p["members"]} == {"e1", "e2"}
    lf.mark(p["members"], "e1_fused")
    assert lf.candidates(evs[1])["status"] == "none"            # the second event's own trigger: already fused


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            print("ok", k)

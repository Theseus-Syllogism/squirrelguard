"""Head direction (squirrel_guard/gaze.py) on synthetic walks with known head yaw.

Run:  .venv/bin/python tests/test_gaze.py      (or pytest)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_walk as sw  # noqa: E402
from squirrel_guard import gaze, trajectory as tj  # noqa: E402
from squirrel_guard.pose import Pose  # noqa: E402

FPS = 8.0


def _track(path, cam, head_turn_deg=0.0, turn_window=None, seed=0):
    """Like synth_walk.synth_track, but the head can be turned relative to the body inside turn_window (s)."""
    rng = np.random.default_rng(seed)
    W, H = cam.size
    trk, truth = [], []
    for i in range(int(path.duration * FPS)):
        t = i / FPS
        pos, hd = path.at(t), path.heading(t)
        J, extra = sw.joints3d(pos, hd, 2 * math.pi * t, path.speed(t))
        turn = head_turn_deg if (turn_window and turn_window[0] <= t < turn_window[1]) else 0.0
        if turn:
            c = (J[3] + J[4]) / 2                                   # rotate the 5 head points about the ear midpoint
            a = math.radians(turn)
            Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            J[:5] = (J[:5] - c) @ Rz.T + c
        truth.append(math.degrees(hd) + turn)
        px = sw.project(cam, np.vstack([J, extra]))
        if not np.isfinite(px).all() or (px[:, 0] < 0).any() or (px[:, 0] >= W).any() or (px[:, 1] < 0).any() or (px[:, 1] >= H).any():
            trk.append(None)
            continue
        # visibility: head points facing away from the camera get low confidence, like a real detector
        hdir = math.radians(truth[-1])
        f = np.array([math.cos(hdir), math.sin(hdir), 0])
        l = np.array([-math.sin(hdir), math.cos(hdir), 0])
        to_cam = cam.C - J[0]
        to_cam /= np.linalg.norm(to_cam)
        normals = [f, f + 0.3 * l, f - 0.3 * l, l, -l]
        conf = np.full(17, 0.9)
        for j, nv in enumerate(normals):
            conf[j] = 0.9 if nv @ to_cam > -0.1 else 0.05
        kp = np.c_[px[:17] + rng.normal(0, 1.0, (17, 2)), conf]
        x1, y1 = px.min(0)
        x2, y2 = px.max(0)
        trk.append(Pose((float(x1), float(y1), float(x2), float(y2)), 0.9, kp))
    return trk, np.array(truth)


def _err(est, truth, f0):
    e = []
    for k, y in enumerate(est["yaw"]):
        if y is not None and (est["conf"][k] or 0) >= 0.4:
            e.append(abs((y - truth[f0 + k] + 180) % 360 - 180))
    return np.array(e)


def test_head_follows_walking_direction():
    cam = sw.wyze_camera()
    trk, truth = _track(sw.curve_with_stop(), cam)
    tr = tj.estimate(trk, FPS, sw.SIZE, cam)
    g = gaze.estimate(trk, tr, cam, sw.SIZE)
    e = _err(g, truth, tr["frame0"])
    print(f"  walking: {len(e)} frames read, median error {np.median(e):.0f} deg, p90 {np.percentile(e, 90):.0f} deg")
    assert len(e) > 20 and np.median(e) < 30


def test_head_turned_while_walking():
    cam = sw.wyze_camera()
    trk, truth = _track(sw.straight(), cam, head_turn_deg=70, turn_window=(2.5, 4.5))
    tr = tj.estimate(trk, FPS, sw.SIZE, cam)
    g = gaze.estimate(trk, tr, cam, sw.SIZE)
    f0 = tr["frame0"]
    inside = [k for k in range(len(g["yaw"])) if 2.8 <= tr["t"][k] < 4.2 and g["yaw"][k] is not None]
    e = np.array([abs((g["yaw"][k] - truth[f0 + k] + 180) % 360 - 180) for k in inside])
    print(f"  head turned 70 deg: {len(e)} frames, median error {np.median(e):.0f} deg; "
          f"off-path share {g['summary']['looking_off_path_share']}")
    assert len(e) >= 4 and np.median(e) < 35
    assert (g["summary"]["looking_off_path_share"] or 0) > 0


if __name__ == "__main__":
    for f in [v for n, v in list(globals().items()) if n.startswith("test_")]:
        f()
        print("ok", f.__name__)

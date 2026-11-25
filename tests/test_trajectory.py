"""Trajectory model against synthetic walks with known ground truth.

Run:  .venv/bin/python -m pytest tests/test_trajectory.py      (pytest optional)
      .venv/bin/python tests/test_trajectory.py                (plain asserts, prints the numbers)
Numbers (and the sample.json for the viewer): scripts/trajectory_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import synth_walk as sw  # noqa: E402
from squirrel_guard import trajectory as tj  # noqa: E402

FPS = 8.0
OCCLUSION = (6.0, 1.5)
GAIT = {"walking": True, "usable": True, "cadence_spm": 112.0, "speed_mps": 1.2, "step_m": 0.64}
CHI2_95 = 5.991


def evaluate(path, seed=0, gait=GAIT, occlusion=OCCLUSION):
    """-> dict of error numbers for one synthetic walk."""
    cam = sw.wyze_camera()
    trk, truth = sw.synth_track(path, cam, fps=FPS, occlusion=occlusion, seed=seed)
    res = tj.estimate(trk, FPS, sw.SIZE, cam, gait=gait)
    assert res is not None
    i0, n = res["frame0"], len(res["t"])
    t = np.array(res["t"])
    e = np.array(res["mean"]) - truth["xy"][i0:i0 + n]
    C = np.array(res["cov"])
    d2 = np.einsum("ni,nij,nj->n", e, np.linalg.inv(C), e)
    occ = (t >= occlusion[0]) & (t < occlusion[0] + occlusion[1]) if occlusion else np.zeros(n, bool)
    out = {"frames": n, "observed": int(sum(res["observed"])),
           "rmse_m": float(np.sqrt((e ** 2).sum(1).mean())),
           "rmse_occlusion_m": float(np.sqrt((e[occ] ** 2).sum(1).mean())) if occ.any() else None,
           "coverage95": float((d2 <= CHI2_95).mean()),
           "coverage95_occlusion": float((d2[occ] <= CHI2_95).mean()) if occ.any() else None,
           "speed_mae_mps": float(np.abs(np.array(res["speed"]) - truth["speed"][i0:i0 + n]).mean()),
           "footprints": len(res["footprints"]), "true_steps": sw.true_steps(path, GAIT["cadence_spm"]),
           "metrics": res["metrics"], "res": res}
    fp = res["footprints"]
    if len(fp) > 2:
        d = [np.hypot(a["x"] - b["x"], a["y"] - b["y"]) for a, b in zip(fp, fp[1:]) if b["t"] - a["t"] < 1.0]
        # consecutive footprints are on opposite sides: expected spacing sqrt(step^2 + (2 lateral)^2)
        step = GAIT["speed_mps"] * 60 / GAIT["cadence_spm"]
        out["footprint_spacing_m"] = float(np.median(d))
        out["footprint_spacing_expected_m"] = float(np.hypot(step, 2 * tj.LATERAL_M))
    pt, pm, pc = (np.array(res["prediction"][k]) for k in ("t", "mean", "cov"))
    for lead in (1.0, 2.0):
        k = int(np.argmin(np.abs(pt - (t[-1] + lead))))
        err = pm[k] - path.at(pt[k])
        out[f"pred_{lead:.0f}s_err_m"] = float(np.linalg.norm(err))
        out[f"pred_{lead:.0f}s_inside95"] = bool(err @ np.linalg.inv(pc[k]) @ err <= CHI2_95)
        out[f"pred_{lead:.0f}s_sd_m"] = float(np.sqrt(np.trace(pc[k]) / 2))
    return out


def test_straight_walk():
    r = evaluate(sw.straight(), occlusion=None)
    assert r["rmse_m"] < 0.25
    assert 0.85 <= r["coverage95"] <= 1.0
    assert r["speed_mae_mps"] < 0.15
    assert abs(r["footprints"] - r["true_steps"]) <= 3
    assert r["metrics"]["stops"] == []
    assert abs(r["metrics"]["heading_entry_deg"] - (-32.0)) < 10   # (-4, 9) -> (8, 1.5)
    assert r["pred_1s_err_m"] < 0.6 and r["pred_1s_inside95"]
    assert r["pred_2s_inside95"]


def test_curve_with_stop_and_occlusion():
    r = evaluate(sw.curve_with_stop())
    assert r["rmse_m"] < 0.3
    assert r["rmse_occlusion_m"] < 0.6                 # occluded while standing still: prediction holds
    assert 0.85 <= r["coverage95"] <= 1.0
    assert r["coverage95_occlusion"] >= 0.8
    stops = r["metrics"]["stops"]
    assert len(stops) == 1 and abs(stops[0]["t0"] - 4.5) < 1.0 and abs(stops[0]["t1"] - 7.5) < 1.0
    assert abs(r["footprints"] - r["true_steps"]) <= 4
    assert abs(r["footprint_spacing_m"] - r["footprint_spacing_expected_m"]) < 0.15
    assert r["pred_2s_inside95"]


def test_moving_occlusion_grows_uncertainty():
    r = evaluate(sw.straight(duration=9.0), occlusion=(3.0, 1.5))
    res, t = r["res"], np.array(r["res"]["t"])
    sd = np.sqrt(np.array(res["cov"])[:, 0, 0] + np.array(res["cov"])[:, 1, 1])
    occ = (t >= 3.0) & (t < 4.5)
    assert sd[occ].max() > 1.5 * np.median(sd[~occ])  # the gap is visibly less certain
    assert r["rmse_occlusion_m"] < 0.5
    assert r["coverage95_occlusion"] >= 0.8


def test_leaving_the_view():
    r = evaluate(sw.leaving())
    assert r["rmse_m"] < 0.3
    assert r["coverage95"] >= 0.8                      # weak spot: near the camera the leading-foot bias shows
    assert r["metrics"]["path_length_m"] > 8
    assert r["pred_1s_inside95"] and r["pred_2s_inside95"]


def test_measurement_covariance_grows_with_range():
    cam = sw.wyze_camera()
    near = tj.measure(cam, (960, 1000), sw.SIZE)
    far = tj.measure(cam, (960, 240), sw.SIZE)        # ~10 m out
    assert near and far
    assert np.trace(far[1]) > 3 * np.trace(near[1])
    w, v = np.linalg.eigh(far[1])
    assert w[1] > 4 * w[0]                              # elongated (std ratio > 2) ...
    ray = far[0] - cam.C[:2]
    assert abs(v[:, 1] @ ray / np.linalg.norm(ray)) > 0.95   # ... along the viewing ray


def test_skeleton_and_json():
    import json
    cam = sw.wyze_camera()
    trk, truth = sw.synth_track(sw.straight(), cam, seed=1)
    sk = tj.skeleton3d(trk, cam, sw.SIZE)
    assert len(sk) == len(trk)
    frames = [s for s in sk if s]
    assert frames
    heads = [j[0][2] for s in frames for j in [s] if j[0]]
    assert 1.3 < np.median(heads) < 1.9              # nose height of a 1.75 m person
    res = tj.estimate(trk, FPS, sw.SIZE, cam, heights_px=True)
    hz = [h for h in res["head_z"] if h is not None]
    assert abs(np.median(hz) - 1.75) < 0.15
    json.dumps(res)                                    # serialisable
    assert tj.estimate([None] * 10, FPS, sw.SIZE, cam) is None


def test_head_plane_when_feet_are_out_of_frame():
    """Camera on a desk (0.8 m) looking UP: feet are below the frame, the head top places the person."""
    import math
    from squirrel_guard.pose import Pose
    from squirrel_guard.geometry import Camera, rot_from_angles
    cam = Camera(sw.nominal_K(*sw.SIZE, hfov_deg=100.0), np.zeros(5), rot_from_angles(-25.0, 0.0), [0, 0, 0.8], sw.SIZE)
    W, H = sw.SIZE
    path = sw.straight()
    rng = np.random.default_rng(3)
    trk, xy = [], []
    for i in range(int(path.duration * FPS)):
        t = i / FPS
        pos = path.at(t)
        J, extra = sw.joints3d(pos, path.heading(t), 2 * math.pi * t, path.speed(t))
        px = sw.project(cam, np.vstack([J, extra]))
        xy.append(pos)
        if not np.isfinite(px).all() or px[:, 1].min() < 0.02 * H or px[:, 0].min() < 0 or px[:, 0].max() >= W:
            trk.append(None)
            continue
        inside = px[:, 1] < H
        kp = np.c_[px[:17] + rng.normal(0, 3.0, (17, 2)), np.where(inside[:17], 0.9, 0.0)]
        x1, y1 = px[inside].min(0) + rng.normal(0, 3.0, 2)
        x2 = px[inside, 0].max() + rng.normal(0, 3.0)
        y2 = min(float(px[:, 1].max()), H - 1.0)
        trk.append(Pose((float(x1), float(y1), float(x2), float(y2)), 0.9, kp))
    seen = [p for p in trk if p is not None]
    assert seen and all(tj.foot_point(p, sw.SIZE) is None for p in seen if p.box[3] >= H - 2)
    res = tj.estimate(trk, FPS, sw.SIZE, cam, head_height=1.75)
    assert res is not None and res["metrics"]["frames_head_anchored"] >= 5
    f0 = res["frame0"]
    truth = np.array(xy)[f0:f0 + len(res["t"])]
    obs = np.array(res["observed"])
    err = np.linalg.norm(np.array(res["mean"])[obs] - truth[obs], axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    print(f"  head-plane: {res['metrics']['frames_head_anchored']} head-anchored frames, rmse {rmse:.3f} m")
    assert rmse < 0.35, rmse
    sk = tj.skeleton3d(trk, cam, sw.SIZE, head_height=1.75)
    assert any(sk)


if __name__ == "__main__":
    import json
    for f in [v for k, v in list(globals().items()) if k.startswith("test_")]:
        f()
        print("ok", f.__name__)
    for name, path, occ in [("straight", sw.straight(), None), ("curve+stop", sw.curve_with_stop(), OCCLUSION),
                            ("leaving", sw.leaving(), OCCLUSION)]:
        r = evaluate(path, occlusion=occ)
        r.pop("res"); r.pop("metrics")
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))

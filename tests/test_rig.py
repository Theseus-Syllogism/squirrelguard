"""rig.fit recovers a synthetic walker: known stature/segments/angles -> RTMW-like keypoints in two cameras (with
lens distortion, noise and dropped joints) -> fit -> stature within 4 %, segments within 10 %, joints within 6 cm,
knees never hyper-extended.

Run:  .venv/bin/python tests/test_rig.py      (or pytest)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squirrel_guard import rig  # noqa: E402
from squirrel_guard.geometry import Camera  # noqa: E402


def _cam(C, target, f=900.0, size=(1280, 720), k1=-0.25):
    C, t = np.array(C, float), np.array(target, float)
    z = t - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], 1)                       # world_from_camera (columns = camera axes)
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]])
    return Camera(K, [k1, 0.05, 0, 0, 0], R, C, size)


def _truth(n=60, dt=0.125):
    import torch
    H = 1.78
    t = np.arange(n) * dt
    root = np.c_[0.9 * t - 2.0, 4.0 + 0.1 * np.sin(t), np.full(n, 0.53 * H)]
    yaw = np.full(n, -math.pi / 2)                   # walking along +x (body forward = +y rotated by yaw)
    ang = np.zeros((n, len(rig.ANGLES)))
    ph = 2 * math.pi * t / 1.1                       # 1.1 s stride
    for s, sg in (("L", 1), ("R", -1)):
        ang[:, rig.ANG_IDX[("hip_flex", s)]] = np.radians(25) * np.sin(ph + (0 if sg > 0 else math.pi))
        ang[:, rig.ANG_IDX[("knee", s)]] = np.radians(30) * (1 + np.sin(ph + (0 if sg > 0 else math.pi) - 1.2))
        ang[:, rig.ANG_IDX[("sh_flex", s)]] = -np.radians(15) * np.sin(ph + (0 if sg > 0 else math.pi))
        ang[:, rig.ANG_IDX[("elbow", s)]] = np.radians(15)
    mult = {"thigh": 1.05, "forearm": 0.95, "sh_half": 1.08}
    seg = {k: torch.tensor(rig.RATIO[k] * H * mult.get(k, 1.0), dtype=torch.float64) for k in rig.SEGS}
    X = rig.forward(torch.tensor(root), torch.tensor(yaw), torch.tensor(ang), seg, torch.tensor(H, dtype=torch.float64))
    return X.numpy(), H, {k: float(v) for k, v in seg.items()}, dt


def test_fit_recovers_synthetic_walker():
    import torch
    X, H, seg, dt = _truth()
    n = len(X)
    rng = np.random.default_rng(0)
    views = []
    for C in ([0.0, -1.0, 2.4], [3.0, 0.5, 2.0]):
        cam = _cam(C, [0.0, 4.0, 0.9])
        uv, _ = rig.project(torch.tensor(X), cam, cam.size)
        uv = uv.numpy() + rng.normal(0, 2.0, uv.shape)
        conf = np.ones(uv.shape[:2])
        conf[rng.random(conf.shape) < 0.15] = 0.0     # 15 % of joints missing
        kp = np.concatenate([uv, conf[..., None]], -1)
        top = uv[..., 1].min(1)
        bot = uv[..., 1].max(1)
        views.append({"cam": cam, "size": cam.size, "kp": kp, "box_h": bot - top})
    init = X[:, [rig.L_HIP, rig.R_HIP], :2].mean(1) + rng.normal(0, 0.3, (n, 2))   # rough start positions
    res = rig.fit(views, n, dt, init_xy=init, stature0=1.70, verbose=False)
    assert abs(res["stature_m"] - H) / H < 0.04, res["stature_m"]
    for k in ("thigh", "shank", "upper_arm", "sh_half", "spine"):
        assert abs(res["segments_m"][k] - seg[k]) / seg[k] < 0.10, (k, res["segments_m"][k], seg[k])
    err = np.linalg.norm(res["markers"][:, :17] - X[:, :17], axis=-1)
    assert np.median(err) < 0.06, np.median(err)
    knees = res["angles_deg"][:, [rig.ANG_IDX[("knee", "L")], rig.ANG_IDX[("knee", "R")]]]
    assert knees.min() > -5.0, knees.min()           # never bent backwards


def test_rig_is_right_handed():
    """Facing +y (yaw 0) in the right-handed world (z up): the subject's left is -x, the nose is in front of the ears,
    knees bend backwards. (A mirrored rig still reproduces 2D keypoints by turning around, so fits cannot catch this.)"""
    import torch
    n = 1
    ang = torch.zeros((n, len(rig.ANGLES)), dtype=torch.float64)
    ang[:, rig.ANG_IDX[("knee", "L")]] = math.radians(60)
    H = torch.tensor(1.75, dtype=torch.float64)
    seg = {k: rig.RATIO[k] * H for k in rig.SEGS}
    X = rig.forward(torch.zeros((n, 3), dtype=torch.float64), torch.zeros(n, dtype=torch.float64), ang, seg, H)[0].numpy()
    assert X[rig.L_SH, 0] < X[rig.R_SH, 0] and X[rig.L_HIP, 0] < X[rig.R_HIP, 0] and X[rig.L_EYE, 0] < X[rig.R_EYE, 0]
    assert X[rig.NOSE, 1] > (X[rig.L_EAR, 1] + X[rig.R_EAR, 1]) / 2
    assert X[rig.L_AN, 1] < X[rig.L_KN, 1]            # flexed knee: the ankle swings back
    assert X[rig.L_BIG, 1] > X[rig.L_HEEL, 1]          # toes forward of the heel


def _rot_err_deg(A, B):
    """Geodesic angle (degrees) between rotation stacks A, B (n, 3, 3)."""
    c = (np.einsum("nij,nij->n", A, B) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _parts_truth(n=40, dt=0.125):
    """A person standing ~2.3 m from two cameras, turning the head, opening the jaw, rotating wrists, curling fingers."""
    import torch
    H = 1.75
    t = np.arange(n) * dt
    root = np.c_[0.2 * np.sin(t), 2.3 + 0.1 * np.sin(0.7 * t), np.full(n, 0.53 * H)]
    yaw = np.full(n, math.pi)                          # facing -y, towards the cameras
    ang = np.zeros((n, len(rig.ANGLES)))
    ang[:, rig.ANG_IDX[("head_twist", None)]] = np.radians(40) * np.sin(2 * math.pi * t / 4.0)
    ang[:, rig.ANG_IDX[("head_flex", None)]] = np.radians(20) * np.sin(2 * math.pi * t / 3.0 + 1)
    ang[:, rig.ANG_IDX[("head_lat", None)]] = np.radians(10) * np.sin(2 * math.pi * t / 2.5)
    for s in ("L", "R"):
        ang[:, rig.ANG_IDX[("elbow", s)]] = np.radians(70)
        ang[:, rig.ANG_IDX[("sh_flex", s)]] = np.radians(25)
    angx = np.zeros((n, len(rig.ANGLES_X)))
    angx[:, rig.ANGX_IDX[("jaw", None)]] = np.radians(10) * (1 + np.sin(2 * math.pi * t / 2.0))
    for s, ph in (("L", 0.0), ("R", 1.3)):
        angx[:, rig.ANGX_IDX[("wr_pron", s)]] = np.radians(55) * np.sin(2 * math.pi * t / 3.5 + ph)
        angx[:, rig.ANGX_IDX[("wr_flex", s)]] = np.radians(30) * np.sin(2 * math.pi * t / 2.7 + ph)
        angx[:, rig.ANGX_IDX[("wr_dev", s)]] = np.radians(10) * np.sin(2 * math.pi * t / 3.1)
        for f in rig.FINGERS:
            angx[:, rig.ANGX_IDX[(f"curl_{f}", s)]] = np.radians(40) * (1 + np.sin(2 * math.pi * t / 4.5 + ph))
    seg = {k: torch.tensor(rig.RATIO[k] * H, dtype=torch.float64) for k in rig.SEGS}
    T = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
    X, fr = rig.forward_parts(T(root), T(yaw), T(ang), T(angx), seg, T(H))
    return X.numpy(), fr, np.degrees(angx), dt


def _parts_views(X, cams_C, noise_body=2.0, noise_parts=1.0, f=900.0):
    import torch
    rng = np.random.default_rng(1)
    views = []
    for C in cams_C:
        cam = _cam(C, [0.0, 2.3, 1.2], f=f, k1=-0.1)
        uv, _ = rig.project(torch.tensor(X), cam, cam.size)
        uv = uv.numpy()
        uv[:, :23] += rng.normal(0, noise_body, uv[:, :23].shape)
        uv[:, 23:] += rng.normal(0, noise_parts, uv[:, 23:].shape)
        conf = np.ones(uv.shape[:2])
        conf[rng.random(conf.shape) < 0.15] = 0.0
        kp = np.concatenate([uv, conf[..., None]], -1)
        top, bot = uv[:, :23, 1].min(1), uv[:, :23, 1].max(1)
        views.append({"cam": cam, "size": cam.size, "kp": kp, "box_h": bot - top})
    return views


def test_parts_recover_head_pose_and_hands():
    """Face (68) and hand (2 x 21) keypoints in two cameras -> head orientation within 4 deg, jaw within 5 deg, hand
    orientation (palm normal, finger direction) within 12 deg (medians); parts usable in every frame."""
    X, fr, angx, dt = _parts_truth()
    n = len(X)
    views = _parts_views(X, ([-0.6, 0.0, 1.6], [0.9, 0.2, 1.4]))
    res = rig.fit(views, n, dt, init_xy=X[:, [rig.L_HIP, rig.R_HIP], :2].mean(1), stature0=1.70, verbose=False,
                  parts=("face", "hands"))
    P = res["parts"]
    assert P["usable_rate"]["face"] > 0.95 and P["usable_rate"]["L"] > 0.9 and P["usable_rate"]["R"] > 0.9, P["usable_rate"]
    e_head = _rot_err_deg(P["head_R"], fr["head"][0].numpy())
    assert np.median(e_head) < 4.0, np.median(e_head)
    jaw_err = np.abs(P["jaw_deg"] - angx[:, rig.ANGX_IDX[("jaw", None)]])
    assert np.median(jaw_err) < 5.0, np.median(jaw_err)
    for s in ("L", "R"):
        Rt = fr[s].numpy()
        sgn = 1.0 if s == "L" else -1.0
        pn_err = np.degrees(np.arccos(np.clip((P["palm_normal"][s] * (Rt[:, :, 0] * sgn)).sum(-1), -1, 1)))
        fd_err = np.degrees(np.arccos(np.clip((P["finger_dir"][s] * -Rt[:, :, 2]).sum(-1), -1, 1)))
        assert np.median(pn_err) < 12.0 and np.median(fd_err) < 12.0, (s, np.median(pn_err), np.median(fd_err))
    ypr_t = rig.head_ypr(fr["head"][0].numpy())
    assert np.median(np.abs(P["head_ypr_deg"][:, 0] - ypr_t[:, 0])) < 4.0


def test_parts_gate_rejects_tiny_hands_and_keeps_api():
    """Cameras 9 m away (hands ~12 px): hands are never used; body-only fits return no "parts" (old API unchanged)."""
    X, fr, angx, dt = _parts_truth(n=16)
    views = _parts_views(X, ([-2.0, -6.5, 2.0], [3.0, -6.0, 2.0]))
    init = X[:, [rig.L_HIP, rig.R_HIP], :2].mean(1)
    res = rig.fit(views, len(X), dt, init_xy=init, verbose=False, iters=(150, 300), parts=("hands",), iters_parts=100)
    assert res["parts"]["usable_rate"]["L"] == 0.0 and res["parts"]["usable_rate"]["R"] == 0.0
    res0 = rig.fit(views, len(X), dt, init_xy=init, verbose=False, iters=(150, 300))
    assert "parts" not in res0 and res0["markers"].shape == (len(X), 23, 3)


if __name__ == "__main__":
    test_rig_is_right_handed()
    print("ok test_rig_is_right_handed")
    test_fit_recovers_synthetic_walker()
    print("ok test_fit_recovers_synthetic_walker")
    test_parts_recover_head_pose_and_hands()
    print("ok test_parts_recover_head_pose_and_hands")
    test_parts_gate_rejects_tiny_hands_and_keeps_api()
    print("ok test_parts_gate_rejects_tiny_hands_and_keeps_api")

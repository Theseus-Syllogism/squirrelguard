"""Map calibration (calib.html points -> geometry.fit_points) on a synthetic mounted Wyze camera.

Run:  .venv/bin/python tests/test_map_calib.py      (or pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_walk as sw  # noqa: E402
from squirrel_guard import trajectory as tj  # noqa: E402
from squirrel_guard.geometry import fit_points, nominal_K  # noqa: E402


def _points(cam, n=10, noise_px=2.0, seed=0, map_noise_m=0.15):
    """Ground points in the camera's view (east/north metres, z = 0), clicked with pixel and map noise."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        g = np.array([rng.uniform(-8, 8), rng.uniform(2.5, 18), 0.0])
        px = sw.project(cam, g[None])[0]
        if np.isfinite(px).all() and 20 < px[0] < cam.size[0] - 20 and 20 < px[1] < cam.size[1] - 20:
            world = g + np.r_[rng.normal(0, map_noise_m, 2), 0]          # aerial-photo click error (~30 cm pixels)
            pts.append({"px": (px + rng.normal(0, noise_px, 2)).tolist(), "world": world.tolist()})
    return pts


def _ground_err(cam_true, cam_fit, size):
    """Median ground-position error (m) over a grid of foot pixels in the lower 2/3 of the frame."""
    W, H = size
    u, v = np.meshgrid(np.linspace(100, W - 100, 9), np.linspace(H * 0.4, H - 40, 7))
    px = np.c_[u.ravel(), v.ravel()]
    a, b = cam_true.ground(px, size), cam_fit.ground(px, size)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1) & (np.linalg.norm(a - cam_true.C[:2], axis=1) < 20)
    return float(np.median(np.linalg.norm(a[ok] - b[ok], axis=1)))


def test_map_points_recover_mounted_camera():
    cam = sw.wyze_camera()                                  # 2.7 m high, 35 deg down, hfov 100, k1 -0.25
    size = cam.size
    pts = _points(cam)
    fixed, st0 = fit_points(pts, size, nominal_K(*size), np.zeros(5), size, refine_lens=False)
    refit, st1 = fit_points(pts, size, nominal_K(*size), np.zeros(5), size, refine_lens=True)
    e0, e1 = _ground_err(cam, fixed, size), _ground_err(cam, refit, size)
    print(f"  nominal lens: height {st0['camera_height_m']} m, reproj {st0['reproj_px_mean']} px, ground error {e0:.2f} m")
    print(f"  lens refit:   height {st1['camera_height_m']} m, reproj {st1['reproj_px_mean']} px, ground error {e1:.2f} m,"
          f" {st1.get('lens_refit')}")
    assert abs(st1["camera_height_m"] - 2.7) < 0.3
    assert e1 < 0.5 and e1 <= e0


def test_trajectory_with_map_calibration():
    cam = sw.wyze_camera()
    size = cam.size
    fit, _ = fit_points(_points(cam, n=10, seed=1), size, nominal_K(*size), np.zeros(5), size, refine_lens=True)
    trk, truth = sw.synth_track(sw.straight(), cam, seed=2)
    res = tj.estimate(trk, 8.0, size, fit)
    f0 = res["frame0"]
    xy = truth["xy"][f0:f0 + len(res["t"])]
    obs = np.array(res["observed"])
    rmse = float(np.sqrt(np.mean(np.sum((np.array(res["mean"])[obs] - xy[obs]) ** 2, 1))))
    print(f"  walk tracked with the map-calibrated camera: path RMSE {rmse:.2f} m")
    assert rmse < 0.4


if __name__ == "__main__":
    for f in [v for n, v in list(globals().items()) if n.startswith("test_")]:
        f()
        print("ok", f.__name__)

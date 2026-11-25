"""Synthetic walks for trajectory tests: a 1.75 m stick person on a known ground path, projected through a
distorted Wyze-like camera into fake kinematics.track() output (Pose|None per frame at ~8 fps)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squirrel_guard.geometry import Camera, nominal_K, rot_from_angles  # noqa: E402
from squirrel_guard.pose import Pose  # noqa: E402

HEIGHT = 1.75
SIZE = (1920, 1080)


def wyze_camera(h=2.7, pitch=35.0, roll=1.0) -> Camera:
    K = nominal_K(*SIZE, hfov_deg=100.0)
    D = np.array([-0.25, 0.05, 0.0, 0.0, 0.0])
    return Camera(K, D, rot_from_angles(pitch, roll), [0.0, 0.0, h], SIZE)


# -- paths: position(t) on the ground, with speed profiles --------------------------------------------------------
class Path2D:
    """Polyline/curve by arc length with a speed profile v(t)."""

    def __init__(self, pts, speed_fn, duration):
        pts = np.asarray(pts, float)
        self.pts = pts
        self.cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        self.speed_fn = speed_fn
        self.duration = duration
        ts = np.arange(0, duration + 5.0 + 1e-9, 0.01)  # past the end: the true continuation
        self._ts = ts
        self._s = np.r_[0, np.cumsum([speed_fn(t) * 0.01 for t in ts[:-1]])]

    def at(self, t):
        s = np.interp(t, self._ts, self._s)
        s = min(s, self.cum[-1])
        x = np.interp(s, self.cum, self.pts[:, 0])
        y = np.interp(s, self.cum, self.pts[:, 1])
        return np.array([x, y])

    def speed(self, t):
        return self.speed_fn(t) if np.interp(t, self._ts, self._s) < self.cum[-1] else 0.0

    def heading(self, t):
        a, b = self.at(max(0, t - 0.05)), self.at(t + 0.05)
        d = b - a
        if np.linalg.norm(d) < 1e-6:
            d = self.at(t + 0.5) - self.at(max(0, t - 0.5))
        if np.linalg.norm(d) < 1e-6:
            return 0.0
        return math.atan2(d[1], d[0])


def bezier(p0, p1, p2, n=200):
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 2 * np.asarray(p0) + 2 * (1 - u) * u * np.asarray(p1) + u ** 2 * np.asarray(p2)


def straight(duration=7.0):
    return Path2D([[-4.0, 9.0], [8.0, 1.5]], lambda t: 1.3, duration)


def curve_with_stop(duration=13.0, stop_at=4.5, stop_len=3.0):
    def v(t):
        base = 1.2
        if stop_at <= t < stop_at + stop_len:
            return 0.0
        if stop_at - 0.4 <= t < stop_at:
            return base * (stop_at - t) / 0.4
        if stop_at + stop_len <= t < stop_at + stop_len + 0.4:
            return base * (t - stop_at - stop_len) / 0.4
        return base
    return Path2D(bezier([-4.5, 8.5], [0.5, 2.0], [7.0, 11.0]), v, duration)


def leaving(duration=8.0):
    return Path2D([[-3.0, 11.0], [4.0, -3.0]], lambda t: 1.4, duration)


# -- body model -----------------------------------------------------------------------------------------------------
def joints3d(pos, heading, phase, speed, height=HEIGHT):
    """COCO-17 joints in world metres. phase: gait phase in radians (one full cycle = two steps)."""
    sc = height / 1.75
    f = np.array([math.cos(heading), math.sin(heading), 0.0])
    l = np.array([-math.sin(heading), math.cos(heading), 0.0])
    up = np.array([0, 0, 1.0])
    amp = min(1.0, speed / 1.2)
    sw = math.sin(phase) * amp
    base = np.array([pos[0], pos[1], 0.0])

    def P(fw, lat, z):
        return base + f * fw * sc + l * lat * sc + up * z * sc

    J = [P(0.08, 0.0, 1.62), P(0.08, 0.03, 1.65), P(0.08, -0.03, 1.65), P(0.0, 0.07, 1.62), P(0.0, -0.07, 1.62),
         P(0.0, 0.19, 1.45), P(0.0, -0.19, 1.45),
         P(-0.10 * sw, 0.22, 1.15), P(0.10 * sw, -0.22, 1.15),
         P(-0.22 * sw, 0.23, 0.85), P(0.22 * sw, -0.23, 0.85),
         P(0.0, 0.10, 0.95), P(0.0, -0.10, 0.95),
         P(0.18 * sw, 0.10, 0.50), P(-0.18 * sw, -0.10, 0.50),
         P(0.32 * sw, 0.10, 0.08), P(-0.32 * sw, -0.10, 0.08)]
    sole_l = P(0.32 * sw, 0.10, 0.0)
    sole_r = P(-0.32 * sw, -0.10, 0.0)
    head = P(0.0, 0.0, 1.0 * height / sc)
    return np.array(J), np.array([sole_l, sole_r, head])


def project(cam: Camera, pts3):
    Rcw = cam.R.T
    rvec, _ = cv2.Rodrigues(Rcw)
    tvec = (-Rcw @ cam.C).reshape(3, 1)
    zc = (pts3 - cam.C) @ cam.R  # camera-frame coordinates
    px, _ = cv2.projectPoints(np.asarray(pts3, float).reshape(-1, 1, 3), rvec, tvec, cam.K, cam.D)
    px = px.reshape(-1, 2)
    px[zc[:, 2] <= 0.05] = np.nan
    return px


def synth_track(path: Path2D, cam: Camera, fps=8.0, cadence_spm=112.0, drop=0.2, occlusion=(6.0, 1.5),
                pix_sigma=3.0, seed=0, height=HEIGHT):
    """-> (track, truth) with track = [Pose|None], truth = dict(t, xy (N,2), speed, in_view)."""
    rng = np.random.default_rng(seed)
    W, H = cam.size
    n = int(path.duration * fps)
    t = np.arange(n) / fps
    track, xy, spd, inview = [], [], [], []
    phase, prev_t = 0.0, 0.0
    for i, ti in enumerate(t):
        pos, v = path.at(ti), path.speed(ti)
        phase += 2 * math.pi * (cadence_spm / 60.0 / 2.0) * (ti - prev_t) * (1.0 if v > 0.05 else 0.0)
        prev_t = ti
        xy.append(pos); spd.append(v)
        J, extra = joints3d(pos, path.heading(ti), phase, v, height)
        px = project(cam, np.vstack([J, extra]))
        vis = np.isfinite(px).all(1)
        allpx = px[vis]
        in_frame = vis.all() and (allpx[:, 0] >= 0).all() and (allpx[:, 0] < W).all() and (allpx[:, 1] >= 0).all() and (allpx[:, 1] < H).all()
        inview.append(in_frame)
        occl = occlusion is not None and occlusion[0] <= ti < occlusion[0] + occlusion[1]
        if not in_frame or occl or rng.random() < drop:
            track.append(None)
            continue
        kp = np.c_[px[:17] + rng.normal(0, pix_sigma, (17, 2)), rng.uniform(0.6, 0.95, 17)]
        x1, y1 = allpx.min(0) + rng.normal(0, pix_sigma, 2)
        x2, y2 = allpx.max(0) + rng.normal(0, pix_sigma, 2)
        track.append(Pose((float(x1), float(y1), float(x2), float(y2)), 0.9, kp))
    return track, {"t": t, "xy": np.array(xy), "speed": np.array(spd), "in_view": np.array(inview)}


def true_steps(path: Path2D, cadence_spm=112.0, fps=100.0):
    """Number of steps the synthetic person actually took (step clock runs while moving)."""
    n, prev, acc = 0, 0.0, 0.0
    for ti in np.arange(0, path.duration, 1 / fps):
        if path.speed(ti) > 0.05:
            acc += (ti - prev) * cadence_spm / 60.0
            while acc >= 1:
                acc -= 1; n += 1
        prev = ti
    return n

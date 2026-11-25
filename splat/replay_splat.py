#!/usr/bin/env python3
"""One empty frame from a fixed camera -> Gaussian-splat scene in a trajectory replay's own world frame.

  ~/src/splat/.venv/bin/python splat/replay_splat.py FRAME.png REPLAY.json [--near 0.4] [--far 6.5] [--step 3]

Depth comes from Depth Anything V2 Small (Apache-2.0, relative inverse depth), made metric with two ASSUMED
reference distances (nearest 2 % of pixels at --near, farthest 3 % at --far). Every kept pixel is cast along the
replay camera's own ray (REPLAY.json "camera": K, D, R, C, so lens distortion and pose match the path, cut-outs
and camera projection exactly) to its depth, and becomes one small isotropic Gaussian. Writes <name>_splat.splat
(antimatter15 layout, 32 bytes per Gaussian) next to REPLAY.json and registers it as assets.splat, which the
viewer loads for the "Gaussian splat" scene. Surfaces the camera cannot see are holes; the scale is approximate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squirrel_guard.geometry import Camera  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from single_view_3d import depth_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame"); ap.add_argument("replay")
    ap.add_argument("--near", type=float, default=0.4, help="metres to the nearest 2%% of pixels")
    ap.add_argument("--far", type=float, default=6.5, help="metres to the farthest 3%% of pixels")
    ap.add_argument("--step", type=int, default=3, help="pixel stride")
    ap.add_argument("--max-dist", type=float, default=12.0)
    a = ap.parse_args()
    rp = Path(a.replay)
    tr = json.loads(rp.read_text())
    c = tr["camera"]
    cam = Camera(c["K"], c["D"], c["R"], c["C"], tuple(c["size"]))
    img = cv2.imread(a.frame)
    H, W = img.shape[:2]
    if (W, H) != tuple(c["size"]):
        img = cv2.resize(img, tuple(c["size"]), interpolation=cv2.INTER_AREA); H, W = img.shape[:2]

    disp = depth_model(img[:, :, ::-1].copy())                       # bigger = nearer
    lo, hi = np.percentile(disp, 3), np.percentile(disp, 98)
    alpha = (1 / a.near - 1 / a.far) / (hi - lo)
    beta = 1 / a.far - alpha * lo
    zdep = 1.0 / np.clip(alpha * disp + beta, 1 / 30.0, None)        # depth along the optical axis (m)

    s = a.step
    ys, xs = np.mgrid[s // 2:H:s, s // 2:W:s]
    gy, gx = np.gradient(np.log(zdep))
    keep = np.hypot(gx, gy)[ys, xs] < 0.04                            # skip depth cliffs (no rubber sheets)
    xs, ys = xs[keep], ys[keep]
    d = cam.rays(np.c_[xs, ys].astype(float), (W, H))                 # unit world rays, distortion included
    fwd = np.asarray(cam.R, float)[:, 2]                              # optical axis in world
    dist = zdep[ys, xs] / np.clip(d @ fwd, 0.05, None)                # z-depth -> distance along the ray
    ok = dist < a.max_dist
    P = (np.asarray(cam.C, float)[None, :] + d * dist[:, None])[ok]
    f = float(cam.K[0, 0])
    sc = (dist[ok] / f * s * 0.7).astype(np.float32)
    rgb = img[ys[ok], xs[ok], ::-1]

    n = len(P)
    rec = np.zeros(n, dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("q", "u1", 4)])
    rec["p"] = P.astype(np.float32)
    rec["s"] = np.repeat(sc[:, None], 3, 1)
    rec["c"][:, :3] = rgb
    rec["c"][:, 3] = 235
    rec["q"] = (255, 128, 128, 128)                                    # identity quaternion (w, x, y, z)
    name = rp.stem
    out = rp.with_name(f"{name}_splat.splat")
    rec.tofile(out)
    tr.setdefault("assets", {})["splat"] = out.name
    tr["splat_note"] = (f"single-frame depth (Depth Anything V2 Small), scale assumed: nearest {a.near} m, "
                        f"farthest {a.far} m; hidden surfaces are holes")
    rp.write_text(json.dumps(tr, separators=(",", ":")))
    z = P[:, 2]
    print(f"{n:,} Gaussians -> {out} ({out.stat().st_size / 1e6:.1f} MB); distance p5-p95 "
          f"{np.percentile(dist[ok], 5):.1f}-{np.percentile(dist[ok], 95):.1f} m, height p5-p95 "
          f"{np.percentile(z, 5):.1f}-{np.percentile(z, 95):.1f} m")


if __name__ == "__main__":
    main()

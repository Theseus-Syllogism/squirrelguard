#!/usr/bin/env python
"""Fit an image-to-3D vehicle mesh (e.g. TripoSR) to a real camera frame: scale it to published dimensions, then
find its orientation, heading and ground position by render-and-compare against the vehicle's silhouette.

  .venv/bin/python scripts/vehicle_fit.py JOB.json

JOB.json:
  {"mesh": "mesh.obj",                       # any orientation/scale (TripoSR output is unit-sized, axes arbitrary)
   "mask": "mask.png",                       # vehicle silhouette (alpha/greyscale) in the crop's pixel space
   "crop": {"frame_xy": [x0, y0], "pad_xy": [0, 136]},   # mask pixel (u, v) -> frame (x0 + u - px, y0 + v - py)
   "frame_to_full": 2.6667,                  # frame pixel -> full-resolution sensor pixel
   "intrinsics": "v4.intrinsics.json",       # K, D (OpenCV), image_size of the full-resolution sensor
   "dims_m": [5.62, 2.0, 1.96],              # published length, width, height
   "camera_height_m": 0.35, "camera_pitch_deg": 3,       # starting values; pitch is refined
   "camera_site": {"position": [x, y], "yaw_deg": 85},   # optional: report the result in site-enu
   "out": "fit.json", "overlay": "fit.png"}

Model frame after alignment: X forward (length), Y left (width), Z up, origin at the centre of the footprint on the
ground. The fit only explains the silhouette: colour, interior and hidden sides come from the generator's guess.
"""
from __future__ import annotations

import itertools
import json
import math
import sys

import cv2
import numpy as np
from scipy.optimize import minimize

SCALE = 0.25          # work at quarter sensor resolution


def load_obj(p):
    v, f = [], []
    for line in open(p):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            f.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(v), np.array(f)


def orientations(V):
    """Candidate model frames: PCA axes, longest = forward; up is either remaining axis, both signs."""
    c = V - V.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    a = vt                                      # rows: principal axes, largest variance first
    out = []
    for up_i, sf, su in itertools.product((1, 2), (1, -1), (1, -1)):
        fwd = a[0] * sf; up = a[up_i] * su; left = np.cross(up, fwd)
        out.append(np.stack([fwd, left, up]))  # rows = model X, Y, Z in mesh coordinates
    return out


def normalise(V, R, dims):
    P = (V - V.mean(0)) @ R.T                   # mesh -> model axes
    lo, hi = P.min(0), P.max(0)
    P = (P - (lo + hi) / 2) / (hi - lo) * np.array(dims)   # bounding box -> published dimensions
    P[:, 2] += dims[2] / 2                      # wheels on the ground
    return P


def project(P, theta, tx, ty, h, pitch, K):
    c, s = math.cos(theta), math.sin(theta)
    x = c * P[:, 0] - s * P[:, 1] + tx; y = s * P[:, 0] + c * P[:, 1] + ty; z = P[:, 2] - h
    sp, cp = math.sin(pitch), math.cos(pitch)
    xc = x; yc = -(-sp * y + cp * z); zc = cp * y + sp * z
    zc = np.maximum(zc, 0.05)
    return np.stack([K[0, 0] * xc / zc + K[0, 2], K[1, 1] * yc / zc + K[1, 2]], 1)


def render(P, F, prm, K, shape):
    uv = project(P, *prm, K)
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [t.astype(np.int32) for t in uv[F]], 1)
    return m


def iou(a, b):
    i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum()
    return i / u if u else 0.0


def main(job):
    J = json.load(open(job))
    intr = json.load(open(J["intrinsics"]))
    K0 = np.array(intr["K"]); D = np.array(intr["D"]); W0, H0 = intr["image_size"]
    # silhouette -> full-res sensor -> undistorted (pinhole) -> working resolution
    mk = (cv2.imread(J["mask"], cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    full = np.zeros((H0, W0), np.uint8)
    x0, y0 = J["crop"]["frame_xy"]; px, py = J["crop"]["pad_xy"]; s = J["frame_to_full"]
    vs, us = np.nonzero(mk)
    fx = ((x0 + us - px) * s).astype(int); fy = ((y0 + vs - py) * s).astype(int)
    ok = (fx >= 0) & (fx < W0) & (fy >= 0) & (fy < H0)
    full[fy[ok], fx[ok]] = 1
    full = cv2.dilate(full, np.ones((4, 4), np.uint8))
    und = cv2.undistort(full * 255, K0, D) > 127
    K = K0.copy(); K[:2] *= SCALE
    shape = (int(H0 * SCALE), int(W0 * SCALE))
    target = cv2.resize(und.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    ys, xs = np.nonzero(target)
    V, F = load_obj(J["mesh"])
    if len(F) > 60000:                          # silhouettes do not need every triangle
        F = F[np.random.default_rng(0).choice(len(F), 60000, replace=False)]
    dims = J["dims_m"]; h = J.get("camera_height_m", 0.35); pitch0 = math.radians(J.get("camera_pitch_deg", 0))
    # ground position seeded from the silhouette's bottom centre (ray hits the ground plane)
    ub, vb = xs.mean(), ys.max()
    rx, ry = (ub - K[0, 2]) / K[0, 0], (vb - K[1, 2]) / K[1, 1]
    sp, cp = math.sin(pitch0), math.cos(pitch0)
    ray = np.array([rx, ry * sp + cp, -ry * cp + sp])          # camera-level frame: x right, y forward, z up
    t = -h / ray[2] if ray[2] < -1e-3 else 10.0 / max(ray[1], 0.1)
    g0 = ray[:2] * min(max(t, 2.0), 30.0)
    best = (-1, None)
    for oi, R in enumerate(orientations(V)):
        P = normalise(V, R, dims)
        for th in np.radians(np.arange(0, 360, 15)):
            for back in (0.0, dims[1] / 2, dims[0] / 2):   # the bottom-centre ray hits the near side, not the centre
                d = g0 / np.linalg.norm(g0)
                prm = (th, g0[0] + d[0] * back, g0[1] + d[1] * back, h, pitch0)
                sc = iou(render(P, F, prm, K, shape), target)
                if sc > best[0]:
                    best = (sc, oi, prm)
    sc0, oi, prm = best
    P = normalise(V, orientations(V)[oi], dims)

    def loss(z):
        return -iou(render(P, F, (z[0], z[1], z[2], h, z[3]), K, shape), target)
    z0 = np.array([prm[0], prm[1], prm[2], prm[4]])
    simplex = np.vstack([z0, z0 + np.diag([0.15, 0.4, 0.6, 0.03])])
    res = minimize(loss, z0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 600, "initial_simplex": simplex})
    th, tx, ty, pitch = res.x; score = -res.fun
    out = {"iou_coarse": round(sc0, 3), "iou": round(score, 3), "orientation": int(oi),
           "camera_frame": {"heading_deg": round(math.degrees(th) % 360, 1), "right_m": round(tx, 2), "forward_m": round(ty, 2),
                            "distance_m": round(math.hypot(tx, ty), 2), "camera_pitch_deg": round(math.degrees(pitch), 2)},
           "dims_m": dims, "note": "pose explains the silhouette only; mesh appearance is generated"}
    cs = J.get("camera_site")
    if cs:
        yaw = math.radians(cs["yaw_deg"]); fwd = np.array([math.sin(yaw), math.cos(yaw)]); right = np.array([math.cos(yaw), -math.sin(yaw)])
        c = np.array(cs["position"][:2]) + right * tx + fwd * ty
        vdir = right * math.cos(th) + fwd * math.sin(th)
        out["site"] = {"center": c.round(2).tolist(), "heading_deg": round(math.degrees(math.atan2(vdir[0], vdir[1])) % 360, 1)}
    np.save(J["out"].replace(".json", "_model.npy"), P)
    json.dump(out, open(J["out"], "w"), indent=1)
    if J.get("overlay"):
        m = render(P, F, (th, tx, ty, h, pitch), K, shape)
        img = np.full((*shape, 3), 255, np.uint8)
        img[target] = (180, 180, 180); img[m.astype(bool)] = (60, 110, 220)
        img[np.logical_and(target, m.astype(bool))] = (40, 160, 90)
        cv2.imwrite(J["overlay"], img)
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1])

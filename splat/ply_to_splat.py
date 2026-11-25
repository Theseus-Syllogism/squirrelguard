#!/usr/bin/env python3
"""Trained 3DGS .ply (make_splat.py train) -> .splat for the replay viewer (antimatter15 layout, 32 bytes each).

  python splat/ply_to_splat.py splat.ply OUT.splat [--min-opacity 0.05] [--max-scale 0.5] [--bbox x0,y0,z0,x1,y1,z1]

Per Gaussian: position f32x3, scale f32x3 (linear), colour u8x4 (RGB from the SH DC term, alpha = sigmoid(opacity)),
rotation u8x4 (w x y z, q*128+128). View-dependent SH bands are dropped (the viewer renders DC only). Sorted by
size x opacity, largest first, as the reference viewer expects. Faint and oversized Gaussians (floaters, which a
3-view splat has many of) are removed; --max-ratio removes needles (spikes: very elongated Gaussians, typical where
only one or two cameras see a surface); --bbox keeps only Gaussians inside a world box (metres).
"""
from __future__ import annotations

import argparse

import numpy as np
from plyfile import PlyData

C0 = 0.28209479177387814


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply"); ap.add_argument("out")
    ap.add_argument("--min-opacity", type=float, default=0.05)
    ap.add_argument("--max-scale", type=float, default=0.5, help="drop Gaussians with an axis longer than this (m)")
    ap.add_argument("--bbox", help="x0,y0,z0,x1,y1,z1 world box to keep")
    ap.add_argument("--max-ratio", type=float, default=0.0,
                    help="drop needles: longest/middle axis ratio above this (0 = off; try 6)")
    a = ap.parse_args()
    v = PlyData.read(a.ply)["vertex"]
    pos = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    sc = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1)).astype(np.float32)
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)
    rgb = np.clip((0.5 + C0 * dc) * 255, 0, 255)
    alpha = 1 / (1 + np.exp(-np.asarray(v["opacity"])))
    q = np.stack([v[f"rot_{i}"] for i in range(4)], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    keep = (alpha >= a.min_opacity) & (sc.max(1) <= a.max_scale) & np.isfinite(pos).all(1)
    ss = np.sort(sc, 1)
    ratio = ss[:, 2] / np.maximum(ss[:, 1], 1e-9)        # longest / middle axis: needles, not flat discs
    if a.max_ratio > 0:
        spikes = keep & (ratio > a.max_ratio)
        print(f"spikes: {int(spikes.sum())} Gaussians with longest/middle axis ratio > {a.max_ratio:g} removed")
        keep &= ratio <= a.max_ratio
    if a.bbox:
        b = np.array([float(t) for t in a.bbox.split(",")])
        keep &= (pos >= b[:3]).all(1) & (pos <= b[3:]).all(1)
    order = np.argsort(-(sc.prod(1) * alpha)[keep])
    idx = np.nonzero(keep)[0][order]
    rec = np.zeros(len(idx), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"], rec["s"] = pos[idx], sc[idx]
    rec["c"] = np.c_[rgb[idx], alpha[idx] * 255].astype(np.uint8)
    rec["r"] = np.clip(q[idx] * 128 + 128, 0, 255).astype(np.uint8)
    rec.tofile(a.out)
    print(f"{len(idx)}/{len(pos)} Gaussians -> {a.out} ({rec.nbytes / 2 ** 20:.1f} MB)")


if __name__ == "__main__":
    main()

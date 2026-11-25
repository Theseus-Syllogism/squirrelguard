#!/usr/bin/env python
"""Turn the site's coloured lidar points (geo_site.py) into a standard 3D Gaussian Splatting PLY, the format Blender
5.3 imports natively as a Gaussian point cloud (and that splat/ tools and the web viewer's converter read).

  .venv/bin/python scripts/site_splat.py data/site/site_points.ply data/site/site_splat.ply

Each lidar point becomes one Gaussian sized from its neighbour spacing (lidar is sparse: ~2-3 pts/m2), ground
returns flattened to discs. Colours come from the NAIP orthophoto (top-down: walls take roof/ground colour).
A lidar Gaussian scene is coarse context (streets, houses, trees at ~0.5 m), not a photoreal capture.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SH0 = 0.28209479177387814


def read_site_ply(p: Path) -> np.ndarray:
    raw = p.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("c", "u1")]
    return np.frombuffer(raw[end:], dtype=dt)


def write_3dgs(p: Path, xyz, rgb01, scale_log, opacity_logit, rot):
    n = len(xyz)
    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    a = np.zeros((n, len(names)), np.float32)
    a[:, 0:3] = xyz
    a[:, 6:9] = (rgb01 - 0.5) / SH0
    a[:, 9] = opacity_logit
    a[:, 10:13] = scale_log
    a[:, 13:17] = rot
    hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    hdr += "".join(f"property float {k}\n" for k in names) + "end_header\n"
    p.write_bytes(hdr.encode() + a.tobytes())


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    pts = read_site_ply(src)
    xyz = np.column_stack([pts["x"], pts["y"], pts["z"]]).astype(np.float32)
    rgb = np.column_stack([pts["r"], pts["g"], pts["b"]]).astype(np.float32) / 255.0
    ground = pts["c"] == 2
    d, _ = cKDTree(xyz).query(xyz, k=5)
    spacing = np.clip(d[:, 1:].mean(1), 0.08, 1.2)
    sig = 0.7 * spacing
    scale = np.column_stack([sig, sig, sig])
    scale[ground, 2] = 0.03                       # ground: flat discs
    rot = np.zeros((len(xyz), 4), np.float32)
    rot[:, 0] = 1.0                               # identity quaternion (w, x, y, z)
    write_3dgs(dst, xyz, rgb, np.log(scale).astype(np.float32), np.full(len(xyz), 2.5, np.float32), rot)
    print(f"{dst}: {len(xyz):,} gaussians, median sigma {np.median(sig):.2f} m")


if __name__ == "__main__":
    main()

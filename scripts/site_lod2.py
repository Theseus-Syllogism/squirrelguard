#!/usr/bin/env python
"""Procedural LOD2 street model from public data: OSM building footprints + USGS lidar -> clean buildings with
fitted roofs, lidar-detected trees and OSM roads, all in site-enu (x east, y north, z up, metres; geo_site.py).

  .venv/bin/python scripts/site_lod2.py data/site-naip            # needs site.json, site_points.ply, osm.json

Writes <site>/lod2.json, which scripts/blender/site_model.py turns into meshes and materials ("lod2" in the job).

Per building: lidar points inside the footprint (non-ground, >= 2 m above the local ground) give the eave height
(25th percentile) and the ridge height (98th). The roof is chosen by comparing how well a flat, gable (ridge on
the long axis of the minimum-area rectangle) or hip plane model explains those points (median absolute error).
Trees: local maxima of the canopy height model outside buildings, >= 3 m, radius from the canopy extent.
This is a model of the street's shape, not a photograph: walls, windows and roof materials are procedural.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.ndimage import maximum_filter, label
from scipy.spatial import cKDTree

LANE_W = {"secondary": 11.0, "residential": 9.0, "service": 4.0, "footway": 1.6}


def read_points(p: Path):
    raw = p.read_bytes(); end = raw.index(b"end_header\n") + 11
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("c", "u1")]
    return np.frombuffer(raw[end:], dtype=dt)


def inside(poly: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Even-odd ray casting; poly (n, 2) closed or open."""
    res = np.zeros(len(x), bool); px, py = poly[:, 0], poly[:, 1]; j = len(poly) - 1
    for i in range(len(poly)):
        c = ((py[i] > y) != (py[j] > y)) & (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-12) + px[i])
        res ^= c; j = i
    return res


def min_rect(poly: np.ndarray):
    """Minimum-area bounding rectangle: centre, long-axis unit vector, half-length, half-width."""
    best = None
    for i in range(len(poly) - 1):
        e = poly[i + 1] - poly[i]; n = np.linalg.norm(e)
        if n < 1e-6:
            continue
        u = e / n; v = np.array([-u[1], u[0]]); a = poly @ u; b = poly @ v
        area = (a.max() - a.min()) * (b.max() - b.min())
        if best is None or area < best[0]:
            best = (area, u, v, a, b)
    _, u, v, a, b = best
    if a.max() - a.min() < b.max() - b.min():            # make u the long axis
        u, v, a, b = v, -u, b, -a
    c = u * (a.max() + a.min()) / 2 + v * (b.max() + b.min()) / 2
    return c, u, (a.max() - a.min()) / 2, (b.max() - b.min()) / 2


def roof_fit(pts, c, u, hl, hw, eave, ridge):
    """Pick flat / gable / hip by the median error of each roof model against the lidar points."""
    rel = pts[:, :2] - c; s = rel @ u; t = rel @ np.array([-u[1], u[0]]); z = pts[:, 2]
    rise = max(ridge - eave, 0.0)
    flat = np.full_like(z, eave + rise * 0.5) if rise < 0.8 else np.full_like(z, ridge)
    gable = ridge - rise * np.clip(np.abs(t) / max(hw, 0.5), 0, 1)
    inset = min(hw, hl)                                                 # hip: 45 deg-ish ends
    hip = ridge - rise * np.clip(np.maximum(np.abs(t) / max(hw, 0.5), (np.abs(s) - (hl - inset)) / max(inset, 0.5)), 0, 1)
    err = {k: float(np.median(np.abs(z - m))) for k, m in (("flat", flat), ("gable", gable), ("hip", hip))}
    kind = "flat" if rise < 0.8 else min(err, key=err.get)
    return kind, err


def main(site: Path):
    meta = json.loads((site / "site.json").read_text()); o = meta["origin"]
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{o['utm_epsg']}", always_xy=True)
    enu = lambda lon, lat: np.array(tf.transform(lon, lat)) - np.array([o["easting"], o["northing"]])
    P = read_points(site / "site_points.ply")
    xyz = np.stack([P["x"], P["y"], P["z"]], 1).astype(np.float64); cls = P["c"]
    gnd = xyz[cls == 2]; gtree = cKDTree(gnd[:, :2])

    def ground(x, y, r=4.0):
        idx = gtree.query_ball_point([x, y], r)
        return float(np.median(gnd[idx, 2])) if idx else float(np.median(gnd[gtree.query([x, y], 8)[1], 2]))

    osm = json.loads((site / "osm.json").read_text())["elements"]
    half = meta["radius_m"]; buildings, roads, used = [], [], np.zeros(len(xyz), bool)
    for e in osm:
        tags = e.get("tags", {}); geom = e.get("geometry")
        if not geom:
            continue
        pts = np.array([enu(g["lon"], g["lat"]) for g in geom])
        if "building" in tags and len(pts) >= 4:
            if np.abs(pts).max() > half:
                continue
            c0 = pts.mean(0); z0 = ground(*c0)
            m = inside(pts, xyz[:, 0], xyz[:, 1]) & (cls != 2) & (xyz[:, 2] > z0 + 2.0)
            used |= inside(pts, xyz[:, 0], xyz[:, 1])
            roofpts = xyz[m]
            if len(roofpts) >= 6:
                eave, ridge = np.percentile(roofpts[:, 2], [25, 98])
                eave = max(eave - 0.4, z0 + 2.4)
            else:                                            # no lidar on it (new build, or too small): one storey
                eave, ridge = z0 + 2.7, z0 + 4.6
            c, u, hl, hw = min_rect(pts)
            kind, err = roof_fit(roofpts, c, u, hl, hw, eave, ridge) if len(roofpts) >= 6 else ("gable", {})
            buildings.append({"id": e["id"], "footprint": pts[:-1].round(3).tolist() if np.allclose(pts[0], pts[-1]) else pts.round(3).tolist(),
                              "ground": round(z0, 2), "eave": round(float(eave), 2), "ridge": round(float(ridge), 2),
                              "roof": kind, "rect": {"c": c.round(3).tolist(), "u": u.round(4).tolist(), "hl": round(hl, 2), "hw": round(hw, 2)},
                              "lidar_points": int(len(roofpts)), "fit_err_m": {k: round(v, 2) for k, v in err.items()},
                              "kind": tags.get("building")})
        elif "highway" in tags:
            w = LANE_W.get(tags["highway"], 5.0)
            roads.append({"kind": tags["highway"], "width": w, "line": [[round(x, 2), round(y, 2), round(ground(x, y, 6), 2)] for x, y in pts]})

    # trees: canopy height model (non-ground, outside buildings) -> local maxima
    cell = 1.0; n = int(2 * half / cell)
    veg = (cls != 2) & ~used
    gz = np.array([0.0]) if not veg.any() else None
    ii = np.clip(((xyz[veg, 0] + half) / cell).astype(int), 0, n - 1); jj = np.clip(((xyz[veg, 1] + half) / cell).astype(int), 0, n - 1)
    gl = np.array([gnd[k, 2] for k in gtree.query(xyz[veg, :2])[1]])
    hag = xyz[veg, 2] - gl
    chm = np.zeros((n, n)); np.maximum.at(chm, (jj, ii), hag)
    peaks = (chm == maximum_filter(chm, size=7)) & (chm >= 3.0)
    lab, nl = label(chm >= 2.0)
    trees = []
    for j, i in zip(*np.nonzero(peaks)):
        x, y = -half + (i + 0.5) * cell, -half + (j + 0.5) * cell
        area = (lab == lab[j, i]).sum() * cell * cell if lab[j, i] else 9.0
        trees.append({"x": round(x, 2), "y": round(y, 2), "z": round(ground(x, y, 3), 2), "h": round(float(chm[j, i]), 2),
                      "r": round(float(min(max(math.sqrt(area / math.pi), 1.2), 6.0)), 2)})
    out = {"version": 1, "frame": "site-enu metres (geo_site.py)", "sources": {
        "footprints": "OpenStreetMap contributors (ODbL)", "heights": meta.get("sources", {}).get("lidar", "USGS 3DEP"),
        "note": "roof shape fitted to lidar; walls, windows and materials are procedural, not observed"},
        "buildings": buildings, "roads": roads, "trees": trees}
    (site / "lod2.json").write_text(json.dumps(out))
    kinds = {}
    for b in buildings:
        kinds[b["roof"]] = kinds.get(b["roof"], 0) + 1
    print(f"{len(buildings)} buildings {kinds}, {len(roads)} roads, {len(trees)} trees "
          f"(median eave {np.median([b['eave'] - b['ground'] for b in buildings]):.1f} m, "
          f"ridge {np.median([b['ridge'] - b['ground'] for b in buildings]):.1f} m above ground)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

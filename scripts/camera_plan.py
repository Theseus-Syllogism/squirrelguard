#!/usr/bin/env python
"""Camera placement planner on the site's GIS base (geo_site.py): for each camera, which ground a 1.7 m person can
stand on and still be seen, and how tall they appear in the DETECTION frame -- the number that decides whether
Frigate's model finds them (roughly: >= 80 px reliable, 40-80 px marginal, < 40 px usually missed).

  .venv/bin/python scripts/camera_plan.py data/site plan.json --out data/site/plan

plan.json:
  {"vehicles": [{"name": "Ram", "center": [x, y], "heading_deg": 90, "length": 5.8, "width": 2.0,
                 "z0": 0.45, "z1": 1.95}],          # parked cars (not in the lidar) as occluding boxes
   "cameras": [{"name": "v4", "position": [x, y, z], "yaw_deg": 85, "pitch_deg": 3, "hfov_deg": 72.2,
                "aspect": 1.7778, "detect_width": 2560, "range_m": 60}]}

Occlusion: lidar non-ground points (houses, trees, fences) and the vehicle boxes, tested on the line from the camera
to the person's chest (1.2 m) and head (1.6 m); a person counts as visible if either is. Lens distortion is ignored
(pinhole), so the edges of a wide lens are approximate.
Writes <out>/<camera>.png (coverage over the orthophoto) and <out>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PERSON_H = 1.7
BANDS = [(80, (40, 200, 60)), (40, (240, 200, 40)), (0, (230, 60, 50))]   # px >= threshold -> colour


def read_points(site: Path) -> np.ndarray:
    raw = (site / "site_points.ply").read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("c", "u1")]
    return np.frombuffer(raw[end:], dtype=dt)


def box_points(v: dict, step: float = 0.15) -> np.ndarray:
    """Sample a vehicle box's surface as occluder points."""
    L, W, z0, z1 = v["length"], v["width"], v.get("z0", 0.4), v.get("z1", 1.9)
    h = math.radians(v.get("heading_deg", 0))
    fwd, side = np.array([math.sin(h), math.cos(h)]), np.array([math.cos(h), -math.sin(h)])
    out = []
    for z in np.arange(z0, z1 + 1e-6, step):
        for a in np.arange(-L / 2, L / 2 + 1e-6, step):
            for b in (-W / 2, W / 2):
                out.append([*(np.array(v["center"]) + a * fwd + b * side), z])
        for b in np.arange(-W / 2, W / 2 + 1e-6, step):
            for a in (-L / 2, L / 2):
                out.append([*(np.array(v["center"]) + a * fwd + b * side), z])
    return np.array(out, np.float32)


def cam_basis(c: dict):
    yaw, pitch = math.radians(c["yaw_deg"]), math.radians(c.get("pitch_deg", 0))
    fwd = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    up = np.cross(right, fwd)
    return fwd, right, up


def depth_buffer(cam: np.ndarray, occ: np.ndarray, radius: np.ndarray, res_deg: float = 0.25):
    """Angular (azimuth x elevation) min-distance buffer of occluder points, each splatted by its angular size."""
    d = occ - cam
    dist = np.linalg.norm(d, axis=1)
    az = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
    el = np.degrees(np.arcsin(np.clip(d[:, 2] / np.maximum(dist, 1e-6), -1, 1)))
    naz, nel = int(360 / res_deg), int(180 / res_deg)
    buf = np.full((nel, naz), np.inf, np.float32)
    ang = np.degrees(np.arctan2(radius, np.maximum(dist, 1e-3)))
    k = np.clip((ang / res_deg).astype(int), 0, 40)
    ia, ie = (az / res_deg).astype(int), ((el + 90) / res_deg).astype(int)
    for kk in np.unique(k):
        m = k == kk
        for dx in range(-kk, kk + 1):
            for dy in range(-kk, kk + 1):
                if dx * dx + dy * dy > kk * kk + kk:
                    continue
                np.minimum.at(buf, (np.clip(ie[m] + dy, 0, nel - 1), (ia[m] + dx) % naz), dist[m])
    return buf, res_deg


def visible(cam, pts, buf, res):
    d = pts - cam
    dist = np.linalg.norm(d, axis=1)
    az = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
    el = np.degrees(np.arcsin(np.clip(d[:, 2] / np.maximum(dist, 1e-6), -1, 1)))
    ia = (az / res).astype(int) % buf.shape[1]
    ie = np.clip(((el + 90) / res).astype(int), 0, buf.shape[0] - 1)
    return dist < buf[ie, ia] - 0.3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site")
    ap.add_argument("plan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=float, default=0.25)
    a = ap.parse_args()
    site, out = Path(a.site), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((site / "site.json").read_text())
    plan = json.loads(Path(a.plan).read_text())
    R = meta["radius_m"]
    ground = np.load(site / meta["ground"]["file"])          # 1 m grid over [-R, R]
    pts = read_points(site)
    ng = pts["c"] != 2
    occ = np.column_stack([pts["x"][ng], pts["y"][ng], pts["z"][ng]]).astype(np.float32)
    rad = np.full(len(occ), 0.35, np.float32)                # lidar spacing ~0.6 m: each point ~0.35 m radius
    vb = [box_points(v) for v in plan.get("vehicles", [])]
    if vb:
        vb = np.concatenate(vb)
        occ = np.concatenate([occ, vb])
        rad = np.concatenate([rad, np.full(len(vb), 0.1, np.float32)])

    def gz(x, y):
        i = np.clip(np.round(y + R).astype(int), 0, ground.shape[0] - 1)
        j = np.clip(np.round(x + R).astype(int), 0, ground.shape[1] - 1)
        return ground[i, j]

    ortho = Image.open(site / meta["ortho"]["file"]).convert("RGB")
    opx = meta["ortho"]["pixel_m"]
    summary = {}
    for c in plan["cameras"]:
        cam = np.array(c["position"], np.float64)
        rng = c.get("range_m", 60)
        fwd, right, up = cam_basis(c)
        hf = math.radians(c["hfov_deg"]) / 2
        vf = math.atan(math.tan(hf) / c.get("aspect", 16 / 9))
        f_px = (c["detect_width"] / 2) / math.tan(hf)
        g = np.arange(-rng, rng + 1e-6, a.cell)
        X, Y = np.meshgrid(cam[0] + g, cam[1] + g)
        X, Y = X.ravel(), Y.ravel()
        inside = (np.abs(X) < R) & (np.abs(Y) < R)
        X, Y = X[inside], Y[inside]
        Z = gz(X, Y)
        feet = np.column_stack([X, Y, Z])
        rel = feet - cam
        depth = rel @ fwd
        ok = (depth > 0.5) & (np.hypot(rel[:, 0], rel[:, 1]) <= rng)
        for zoff in (0.0, PERSON_H):          # whole person inside the frame (feet and head)
            p = rel + np.array([0, 0, zoff])
            dd = p @ fwd
            ok &= (dd > 0) & (np.abs(np.arctan2(p @ right, dd)) <= hf) & (np.abs(np.arctan2(p @ up, dd)) <= vf)
        buf, res = depth_buffer(cam, occ, rad)
        vis = np.zeros(len(X), bool)
        for zoff in (1.2, 1.6):
            vis |= visible(cam, feet + np.array([0, 0, zoff]), buf, res)
        seen = ok & vis
        px = f_px * PERSON_H / np.maximum(depth, 1e-3)
        # overlay
        crop_r = rng + 5
        x0, y1 = cam[0] - crop_r, cam[1] + crop_r
        box = (int((x0 + R) / opx), int((R - y1) / opx), int((x0 + 2 * crop_r + R) / opx),
               int((R - (y1 - 2 * crop_r)) / opx))
        base = ortho.crop(box)
        scale = 900 / base.size[0]
        base = base.resize((900, 900)).convert("RGBA")
        lay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        cpx = max(1, int(a.cell / opx * scale))
        counts = {}
        for thr, col in BANDS:
            m = seen & (px >= thr)
            sel = m & ~counts.get("_done", np.zeros(len(px), bool))
            counts["_done"] = counts.get("_done", np.zeros(len(px), bool)) | sel
            counts[f">={thr}px"] = round(float(sel.sum()) * a.cell ** 2, 1)
            for x, y in zip(X[sel], Y[sel]):
                u, v = (x - x0) / opx * scale, (y1 - y) / opx * scale
                d.rectangle((u, v, u + cpx, v + cpx), fill=(*col, 120))
        counts.pop("_done")
        cu, cv = (cam[0] - x0) / opx * scale, (y1 - cam[1]) / opx * scale
        yaw = math.radians(c["yaw_deg"])
        for s in (-hf, hf):
            L = rng / opx * scale
            d.line((cu, cv, cu + L * math.sin(yaw + s), cv - L * math.cos(yaw + s)), fill=(255, 255, 255, 220), width=2)
        d.ellipse((cu - 7, cv - 7, cu + 7, cv + 7), fill=(0, 120, 255, 255), outline=(255, 255, 255, 255), width=2)
        for v in plan.get("vehicles", []):
            h = math.radians(v.get("heading_deg", 0))
            fw, sd = np.array([math.sin(h), math.cos(h)]), np.array([math.cos(h), -math.sin(h)])
            cs = [np.array(v["center"]) + p * v["length"] / 2 * fw + q * v["width"] / 2 * sd
                  for p, q in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
            d.polygon([((p[0] - x0) / opx * scale, (y1 - p[1]) / opx * scale) for p in cs], outline=(255, 255, 255, 255))
        img = Image.alpha_composite(base, lay).convert("RGB")
        ImageDraw.Draw(img).text((10, 10), f"{c['name']}: person height in {c['detect_width']} px detect frame  "
                                           "green >=80  yellow 40-80  red <40", fill=(255, 255, 255))
        img.save(out / f"{c['name']}.png")
        summary[c["name"]] = {"visible_area_m2_by_person_px": counts, "focal_px": round(f_px, 1)}
        print(c["name"], summary[c["name"]])
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

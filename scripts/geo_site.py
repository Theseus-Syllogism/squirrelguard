#!/usr/bin/env python
"""Build the site's geospatial base: a local metric frame around one WGS84 point, USGS 3DEP lidar points
(houses, trees, ground) coloured from an aerial orthophoto (USGS NAIP by default; local sources can be added to ORTHOS),
and a ground model.

  .venv/bin/python scripts/geo_site.py --lat 39.7392 --lon -104.9903 --radius 110 --out data/site

Local frame "site-enu": x = metres east, y = metres north (UTM zone grid), z = metres above the lidar ground at the
origin point. Outputs (data/ is git-ignored: the site location is private):
  site.json        origin (lat/lon, UTM, ground elevation), CRS, bounds, data sources and their dates
  site_points.ply  lidar points in site-enu: x y z red green blue classification (2 ground, 3-5 vegetation, 6 building)
  ortho.png        orthophoto, north-up, covering [-radius, radius] in x and y (pixel size in site.json)
  ground.npy       ground height grid (z, site-enu) on the ortho's footprint at 1 m, from lidar ground returns
"""
from __future__ import annotations

import argparse
import io
import json
import math
import socket
import time
import urllib.request
from pathlib import Path

import laspy
import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt

EPT = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{name}/"
NAIP = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
ORTHOS = {   # name -> (export URL, native pixel m, provider/licence note)
    "naip": (NAIP, 0.3, "USDA NAIP via USGS The National Map (public domain)"),
}
UA = {"User-Agent": "site-geo/1.0"}

_gai = socket.getaddrinfo


def _ipv4_only(host, *a, **k):        # the home network advertises IPv6 that does not route: SYNs hang for minutes
    return [r for r in _gai(host, *a, **k) if r[0] == socket.AF_INET] or _gai(host, *a, **k)


socket.getaddrinfo = _ipv4_only


def get(url: str, tries: int = 4) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 + 3 * i)


def utm_epsg(lat: float, lon: float) -> int:
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def ept_points(name: str, box3857: tuple[float, float, float, float]) -> np.ndarray:
    """All points of an EPT dataset inside an EPSG:3857 box -> (N, 4) x, y, z, classification."""
    base = EPT.format(name=name)
    meta = json.loads(get(base + "ept.json"))
    b = meta["bounds"]
    hier: dict[str, int] = {}
    x0, y0, x1, y1 = box3857

    def hits(key: str) -> bool:
        d, X, Y, _Z = map(int, key.split("-"))
        size = (b[3] - b[0]) / 2 ** d
        nx0, ny0 = b[0] + X * size, b[1] + Y * size
        return not (nx0 > x1 or nx0 + size < x0 or ny0 > y1 or ny0 + size < y0)

    def load_hier(key: str):
        h = json.loads(get(base + f"ept-hierarchy/{key}.json"))
        for k, v in h.items():
            if not hits(k):
                continue              # only the octree branch over the site
            if v == -1:
                load_hier(k)          # sub-hierarchy page
            else:
                hier[k] = v
    load_hier("0-0-0-0")
    print(f"  {len(hier)} octree nodes over the site", flush=True)
    out = []
    for key, n in hier.items():
        if n <= 0:
            continue
        las = laspy.read(io.BytesIO(get(base + f"ept-data/{key}.laz")))
        p = np.column_stack([las.x, las.y, las.z, np.asarray(las.classification)])
        m = (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)
        if m.any():
            out.append(p[m])
    return np.concatenate(out) if out else np.zeros((0, 4))


def ortho(e0, n0, e1, n1, epsg: int, px: float, source: str) -> tuple[Image.Image, dict]:
    url, native, note = ORTHOS[source]
    w, h = int(round((e1 - e0) / px)), int(round((n1 - n0) / px))
    tiles, step = [], 2000                         # services cap image size (4096): fetch in tiles
    img = Image.new("RGB", (w, h))
    for ty in range(0, h, step):
        for tx in range(0, w, step):
            tw, th = min(step, w - tx), min(step, h - ty)
            be0, bn1 = e0 + tx * px, n1 - ty * px
            q = (f"?bbox={be0},{bn1 - th * px},{be0 + tw * px},{bn1}&bboxSR={epsg}&imageSR={epsg}"
                 f"&size={tw},{th}&format=png&f=image")
            if "ImageServer" in url:
                q += "&interpolation=RSP_BilinearInterpolation"
            img.paste(Image.open(io.BytesIO(get(url + q))).convert("RGB"), (tx, ty))
    return img, {"source": source, "service": url.rsplit("/", 1)[0], "native_pixel_m": native, "pixel_m": px,
                 "provider": note}


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, cls: np.ndarray):
    n = len(xyz)
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar classification\n"
           "end_header\n").encode()
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"),
                             ("c", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    rec["c"] = cls
    path.write_bytes(hdr + rec.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--radius", type=float, default=110.0, help="half-width of the square site, metres")
    ap.add_argument("--lidar", default="CO_CameronPkFire_1_2021", help="USGS 3DEP EPT dataset name")
    ap.add_argument("--ortho-px", type=float, default=0.1, help="ortho pixel size, metres")
    ap.add_argument("--out", default="data/site")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    epsg = utm_epsg(a.lat, a.lon)
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True)
    m2utm = Transformer.from_crs(3857, epsg, always_xy=True)
    utm2m = Transformer.from_crs(epsg, 3857, always_xy=True)
    E0, N0 = to_utm.transform(a.lon, a.lat)
    r = a.radius
    e0, n0, e1, n1 = E0 - r, N0 - r, E0 + r, N0 + r

    # lidar: query a 3857 box that covers the UTM square, then clip in UTM
    cx, cy = utm2m.transform([e0, e1, e0, e1], [n0, n0, n1, n1])
    print("lidar ...", flush=True)
    p = ept_points(a.lidar, (min(cx) - 5, min(cy) - 5, max(cx) + 5, max(cy) + 5))
    E, N = m2utm.transform(p[:, 0], p[:, 1])
    E, N = np.asarray(E), np.asarray(N)
    keep = (E >= e0) & (E <= e1) & (N >= n0) & (N <= n1) & (p[:, 3] != 7) & (p[:, 3] != 18)   # drop noise
    E, N, Z, C = E[keep], N[keep], p[keep, 2], p[keep, 3].astype(np.uint8)
    print(f"  {len(E):,} points", flush=True)

    # ground grid (1 m) from ground returns; origin height = ground at the origin point
    g = C == 2
    gx = np.arange(-r, r + 1e-6, 1.0)
    GX, GY = np.meshgrid(gx, gx)
    G = griddata((E[g] - E0, N[g] - N0), Z[g], (GX, GY), method="linear")
    if np.isnan(G).any():
        idx = distance_transform_edt(np.isnan(G), return_distances=False, return_indices=True)
        G = G[tuple(idx)]
    z0 = float(griddata((E[g] - E0, N[g] - N0), Z[g], [(0.0, 0.0)], method="linear")[0])
    np.save(out / "ground.npy", (G - z0).astype(np.float32))

    print(f"ortho {a.ortho} ...", flush=True)
    img, nmeta = ortho(e0, n0, e1, n1, epsg, a.ortho_px, a.ortho)
    img.save(out / "ortho.png")
    A = np.asarray(img)
    col = np.clip(((E - e0) / a.ortho_px).astype(int), 0, A.shape[1] - 1)
    row = np.clip(((n1 - N) / a.ortho_px).astype(int), 0, A.shape[0] - 1)
    rgb = A[row, col]

    xyz = np.column_stack([E - E0, N - N0, Z - z0]).astype(np.float32)
    write_ply(out / "site_points.ply", xyz, rgb, C)
    lat_lon_check = Transformer.from_crs(epsg, 4326, always_xy=True).transform(E0, N0)
    site = {
        "frame": "site-enu",
        "origin": {"lat": a.lat, "lon": a.lon, "utm_epsg": epsg, "easting": E0, "northing": N0,
                   "ground_elev_m": z0, "elev_datum": "lidar dataset vertical datum (NAVD88 for USGS 3DEP)"},
        "axes": "x east, y north (UTM grid), z up; metres",
        "radius_m": r,
        "ortho": {"file": "ortho.png", "pixel_m": a.ortho_px, "extent_xy": [-r, r, -r, r],
                  "size": list(img.size)},
        "ground": {"file": "ground.npy", "step_m": 1.0, "extent_xy": [-r, r, -r, r]},
        "points": {"file": "site_points.ply", "count": int(len(xyz)),
                   "classes": {str(k): int((C == k).sum()) for k in np.unique(C)}},
        "sources": {
            "lidar": {"dataset": a.lidar, "url": EPT.format(name=a.lidar), "provider": "USGS 3DEP (public domain)"},
            "ortho": nmeta,
        },
        "check_origin_lonlat": lat_lon_check,
        "built": time.strftime("%Y-%m-%d %H:%M"),
    }
    (out / "site.json").write_text(json.dumps(site, indent=1))
    print(json.dumps({k: site[k] for k in ("points", "ortho")}, indent=1))


if __name__ == "__main__":
    main()

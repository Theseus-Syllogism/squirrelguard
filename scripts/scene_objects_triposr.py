#!/usr/bin/env python3
"""Comparison route for scripts/scene_objects_pixarmesh.py: one object crop -> TripoSR mesh, sized from depth.

  ~/src/pixarmesh/.venv/bin/python scripts/scene_objects_triposr.py FRAME.png REPLAY.json --out DIR \
      [--triposr ~/src/pixarmesh/TripoSR] [--objects 0,1,3] [--ceiling 2.5]

Re-runs the same undistort + depth + Grounding-DINO/SAM step as the PixARMesh script (same object indices as its
debug/masks.jpg), cuts each chosen object out on white with its SAM mask, and runs TripoSR (MIT, Stability AI +
Tripo AI). TripoSR has no metric scale and no pose: the mesh is scaled so its height matches the world-Z extent of
the object's own depth points (5-95th percentile) and placed at their centre, upright, unrotated.
Writes DIR/triposr/<i>_<label>.glb (glTF Y-up) and DIR/triposr.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_objects_pixarmesh as sp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame"); ap.add_argument("replay")
    ap.add_argument("--out", default=str(Path.home() / ".cache/squirrel-guard-test/scene-objects"))
    ap.add_argument("--triposr", default=str(Path.home() / "src/pixarmesh/TripoSR"))
    ap.add_argument("--objects", default="0,1,3")
    ap.add_argument("--prompt", default="chair. desk. table. computer monitor. shelf. lamp. coat. cat tree.")
    ap.add_argument("--ceiling", type=float, default=2.5)
    ap.add_argument("--mc-resolution", type=int, default=256)
    a = ap.parse_args()
    import torch
    import trimesh
    from PIL import Image
    dev = "cuda"
    out = Path(a.out).expanduser(); (out / "triposr").mkdir(parents=True, exist_ok=True)
    c = json.loads(Path(a.replay).read_text())["camera"]
    K, D, R, C = (np.asarray(c[k], float) for k in "KDRC")
    und = cv2.undistort(cv2.imread(a.frame), K, D, None, K); rgb = und[:, :, ::-1].copy(); H, W = rgb.shape[:2]
    depth = sp.depth_metric(rgb, dev)
    ys, xs = np.mgrid[0:H, 0:W]
    rays = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones((H, W))], -1)
    ceil = float(np.median((C + (rays * depth[..., None]) @ R.T)[: H // 6, W // 4: 3 * W // 4, 2]))
    depth *= (a.ceiling - C[2]) / (ceil - C[2]) if a.ceiling > 0 else 1.0
    Pw = C + (rays * depth[..., None]) @ R.T
    boxes, scores, labels, masks = sp.detect_and_segment(rgb, a.prompt, dev)
    keep = [i for i in range(len(boxes)) if masks[i].mean() >= 0.004]

    sys.path.insert(0, str(Path(a.triposr).expanduser()))
    import types
    sys.modules.setdefault("rembg", types.ModuleType("rembg"))   # background removal not needed: SAM mask used
    from tsr.system import TSR
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192); model.to(dev)
    load_s = time.time() - t0
    res = []
    for j in [int(x) for x in a.objects.split(",")]:
        i = keep[j]; t0 = time.time()
        x0, y0, x1, y1 = boxes[i].astype(int); m = masks[i]
        s = int(max(x1 - x0, y1 - y0) * 1.15); cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        canvas = np.full((s, s, 3), 255, np.uint8)       # white background, object centred (TripoSR's expected input)
        sx0, sy0 = cx - s // 2, cy - s // 2
        ya, yb, xa, xb = max(sy0, 0), min(sy0 + s, H), max(sx0, 0), min(sx0 + s, W)
        sub = rgb[ya:yb, xa:xb].copy(); sub[~m[ya:yb, xa:xb]] = 255
        canvas[ya - sy0:yb - sy0, xa - sx0:xb - sx0] = sub
        img = Image.fromarray(canvas).resize((512, 512))
        img.save(out / "triposr" / f"{j:02d}_input.png")
        with torch.no_grad():
            codes = model([img], device=dev)
            mesh = model.extract_mesh(codes, True, resolution=a.mc_resolution)[0]
        # TripoSR frame: Z up (object faces +X). Scale to measured height, centre on the depth points.
        pts = Pw[m & (depth > 0.05) & (depth < 10)]
        lo, hi = np.percentile(pts, 5, 0), np.percentile(pts, 95, 0)
        height = float(hi[2] - lo[2]); pos = 0.5 * (lo + hi)
        v = mesh.vertices - 0.5 * (mesh.vertices.min(0) + mesh.vertices.max(0))
        k = height / max(np.ptp(mesh.vertices[:, 2]), 1e-6); v *= k
        g = trimesh.Trimesh(np.c_[v[:, 0], v[:, 2], -v[:, 1]], mesh.faces, vertex_colors=mesh.visual.vertex_colors, process=False)
        lab = labels[i].replace(" ", "_"); fn = f"{j:02d}_{lab}.glb"; g.export(out / "triposr" / fn)
        dt = time.time() - t0
        res.append(dict(file=f"triposr/{fn}", label=labels[i], position=np.round(pos, 3).tolist(), rotation=[0, 0, 0, 1],
                        scale=[1, 1, 1], size_m=np.round(np.ptp(v, 0), 3).tolist(), depth_extent_m=np.round(hi - lo, 3).tolist(),
                        faces=int(len(mesh.faces)), seconds=round(dt, 1)))
        print(j, labels[i], res[-1]["size_m"], f"{dt:.1f}s", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**20
    (out / "triposr.json").write_text(json.dumps(dict(load_s=round(load_s, 1), peak_vram_mib=round(peak), objects=res), indent=1))
    print(f"peak {peak:.0f} MiB")


if __name__ == "__main__":
    main()

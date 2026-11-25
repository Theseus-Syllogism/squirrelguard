#!/usr/bin/env python3
"""One fixed-camera frame -> approximate 3D Gaussian scene -> renders from new viewpoints (gsplat on the GPU).

A fixed camera has no parallax, so geometry comes from a monocular depth model (Depth Anything V2 Small, Apache-2.0,
relative depth). It is turned into metres with two assumed reference distances (nearest objects / far wall),
so the scale is APPROXIMATE. Each pixel becomes a small Gaussian sized to its pixel footprint. New viewpoints show
real depth structure near the original angle; surfaces hidden from the camera are holes.

  ~/src/splat/.venv/bin/python single_view_3d.py IMAGE OUTDIR [--hfov 105] [--k1 -0.25] [--near 0.9] [--far 7]
"""
from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch


def depth_model(img_rgb: np.ndarray) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    name = "depth-anything/Depth-Anything-V2-Small-hf"
    proc, model = AutoImageProcessor.from_pretrained(name), AutoModelForDepthEstimation.from_pretrained(name).cuda().eval()
    with torch.no_grad():
        out = model(**{k: v.cuda() for k, v in proc(images=img_rgb, return_tensors="pt").items()}).predicted_depth
    disp = torch.nn.functional.interpolate(out[None], size=img_rgb.shape[:2], mode="bicubic", align_corners=False)[0, 0]
    return disp.float().cpu().numpy()                      # relative inverse depth (bigger = nearer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("out")
    ap.add_argument("--hfov", type=float, default=105.0); ap.add_argument("--k1", type=float, default=-0.25)
    ap.add_argument("--near", type=float, default=0.9, help="metres to the nearest 2%% of pixels")
    ap.add_argument("--far", type=float, default=7.0, help="metres to the farthest 3%% of pixels")
    ap.add_argument("--step", type=int, default=3, help="pixel stride for Gaussians")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(a.image)
    H, W = img.shape[:2]
    f = (W / 2) / math.tan(math.radians(a.hfov) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float64)
    D = np.array([a.k1, 0.1 * abs(a.k1), 0, 0, 0])
    und = cv2.undistort(img, K, D)                        # straighten the wide-angle lens (beams become straight)
    cv2.imwrite(str(out / "undistorted.jpg"), und)
    disp = depth_model(und[:, :, ::-1].copy())
    valid = cv2.undistort(np.ones((H, W), np.uint8) * 255, K, D) > 128
    lo, hi = np.percentile(disp[valid], 3), np.percentile(disp[valid], 98)
    # depth = 1 / (alpha * disp + beta), pinned so the far 3 % sit at --far and the near 2 % at --near
    alpha = (1 / a.near - 1 / a.far) / (hi - lo)
    beta = 1 / a.far - alpha * lo
    depth = 1.0 / np.clip(alpha * disp + beta, 1 / 30.0, None)
    vis = cv2.applyColorMap(cv2.normalize(-np.log(depth), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid] = 0
    cv2.imwrite(str(out / "depth.jpg"), vis)

    # Gaussians: one per `step` pixels, skipping lens-undistortion borders and depth cliffs (avoid rubber sheets)
    s = a.step
    ys, xs = np.mgrid[s // 2:H:s, s // 2:W:s]
    d = depth[ys, xs]
    gy, gx = np.gradient(np.log(depth))
    cliff = np.hypot(gx, gy)[ys, xs] > 0.04
    keep = valid[ys, xs] & ~cliff
    xs, ys, d = xs[keep], ys[keep], d[keep]
    P = np.stack([(xs - K[0, 2]) / f * d, (ys - K[1, 2]) / f * d, d], 1).astype(np.float32)   # OpenCV camera frame
    rgb = und[ys, xs, ::-1].astype(np.float32) / 255.0
    dev = "cuda"
    means = torch.from_numpy(P).to(dev)
    sc = torch.from_numpy((d / f * s * 0.75).astype(np.float32)).to(dev)[:, None].repeat(1, 3)
    quats = torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(len(P), 1)
    opac = torch.full((len(P),), 0.95, device=dev)
    cols = torch.from_numpy(rgb).to(dev)
    print(f"{len(P):,} Gaussians; depth {np.percentile(d, 5):.1f}-{np.percentile(d, 95):.1f} m (assumed scale)")

    from gsplat import rasterization
    pivot = np.array([0.0, 0.0, float(np.median(d))])      # orbit around the middle of the room

    def view(yaw_deg, pitch_deg, back=0.0, size=(960, 540)):
        """Camera orbiting the pivot: yaw left/right, pitch from above; returns an RGB image."""
        yaw, pit = math.radians(yaw_deg), math.radians(pitch_deg)
        Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
        Rx = np.array([[1, 0, 0], [0, math.cos(pit), -math.sin(pit)], [0, math.sin(pit), math.cos(pit)]])
        Rwc = Ry @ Rx                                       # camera-to-world rotation about the pivot
        C = pivot + Rwc @ (np.array([0, 0, 0.0]) - pivot + np.array([0, 0, -back]))
        V = np.eye(4); V[:3, :3] = Rwc.T; V[:3, 3] = -Rwc.T @ C
        w, h = size
        k = np.array([[f * w / W, 0, w / 2], [0, f * w / W, h / 2], [0, 0, 1]], np.float32)
        r, alpha, _ = rasterization(means, quats, sc, opac, cols, torch.from_numpy(V.astype(np.float32)).to(dev)[None],
                                    torch.from_numpy(k).to(dev)[None], w, h, packed=True)
        bg = torch.tensor([0.07, 0.08, 0.1], device=dev)
        r = r[0] + (1 - alpha[0]) * bg                      # composite over a dark background (holes stay visible)
        return (r.clamp(0, 1).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8)

    stills = {"original_viewpoint": view(0, 0), "orbit_left_25deg": view(-25, 0, 0.5),
              "orbit_right_25deg": view(25, 0, 0.5), "from_above_35deg": view(0, -35, 1.0)}
    for n, im in stills.items():
        cv2.putText(im, n.replace("_", " "), (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{n}.jpg"), im)
    grid = np.vstack([np.hstack([stills["original_viewpoint"], stills["from_above_35deg"]]),
                      np.hstack([stills["orbit_left_25deg"], stills["orbit_right_25deg"]])])
    cv2.imwrite(str(out / "views_grid.jpg"), cv2.resize(grid, (1440, 810)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    fr = out / "orbit"; fr.mkdir(exist_ok=True)
    for i, t in enumerate(np.linspace(0, 2 * math.pi, 48, endpoint=False)):
        cv2.imwrite(str(fr / f"{i:03d}.png"), view(28 * math.sin(t), -12 + 10 * math.cos(t), 0.6, (480, 270)))
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-framerate", "12", "-i", str(fr / "%03d.png"),
                    "-vf", "split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer", "-loop", "0",
                    str(out / "orbit.gif")], check=True)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()

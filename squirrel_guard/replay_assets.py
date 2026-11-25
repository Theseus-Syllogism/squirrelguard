"""Browser assets for the 3D replay of one visit: the person as video cut-outs on perspective-correct 3D cards, and
the camera's own picture projected onto the ground (deploy/recorder/replay/viewer.js draws both).

Runtime: numpy + cv2 + onnxruntime (person mask) only.

Pipeline
    Collector   on_frame(frame, pose) callback for kinematics.track()/analyse(). While the clip is decoded it keeps
                (a) a person crop of every `every`th analysed frame that has a pose (box grown 12 % / 4 %, at most
                crop_h rows, thinned evenly past 2 x max_cutouts) and (b) a downscaled full frame every bg_every
                frames together with the person box (thinned evenly past 2 x bg_frames). A frame without a pose
                costs one integer increment (and, every bg_every frames, one resize). Peak memory 40-80 MB.
    build()     after trajectory.estimate():
      cut-outs  each crop -> YOLOv8-seg person mask (GaitEmbedder._mask, shared) -> eroded 1 px + Gaussian feather
                -> RGBA tile (<= 96 x 192, feet at the tile bottom) -> ONE WebP atlas with alpha (q80). Each cut-out
                gets a 3D card: the four crop-corner pixel rays (Camera.rays, lens distortion included) meet the
                vertical plane that faces the camera through the person's ground point at that frame. The ground
                point is the SMOOTHED trajectory mean (anchor="smoothed", default): the card then stands on the
                drawn path, frames whose feet are cut off by the image edge still get a card, and because any plane
                works from the camera's viewpoint the error is only vertical: the feet float/sink by about
                camera_height x residual / range (a few cm). anchor="measured" uses the frame's own foot pixel.
                Seen from the Wyze camera the card reproduces the original pixels; elsewhere it is a flat card.
                A card is dropped when the person's confident keypoints land at implausible heights on it (median
                z / nominal adult height outside 0.45-1.6, KP_NOMINAL_Z): a misplaced anchor, typically a close-up
                whose feet are cut off by the frame edge; also when nothing can be checked (no confident keypoint
                above the ankles) and the body runs off the top of the frame.
      ground    a clean background: per-pixel median of the kept full frames with the person's (grown) box masked
                out of each, so the person disappears even when slow; where every frame had the person, the plain
                median. A ground mesh: 0.5 m grid on Z = 0 over the visible ground up to max_range (30 m), plus
                an optional vertical backdrop arc at max_range for everything above the horizon. Per-vertex UVs are
                cv2.projectPoints WITH distortion, divided by the image size (v down: the raw distorted frame is
                the texture, flipY = false). Per-vertex UVs are not perspective-exact inside a triangle; the viewer
                therefore projects every fragment through the camera (same K, D, R, C) and uses the mesh only as
                geometry. Cells are kept when any corner projects inside the image; the shader clips exactly at
                the image border and at max_range.

Formats (all relative to the trajectory JSON, see attach()):
    traj["replay"] = {"version", "frame_size": [W, H], "r2max",
                      "cutouts": {"atlas_size", "tile", "items": [{"t", "frame", "uv": [u0, v0, u1, v1],
                                  "crop": [x1, y1, x2, y2], "corners": [[x, y, z] x 4], "ground": [x, y]}]},
                      "ground": {"image_size", "step", "max_range_m", "cells", "backdrop_cells"},
                      "stats": {...}}
        uv      the tile in the atlas, normalised, v down (image convention); corners TL, TR, BR, BL match
                (u0, v0), (u1, v0), (u1, v1), (u0, v1) and crop's corners in frame pixels
        r2max   largest undistorted normalised radius^2 inside the image (x 1.02): projective texturing beyond it
                would fold back through the distortion polynomial
        t/frame seconds / analysed-frame index on the trajectory's clock (t = frame / fps, like traj["t"])
    traj["assets"] = {"atlas": "<prefix>_replay_atlas.webp", "background": "<prefix>_replay_bg.jpg",
                      "ground": "<prefix>_replay_ground.json"}
    ground JSON   {"version", "image_size", "ground": {"vertices", "uvs", "indices"}, "backdrop": {...} | absent}
                  vertices = flat [x, y, z, ...] world metres (Z up), uvs = flat [u, v, ...], indices = triangles

Sizes: fog demo (1920x1080, 28 s at 7.5 fps, ~100 cut-outs): atlas 1440x1344 WebP 255-265 KB (~2.5 KB a tile),
background 960x540 JPEG q80 55 KB, ground JSON ~120 KB (1840 cells + 260 backdrop cells), cut-out metadata ~25 KB
inside the trajectory JSON; ~0.45 MB on top of the trajectory JSON. Worst case per visit: 120 tiles (~600 KB at
busy detail) + 150 KB background + ~250 KB ground (wide views) < 1.1 MB.
CPU (laptop, i7 class): collector ~0.1 s per visit (crops and background copies only), build ~4-8 s: YOLOv8-seg
~40-75 ms a cut-out (by threads), clean background ~0.5 s, mesh + WebP encode ~0.3 s.

Limits
    flat ground (Z = 0) everywhere; the cut-out is a flat card facing the camera: right from the camera, a paper
    figure from the side and edge-on from 90 degrees; the projection paints everything onto the ground or the
    backdrop, so anything standing up (trees, walls, cars, a parked bike) is smeared away from the camera, and
    shadows/lighting are frozen at the background frames; a person who stands still in the same spot for the whole
    clip stays in the background; segmentation misses (dark, IR, heavy occlusion) drop that cut-out.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .geometry import Camera
from .trajectory import foot_point

VERSION = 1
MAX_RANGE_M = 30.0
# plausibility of a card: confident keypoints must land at a sane height on it (a bad anchor puts e.g. the knees of
# a legs-only close-up 1.3 m up). Nominal adult heights (m) of COCO keypoints; the median ratio z / nominal must be
# within KP_RATIO (0.45 admits a ~0.8 m child, 1.6 a person on a step); ankles are too close to 0 to judge.
KP_NOMINAL_Z = {0: 1.62, 5: 1.43, 6: 1.43, 7: 1.08, 8: 1.08, 9: 0.83, 10: 0.83, 11: 0.92, 12: 0.92, 13: 0.50, 14: 0.50}
KP_RATIO = (0.45, 1.6)


# -- segmentation -----------------------------------------------------------------------------------------------
class PersonMasker:
    """Standalone YOLOv8-seg person masker (models/yolov8s-seg-320.onnx) for when no GaitEmbedder is loaded.
    It reuses GaitEmbedder._mask unchanged: this object carries the four attributes that method reads
    (seg, seg_inp, size, seg_conf). The profiler can pass its GaitEmbedder to build() instead."""

    def __init__(self, seg_model: str, threads: int = 2, seg_conf: float = 0.25):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.seg = ort.InferenceSession(seg_model, so, providers=["CPUExecutionProvider"])
        self.seg_inp = self.seg.get_inputs()[0].name
        self.size = int(self.seg.get_inputs()[0].shape[2])
        self.seg_conf = seg_conf

    def _mask(self, img: np.ndarray, box) -> np.ndarray | None:
        from .gait_embed import GaitEmbedder
        return GaitEmbedder._mask(self, img, box)


# -- collection while the clip is decoded ------------------------------------------------------------------------
class Collector:
    """Pass `collector.on_frame` (or the object itself) as on_frame to kinematics.track()/analyse().

    every       keep a crop of every Nth analysed frame with a pose (8 fps analysis, every=2 -> 4 cut-outs/s)
    max_cutouts build() keeps at most this many; collection thins evenly beyond twice as many
    crop_h      crops are downscaled to at most this many rows (2x the atlas tile height)
    bg_frames   full frames for the background median; one is kept every bg_every frames (doubling as needed)
    Frame indices count every call: with several clip parts the profiler passes frame_offset to build()."""

    def __init__(self, every: int = 2, max_cutouts: int = 120, crop_h: int = 384, bg_frames: int = 9,
                 bg_every: int = 8, bg_width: int = 960):
        self.every, self.max_cutouts, self.crop_h = max(1, every), max_cutouts, crop_h
        self.bg_frames, self.bg_every, self.bg_width = bg_frames, max(1, bg_every), bg_width
        self.crops: list[dict] = []
        self.bg: list[tuple[int, np.ndarray, tuple | None]] = []   # (frame, small frame, person box in small px)
        self.size = (0, 0)
        self.n = 0
        self.ms = 0.0

    def __call__(self, frame: np.ndarray, pose) -> None:
        self.on_frame(frame, pose)

    def on_frame(self, frame: np.ndarray, pose) -> None:
        i = self.n
        self.n += 1
        take_bg = i % self.bg_every == 0
        if pose is None and not take_bg:
            return
        t0 = time.perf_counter()
        H, W = frame.shape[:2]
        self.size = (W, H)
        if take_bg:
            sc = min(1.0, self.bg_width / W)
            small = cv2.resize(frame, (int(round(W * sc)), int(round(H * sc))), interpolation=cv2.INTER_AREA) \
                if sc < 1 else frame.copy()
            box = None if pose is None else tuple(v * sc for v in pose.box)
            self.bg.append((i, small, box))
            if len(self.bg) >= 2 * self.bg_frames:            # keep the spread over the whole visit
                self.bg = self.bg[::2]
                self.bg_every *= 2
        if pose is not None and i % self.every == 0:
            self._crop(i, frame, pose)
        self.ms += (time.perf_counter() - t0) * 1000

    def _crop(self, i: int, frame: np.ndarray, pose) -> None:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = pose.box
        bw, bh = x2 - x1, y2 - y1
        if bh < 0.05 * H or bw < 4:
            return
        X1, Y1 = int(max(0, math.floor(x1 - 0.12 * bw))), int(max(0, math.floor(y1 - 0.04 * bh)))
        X2, Y2 = int(min(W, math.ceil(x2 + 0.12 * bw))), int(min(H, math.ceil(y2 + 0.04 * bh)))
        if X2 - X1 < 8 or Y2 - Y1 < 16:
            return
        sc = min(1.0, self.crop_h / (Y2 - Y1))
        crop = frame[Y1:Y2, X1:X2]
        crop = cv2.resize(crop, (max(1, int(round((X2 - X1) * sc))), max(1, int(round((Y2 - Y1) * sc)))),
                          interpolation=cv2.INTER_AREA) if sc < 1 else crop.copy()
        fp = foot_point(pose, (W, H))
        kp = pose.kp
        check = [(float(kp[j, 0]), float(kp[j, 1]), z) for j, z in KP_NOMINAL_Z.items() if kp[j, 2] >= 0.5]
        self.crops.append({"frame": i, "rect": (X1, Y1, X2, Y2), "img": crop, "foot": fp[0] if fp else None,
                           "check": check, "top_cut": y1 <= 0.01 * H, "box": ((x1 - X1) * sc, (y1 - Y1) * sc, (x2 - X1) * sc, (y2 - Y1) * sc)})
        if len(self.crops) >= 2 * self.max_cutouts:           # thin evenly, keep the time span
            self.crops = self.crops[::2]
            self.every *= 2


# -- geometry -------------------------------------------------------------------------------------------------------
def facing_plane_hits(cam: Camera, pts_px, frame_size, ground_xy) -> np.ndarray | None:
    """Pixels -> (N, 3) world points on the vertical plane through ground_xy whose normal points at the camera
    (the same construction as trajectory.skeleton3d). None when a ray misses the plane (behind / parallel)."""
    G = np.array([ground_xy[0], ground_xy[1], 0.0])
    nrm = np.array([cam.C[0] - G[0], cam.C[1] - G[1], 0.0])
    nn = np.linalg.norm(nrm)
    if nn < 1e-6:
        return None
    nrm /= nn
    d = cam.rays(pts_px, frame_size)
    den = d @ nrm
    if (np.abs(den) < 1e-6).any():
        return None
    s = float(nrm @ (G - cam.C)) / den
    if (s <= 0).any():
        return None
    return cam.C[None, :] + s[:, None] * d


def project(cam: Camera, pts3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points -> (pixels at cam.size WITH distortion, camera depth, undistorted normalised radius^2)."""
    P = np.asarray(pts3, np.float64).reshape(-1, 3)
    Rcw = cam.R.T
    rvec, _ = cv2.Rodrigues(Rcw)
    tvec = (-Rcw @ cam.C).reshape(3, 1)
    px, _ = cv2.projectPoints(P.reshape(-1, 1, 3), rvec, tvec, cam.K, cam.D)
    pc = (P - cam.C) @ cam.R                       # camera coordinates (x right, y down, z forward)
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = (pc[:, 0] ** 2 + pc[:, 1] ** 2) / (z * z)
    return px.reshape(-1, 2), z, r2


def image_r2max(cam: Camera) -> float:
    """Largest undistorted normalised radius^2 over the image border (x 1.02)."""
    W, H = cam.size
    t = np.linspace(0, 1, 33)
    border = np.r_[np.c_[t * W, np.zeros_like(t)], np.c_[t * W, np.full_like(t, H)],
                   np.c_[np.zeros_like(t), t * H], np.c_[np.full_like(t, W), t * H]]
    n = cv2.undistortPoints(border.reshape(-1, 1, 2), cam.K, cam.D).reshape(-1, 2)
    return float((n ** 2).sum(1).max() * 1.02)


def _lattice_mesh(cam: Camera, P: np.ndarray, rows: int, cols: int, r2max: float, valid=None) -> dict:
    """(rows x cols) lattice of world points -> {vertices, uvs, indices} with the cells that have at least one
    corner inside the image and all four in front of the camera. UVs = projected pixel / image size (v down)."""
    W, H = cam.size
    px, z, r2 = project(cam, P)
    front = z > 0.05
    inside = front & (r2 <= r2max) & (px[:, 0] >= 0) & (px[:, 0] <= W) & (px[:, 1] >= 0) & (px[:, 1] <= H)
    if valid is not None:
        inside &= valid
    a = (np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)[None, :]).ravel()
    quad = np.stack([a, a + 1, a + cols, a + cols + 1], 1)
    keep = front[quad].all(1) & inside[quad].any(1)
    quad = quad[keep]
    used = np.unique(quad)
    remap = np.full(len(P), -1, np.int64)
    remap[used] = np.arange(len(used))
    q = remap[quad]
    idx = np.stack([q[:, 0], q[:, 2], q[:, 1], q[:, 1], q[:, 2], q[:, 3]], 1).ravel()
    uv = np.c_[px[used, 0] / W, px[used, 1] / H]
    uv = np.where(np.isfinite(uv), uv, 0.0)
    return {"vertices": np.round(P[used], 2).ravel().tolist(), "uvs": np.round(uv, 4).ravel().tolist(),
            "indices": idx.astype(int).tolist(), "cells": int(len(quad))}


def ground_mesh(cam: Camera, step: float = 0.5, max_range: float = MAX_RANGE_M, backdrop: bool = True) -> dict:
    """Mesh of the camera's visible ground (Z = 0, <= max_range from the camera) on a `step` grid, plus a vertical
    backdrop arc at max_range covering what lies above the horizon. World metres; see the module docstring."""
    W, H = cam.size
    r2max = image_r2max(cam)
    uu, vv = np.meshgrid(np.linspace(0, W, 49), np.linspace(0, H, 28))
    g = cam.ground(np.c_[uu.ravel(), vv.ravel()], (W, H))
    g = g[np.isfinite(g).all(1)]
    if len(g):
        d = g - cam.C[None, :2]
        rng = np.linalg.norm(d, axis=1)
        g = np.r_[g[rng <= max_range], cam.C[None, :2] + d[rng > max_range] * (max_range / rng[rng > max_range, None])]
    out: dict = {"step": step, "max_range_m": max_range, "r2max": r2max}
    if len(g) >= 3:
        x0, y0 = np.floor(g.min(0) / step) * step - step
        x1, y1 = np.ceil(g.max(0) / step) * step + step
        while ((x1 - x0) / step + 1) * ((y1 - y0) / step + 1) > 60000:   # very wide views: coarsen
            step *= 2
        xs, ys = np.arange(x0, x1 + 1e-9, step), np.arange(y0, y1 + 1e-9, step)
        X, Y = np.meshgrid(xs, ys)
        P = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
        near = np.linalg.norm(P[:, :2] - cam.C[None, :2], axis=1) <= max_range + step
        out["step"] = step
        out["ground"] = _lattice_mesh(cam, P, len(ys), len(xs), r2max, valid=near)
    if backdrop:
        # azimuth span of the image border rays, height span of all rays at max_range
        t = np.linspace(0, 1, 25)
        pts = np.r_[np.c_[t * W, np.zeros_like(t)], np.c_[t * W, np.full_like(t, H)],
                    np.c_[np.zeros_like(t), t * H], np.c_[np.full_like(t, W), t * H]]
        d = cam.rays(pts, (W, H))
        hor = np.hypot(d[:, 0], d[:, 1])
        okh = hor > 1e-3
        fwd = cam.R[:, 2]
        az0 = math.atan2(fwd[1], fwd[0]) if math.hypot(fwd[0], fwd[1]) > 1e-3 else 0.0
        az = (np.arctan2(d[okh, 1], d[okh, 0]) - az0 + np.pi) % (2 * np.pi) - np.pi
        z_at = cam.C[2] + max_range * d[okh, 2] / hor[okh]
        z_top = float(min(80.0, z_at.max())) if okh.any() else 0.0
        if z_top > 0.3:
            a_lo, a_hi = float(az.min()) - math.radians(4), float(az.max()) + math.radians(4)
            na = max(8, int(math.ceil((a_hi - a_lo) / math.radians(2.5))) + 1)
            nz = max(3, int(math.ceil(z_top / 1.5)) + 1)
            A, Z = np.meshgrid(az0 + np.linspace(a_lo, a_hi, na), np.linspace(-0.25, z_top * 1.03, nz))   # 25 cm skirt below the ground: no seam at grazing angles
            Q = np.c_[cam.C[0] + max_range * np.cos(A.ravel()), cam.C[1] + max_range * np.sin(A.ravel()), Z.ravel()]
            bd = _lattice_mesh(cam, Q, nz, na, r2max)
            if bd["cells"]:
                out["backdrop"] = bd
    return out


def clean_background(bg: list) -> np.ndarray | None:
    """[(frame, image, person box | None)] -> per-pixel median with each frame's (grown) person box excluded;
    pixels covered by the person in every frame fall back to the plain median."""
    if not bg:
        return None
    imgs = np.stack([b[1] for b in bg])                      # (N, h, w, 3) uint8
    N, h, w = imgs.shape[:3]
    masked = np.zeros((N, h, w), bool)
    for k, (_, _, box) in enumerate(bg):
        if box is None:
            continue
        x1, y1, x2, y2 = box
        gx, gy = 0.15 * (x2 - x1) + 4, 0.08 * (y2 - y1) + 4
        masked[k, int(max(0, y1 - gy)):int(min(h, y2 + gy)), int(max(0, x1 - gx)):int(min(w, x2 + gx))] = True
    plain = np.median(imgs, axis=0).astype(np.uint8) if N > 1 else imgs[0]
    if not masked.any():
        return plain
    valid = (~masked).sum(0)                                 # (h, w)
    # masked samples sort to the end (value 256 in uint16); median index over the valid ones
    s = np.where(masked[..., None], np.uint16(256), imgs.astype(np.uint16))
    s.sort(axis=0)
    lo = np.clip((valid - 1) // 2, 0, N - 1)
    hi = np.clip(valid // 2, 0, N - 1)
    a = np.take_along_axis(s, np.broadcast_to(lo[None, :, :, None], (1, h, w, 3)), 0)[0]
    b = np.take_along_axis(s, np.broadcast_to(hi[None, :, :, None], (1, h, w, 3)), 0)[0]
    med = ((a + b + 1) // 2).astype(np.uint8)
    return np.where((valid > 0)[..., None], med, plain)


# -- build ------------------------------------------------------------------------------------------------------------
def _cutout_rgba(img: np.ndarray, m: np.ndarray, tile: tuple[int, int]) -> np.ndarray:
    tw, th = tile
    h, w = img.shape[:2]
    sc = min(tw / w, th / h, 1.0)
    nw, nh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
    m = cv2.erode(m, np.ones((3, 3), np.uint8))               # pull the edge in by ~1 px before feathering
    rgba = cv2.resize(np.dstack([img, m]), (nw, nh), interpolation=cv2.INTER_AREA)
    rgba[..., 3] = cv2.GaussianBlur(rgba[..., 3], (5, 5), 1.0)
    return rgba


def build(collector: Collector, cam: Camera, traj: dict, masker, fps: float | None = None, frame_offset: int = 0,
          max_cutouts: int | None = None, tile=(96, 192), atlas_quality: int = 80, bg_quality: int = 80,
          anchor: str = "smoothed", ground_step: float = 0.5, max_range: float = MAX_RANGE_M,
          backdrop: bool = True) -> tuple[dict, bytes | None, bytes | None]:
    """-> (replay, atlas_webp | None, background_jpeg | None); replay = {"cutouts", "ground", ...} (module docstring),
    with the ground mesh itself under replay["ground"]["mesh"] (attach() moves it to its own file).

    traj          trajectory.estimate() output for the clip part whose first analysed frame was collector frame
                  frame_offset (0 for single-clip events; else the summed track lengths of the earlier parts)
    masker        anything with ._mask(img, box) -> uint8 mask | None: the profiler's GaitEmbedder or a
                  PersonMasker; None = no cut-outs (background and ground only)
    fps           analysed fps (default traj["fps"])"""
    t_start = time.perf_counter()
    fps = float(fps or traj["fps"])
    W, H = collector.size
    max_cutouts = max_cutouts or collector.max_cutouts
    n_t = len(traj["t"])
    f0 = int(traj["frame0"])
    mean = traj["mean"]
    lo, hi = f0 + frame_offset, f0 + n_t + frame_offset
    cands = [c for c in collector.crops if lo <= c["frame"] < hi]
    if len(cands) > max_cutouts:
        sel = np.unique(np.linspace(0, len(cands) - 1, max_cutouts).round().astype(int))
        cands = [cands[i] for i in sel]
    tiles, items = [], []
    seg_ms, missed, implausible = 0.0, 0, 0
    for c in cands if masker is not None else []:
        local = c["frame"] - frame_offset
        if anchor == "measured" and c["foot"] is not None:
            g = cam.ground([c["foot"]], (W, H))[0]
        else:
            g = np.asarray(mean[local - f0], float)
        if not np.isfinite(g).all():
            continue
        X1, Y1, X2, Y2 = c["rect"]
        P = facing_plane_hits(cam, [[X1, Y1], [X2, Y1], [X2, Y2], [X1, Y2]] + [[u, v] for u, v, _ in c["check"]],
                              (W, H), g)
        if P is None:
            continue
        if c["check"]:
            ratio = float(np.median(P[4:, 2] / np.array([z for _, _, z in c["check"]])))
            if not KP_RATIO[0] <= ratio <= KP_RATIO[1]:
                implausible += 1
                continue
        elif c["top_cut"]:              # nothing to verify and the body runs off the top: a close-up, unreliable
            implausible += 1
            continue
        P = P[:4]
        t0 = time.perf_counter()
        m = masker._mask(c["img"], c["box"])
        seg_ms += (time.perf_counter() - t0) * 1000
        if m is None or m.max() == 0:
            missed += 1
            continue
        tiles.append(_cutout_rgba(c["img"], m, tile))
        items.append({"t": round(local / fps, 3), "frame": int(local), "crop": [int(X1), int(Y1), int(X2), int(Y2)],
                      "corners": np.round(P, 3).tolist(), "ground": [round(float(g[0]), 3), round(float(g[1]), 3)]})
    atlas_bytes, atlas_size = None, [0, 0]
    tw, th = tile
    if tiles:
        cols = max(1, min(len(tiles), int(math.ceil(math.sqrt(len(tiles) * th / tw)))))
        rows = int(math.ceil(len(tiles) / cols))
        atlas = np.zeros((rows * th, cols * tw, 4), np.uint8)
        AW, AH = cols * tw, rows * th
        for k, (im, it) in enumerate(zip(tiles, items)):
            r, cc = divmod(k, cols)
            nh, nw = im.shape[:2]
            x, y = cc * tw + (tw - nw) // 2, r * th + (th - nh)        # feet at the tile bottom
            atlas[y:y + nh, x:x + nw] = im
            it["uv"] = [round(x / AW, 5), round(y / AH, 5), round((x + nw) / AW, 5), round((y + nh) / AH, 5)]
        ok, buf = cv2.imencode(".webp", atlas, [cv2.IMWRITE_WEBP_QUALITY, atlas_quality])
        if ok:
            atlas_bytes, atlas_size = buf.tobytes(), [AW, AH]
    t_bg = time.perf_counter()
    med = clean_background(collector.bg)
    bg_bytes, bg_size = None, [0, 0]
    if med is not None:
        ok, buf = cv2.imencode(".jpg", med, [cv2.IMWRITE_JPEG_QUALITY, bg_quality])
        if ok:
            bg_bytes, bg_size = buf.tobytes(), [med.shape[1], med.shape[0]]
    bg_ms = (time.perf_counter() - t_bg) * 1000
    mesh = ground_mesh(cam, ground_step, max_range, backdrop)
    replay = {"version": VERSION, "frame_size": [W, H], "r2max": round(mesh["r2max"], 5), "fps": fps,
              "cutouts": {"atlas_size": atlas_size, "tile": [tw, th], "anchor": anchor, "items": items,
                          "note": "flat cards facing the camera; exact only from the camera's viewpoint"},
              "ground": {"image_size": bg_size, "step": mesh["step"], "max_range_m": max_range,
                         "cells": (mesh.get("ground") or {}).get("cells", 0),
                         "backdrop_cells": (mesh.get("backdrop") or {}).get("cells", 0),
                         "mesh": {k: mesh[k] for k in ("ground", "backdrop") if k in mesh},
                         "note": "flat ground; anything standing up is smeared onto the ground or the backdrop"},
              "stats": {"cutouts": len(items), "mask_missed": missed, "implausible": implausible, "segmentation_ms": round(seg_ms, 1),
                        "background_ms": round(bg_ms, 1), "collect_ms": round(collector.ms, 1),
                        "build_s": round(time.perf_counter() - t_start, 2),
                        "atlas_bytes": len(atlas_bytes or b""), "background_bytes": len(bg_bytes or b"")}}
    return replay, atlas_bytes, bg_bytes


def attach(traj: dict, replay: dict, atlas: bytes | None, background: bytes | None, out_dir: str | Path,
           prefix: str) -> dict:
    """Write <prefix>_replay_atlas.webp, <prefix>_replay_bg.jpg, <prefix>_replay_ground.json into out_dir (next to
    the trajectory JSON) and add traj["replay"] (cut-out metadata) and traj["assets"] (file names relative to the
    trajectory JSON's URL). Returns traj. Assets that could not be built are simply absent."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    rp = {k: v for k, v in replay.items() if k != "ground"}
    gr = dict(replay["ground"])
    mesh = gr.pop("mesh", {})
    rp["ground"] = gr
    assets = {}
    if atlas and replay["cutouts"]["items"]:
        (d / f"{prefix}_replay_atlas.webp").write_bytes(atlas)
        assets["atlas"] = f"{prefix}_replay_atlas.webp"
    if background:
        (d / f"{prefix}_replay_bg.jpg").write_bytes(background)
        assets["background"] = f"{prefix}_replay_bg.jpg"
    if background and mesh.get("ground"):
        g = {"version": VERSION, "image_size": gr["image_size"], **mesh}
        (d / f"{prefix}_replay_ground.json").write_text(json.dumps(g, separators=(",", ":")))
        assets["ground"] = f"{prefix}_replay_ground.json"
    rp["stats"] = dict(rp["stats"], ground_bytes=(d / assets["ground"]).stat().st_size if "ground" in assets else 0)
    traj["replay"] = rp
    traj["assets"] = assets
    return traj

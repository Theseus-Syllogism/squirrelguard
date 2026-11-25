#!/usr/bin/env python
"""Offline "3D body" for the trajectory replay: WHAM (world-grounded SMPL body from video) on a static-camera clip.

  ~/src/wham/.venv/bin/python scripts/body3d_wham.py CLIP.mp4 [--replay deploy/recorder/replay/demo/basement.json]
        [--out OUT.json] [--name basement] [--wham ~/src/wham] [--kp2d vitpose|yolo] [--flip] [--device cuda|cpu]
  ~/src/wham/.venv/bin/python scripts/body3d_wham.py --setup-smpl models/smpl     (once: SMPL .npz -> WHAM .pkl)

Runs in WHAM's own venv (~/src/wham/.venv: torch + smplx, no mmcv/mmpose/DPVO), never in the project runtime.
Output: the replay JSON (copy) with an optional "body3d" block, plus OUT_DIR/<name>_body.bin (float16 vertices per
mesh frame, relative to that frame's root) and OUT_DIR/smpl_faces_u16.bin (shared triangle list). The .bin files
hold SMPL-derived data (non-commercial licence): they are git-ignored and must stay on the user's machines.

Pipeline (static camera, so WHAM's SLAM/DPVO step is skipped: camera angular velocity = 0, as WHAM's own
--estimate_local_only path does):
  1. person track: squirrel_guard.kinematics.track at the clip's full frame rate (YOLOv8s-pose ONNX, the same
     single-person tracker as the replay) -> one box per frame
  2. 2D keypoints: ViTPose-H (WHAM's checkpoint vitpose-h-multi-coco.pth) re-implemented in plain torch here (WHAM's
     own ViT backbone + the 2-deconv heatmap head, UDP crop, DARK decoding) so mmcv/mmpose are not needed;
     --kp2d yolo uses the tracker's YOLOv8s-pose keypoints instead (noisier, no extra model)
  3. image features + initial SMPL: HMR2.0 ViT-H (hmr2a.ckpt), as WHAM's FeatureExtractor, fp32 (GPU or CPU)
  4. WHAM (wham_vit_bedlam_w_3dpw) on each run of >= 30 frames (gaps <= GAP_FILL frames are bridged with masked
     keypoints); camera intrinsics = the replay's K (keypoints stay in image coordinates: see --undistort), else
     WHAM's default focal sqrt(W^2 + H^2)
  5. placement in the replay world (Z up, metres, ground Z = 0). Default --root fit (see fit_world): walking-leg
     prior where the legs are unseen, then a robust reprojection fit of every body through the replay camera
     (K, D, R, C): first in camera coordinates, then in the world (upright prior); depth from body size + focal
     length. --solve-camera re-estimates the lens, pitch and height from those bodies. The older placements
     (--root camera/image/world) use the ALIGNMENT below.
  6. sampled at the replay's analysed frames (t), written as "body3d"

ALIGNMENT. WHAM returns two things per frame: (a) the body in CAMERA coordinates (global orient + translation from
a full-perspective camera with our focal length; depth comes from the SMPL body size, i.e. from the betas), and
(b) a WORLD trajectory: root orientation + root velocity integrated over time in WHAM's gravity-aligned frame
(Y up; its origin is the first frame's pelvis and its yaw is arbitrary). Our replay world is fixed by the
calibrated camera (R, C, K). We use:
  camera placement  X = R @ x_cam + C for every vertex/joint: exactly consistent with the image, but its depth is
                    noisy frame to frame and it inherits our camera's pitch/height assumptions;
  world placement   WHAM's world motion, rotated so WHAM's up = our +Z ("gravity" mode, default), then yaw and XY
                    translation fitted by 2D Procrustes (no scale) to the camera placement's pelvis path over all
                    frames, then shifted in Z so the feet during predicted contact rest on Z = 0. Smooth, foot-
                    contact aware, upright. --align camera instead maps WHAM's world through WHAM's own estimate of
                    the camera rotation and then our R (uses our gravity instead of WHAM's).
  The angle between WHAM's gravity and our camera's (from R) is reported: it is an independent check of the
  camera pitch, which for the basement clip was itself solved from an assumed 1.75 m walker.
SCALE. WHAM never sees a metric cue except the human body: the SMPL betas fix the body size, and the camera
depth follows from body size x focal / pixel size. The implied stature (SMPL vertex height in the zero pose with
the clip's mean betas) is reported and compared with the 1.75 m assumed by the replay; the ratio of WHAM's
camera distances to the trajectory's says how the two scales differ.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VIS_THRESH = 0.3          # WHAM: keypoints below this confidence are masked
MIN_FRAMES = 30           # WHAM's minimum track length
HMR2_FP32_BYTES = 3.1 * 2 ** 30   # free VRAM needed to run HMR2 (ViT-H) in fp32 on the GPU
ROOT_SMOOTH_S = 0.3      # s: Gaussian sigma for the camera-placed pelvis path (--root camera)
GAP_FILL = 8              # bridge detection gaps up to this many frames (masked keypoints, interpolated box)
VITPOSE_IN = (192, 256)   # W, H
FACES_FILE = "smpl_faces_u16.bin"
VERSION = "1"


# -- SMPL setup ----------------------------------------------------------------------------------------------------
def setup_smpl(src: Path, wham: Path) -> None:
    """SMPL v1.1 NEUTRAL .npz (user-registered download) -> the .pkl smplx loads from WHAM's
    dataset/body_models/smpl. Plain numpy arrays (no chumpy). Only the neutral model is converted: WHAM's inference,
    the fit and every exported mesh/joint use SMPL_NEUTRAL (build_body_model's default gender)."""
    dst = wham / "dataset/body_models/smpl"
    dst.mkdir(parents=True, exist_ok=True)
    for g in ("NEUTRAL",):
        f = src / f"SMPL_{g}.npz"
        if not f.is_file():
            print(f"missing {f}")
            continue
        d = np.load(f, allow_pickle=True)
        out = {k: np.asarray(d[k]) for k in ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "f")}
        out["kintree_table"] = np.asarray(d["kintree_table"]).astype(np.int64)
        out["f"] = out["f"].astype(np.int64)
        with open(dst / f"SMPL_{g}.pkl", "wb") as fh:
            pickle.dump(out, fh, protocol=4)
        print(f"wrote {dst / f'SMPL_{g}.pkl'} (shapedirs {out['shapedirs'].shape})")


# -- video ---------------------------------------------------------------------------------------------------------
def frames(clip: str, n_max: int):
    cap = cv2.VideoCapture(clip)
    try:
        i = 0
        while i < n_max:
            ok, fr = cap.read()
            if not ok:
                break
            yield fr
            i += 1
    finally:
        cap.release()


def simple_track(frs, pose, max_lost: int = 8):
    """Fallback single-person tracker (no torso-quality gate, no appearance check): start with the largest person,
    then follow the detection with the best IoU with the last box (or the nearest centre within one box size).
    For clips where kinematics.track ends early (e.g. the fog stock walk: a low-contrast torso trips its
    weak-torso gate). Returns one Pose|None per frame."""
    out, last, lost = [], None, 0
    for fr in frs:
        cand = pose(fr)
        p = None
        if cand:
            if last is None or lost > max_lost:
                p = max(cand, key=lambda c: (c.box[2] - c.box[0]) * (c.box[3] - c.box[1]))
            else:
                best = max(cand, key=lambda c: _iou(c.box, last))
                if _iou(best.box, last) > 0.1:
                    p = best
                else:
                    lc = np.array([(last[0] + last[2]) / 2, (last[1] + last[3]) / 2])
                    near = min(cand, key=lambda c: np.hypot((c.box[0] + c.box[2]) / 2 - lc[0], (c.box[1] + c.box[3]) / 2 - lc[1]))
                    d = np.hypot((near.box[0] + near.box[2]) / 2 - lc[0], (near.box[1] + near.box[3]) / 2 - lc[1])
                    if d < max(last[2] - last[0], last[3] - last[1]):
                        p = near
        if p is not None:
            last, lost = p.box, 0
        else:
            lost += 1
        out.append(p)
    return out


def _iou(a, b) -> float:
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


# -- ViTPose-H in plain torch -------------------------------------------------------------------------------------
def build_vitpose(ckpt: str, device, dtype):
    import torch
    import torch.nn as nn
    from lib.models.preproc.backbone.vit import vit

    class ViTPose(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = vit()          # ViT-H/16, 256x192: the same backbone as ViTPose-H
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(1280, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
            self.final = nn.Conv2d(256, 17, 1)

        def forward(self, x):
            return self.final(self.deconv(self.backbone(x)))

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    m = ViTPose()
    new = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            new[k] = v
        elif k.startswith("keypoint_head.deconv_layers."):
            new["deconv." + k[len("keypoint_head.deconv_layers."):]] = v
        elif k.startswith("keypoint_head.final_layer."):
            new["final." + k[len("keypoint_head.final_layer."):]] = v
    missing, unexpected = m.load_state_dict(new, strict=False)
    missing = [k for k in missing if "num_batches" not in k]
    if missing:
        raise RuntimeError(f"ViTPose weights missing: {missing[:5]}")
    del sd
    return m.eval().to(device=device, dtype=dtype)


def _udp_box(box, aspect=VITPOSE_IN[0] / VITPOSE_IN[1], pad=1.25):
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    return cx, cy, w * pad, h * pad


def _dark(hm: np.ndarray) -> np.ndarray:
    """(17, h, w) heatmaps -> (17, 3) [x, y, conf] in heatmap pixels: argmax + DARK Taylor refinement."""
    K, h, w = hm.shape
    out = np.zeros((K, 3), np.float32)
    for k in range(K):
        m = hm[k]
        idx = int(np.argmax(m))
        y, x = divmod(idx, w)
        conf = float(m[y, x])
        px, py = float(x), float(y)
        if 1 <= x < w - 1 and 1 <= y < h - 1:
            b = cv2.GaussianBlur(m, (11, 11), 0)
            b = np.log(np.maximum(b, 1e-10))
            dx = 0.5 * (b[y, x + 1] - b[y, x - 1])
            dy = 0.5 * (b[y + 1, x] - b[y - 1, x])
            dxx = b[y, x + 1] - 2 * b[y, x] + b[y, x - 1]
            dyy = b[y + 1, x] - 2 * b[y, x] + b[y - 1, x]
            dxy = 0.25 * (b[y + 1, x + 1] - b[y + 1, x - 1] - b[y - 1, x + 1] + b[y - 1, x - 1])
            det = dxx * dyy - dxy * dxy
            if abs(det) > 1e-12:
                ox = -(dyy * dx - dxy * dy) / det
                oy = -(dxx * dy - dxy * dx) / det
                if abs(ox) < 1 and abs(oy) < 1:
                    px, py = px + ox, py + oy
        out[k] = (px, py, conf)
    return out


def vitpose_keypoints(clip, boxes, n, model, device, dtype, batch=16):
    """boxes: per frame xyxy or None -> per frame (17, 3) keypoints in image pixels (or None)."""
    import torch
    Wi, Hi = VITPOSE_IN
    mean = np.array([0.485, 0.456, 0.406], np.float32) * 255
    std = np.array([0.229, 0.224, 0.225], np.float32) * 255
    out = [None] * n
    buf, meta = [], []

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).to(device=device, dtype=dtype)
        with torch.no_grad():
            hm = model(x).float().cpu().numpy()
        for (i, cx, cy, bw, bh), h in zip(meta, hm):
            kp = _dark(h)
            hh, hw = h.shape[1:]
            # UDP: heatmap px -> input px (scale (W-1)/(hw-1)) -> image px
            ux = kp[:, 0] * (Wi - 1) / (hw - 1)
            uy = kp[:, 1] * (Hi - 1) / (hh - 1)
            kp[:, 0] = (ux - (Wi - 1) / 2) * bw / (Wi - 1) + cx
            kp[:, 1] = (uy - (Hi - 1) / 2) * bh / (Hi - 1) + cy
            out[i] = kp
        buf.clear()
        meta.clear()

    for i, fr in enumerate(frames(clip, n)):
        if boxes[i] is None:
            continue
        cx, cy, bw, bh = _udp_box(boxes[i])
        sx, sy = (Wi - 1) / bw, (Hi - 1) / bh
        M = np.array([[sx, 0, (Wi - 1) / 2 - sx * cx], [0, sy, (Hi - 1) / 2 - sy * cy]], np.float32)
        crop = cv2.warpAffine(fr, M, (Wi, Hi), flags=cv2.INTER_LINEAR)
        crop = (crop[:, :, ::-1].astype(np.float32) - mean) / std
        buf.append(crop.transpose(2, 0, 1))
        meta.append((i, cx, cy, bw, bh))
        if len(buf) >= batch:
            flush()
    flush()
    return out


# -- WHAM inputs ---------------------------------------------------------------------------------------------------
def segments(valid: np.ndarray, gap: int = GAP_FILL, min_len: int = MIN_FRAMES) -> list[tuple[int, int]]:
    """Runs of valid frames, bridging gaps <= gap; only runs >= min_len frames. Returns [(start, end_exclusive)]."""
    idx = np.flatnonzero(valid)
    if not len(idx):
        return []
    runs, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev - 1 > gap:
            runs.append((s, prev + 1))
            s = i
        prev = i
    runs.append((s, prev + 1))
    return [(a, b) for a, b in runs if b - a >= min_len]


def kp_boxes(kp: np.ndarray, valid: np.ndarray, fps: float, s_factor: float = 1.2) -> np.ndarray:
    """WHAM's DetectionModel.compute_bboxes_from_keypoints (+ the median smoothing of process()): square boxes
    (cx, cy, scale = side / 200) from the confident keypoints, interpolated across bridged gaps."""
    import scipy.signal as signal
    n = len(kp)
    bb = np.full((n, 3), np.nan)
    for i in range(n):
        if not valid[i]:
            continue
        m = kp[i, :, 2] > VIS_THRESH
        if m.sum() < 2:
            continue
        x0, y0 = kp[i, m, 0].min(), kp[i, m, 1].min()
        x1, y1 = kp[i, m, 0].max(), kp[i, m, 1].max()
        bb[i] = ((x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * s_factor / 200.0)
    ok = np.isfinite(bb[:, 0])
    t = np.arange(n)
    for c in range(3):
        bb[:, c] = np.interp(t, t[ok], bb[ok, c])
    k = int(int(fps / 2) / 2) * 2 + 1
    return np.array([signal.medfilt(p, k) for p in bb.T]).T


def undistort_kp(kp: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    out = kp.copy()
    if not np.any(D):
        return out
    pts = kp[..., :2].reshape(-1, 1, 2).astype(np.float64)
    u = cv2.undistortPoints(pts, K, D, P=K).reshape(kp.shape[:-1] + (2,))
    out[..., :2] = u
    return out


def hmr2_features(clip, n, seg_frames: dict, bboxes: np.ndarray, model, device, dtype, batch=16, flip=False):
    """WHAM's FeatureExtractor: 256x256 crops at the keypoint boxes -> 1024-d tokens; initial SMPL from the first
    frame of every segment. seg_frames: frame -> segment id."""
    import torch
    from lib.models.preproc.backbone.utils import process_image
    feats, ffeats, inits, finits = {}, {}, {}, {}
    firsts = {}
    for f, s in sorted(seg_frames.items()):
        firsts.setdefault(s, f)
    buf, ids = [], []

    def run(x, ids_, store, init_store):
        # autocast: HMR2's SMPL head builds float32 tokens internally; with fp16 weights they must be cast too
        with torch.no_grad(), torch.autocast(device_type=x.device.type, dtype=x.dtype, enabled=x.dtype == torch.float16):
            tok = model(x, encode=True).float().cpu()
            for j, f in enumerate(ids_):
                store[f] = tok[j]
            fi = [j for j, f in enumerate(ids_) if f in firsts.values()]
            if fi:
                go, bp, be, _ = model(x[fi], encode=False)
                for jj, j in enumerate(fi):
                    init_store[ids_[j]] = (go[jj:jj + 1].float().cpu(), bp[jj:jj + 1].float().cpu(),
                                           be[jj:jj + 1].float().cpu())

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).to(device=device, dtype=dtype)
        run(x, ids, feats, inits)
        if flip:
            run(torch.flip(x, (3,)), ids, ffeats, finits)
        buf.clear()
        ids.clear()

    for i, fr in enumerate(frames(clip, n)):
        if i not in seg_frames:
            continue
        cx, cy, sc = bboxes[i]
        norm, _ = process_image(fr[..., ::-1], [cx, cy], sc, 256, 256)
        buf.append(norm.astype(np.float32))
        ids.append(i)
        if len(buf) >= batch:
            flush()
    flush()
    return feats, inits, ffeats, finits


# -- geometry helpers ----------------------------------------------------------------------------------------------
def procrustes2d(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None):
    """Rotation (yaw) + translation minimising sum w |R a + t - b|^2 for (N, 2) point sets. Returns (R2, t)."""
    w = np.ones(len(a)) if w is None else w
    w = w / w.sum()
    ma, mb = (w[:, None] * a).sum(0), (w[:, None] * b).sum(0)
    H = ((a - ma) * w[:, None]).T @ (b - mb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, mb - R @ ma


def rot_to_aa(Rm: np.ndarray) -> np.ndarray:
    return np.array([cv2.Rodrigues(r)[0].reshape(3) for r in Rm.reshape(-1, 3, 3)]).reshape(Rm.shape[:-2] + (3,))


def foot_sides(smpl, device) -> tuple[list[int], list[int]]:
    """Which of WHAM's 4 foot points (J_regressor_feet) are left/right: SMPL's left is +x in the rest pose."""
    import torch
    with torch.no_grad():
        o = smpl.get_output(global_orient=torch.zeros(1, 3, device=device), body_pose=torch.zeros(1, 69, device=device),
                            betas=torch.zeros(1, 10, device=device))
    x = o.feet[0, :, 0].cpu().numpy()
    return [i for i in range(4) if x[i] > 0], [i for i in range(4) if x[i] <= 0]


# -- image-aligned body: leg prior + reprojection fit + camera re-solve -------------------------------------------
LEG_BP = {"lhip": 0, "rhip": 1, "lknee": 3, "rknee": 4, "lank": 6, "rank": 7, "lfoot": 9, "rfoot": 10}  # body_pose idx
FIT_SIGMA_FRAC = 0.03     # Cauchy scale of the 2D residual, as a fraction of the person's box height (px)
FIT_DELTA_DEG = float(os.environ.get("BODY3D_FIT_DELTA_DEG", 8.0))      # prior std of the per-frame global-orientation correction
FIT_ACC = 3.0             # m/s^2: prior std of the pelvis acceleration (temporal smoothness)
CAMFRAME_DELTA_DEG = 25.0  # camera-frame fit: orientation prior around WHAM's camera-frame root
FIT_DDELTA_DEG = 1.0      # deg per frame: prior std of the change of the orientation correction


def _rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def legs_visible(kpf: np.ndarray) -> bool:
    """Both knees and both ankles detected (ViTPose conf >= VIS_THRESH)."""
    return bool((kpf[[13, 14, 15, 16], 2] >= VIS_THRESH).all())


def walk_legs(t: np.ndarray, speed: np.ndarray, cadence_spm: float) -> np.ndarray:
    """Straight-leg walking prior: (F, 8, 3, 3) rotations for the body_pose slots in LEG_BP order. Stride phase
    integrates cadence (steps/min -> strides/s = cadence / 120); the swing amplitude follows the trajectory speed
    (0 at a standstill: straight legs, 5 deg knee bend). SMPL: +z = forward, hip flexion forward = -x rotation,
    knee flexion = +x rotation."""
    f = cadence_spm / 120.0
    phase = 2 * math.pi * f * (t - t[0])
    amp = np.clip(speed / 1.2, 0.0, 1.0)
    out = np.tile(np.eye(3), (len(t), 8, 1, 1))
    for k in range(len(t)):
        for side, ph in ((0, phase[k]), (1, phase[k] + math.pi)):
            hip = -amp[k] * 22.0 * math.sin(ph)                         # + forward
            knee = 5.0 + amp[k] * 35.0 * max(0.0, math.sin(ph + math.pi / 3)) ** 2
            out[k, side] = _rx(hip)                                     # lhip / rhip
            out[k, 2 + side] = _rx(knee)                                # lknee / rknee
            out[k, 4 + side] = _rx(-0.4 * hip)                          # ankles: keep the foot roughly flat
    return out


def knee_flexion_deg(J: np.ndarray) -> np.ndarray:
    """(F, 17, 3) COCO joints -> (F, 2) knee flexion (0 = straight leg)."""
    out = np.zeros((len(J), 2))
    for i, (h, k, a_) in enumerate(((11, 13, 15), (12, 14, 16))):
        u, v = J[:, h] - J[:, k], J[:, a_] - J[:, k]
        c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
        out[:, i] = 180.0 - np.degrees(np.arccos(np.clip(c, -1, 1)))
    return out


def upright(Rroot: np.ndarray) -> np.ndarray:
    """Minimal rotation of each (3, 3) root orientation so the body's up axis (SMPL +y) is world +Z (yaw kept)."""
    out = Rroot.copy()
    z = np.array([0, 0, 1.0])
    for i, R in enumerate(Rroot):
        u = R @ np.array([0, 1.0, 0])
        ax = np.cross(u, z)
        sn, cs = np.linalg.norm(ax), float(u @ z)
        if sn < 1e-8:
            continue
        rv = ax / sn * math.atan2(sn, cs)
        out[i] = cv2.Rodrigues(rv)[0] @ R
    return out


def project_np(X: np.ndarray, Rw: np.ndarray, Cw: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """World points (..., 3) -> distorted pixels (..., 2) (OpenCV model; NaN behind the camera)."""
    Xc = (X - Cw) @ Rw                                          # = R^T (X - C)
    z = Xc[..., 2:3]
    xy = Xc[..., :2] / np.where(z > 1e-6, z, np.nan)
    d = np.r_[np.asarray(D, float).reshape(-1), np.zeros(5)][:5]
    x, y = xy[..., 0], xy[..., 1]
    r2 = x * x + y * y
    rad = 1 + d[0] * r2 + d[1] * r2 * r2 + d[4] * r2 ** 3
    xd = x * rad + 2 * d[2] * x * y + d[3] * (r2 + 2 * x * x)
    yd = y * rad + d[2] * (r2 + 2 * y * y) + 2 * d[3] * x * y
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], -1)


def reproj_err(Jw: np.ndarray, kpf: np.ndarray, Rw, Cw, K, D) -> np.ndarray:
    """Per visible joint (conf >= VIS_THRESH) pixel distance, flattened over frames."""
    P = project_np(Jw, Rw, Cw, K, D)
    m = kpf[..., 2] >= VIS_THRESH
    e = np.linalg.norm(P - kpf[..., :2], axis=-1)
    return e[m & np.isfinite(e)]


def image_fit(Jrel, Vrel, T0, kpf, Rw, Cw, K, D, fps, iters=300, delta_deg=None):
    """Per frame: pelvis translation T (3) and a small orientation correction (axis-angle, about the pelvis)
    minimising the Cauchy-robust, confidence-weighted 2D reprojection error of the 17 joints through the replay
    camera (K, D, R, C), with priors: correction ~ N(0, FIT_DELTA_DEG), smooth correction, smooth pelvis path
    (acceleration ~ N(0, FIT_ACC)). Joints with conf < VIS_THRESH get no weight. Returns (J, V, T, delta_deg)."""
    import torch
    tt = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
    Jr, kp_, Kt, Rt, Ct = tt(Jrel), tt(kpf), tt(K), tt(Rw), tt(Cw)
    d = np.r_[np.asarray(D, float).reshape(-1), np.zeros(5)][:5]
    F = len(Jrel)
    w = torch.where(kp_[..., 2] >= VIS_THRESH, kp_[..., 2], torch.zeros_like(kp_[..., 2]))
    # residual scale per frame from the visible keypoint extent (person size in pixels)
    hpx = np.array([np.ptp(k[k[:, 2] >= VIS_THRESH, 1]) if (k[:, 2] >= VIS_THRESH).sum() >= 2 else 200.0 for k in kpf])
    hpx_ = np.maximum(hpx, 60.0)
    sig = tt(np.maximum(FIT_SIGMA_FRAC * hpx_, 3.0))[:, None]
    T = tt(T0).clone().requires_grad_(True)
    rv = torch.zeros(F, 3, dtype=torch.float64, requires_grad=True)
    dt = 1.0 / fps

    def rodrigues(v):
        th = v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        k = v / th
        Kx = torch.zeros(F, 3, 3, dtype=v.dtype)
        Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
        Kx[:, 1, 2], Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
        th = th[..., None]
        return torch.eye(3, dtype=v.dtype) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx

    def proj(X):
        Xc = (X - Ct) @ Rt
        z = Xc[..., 2].clamp_min(0.05)
        x, y = Xc[..., 0] / z, Xc[..., 1] / z
        r2 = x * x + y * y
        rad = 1 + d[0] * r2 + d[1] * r2 * r2 + d[4] * r2 ** 3
        xd = x * rad + 2 * d[2] * x * y + d[3] * (r2 + 2 * x * x)
        yd = y * rad + d[2] * (r2 + 2 * y * y) + 2 * d[3] * x * y
        return torch.stack([Kt[0, 0] * xd + Kt[0, 2], Kt[1, 1] * yd + Kt[1, 2]], -1)

    sd = math.radians(FIT_DELTA_DEG if delta_deg is None else delta_deg)
    sdd = math.radians(FIT_DDELTA_DEG)

    def closure():
        opt.zero_grad()
        X = torch.einsum("fij,fkj->fki", rodrigues(rv), Jr) + T[:, None]
        r = (proj(X) - kp_[..., :2]) / sig[..., None]
        data = (w * torch.log1p((r * r).sum(-1))).sum()
        prior = ((rv / sd) ** 2).sum()
        if F > 1:
            prior = prior + (((rv[1:] - rv[:-1]) / sdd) ** 2).sum()
        if F > 2:
            acc = (T[2:] - 2 * T[1:-1] + T[:-2]) / (dt * dt)
            prior = prior + ((acc / FIT_ACC) ** 2).sum()
        loss = data + 0.5 * prior
        loss.backward()
        return loss

    # coarse to fine: a wide robust scale first (the start can be 100+ px off), then the working scale
    for frac in (0.3, 0.1, FIT_SIGMA_FRAC):
        sig = tt(np.maximum(frac * hpx_, 3.0))[:, None]
        opt = torch.optim.LBFGS([T, rv], lr=1, max_iter=iters, line_search_fn="strong_wolfe", tolerance_change=1e-12)
        opt.step(closure)
    with torch.no_grad():
        Rd = rodrigues(rv).numpy()
    Tn = T.detach().numpy()
    J = np.einsum("fij,fkj->fki", Rd, Jrel) + Tn[:, None]
    V = np.einsum("fij,fkj->fki", Rd, Vrel) + Tn[:, None]
    return J, V, Tn, Rd, np.degrees(np.linalg.norm(rv.detach().numpy(), axis=1))


def lens_reach(K: np.ndarray, D, W: int, H: int) -> float:
    """How far the radial model d(r) = r (1 + k1 r^2 + k2 r^4 + k3 r^6) stays invertible, as a fraction of the image
    corner's distorted radius (>= 1: every pixel has a ray; < 1: pixels beyond that radius fold over, so
    cv2.undistortPoints - and the trajectory's rays - return garbage there)."""
    d = np.r_[np.asarray(D, float).reshape(-1), np.zeros(5)][:5]
    r = np.linspace(0, 4, 4001)
    dr = r * (1 + d[0] * r ** 2 + d[1] * r ** 4 + d[4] * r ** 6)
    inc = np.diff(dr) > 0
    stop = int(np.argmin(inc)) if not inc.all() else len(inc)
    corner = math.hypot(max(K[0, 2], W - K[0, 2]) / K[0, 0], max(K[1, 2], H - K[1, 2]) / K[1, 1])
    return float(dr[: stop + 1].max() / corner)


def solve_camera(samples, pitch0: float):
    """samples: list of (P_cam (3,), height above the soles (m)) for pelvis + nose of every fitted frame, positions in
    OpenCV camera coordinates. Finds pitch (deg, + = down; roll 0, as replay_demo's --pitch) and camera height h so
    that world z = (R(pitch) P)_z + h matches those heights (a person of the fitted size walking on a level floor).
    Returns (pitch, h, rms residual m)."""
    from squirrel_guard.geometry import rot_from_angles
    P = np.array([s[0] for s in samples])
    hz = np.array([s[1] for s in samples])
    best = None
    for p in np.arange(pitch0 - 40, pitch0 + 40.01, 0.1):
        if not -89 < p < 89:
            continue
        z = P @ rot_from_angles(float(p), 0.0)[2]
        hc = float(np.median(hz - z))
        res = hz - z - hc
        rms = float(np.sqrt(np.mean(np.minimum(res * res, 0.09))))     # robust (30 cm cap)
        if best is None or rms < best[2]:
            best = (float(p), hc, rms)
    return best


# -- main ----------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--replay", help="replay JSON (scripts/replay_demo.py output): camera model, time base, trajectory")
    ap.add_argument("--out", help="output JSON (default: <replay stem>_body3d.json next to the replay, else "
                    "<clip stem>_body3d.json in the current directory)")
    ap.add_argument("--name", help="asset name prefix (default: the replay's stem)")
    ap.add_argument("--wham", default=str(Path.home() / "src/wham"))
    ap.add_argument("--kp2d", choices=["vitpose", "yolo"], default="vitpose")
    ap.add_argument("--tracker", choices=["kinematics", "simple"], default="kinematics",
                    help="person track; kinematics falls back to simple when it keeps < 30%% of the frames")
    ap.add_argument("--checkpoint", default="checkpoints/wham_vit_bedlam_w_3dpw.pth.tar")
    ap.add_argument("--pose-model", default=str(ROOT / "models/yolov8s-pose.onnx"))
    ap.add_argument("--align", choices=["gravity", "camera"], default="gravity")
    ap.add_argument("--ground", choices=["snap", "constant"], default="snap")
    ap.add_argument("--root", choices=["fit", "camera", "image", "world"], default="fit",
                    help="fit = per-frame reprojection fit of the body (translation + small rotation, walking-leg prior "
                         "where the legs are unseen) through the replay camera; camera = WHAM's camera-frame depth "
                         "(smoothed); image = grounded body, horizontal shift only; world = WHAM's integrated path")
    ap.add_argument("--solve-hfov", default="95", help="--solve-camera: candidate hfov values (deg), comma-separated")
    ap.add_argument("--solve-k1", default="-0.25", help="--solve-camera: candidate k1 values, comma-separated")
    ap.add_argument("--solve-camera", action="store_true", help="with --root fit: re-estimate the camera pitch and "
                    "height from the image-fitted bodies (report only; the output keeps the replay's camera)")
    ap.add_argument("--flip", action="store_true", help="flip test-time augmentation (2x feature time)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true", help="run ViTPose in fp32 (default fp16 on CUDA: keypoints match fp32 "
                    "to 0.01 px median)")
    ap.add_argument("--undistort", action="store_true", help="feed WHAM undistorted keypoints (only for a trustworthy lens "
                    "model: the replay's assumed k1 folds over near the image edge and inflates the person)")
    ap.add_argument("--hmr2-fp16", action="store_true", help="HMR2 features in fp16 (fits next to llama-server, but biases "
                    "the body shape; see HMR2_FP32_BYTES)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-s", type=float, default=40.0)
    ap.add_argument("--mesh-step", type=int, default=1, help="write the mesh every N-th replay frame")
    ap.add_argument("--no-mesh", action="store_true")
    ap.add_argument("--cache", help="directory for the preprocessing (track, keypoints, features) to reuse on re-runs")
    ap.add_argument("--height", type=float, default=1.75, help="stature assumed by the replay (for the report)")
    ap.add_argument("--setup-smpl", metavar="DIR", help="convert DIR/SMPL_{NEUTRAL,MALE}.npz for WHAM and exit")
    a = ap.parse_args()

    wham = Path(a.wham).expanduser().resolve()
    if a.setup_smpl:
        setup_smpl(Path(a.setup_smpl).resolve(), wham)
        return
    if not a.clip:
        ap.error("clip required")
    clip = str(Path(a.clip).resolve())
    replay_path = Path(a.replay).resolve() if a.replay else None
    out_path = Path(a.out).resolve() if a.out else (
        replay_path.with_name(replay_path.stem + "_body3d.json") if replay_path else Path.cwd() / (Path(clip).stem + "_body3d.json"))
    name = a.name or (replay_path.stem if replay_path else Path(clip).stem)
    pose_model = str(Path(a.pose_model).resolve())

    # WHAM resolves its data with relative paths ('dataset/...', 'configs/...')
    os.chdir(wham)
    sys.path.insert(0, str(wham))
    import torch
    from squirrel_guard import kinematics
    from squirrel_guard.pose import PoseEstimator

    dev = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
    dtype = torch.float16 if (dev.type == "cuda" and not a.fp32) else torch.float32
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timing, vram = {}, {}

    def stage_vram(k):
        if dev.type == "cuda":
            vram[k] = round(torch.cuda.max_memory_reserved() / 2 ** 20)
            torch.cuda.reset_peak_memory_stats()

    cap = cv2.VideoCapture(clip)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    n = min(n_total, int(round(a.max_s * src_fps)))

    rep, cam = None, None
    if replay_path:
        rep = json.loads(replay_path.read_text())
        from squirrel_guard.geometry import Camera
        c = rep["camera"]
        cam = Camera(c["K"], c.get("D"), c["R"], c["C"], c["size"])
    if cam is not None:
        sx, sy = W / cam.size[0], H / cam.size[1]
        K = cam.K.copy()
        K[0] *= sx
        K[1] *= sy
        D = cam.D
    else:
        f = math.hypot(W, H)
        K, D = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]]), np.zeros(5)

    cache = Path(a.cache).resolve() / f"{Path(clip).stem}_{a.kp2d}{'_flip' if a.flip else ''}{'_h16' if a.hmr2_fp16 else ''}_{n}.pt" if a.cache else None
    if cache is not None and cache.is_file():
        c_ = torch.load(cache, weights_only=False)
        nf, kp, valid, segs = c_["nf"], c_["kp"], c_["valid"], c_["segs"]
        feats, inits, ffeats, finits = c_["feats"], c_["inits"], c_["ffeats"], c_["finits"]
        timing.update(c_["timing"])
        vram.update(c_["vram"])
        kp_u = undistort_kp(kp, K, D)
        kp_net = kp_u if a.undistort else kp
        bb_net = kp_boxes(kp_net, valid, src_fps)
        print(f"preprocessing loaded from {cache}: segments {segs}")
    else:
        # 1. track --------------------------------------------------------------------------------------------------
        t0 = time.perf_counter()
        # the tracker is tuned for the replay's ~8 fps (its "weak torso" run length and re-acquire logic count
        # frames), so it runs on the same subsampled frames as the replay; boxes in between are interpolated
        step = max(1, int(round(src_fps / (rep["fps"] if rep is not None else 8.0))))
        pe = PoseEstimator(pose_model)
        sub, _ = kinematics.track((fr for i, fr in enumerate(frames(clip, n)) if i % step == 0), pe)
        if a.tracker == "simple" or sum(p is not None for p in sub) < 0.3 * len(sub):
            simple = simple_track((fr for i, fr in enumerate(frames(clip, n)) if i % step == 0), pe)
            print(f"tracker: kinematics.track kept {sum(p is not None for p in sub)} of {len(sub)} frames, "
                  f"simple IoU tracker {sum(p is not None for p in simple)}")
            if a.tracker == "simple" or sum(p is not None for p in simple) > sum(p is not None for p in sub):
                sub = simple
                timing["tracker"] = "simple"
        nf = n
        boxes = [None] * nf
        for j, p in enumerate(sub):
            if p is not None and j * step < nf:
                boxes[j * step] = np.asarray(p.box, float)
        for j in range(len(sub) - 1):
            b0, b1 = sub[j], sub[j + 1]
            if b0 is None or b1 is None:
                continue
            for d in range(1, step):
                f = j * step + d
                if f < nf:
                    boxes[f] = (1 - d / step) * np.asarray(b0.box, float) + d / step * np.asarray(b1.box, float)
        timing["track_s"] = round(time.perf_counter() - t0, 1)
        print(f"track: {len(sub)} frames @ {src_fps / step:.2f} fps, {sum(b is not None for b in boxes)} of {nf} "
              f"full-rate frames with a person box "
              f"({timing['track_s']} s)")

        # 2. 2D keypoints -------------------------------------------------------------------------------------------
        t0 = time.perf_counter()
        if a.kp2d == "vitpose":
            vp = build_vitpose("checkpoints/vitpose-h-multi-coco.pth", dev, dtype)
            kps = vitpose_keypoints(clip, boxes, nf, vp, dev, dtype, a.batch)
            del vp
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        else:
            kps = [None] * nf                      # tracker keypoints exist on the subsampled frames only
            for j, p in enumerate(sub):
                if p is not None and j * step < nf:
                    kps[j * step] = p.kp.astype(np.float32)
        timing["kp2d_s"] = round(time.perf_counter() - t0, 1)
        stage_vram("kp2d")
        kp = np.zeros((nf, 17, 3), np.float32)
        valid = np.zeros(nf, bool)
        for i, k in enumerate(kps):
            if k is not None and (k[:, 2] > VIS_THRESH).sum() >= 6:
                kp[i], valid[i] = k, True
        segs = segments(valid)
        print(f"2D keypoints ({a.kp2d}): {valid.sum()} frames usable, segments {segs} ({timing['kp2d_s']} s)")
        if not segs:
            sys.exit("no segment of >= 30 frames with a usable person")
        # boxes: image (for the crops) and the network input (distorted image coordinates by default: the assumed
        # lens model can be non-invertible near the image edge, see --undistort)
        kp_u = undistort_kp(kp, K, D)
        kp_net = kp_u if a.undistort else kp
        bb_img = kp_boxes(kp, valid, src_fps)
        bb_net = kp_boxes(kp_net, valid, src_fps)
        seg_of = {f: s for s, (a0, a1) in enumerate(segs) for f in range(a0, a1)}

        # 3. features -----------------------------------------------------------------------------------------------
        t0 = time.perf_counter()
        from lib.models.preproc.backbone.hmr2 import hmr2
        # HMR2 runs in fp32: its fp16 tokens differ from fp32 by only ~0.2 % (cosine 0.99999) but the error is
        # systematic, and WHAM's shape output is sensitive to it (basement clip: stature 1.47 m from fp16 features
        # vs 1.76 m from fp32; 0.2 % random noise changes nothing). fp32 ViT-H needs ~2.6 GB of weights: on the GPU
        # when that much is free, else on the CPU (~0.8 s per frame).
        h_dev, h_dtype = dev, torch.float32
        if a.hmr2_fp16 and dev.type == "cuda":
            h_dtype = torch.float16
        elif dev.type == "cuda" and torch.cuda.mem_get_info()[0] < HMR2_FP32_BYTES:
            print(f"features: only {torch.cuda.mem_get_info()[0] / 2 ** 30:.1f} GiB free on the GPU, HMR2 fp32 on the CPU")
            h_dev = torch.device("cpu")
        timing["features_device"] = f"{h_dev.type} {str(h_dtype).split('.')[-1]}"
        hm = hmr2("checkpoints/hmr2a.ckpt").eval().to(device=h_dev, dtype=h_dtype)
        feats, inits, ffeats, finits = hmr2_features(clip, nf, seg_of, bb_img, hm, h_dev, h_dtype,
                                                     a.batch if h_dtype == torch.float16 else max(1, a.batch // 4), a.flip)
        del hm
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        timing["features_s"] = round(time.perf_counter() - t0, 1)
        stage_vram("features")
        print(f"features: {len(feats)} frames ({timing['features_s']} s)")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"nf": nf, "kp": kp, "valid": valid, "segs": segs, "feats": feats, "inits": inits,
                        "ffeats": ffeats, "finits": finits, "timing": dict(timing), "vram": dict(vram)}, cache)

    # 4. WHAM -------------------------------------------------------------------------------------------------------
    t0 = time.perf_counter()
    from configs.config import get_cfg_defaults
    from lib.data.datasets import CustomDataset
    from lib.models import build_body_model, build_network
    from lib.utils.imutils import avg_preds, flip_bbox, flip_kp
    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/yamls/demo.yaml")
    cfg.TRAIN.CHECKPOINT = a.checkpoint
    cfg.DEVICE = dev.type
    cfg.FLIP_EVAL = a.flip
    smpl = build_body_model(dev.type, cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN, gender="neutral")
    assert getattr(smpl, "gender", "neutral") == "neutral", "WHAM must run on SMPL_NEUTRAL"
    _load = torch.load                       # WHAM's build_network torch.load()s without map_location (CUDA tensors)
    torch.load = lambda f, *args, **kw: _load(f, *args, **{"map_location": dev, **kw})
    try:
        network = build_network(cfg, smpl)
    finally:
        torch.load = _load
    network.eval()
    left_ft, right_ft = foot_sides(smpl, dev)

    tracking = {}
    for s, (s0, s1) in enumerate(segs):
        fr_ids = np.arange(s0, s1)
        kpn = kp_net[s0:s1].copy()
        kpn[~valid[s0:s1], :, 2] = 0.0                       # bridged frames: all keypoints masked
        r = {"frame_id": fr_ids, "keypoints": kpn, "bbox": bb_net[s0:s1],
             "features": torch.stack([feats[f] for f in fr_ids])}
        go, bp, be = inits[s0]
        r.update(init_global_orient=go, init_body_pose=bp, init_betas=be)
        if a.flip:
            r["flipped_keypoints"] = flip_kp(kpn.copy(), W)
            r["flipped_bbox"] = np.stack([flip_bbox(b, W, H) for b in bb_net[s0:s1]])
            r["flipped_features"] = torch.stack([ffeats[f] for f in fr_ids])
            go, bp, be = finits[s0]
            r.update(flipped_init_global_orient=go, flipped_init_body_pose=bp, flipped_init_betas=be)
        tracking[s] = r
    slam = np.zeros((nf, 7))
    slam[:, 3] = 1.0                                         # static camera: identity rotation every frame
    ds = CustomDataset(cfg, tracking, slam, W, H, src_fps)
    ds.intrinsics = torch.from_numpy(K).float().unsqueeze(0)

    seg_out = []
    with torch.no_grad():
        for s in range(len(segs)):
            batch = ds.load_data(s)
            _id, x, inits_, features, mask, init_root, cam_angvel, frame_id, kw = batch
            cam_angvel = cam_angvel[:, :x.shape[1]]
            if a.flip:
                fb = ds.load_data(s, True)
                _, fx, finit, ffeat, fmask, finit_root, _, _, fkw = fb
                fpred = network(fx, finit, ffeat, mask=fmask, init_root=finit_root, cam_angvel=cam_angvel,
                                return_y_up=True, **fkw)
                pred = network(x, inits_, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel,
                               return_y_up=True, **kw)
                fp_, fs_ = fpred["pose"].squeeze(0).reshape(-1, 24, 6), fpred["betas"].squeeze(0)
                p_, s_ = pred["pose"].squeeze(0).reshape(-1, 24, 6), pred["betas"].squeeze(0)
                ap_, as_ = avg_preds(p_, s_, fp_, fs_)
                ac = (fpred["contact"][..., [2, 3, 0, 1]] + pred["contact"]) / 2
                network.pred_pose = ap_.reshape(-1, 144).view_as(network.pred_pose)
                network.pred_shape = as_.view_as(network.pred_shape)
                network.pred_contact = ac.view_as(network.pred_contact)
                out = network.forward_smpl(**kw)
                pred = network.refine_trajectory(out, cam_angvel, return_y_up=True)
            else:
                pred = network(x, inits_, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel,
                               return_y_up=True, **kw)
            F = len(frame_id)
            body = pred["poses_body"].reshape(F, 23, 3, 3)
            root_c = pred["poses_root_cam"].reshape(F, 3, 3)
            root_w = pred["poses_root_world"].reshape(F, 3, 3)
            betas = pred["betas"].reshape(F, 10)
            trans_c = pred["trans_cam"].reshape(F, 3)
            trans_w = pred["trans_world"].reshape(F, 3)
            contact = network.pred_contact.reshape(F, 4)
            oc = smpl.get_output(global_orient=root_c[:, None], body_pose=body, betas=betas, pose2rot=False)
            ow = smpl.get_output(global_orient=root_w[:, None], body_pose=body, betas=betas, pose2rot=False)
            seg_out.append({
                "frames": np.asarray(frame_id),
                "body_pose": body.cpu().numpy(), "root_cam": root_c.cpu().numpy(), "root_world": root_w.cpu().numpy(),
                "betas": betas.cpu().numpy(), "trans_cam": trans_c.cpu().numpy(), "trans_world": trans_w.cpu().numpy(),
                "contact": contact.cpu().numpy(),
                "verts_cam": (oc.vertices + trans_c[:, None]).cpu().numpy(),      # hip-centred + camera translation
                "joints_cam": (oc.joints[:, :17] + trans_c[:, None]).cpu().numpy(),
                "feet_cam": (oc.feet + trans_c[:, None]).cpu().numpy(),
                "verts_w": (ow.vertices + trans_w[:, None]).cpu().numpy(),
                "joints_w": (ow.joints[:, :17] + trans_w[:, None]).cpu().numpy(),
                "feet_w": (ow.feet + trans_w[:, None]).cpu().numpy(),
            })
    timing["wham_s"] = round(time.perf_counter() - t0, 2)
    stage_vram("wham")
    print(f"WHAM: {len(segs)} segment(s), {sum(len(s['frames']) for s in seg_out)} frames ({timing['wham_s']} s)")

    # 5. alignment --------------------------------------------------------------------------------------------------
    Rw = cam.R if cam is not None else np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)   # level camera looking +Y
    Cw = cam.C if cam is not None else np.array([0, 0, 0.0])
    YUP2ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)      # WHAM world (Y up) -> Z up
    YDOWN = np.diag([1.0, -1.0, -1.0])                                  # WHAM's yup2ydown (pi about x)
    to_w = lambda P, A, t: P @ A.T + t                                  # noqa: E731
    faces = smpl.faces.astype(np.int64)
    import copy

    def place(Rw, Cw, seg_in):
        """Align WHAM's output into the replay world for camera (Rw, Cw); returns (segments, report)."""
        seg_out = copy.deepcopy(seg_in)
        report_seg = []
        for so in seg_out:
            F = len(so["frames"])
            # camera placement: OpenCV camera -> replay world
            for k in ("verts", "joints", "feet"):
                so[k + "_camplace"] = to_w(so[k + "_cam"], Rw, Cw)
            pel_cam = so["joints_camplace"][:, 11:13].mean(1)              # hip midpoint (WHAM's root offset point)
            # WHAM's own camera rotation: cam_from_worldYdown = root_cam @ root_worldYdown^T
            rw_ydown = YDOWN @ so["root_world"]
            Rcw = so["root_cam"] @ np.transpose(rw_ydown, (0, 2, 1))
            g_cam = Rcw @ np.array([0, 1.0, 0])                             # WHAM gravity (down) in camera coords
            pitch_w = np.degrees(np.arcsin(np.clip(g_cam[:, 2], -1, 1)))    # + = camera looks down
            g_ours = Rw.T @ np.array([0, 0, -1.0])
            tilt = np.degrees(np.arccos(np.clip(g_cam @ g_ours, -1, 1)))
            if a.align == "gravity":
                pw = so["joints_w"][:, 11:13].mean(1) @ YUP2ZUP.T
                R2, t2 = procrustes2d(pw[:, :2], pel_cam[:, :2])
                A = np.eye(3)
                A[:2, :2] = R2
                A = A @ YUP2ZUP
                t = np.r_[t2, 0.0]
            else:
                Rm = _mean_rot(Rcw)
                A = Rw @ Rm @ YDOWN
                pw = so["joints_w"][:, 11:13].mean(1) @ A.T
                t = np.r_[np.median(pel_cam - pw, 0)[:2], 0.0]
            for k in ("verts", "joints", "feet"):
                so[k + "_world"] = to_w(so[k + "_w"], A, t)
            pel_w = so["joints_world"][:, 11:13].mean(1)
            so["pel_integrated"] = pel_w.copy()                              # WHAM's own (integrated) path, aligned
            if a.root in ("camera", "image", "fit"):
                # the pelvis XY follows the camera placement (smoothed over ROOT_SMOOTH_S): where the image puts the
                # person at WHAM's depth; WHAM's integrated root velocity is ignored (it under-walks when the legs are
                # out of frame, see docs/body3d.md)
                sig = max(1.0, ROOT_SMOOTH_S * src_fps)
                ker = np.exp(-0.5 * (np.arange(-int(3 * sig), int(3 * sig) + 1) / sig) ** 2)
                ker /= ker.sum()
                sm = np.stack([np.convolve(np.pad(pel_cam[:, c], len(ker) // 2, mode="edge"), ker, mode="valid")
                               for c in range(2)], 1)
                shift = np.c_[sm - pel_w[:, :2], np.zeros(F)]
                for k in ("verts", "joints", "feet"):
                    so[k + "_world"] = so[k + "_world"] + shift[:, None, :]
            # ground: the lowest vertex during predicted foot contact -> Z = 0. --ground snap (default) does this per
            # frame (offsets interpolated between contact frames, Gaussian-smoothed): it removes the vertical drift of
            # WHAM's integrated root path, which appears when the real camera is not level (WHAM's static-camera world
            # is level with the camera). --ground constant uses one offset for the whole run.
            con = so["contact"] > 0.5
            low = so["verts_world"][:, :, 2].min(1)
            anyc = con.any(1)
            if anyc.sum() < 3:
                anyc = low <= np.percentile(low, 10)
            dz_c = -float(np.median(low[anyc]))
            fi = np.arange(len(low))
            dz = np.interp(fi, fi[anyc], -low[anyc])
            sig = max(1.0, 0.2 * src_fps)
            ker = np.exp(-0.5 * (np.arange(-int(3 * sig), int(3 * sig) + 1) / sig) ** 2)
            ker /= ker.sum()
            dz = np.convolve(np.pad(dz, len(ker) // 2, mode="edge"), ker, mode="valid")
            drift = float(dz.max() - dz.min())        # how far WHAM's own path leaves a level floor
            if a.ground != "snap":
                dz = np.full(len(low), dz_c)
            for k in ("verts", "joints", "feet"):
                so[k + "_world"][..., 2] += dz[:, None]
            if a.root == "image":
                # grounded AND image-consistent: keep the upright, floor-standing body and solve, per frame, the
                # horizontal shift (tx, ty) that best re-projects its 17 joints onto the 2D keypoints through OUR camera
                # (linear least squares, confidence-weighted, on undistorted normalised coordinates), then smooth it.
                # Distance from the camera then follows from WHAM's body size + the floor + our camera, the same
                # ingredients as the trajectory's head-plane placement but with the fitted body instead of 1.75 m.
                Rcw_ours = Rw.T
                Kinv = np.linalg.inv(K)
                T = np.full((F, 2), np.nan)
                for j, f in enumerate(so["frames"]):
                    if not valid[f]:
                        continue
                    kpf = kp_u[f]
                    ok = kpf[:, 2] > VIS_THRESH
                    if ok.sum() < 4:
                        continue
                    uv = (np.c_[kpf[ok, :2], np.ones(ok.sum())] @ Kinv.T)[:, :2]
                    P = so["joints_world"][j][ok] - Cw
                    w = np.sqrt(kpf[ok, 2])
                    rows, rhs = [], []
                    for (u, v), Pj, wj in zip(uv, P, w):
                        for rr in (Rcw_ours[0] - u * Rcw_ours[2], Rcw_ours[1] - v * Rcw_ours[2]):
                            rows.append(wj * rr[:2])
                            rhs.append(-wj * rr @ Pj)
                    sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
                    T[j] = sol
                okT = np.isfinite(T[:, 0])
                if okT.sum() >= 3:
                    fi = np.arange(F)
                    T = np.stack([np.interp(fi, fi[okT], T[okT, c]) for c in range(2)], 1)
                    T = np.stack([_smooth(T[:, c], ROOT_SMOOTH_S * src_fps) for c in range(2)], 1)
                    for k in ("verts", "joints", "feet"):
                        so[k + "_world"][..., :2] += T[:, None, :]
            so["A"], so["t"] = A, t
            so["root_world_ours"] = A @ so["root_world"]                    # (F, 3, 3) global orient in replay world
            report_seg.append({"frames": [int(so["frames"][0]), int(so["frames"][-1])],
                               "wham_pitch_deg_median": round(float(np.median(pitch_w)), 1),
                               "gravity_vs_camera_deg_median": round(float(np.median(tilt)), 1),
                               "ground_shift_m_median": round(dz_c, 3),
                               "vertical_drift_m": round(drift, 3)})
        return seg_out, report_seg

    def refit(seg_out, Rw, Cw, K_=None, D_=None, camframe=False):
        """5b. image-aligned body (--root fit): walking-leg prior where the legs are not seen, upright torso there,
        then the reprojection fit of every frame through the replay camera. Adds before/after numbers."""
        K_ = K if K_ is None else K_
        D_ = D if D_ is None else D_
        info = {"err_before": [], "err_after": [], "knee_before": [], "knee_after": [], "knee_before_all": [],
                "replaced": 0, "frames": 0, "delta_deg": [], "sole_z_legs_seen": [], "samples": []}
        rep_t = np.asarray(rep["t"], float) if rep is not None else None
        rep_speed = np.asarray(rep["speed"], float) if rep is not None else None
        cad = float(rep["metrics"].get("cadence_spm") or 110.0) if rep is not None else 110.0
        for so in seg_out:
            fs = so["frames"]
            F = len(fs)
            kpf = kp[fs].copy()
            kpf[~valid[fs], :, 2] = 0.0
            if camframe:                          # camera coordinates: WHAM's camera-frame body is the start
                Rw, Cw = np.eye(3), np.zeros(3)
                so["joints_world"] = so["joints_cam"]
                so["root_world_ours"] = so["root_cam"]
            info["err_before"].append(reproj_err(so["joints_world"], kpf, Rw, Cw, K_, D_))
            kb = knee_flexion_deg(so["joints_world"])
            seen = np.array([legs_visible(k) for k in kpf])
            tsec = fs / src_fps
            speed = np.interp(tsec, rep_t, rep_speed) if rep is not None else np.ones(F)
            prior = walk_legs(tsec, speed, cad)
            bp = so["body_pose"].copy()
            root = so["root_world_ours"].copy()
            rep_idx = ~seen
            for slot_i, slot in enumerate(LEG_BP.values()):
                if slot_i < 8:
                    bp[rep_idx, slot] = prior[rep_idx, slot_i] if slot_i < 6 else np.eye(3)
            if not camframe:
                root[rep_idx] = upright(root[rep_idx])
            with torch.no_grad():
                o = smpl.get_output(global_orient=torch.tensor(root[:, None], dtype=torch.float32, device=dev),
                                    body_pose=torch.tensor(bp, dtype=torch.float32, device=dev),
                                    betas=torch.tensor(so["betas"], dtype=torch.float32, device=dev), pose2rot=False)
            Jrel, Vrel, Frel = (o.joints[:, :17].cpu().numpy().astype(np.float64), o.vertices.cpu().numpy().astype(np.float64),
                                o.feet.cpu().numpy().astype(np.float64))
            T0 = so["joints_world"][:, 11:13].mean(1)
            J, V, T, Rd, ddeg = image_fit(Jrel, Vrel, T0, kpf, Rw, Cw, K_, D_, src_fps,
                                          delta_deg=CAMFRAME_DELTA_DEG if camframe else None)
            so["joints_world"], so["verts_world"] = J, V
            so["feet_world"] = np.einsum("fij,fkj->fki", Rd, Frel) + T[:, None]
            so["root_world_ours"] = Rd @ root
            so["body_pose"] = bp
            so["legs_seen"] = seen
            info["err_after"].append(reproj_err(J, kpf, Rw, Cw, K_, D_))
            ka = knee_flexion_deg(J)
            info["knee_before"].append(kb[rep_idx].ravel())
            info["knee_after"].append(ka[rep_idx].ravel())
            info["knee_before_all"].append(kb.ravel())
            info["replaced"] += int(rep_idx.sum())
            info["frames"] += F
            info["delta_deg"].append(ddeg)
            info["sole_z_legs_seen"].append(V[seen].min(1)[:, 2] if seen.any() else np.zeros(0))
            # camera re-solve samples: pelvis and nose in camera coordinates with their height above the soles,
            # measured along the body's own up axis (SMPL +y of the fitted root)
            up = (Rd @ root)[:, :, 1]
            lowz = np.einsum("fkj,fj->fk", np.einsum("fij,fkj->fki", Rd, Vrel), up).min(1)
            Jz = np.einsum("fkj,fj->fk", np.einsum("fij,fkj->fki", Rd, Jrel), up)
            for j in range(F):
                k_ = kpf[j]
                if (k_[[11, 12], 2] >= VIS_THRESH).all():
                    info["samples"].append(((T[j] - Cw) @ Rw, float(0.0 - lowz[j])))
                if k_[0, 2] >= VIS_THRESH:
                    info["samples"].append(((J[j, 0] - Cw) @ Rw, float(Jz[j, 0] - lowz[j])))
        return seg_out, info

    def fit_world(Rw, Cw, K_=None, D_=None, seg_old=None):
        """--root fit: (1) every body fitted in CAMERA coordinates through the lens (WHAM's camera-frame body as the
        start; depth from body size and focal length, not from the camera height or pitch), (2) mapped into the
        world with (R, C), torso made upright where the legs were replaced, (3) the reprojection fit again in the
        world with a small orientation prior. WHAM's integrated world path and its yaw are not used for placement.
        seg_old: the previous (gravity + contact) placement, for the 'before' error."""
        seg_c, info_c = refit(copy.deepcopy(seg_wham), None, None, K_, D_, camframe=True)
        for so in seg_c:
            for k in ("joints", "verts", "feet"):
                so[k + "_world"] = so[k + "_world"] @ Rw.T + Cw
            so["root_world_ours"] = Rw @ so["root_world_ours"]
            so["joints_camplace"] = so["joints_cam"] @ Rw.T + Cw
        seg_w, info_w = refit(seg_c, Rw, Cw, K_, D_)
        info = dict(info_w)
        info["knee_before"], info["knee_before_all"] = info_c["knee_before"], info_c["knee_before_all"]
        info["err_camframe"] = info_c["err_after"]
        if seg_old is not None:
            K2 = K if K_ is None else K_
            D2 = D if D_ is None else D_
            info["err_before"] = []
            for so in seg_old:
                kpf = kp[so["frames"]].copy()
                kpf[~valid[so["frames"]], :, 2] = 0.0
                info["err_before"].append(reproj_err(so["joints_world"], kpf, Rw, Cw, K2, D2))
        return seg_w, info

    def summ(info):
        cat = lambda k: np.concatenate(info[k]) if info[k] else np.zeros(0)  # noqa: E731
        eb, ea = cat("err_before"), cat("err_after")
        kb, ka = cat("knee_before"), cat("knee_after")
        sz = cat("sole_z_legs_seen")
        r = lambda v: round(float(v), 1)  # noqa: E731
        return {"reproj_px_before": {"mean": r(eb.mean()), "median": r(np.median(eb)), "n": int(len(eb))} if len(eb) else None,
                "reproj_px_after": {"mean": r(ea.mean()), "median": r(np.median(ea)), "n": int(len(ea))} if len(ea) else None,
                "reproj_px_camera_frame_fit": ({"mean": r(cat("err_camframe").mean()), "median": r(np.median(cat("err_camframe")))}
                                               if info.get("err_camframe") else None),
                "frames_legs_replaced": info["replaced"], "frames": info["frames"],
                "knee_flexion_deg_replaced_frames": {"before_median": r(np.median(kb)), "after_median": r(np.median(ka))} if len(kb) else None,
                "knee_flexion_deg_all_frames_before_median": r(np.median(cat("knee_before_all"))),
                "orientation_correction_deg": {"median": r(np.median(cat("delta_deg"))), "p90": r(np.percentile(cat("delta_deg"), 90))},
                "sole_z_m_legs_seen_median": round(float(np.median(sz)), 3) if len(sz) else None}

    seg_out, report_seg = place(Rw, Cw, seg_wham := seg_out)
    fit_info, camera_solve = None, None
    if a.root == "fit":
        seg_out, fit_info = fit_world(Rw, Cw, seg_old=seg_out)
        print("image fit:", json.dumps(summ(fit_info)))
        if a.solve_camera and cam is not None:
            # camera re-solve, decoupled from the replay's camera: (1) per candidate lens (hfov, k1 with replay_demo's
            # k2 = 0.1 |k1|) fit every body in CAMERA coordinates (WHAM's camera-frame body as the start, orientation
            # free within CAMFRAME_DELTA_DEG, walking-leg prior where the legs are unseen); the lens with the lowest
            # median joint reprojection error wins; (2) the fitted pelvis + nose positions (metric: depth from body
            # size and focal length) must lie at their standing heights above a level floor: least squares for pitch
            # (roll 0) and camera height; (3) the upright body is fitted again through the solved camera
            from squirrel_guard.geometry import nominal_K, rot_from_angles
            fwd = cam.R[:, 2]
            p0 = math.degrees(math.asin(-fwd[2]))
            grid, best = [], None
            for hf in [float(v) for v in a.solve_hfov.split(",")]:
                for k1 in [float(v) for v in a.solve_k1.split(",")]:
                    K_l = nominal_K(W, H, hf)
                    D_l = np.array([k1, 0.1 * abs(k1), 0, 0, 0])
                    _, inf = refit(copy.deepcopy(seg_wham), None, None, K_l, D_l, camframe=True)
                    pitch, hc, rms = solve_camera(inf["samples"], p0)
                    sm_ = summ(inf)
                    row = {"hfov_deg": hf, "k1": k1, "lens_reach": round(lens_reach(K_l, D_l, W, H), 3),
                           "camframe_reproj_px_median": sm_["reproj_px_after"]["median"],
                           "camframe_reproj_px_mean": sm_["reproj_px_after"]["mean"],
                           "pitch_deg": round(pitch, 2), "height_m": round(hc, 3), "floor_rms_m": round(rms, 3)}
                    print("camera solve:", json.dumps(row))
                    grid.append(row)
                    ok_lens = row["lens_reach"] >= 1.0          # an invertible lens model over the whole image
                    if ok_lens and (best is None or row["camframe_reproj_px_median"] < best["camframe_reproj_px_median"]):
                        best = row
            K_b = nominal_K(W, H, best["hfov_deg"])
            D_b = np.array([best["k1"], 0.1 * abs(best["k1"]), 0, 0, 0])
            Rs, Cs = rot_from_angles(best["pitch_deg"], 0.0), np.array([0.0, 0.0, best["height_m"]])
            _, inf = fit_world(Rs, Cs, K_b, D_b, seg_old=place(Rs, Cs, seg_wham)[0])
            camera_solve = {"best": best, "fit_with_best_camera": summ(inf), "grid": grid,
                            "note": "lens = argmin median joint reprojection error of the camera-frame fit; pitch/height "
                                    "= pelvis + nose of those bodies at their standing heights over a level floor (roll "
                                    "0); the output below still uses the replay's own camera"}
            print("camera solve best:", json.dumps(best), json.dumps(summ(inf)))

    # 6. sample at the replay's frames + numbers ---------------------------------------------------------------------
    fmap = {}
    for s, so in enumerate(seg_out):
        for j, f in enumerate(so["frames"]):
            fmap[int(f)] = (s, j)
    if rep is not None:
        step = max(1, int(round(src_fps / rep["fps"])))
        src_frames = [(rep["frame0"] + k) * step for k in range(len(rep["t"]))]
    else:
        step = max(1, int(round(src_fps / 8.0)))
        src_frames = list(range(0, nf, step))
    joints, roots, contacts, smpl_go, smpl_bp, smpl_tr, mesh_idx, mesh_verts = [], [], [], [], [], [], [], []
    legs_prior = []
    for k, f in enumerate(src_frames):
        if f not in fmap:
            for L in (joints, roots, contacts, smpl_go, smpl_bp, smpl_tr, legs_prior):
                L.append(None)
            continue
        s, j = fmap[f]
        so = seg_out[s]
        J = so["joints_world"][j]
        root = J[11:13].mean(0)
        joints.append(np.round(J, 3).tolist())
        roots.append(np.round(root, 3).tolist())
        c = so["contact"][j]
        contacts.append([round(float(c[left_ft].max()), 2), round(float(c[right_ft].max()), 2)])
        legs_prior.append(bool(not so["legs_seen"][j]) if "legs_seen" in so else False)
        smpl_go.append(np.round(rot_to_aa(so["root_world_ours"][j]), 4).tolist())
        smpl_bp.append(np.round(rot_to_aa(so["body_pose"][j]).reshape(-1), 4).tolist())
        smpl_tr.append(None)    # filled below in one batched SMPL call
        if not a.no_mesh and (k % a.mesh_step == 0):
            mesh_idx.append(k)
            mesh_verts.append((so["verts_world"][j] - np.asarray(roots[-1])).astype(np.float16))
    # SMPL 'transl' in the replay world: verts = SMPL(global_orient, body_pose, betas) + transl (standard smplx)
    ks = [k for k, v in enumerate(joints) if v is not None]
    if ks:
        go = torch.tensor(np.array([smpl_go[k] for k in ks]), dtype=torch.float32, device=dev)
        bp = torch.tensor(np.array([smpl_bp[k] for k in ks]), dtype=torch.float32, device=dev)
        be = torch.tensor(np.array([seg_out[fmap[src_frames[k]][0]]["betas"][fmap[src_frames[k]][1]] for k in ks]),
                          dtype=torch.float32, device=dev)
        with torch.no_grad():
            raw = super(type(smpl), smpl).forward(global_orient=go, body_pose=bp, betas=be).vertices.cpu().numpy()
        for r, k in zip(raw, ks):
            s, j = fmap[src_frames[k]]
            smpl_tr[k] = np.round((seg_out[s]["verts_world"][j] - r).mean(0), 4).tolist()

    # stature from the mean betas (zero pose), overall and per segment
    def stature_of(b):
        with torch.no_grad():
            o = super(type(smpl), smpl).forward(betas=torch.tensor(b[None], dtype=torch.float32, device=dev),
                                                body_pose=torch.zeros(1, 69, device=dev),
                                                global_orient=torch.zeros(1, 3, device=dev))
        v0 = o.vertices[0].cpu().numpy()
        return float(v0[:, 1].max() - v0[:, 1].min())
    allb = np.concatenate([so["betas"] for so in seg_out])
    mb = allb.mean(0)
    stature = stature_of(mb)
    for so, rs in zip(seg_out, report_seg):
        rs["stature_m"] = round(stature_of(so["betas"].mean(0)), 3)

    stats = {"frames_src": nf, "frames_wham": int(sum(len(s["frames"]) for s in seg_out)),
             "replay_frames_with_body": len(ks), "replay_frames": len(src_frames),
             "segments": report_seg, "stature_m": round(stature, 3), "stature_assumed_m": a.height,
             "betas_std_over_time": round(float(allb.std(0).mean()), 3), "timing_s": timing,
             "peak_vram_mib_by_stage": vram, "device": dev.type, "dtype": str(dtype).split(".")[-1],
             "kp2d": a.kp2d, "align": a.align, "ground": a.ground, "root": a.root, "flip": a.flip,
             "model": "SMPL neutral"}
    if fit_info is not None:
        stats["image_fit"] = summ(fit_info)
    if camera_solve is not None:
        stats["camera_solve"] = camera_solve
    # foot contacts: share of frames with contact, foot slide while in contact, contact-foot height spread
    con_all = np.concatenate([s["contact"] for s in seg_out]) > 0.5
    feet_w = np.concatenate([s["feet_world"] for s in seg_out])
    slide = []
    for so in seg_out:
        fw, cc = so["feet_world"], so["contact"] > 0.5
        d = np.linalg.norm(np.diff(fw[:, :, :2], axis=0), axis=2) * src_fps        # m/s per foot point
        m = cc[1:] & cc[:-1]
        if m.any():
            slide.extend(d[m].tolist())
    stats["contact"] = {"frames_any_contact_pct": round(100 * float(con_all.any(1).mean()), 1),
                        "frames_left_pct": round(100 * float(con_all[:, left_ft].any(1).mean()), 1),
                        "frames_right_pct": round(100 * float(con_all[:, right_ft].any(1).mean()), 1),
                        "slide_mps_median": round(float(np.median(slide)), 3) if slide else None,
                        "slide_mps_p90": round(float(np.percentile(slide, 90)), 3) if slide else None,
                        "contact_foot_z_iqr_m": [round(float(v), 3) for v in np.percentile(feet_w[:, :, 2][con_all], [25, 75])]
                        if con_all.any() else None}
    if rep is not None:
        obs = np.array(rep["observed"], bool)
        mean = np.array(rep["mean"])
        sel = [k for k in ks if obs[k]]
        if len(sel) >= 3:
            r_w = np.array([roots[k][:2] for k in sel])
            r_c = []
            for k in sel:
                s, j = fmap[src_frames[k]]
                r_c.append(seg_out[s]["joints_camplace"][j, 11:13].mean(0)[:2])
            r_c = np.array(r_c)
            m_ = mean[sel]
            rmse = lambda u, v: round(float(np.sqrt(np.mean(np.sum((u - v) ** 2, 1)))), 3)  # noqa: E731
            R2, t2 = procrustes2d(r_w, m_)
            Cxy = np.asarray(Cw[:2])
            dist_w = np.median(np.linalg.norm(r_w - Cxy, axis=1))
            dist_t = np.median(np.linalg.norm(m_ - Cxy, axis=1))
            stats["vs_trajectory"] = {
                "frames": len(sel),
                "rmse_world_m": rmse(r_w, m_), "rmse_camera_placement_m": rmse(r_c, m_),
                "rmse_world_after_rigid_fit_m": rmse(r_w @ R2.T + t2, m_),
                # WHAM's ranges scale with its body size: rescale about the camera to the replay's assumed stature
                "rmse_world_scaled_to_assumed_height_m": rmse(Cxy + (r_w - Cxy) * (a.height / stature), m_),
                "median_ground_distance_from_camera_m": {"wham": round(float(dist_w), 2), "trajectory": round(float(dist_t), 2)},
                "note": "WHAM pelvis (hip midpoint) XY vs trajectory mean over the trajectory's observed frames; "
                        "world = aligned WHAM world motion (its yaw/XY fitted to WHAM's own camera placement, not to "
                        "the trajectory); rigid_fit = after a best 2D rotation+translation onto the trajectory (path "
                        "shape only)"}
    print("stats:", json.dumps(stats))

    # 7. write ------------------------------------------------------------------------------------------------------
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    body3d = {
        "version": VERSION,
        "method": "WHAM (github.com/yohanshin/WHAM, MIT) bedlam+3DPW checkpoint, static camera (no SLAM), "
                  f"2D keypoints {'ViTPose-H' if a.kp2d == 'vitpose' else 'YOLOv8s-pose'}, HMR2.0 features; "
                  "SMPL neutral, 10 betas",
        "licence": "SMPL-derived (non-commercial, user's own registration): keep on the user's machines",
        "frame": "replay world: metres, Z up, ground Z = 0; per entry of t (null = no body for that frame)",
        "model": "SMPL neutral (v1.1, first 10 betas)",
        "align": a.align, "ground": a.ground, "root_mode": a.root,
        "joints_format": "COCO-17 (same order as skeleton)",
        "joints": joints, "root": roots,
        "contact": contacts, "contact_note": "[left, right] foot contact probability (WHAM), > 0.5 = planted",
        "legs_prior": legs_prior,
        "legs_prior_note": "true = knees/ankles not seen (ViTPose conf < 0.3): straight-leg walking prior instead of "
                           "WHAM's guessed legs, torso upright, feet NOT forced onto the floor",
        "smpl": {"betas": np.round(mb, 4).tolist(), "global_orient": smpl_go, "body_pose": smpl_bp, "transl": smpl_tr,
                 "note": "axis-angle; vertices = SMPL_neutral(global_orient, body_pose, betas) + transl in the replay world"},
        "stats": stats,
    }
    if mesh_idx:
        vb = np.stack(mesh_verts)
        (out_dir / f"{name}_body.bin").write_bytes(vb.astype("<f2").tobytes())
        (out_dir / FACES_FILE).write_bytes(faces.astype("<u2").tobytes())
        body3d["mesh"] = {"vertices": f"{name}_body.bin", "faces": FACES_FILE, "vertex_count": int(vb.shape[1]),
                          "face_count": int(len(faces)), "index": mesh_idx,
                          "encoding": "float16 little-endian, (len(index), vertex_count, 3), relative to root[index[i]]"}
    doc = dict(rep) if rep is not None else {"t": [round(f / src_fps, 3) for f in src_frames]}
    doc["body3d"] = body3d
    out_path.write_text(json.dumps(doc, separators=(",", ":")))
    sizes = {out_path.name: out_path.stat().st_size}
    if mesh_idx:
        sizes[f"{name}_body.bin"] = (out_dir / f"{name}_body.bin").stat().st_size
    print("wrote", ", ".join(f"{k} {v / 1024:.0f} KB" for k, v in sizes.items()), "in", out_dir)


def _smooth(x: np.ndarray, sig: float) -> np.ndarray:
    sig = max(1.0, sig)
    ker = np.exp(-0.5 * (np.arange(-int(3 * sig), int(3 * sig) + 1) / sig) ** 2)
    ker /= ker.sum()
    return np.convolve(np.pad(x, len(ker) // 2, mode="edge"), ker, mode="valid")


def _mean_rot(Rs: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(Rs.mean(0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


if __name__ == "__main__":
    main()

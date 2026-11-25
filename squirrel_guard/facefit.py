"""One consistent 3D face per visit, fitted across many frames and two calibrated cameras (explanatory evidence;
NEVER an identity decision).

facegeom.py meshes the 8 best near-frontal crops of a visit independently and takes per-feature medians. This module
instead fits ONE shape to every usable frame of the visit:

  per frame   head window (COCO head keypoints, faceid.head_window) -> square crop (two passes: the second re-centred
              on the first pass's landmarks) -> MediaPipe face mesh (478 landmarks, z RELATIVE) -> FrameObs: landmarks
              in frame px (when calibrated: lens undistorted AND re-projected into a virtual pinhole view centred on
              the face, so off-axis faces are not stretched), IPD px, crop size, sharpness,
              blendshape scores when the landmarker gives them, grayscale/IR flag. IR/grayscale crops are turned into
              3 equal channels; a CLAHE-equalised retry only when that finds no face.
  per visit   identity fit on the MediaPipe canonical topology (468 surface vertices + 10 iris points):
                X_f ~= s_f R_f U + t_f      for every frame f (similarity: rotation, translation, scale)
              solved by alternating weighted Procrustes (poses) and a linear least-squares shape step with
                - x/y residuals at full weight, MediaPipe's learned z at z_weight (it is a prior, not a measurement),
                - Huber weights per frame x vertex and per frame (wrong face, bad crop),
                - weights for face size, sharpness, pose (cos^2 yaw cos pitch, hard cut at max_yaw/max_pitch),
                  per-vertex visibility (normal vs view direction: the hidden cheek of a turned face is guessed),
                - region down-weighting for expression (blendshapes jawOpen/smile/brow/blink when present, else
                  geometric proxies relative to the visit: mouth opening, mouth width, eye opening, brow height),
                - a Laplacian regulariser ||L (U - M)||^2 toward a prior M (the canonical face from the
                  face_landmarker.task bundle when available, else the visit's per-frame median shape after Taubin
                  smoothing), so a few small faces give a smooth face rather than noise.
              Uncertainty: bootstrap over frames (per-vertex RMS and per-feature sd).
  two views   frames of two calibrated cameras at the same instant: the 2D landmarks are cast as rays
              (geometry.Camera.rays, lens undistorted) and triangulated (midpoint) -> METRIC 3D landmarks. A
              similarity + one depth scale (face z axis) fitted per pair gives millimetres per IPD and how much
              MediaPipe's relative depth must be stretched; the medians over pairs fix the scale/depth of the shape
              when they are consistent (else single-view, and the result says why).
  outputs     features (facegeom.FEATURES via facegeom.features_of on the fitted shape, + metric mm when two views
              gave a scale) with bootstrap sd, per-frame reprojection error, and a quantised glTF with LODs:
              LOD0 ~250 triangles (QEM decimation that keeps every landmark features_of uses), LOD1 the native
              898-triangle topology, LOD2/LOD3 Loop subdivision (3592 / 14368 triangles).

POLICY (same rules as faceid.py / facegeom.py)
  - Explanatory only: who someone is stays with the 512-d face embedding (facematch.decide). Geometry is only ever
    compared with the same person's own enrolled geometry (facegeom.compare / merge work on these features too).
  - No ethnicity/race or any group inference; no outside lookups. Expression is never classified here: blendshape
    scores / proxies only DOWN-WEIGHT frames, and the summary reports "apparent expression" shares, description-only.
  - Depth from one view is MediaPipe's learned relative estimate; only the two-view triangulation measures it.
  - FLAME/DECA/EMOCA/MICA/SMIRK (non-commercial licences) are NOT used; nothing from FASR++ is used.

Dependencies: numpy for all fitting, subdivision, decimation and glTF code (tests run without cv2/mediapipe);
cv2 for crops/undistortion/projection; mediapipe only inside the landmarker classes (lazy imports). The MediaPipe
face mesh topology below (triangles derived from mediapipe's FACEMESH_TESSELATION, Apache-2.0) is embedded so no
model file is needed for it.
"""
from __future__ import annotations

import base64
import dataclasses
import functools
import heapq
import json
import logging
import math
import struct
import time
import zipfile
import zlib
from typing import Callable, Sequence

import numpy as np

from . import facegeom as fg

log = logging.getLogger(__name__)
VERSION = "facefit/1"
N_SURF, N_ALL = 468, 478

# -- MediaPipe canonical face mesh topology (Apache-2.0) ----------------------------------------------------------------
# 852 triangles over the 468 surface vertices (uint16 little endian, zlib, base64), oriented so that, in facegeom's
# frontal frame (x image-right, y down, z toward the camera), cross(b - a, c - a) points toward the camera. Derived
# from mediapipe.python.solutions.face_mesh_connections.FACEMESH_TESSELATION (3-cliques of its edge graph, the 2
# non-face cliques removed): boundaries = face oval, two eye openings, inner lips. Closing those three holes gives the
# 898 triangles of MediaPipe's canonical_face_model.
_TRI_B64 = (
    "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"
)
IRIS_RINGS = ((468, (469, 470, 471, 472)), (473, (474, 475, 476, 477)))
LIPS = (0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
        308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415)
EYE_L = (249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466)     # subject's left
EYE_R = (7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246)
BROW_L = (276, 282, 283, 285, 293, 295, 296, 300, 334, 336)
BROW_R = (46, 52, 53, 55, 63, 65, 66, 70, 105, 107)
FACE_OVAL = (10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 234, 251, 284, 288,
             297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454)
LIP_IN_TOP, LIP_IN_BOTTOM = 13, 14
EYE_R_BOT, EYE_L_BOT = 145, 374
HOLE_START = (33, 263, 78)          # start the strip triangulation of each hole at a corner (eyes: outer, lips: inner)
# every surface landmark facegeom.features_of reads (decimation must keep them)
FEATURE_VERTS = tuple(sorted({fg.EYE_R_OUT, fg.EYE_R_IN, fg.EYE_L_IN, fg.EYE_L_OUT, fg.EYE_R_TOP, fg.EYE_L_TOP,
                              fg.BROW_R, fg.BROW_L, fg.NASION, fg.NOSE_TIP, fg.SUBNASALE, fg.ALA_R, fg.ALA_L,
                              fg.MOUTH_R, fg.MOUTH_L, fg.LIP_TOP, fg.LIP_BOTTOM, fg.CHEEK_R, fg.CHEEK_L, fg.CHEEKS_R,
                              fg.CHEEKS_L, fg.JAW_R, fg.JAW_L, fg.CHIN, fg.GLABELLA, *fg.MIDLINE}))


@functools.lru_cache(maxsize=None)
def _tri_open() -> np.ndarray:
    raw = zlib.decompress(base64.b64decode("".join(_TRI_B64)))
    return np.frombuffer(raw, "<u2").reshape(-1, 3).astype(np.int64)


def boundary_loops(T: np.ndarray) -> list[list[int]]:
    """Ordered boundary loops of an oriented triangle mesh, each following its boundary edges' direction."""
    de = {(int(a), int(b)) for t in T for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    nxt = {a: b for a, b in de if (b, a) not in de}
    loops, seen = [], set()
    for v0 in sorted(nxt):
        if v0 in seen:
            continue
        loop, v = [], v0
        while v not in seen:
            seen.add(v)
            loop.append(v)
            v = nxt[v]
        loops.append(loop)
    return loops


def _close_loop(loop: list[int], start: int | None) -> list[list[int]]:
    """Strip-triangulate a hole whose loop runs along the mesh boundary direction (the fill runs the other way)."""
    q = loop[::-1]
    if start in q:
        k = q.index(start)
        q = q[k:] + q[:k]
    tris, lo, hi, left = [], 0, len(q) - 1, True
    while hi - lo >= 2:
        if left:
            tris.append([q[lo], q[lo + 1], q[hi]])
            lo += 1
        else:
            tris.append([q[lo], q[hi - 1], q[hi]])
            hi -= 1
        left = not left
    return tris


@functools.lru_cache(maxsize=None)
def _topology(closed: bool) -> np.ndarray:
    T = _tri_open()
    if not closed:
        return T
    loops = sorted(boundary_loops(T), key=len)[:-1]           # the longest loop is the face outline: stays open
    extra = []
    for lp in loops:
        st = next((s for s in HOLE_START if s in lp), None)
        extra += _close_loop(lp, st)
    return np.vstack([T, np.array(extra, np.int64)])


def topology(closed: bool = True) -> np.ndarray:
    """(F, 3) triangles over the 468 surface vertices; closed=True fills the eyes and the mouth (898 triangles, as
    MediaPipe's canonical face model), closed=False leaves them open (852)."""
    return _topology(bool(closed)).copy()


def unique_edges(T: np.ndarray) -> np.ndarray:
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    return np.unique(e, axis=0)


@functools.lru_cache(maxsize=None)
def _laplacian_sq() -> np.ndarray:
    """L^T L for the uniform graph Laplacian over all 478 points (surface edges + iris rings and spokes)."""
    E = [tuple(e) for e in unique_edges(_tri_open())]
    for c, ring in IRIS_RINGS:
        E += [(c, r) for r in ring] + [(ring[i], ring[(i + 1) % 4]) for i in range(4)]
    A = np.zeros((N_ALL, N_ALL))
    for a, b in E:
        A[a, b] = A[b, a] = 1.0
    deg = A.sum(1)
    L = np.eye(N_ALL) - A / np.maximum(deg, 1)[:, None]
    L[deg == 0] = 0
    return L.T @ L


# -- canonical face from the face_landmarker.task bundle (optional prior) ------------------------------------------------
def _pb_fields(buf: bytes):
    """Minimal protobuf reader: yields (field number, wire type, value) with value = int (varint), bytes (length-
    delimited), or raw 4/8 bytes (fixed32/fixed64)."""
    i, n = 0, len(buf)

    def varint(i):
        v, sh = 0, 0
        while True:
            b = buf[i]
            i += 1
            v |= (b & 0x7F) << sh
            sh += 7
            if b < 0x80:
                return v, i
    while i < n:
        key, i = varint(i)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(i)
        elif wt == 2:
            ln, i = varint(i)
            v, i = buf[i:i + ln], i + ln
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield fno, wt, v


def parse_canonical_mesh(buf: bytes) -> tuple[np.ndarray, np.ndarray] | None:
    """GeometryPipelineMetadata (canonical_mesh = 1: Mesh3d{vertex_buffer = 3 (float, 5 per vertex: x y z u v),
    index_buffer = 4 (uint32)}) -> (V (468, 3) in its own units, y up, z toward the viewer; T (F, 3)) or None."""
    mesh = next((v for f, wt, v in _pb_fields(buf) if f == 1 and wt == 2), None)
    if mesh is None:
        return None
    vb, ib = [], []
    for f, wt, v in _pb_fields(mesh):
        if f == 3:
            vb += list(struct.unpack(f"<{len(v) // 4}f", v)) if wt == 2 else [struct.unpack("<f", v)[0]]
        elif f == 4:
            if wt == 2:
                ib += _unpack_varints(v)
            else:
                ib.append(int(v))
    if len(vb) % 5 or len(ib) % 3:
        return None
    V = np.array(vb, np.float64).reshape(-1, 5)[:, :3]
    T = np.array(ib, np.int64).reshape(-1, 3)
    if len(V) != N_SURF or (len(T) and T.max() >= N_SURF):
        return None
    return V, T


def _enc_varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def _unpack_varints(buf: bytes) -> list[int]:
    out, v, sh = [], 0, 0
    for b in buf:
        v |= (b & 0x7F) << sh
        sh += 7
        if b < 0x80:
            out.append(v)
            v, sh = 0, 0
    return out


def canonical_from_task(task_path: str) -> np.ndarray | None:
    """MediaPipe's canonical face (468 vertices) from a face_landmarker.task bundle, in facegeom's frontal frame (IPD
    from the eye-corner centres = 1), or None when the bundle has no geometry pipeline metadata."""
    try:
        with zipfile.ZipFile(task_path) as z:
            names = [n for n in z.namelist() if "geometry_pipeline_metadata" in n]
            if not names:
                return None
            got = parse_canonical_mesh(z.read(names[0]))
    except (OSError, zipfile.BadZipFile, ValueError, IndexError, struct.error):
        log.exception("canonical face: cannot read %s", task_path)
        return None
    if got is None:
        return None
    V, _ = got
    P = V * (1, -1, -1)                          # y up / z toward viewer -> image-like (y down, z into the scene)
    Q, _ = fg.frontalise(P)
    return Q


# -- per-frame landmarks ---------------------------------------------------------------------------------------------
@dataclasses.dataclass
class FrameObs:
    cam: str
    idx: int                   # frame index in the decoded clip
    t: float
    P: np.ndarray              # (478, 3) landmarks in frame px (x, y; z in px, smaller = nearer), as detected
    ipd_px: float
    crop_side: float           # crop side in frame px handed to the mesh model
    sharp: float               # variance of the Laplacian of the face crop at 128 px
    gray: bool                 # the crop was grayscale (IR) or forced to grayscale
    yaw: float = 0.0           # quick pose from the mesh (facegeom.pose_of), degrees
    pitch: float = 0.0
    blend: dict | None = None  # blendshape scores (Tasks landmarker only)
    ms: float = 0.0            # landmarker time for this frame


def is_grayscale(img: np.ndarray, tol: float = 3.0) -> bool:
    """True for single-channel images and 3-channel ones whose channels are (nearly) equal (Wyze IR night mode)."""
    if img.ndim == 2 or img.shape[2] == 1:
        return True
    s = img[::4, ::4].astype(np.int16)
    return float(np.abs(s[..., 0] - s[..., 1]).mean() + np.abs(s[..., 1] - s[..., 2]).mean()) / 2 < tol


def ir_prepare(img: np.ndarray, clahe: bool = True) -> np.ndarray:
    """Grayscale/IR crop -> 3 equal channels, optionally contrast-equalised (CLAHE) for the mesh model."""
    import cv2
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if clahe:
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)
    return cv2.merge([g, g, g])


def sharpness(crop: np.ndarray) -> float:
    import cv2
    g = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


class TaskLandmarker:
    """MediaPipe Tasks FaceLandmarker (face_landmarker.task, as in the profiler image) with blendshape scores.
    crop (BGR) -> ((478, 3) crop px, {blendshape: score}) or None."""

    def __init__(self, model_path: str, min_conf: float = 0.3):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        self.mp = mp
        opt = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), num_faces=1,
                                           min_face_detection_confidence=min_conf,
                                           min_face_presence_confidence=min_conf, output_face_blendshapes=True)
        self.fl = vision.FaceLandmarker.create_from_options(opt)

    def __call__(self, crop_bgr: np.ndarray):
        import cv2
        h, w = crop_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        r = self.fl.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        if not r.face_landmarks:
            return None
        P = np.array([(p.x * w, p.y * h, p.z * w) for p in r.face_landmarks[0]], np.float32)
        bl = None
        if getattr(r, "face_blendshapes", None):
            bl = {c.category_name: round(float(c.score), 4) for c in r.face_blendshapes[0]}
        return P, bl


class SolutionLandmarker:
    """Legacy mediapipe.solutions.face_mesh (refine_landmarks: 478 points, same topology; bundled with mediapipe
    <= 0.10.21, no model download). Used by the evaluation when face_landmarker.task is not at hand; no blendshapes."""

    def __init__(self, min_conf: float = 0.3):
        import mediapipe as mp
        self.fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1,
                                                  min_detection_confidence=min_conf)

    def __call__(self, crop_bgr: np.ndarray):
        import cv2
        h, w = crop_bgr.shape[:2]
        r = self.fm.process(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        if not r.multi_face_landmarks:
            return None
        return np.array([(p.x * w, p.y * h, p.z * w) for p in r.multi_face_landmarks[0].landmark], np.float32), None


def make_landmarker(model: str | None = None, min_conf: float = 0.3):
    """face_landmarker.task path -> TaskLandmarker; None/"legacy" -> SolutionLandmarker."""
    if model and model != "legacy":
        return TaskLandmarker(model, min_conf)
    return SolutionLandmarker(min_conf)


def _square_crop(frame: np.ndarray, cx: float, cy: float, side: float, out_px: int):
    import cv2
    H, W = frame.shape[:2]
    S = max(8, int(round(side)))
    X0, Y0 = int(round(cx - S / 2)), int(round(cy - S / 2))
    sub = frame[max(0, Y0):min(H, Y0 + S), max(0, X0):min(W, X0 + S)]
    if sub.size == 0:
        return None, (X0, Y0), 1.0
    pad = [max(0, -Y0), max(0, Y0 + S - H), max(0, -X0), max(0, X0 + S - W)]
    if any(pad):
        sub = cv2.copyMakeBorder(sub, *pad, cv2.BORDER_REPLICATE)
    sc = out_px / S
    crop = cv2.resize(sub, (out_px, out_px), interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)
    return crop, (X0, Y0), sc


def landmarks_in_window(frame: np.ndarray, window, landmarker: Callable, crop_px: int = 256, ir: str = "auto",
                        refine: bool = True, cam: str = "", idx: int = 0, t: float = 0.0,
                        zooms: Sequence[float] = (1.0, 0.7, 1.4)) -> FrameObs | None:
    """Face mesh inside a head window (x1, y1, x2, y2 frame px). ir: "auto" (grayscale/IR crops -> 3 equal channels;
    CLAHE-equalised retry when no face is found), "off", "force" (always grayscale + CLAHE), "gray" (always
    grayscale, no CLAHE). Pass 2 re-crops around the first
    pass's landmarks (face extent x 1.6) so the mesh model sees a centred face, as in MediaPipe's own tracking."""
    x1, y1, x2, y2 = [float(v) for v in window]
    cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
    t0 = time.perf_counter()

    def run(cx, cy, side):
        crop, (X0, Y0), sc = _square_crop(frame, cx, cy, side, crop_px)
        if crop is None:
            return None
        gray = is_grayscale(crop)
        if ir == "force":
            crop, gray = ir_prepare(crop, True), True
        elif ir == "gray" or (ir == "auto" and gray):
            crop, gray = ir_prepare(crop, False), True
        got = landmarker(crop)
        if got is None and ir == "auto" and gray:      # IR: CLAHE only as a retry (finds more small faces, but
            crop = ir_prepare(crop, True)              # its landmarks jitter ~30 % more; docs/facefit.md)
            got = landmarker(crop)
        if got is None:
            return None
        P, bl = got
        if P is None or len(P) < N_SURF:
            return None
        if len(P) < N_ALL:                       # no iris: eye-corner centres stand in
            P = np.vstack([P, np.repeat(P[[fg.EYE_R_OUT, fg.EYE_R_IN]].mean(0, keepdims=True), 5, 0),
                           np.repeat(P[[fg.EYE_L_OUT, fg.EYE_L_IN]].mean(0, keepdims=True), 5, 0)])
        Pf = np.asarray(P, np.float64).copy()
        Pf[:, :2] = Pf[:, :2] / sc + (X0, Y0)
        Pf[:, 2] /= sc
        return Pf, bl, crop, side, gray

    got = None
    for z in zooms:
        got = run(cx, cy, side * z)
        if got is not None:
            break
    if got is None:
        return None
    if refine:
        xy = got[0][:N_SURF, :2]
        lo, hi = xy.min(0), xy.max(0)
        g2 = run((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 1.6 * float(max(hi - lo)))
        if g2 is not None:
            got = g2
    P, bl, crop, side_used, gray = got
    ipd = float(np.linalg.norm(P[fg.IRIS_R, :2] - P[fg.IRIS_L, :2]))
    yaw, pitch = fg.pose_of(P)
    return FrameObs(cam, int(idx), float(t), P.astype(np.float32), ipd, float(side_used), sharpness(crop), bool(gray),
                    round(yaw, 1), round(pitch, 1), bl, (time.perf_counter() - t0) * 1e3)


def undistort_landmarks(P: np.ndarray, cam, frame_size, recentre: bool = True) -> np.ndarray:
    """Landmarks in frame px -> the same seen by an ideal pinhole camera with the camera's focal length, lens
    distortion removed and (recentre=True) the view rotated so the face is on the optical axis. Plain undistortion
    to a rectilinear image stretches an off-axis face radially (~1/cos of the off-axis angle: 15 % at 30 deg),
    which a per-frame similarity cannot absorb; the re-centred virtual view does not. Output coordinates are
    centred on the face (only relative geometry matters to the fit); z is scaled by the local change of the IPD."""
    import cv2
    P = np.asarray(P, np.float64)
    sx, sy = cam.size[0] / frame_size[0], cam.size[1] / frame_size[1]
    xy = P[:, :2] * (sx, sy)
    n = cv2.undistortPoints(xy.reshape(-1, 1, 2), cam.K, cam.D).reshape(-1, 2)      # normalised image coords
    f = float(cam.K[0, 0]) / sx
    if recentre:
        v = np.c_[n, np.ones(len(n))]
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        c = v[:N_SURF].mean(0)
        c /= np.linalg.norm(c)
        x = np.cross([0.0, 1.0, 0.0], c)
        x /= np.linalg.norm(x)
        y = np.cross(c, x)
        w = v @ np.stack([x, y, c], 1)                                   # rays in the face-centred view
        u = f * w[:, :2] / w[:, 2:3]
    else:
        u = f * n + (cam.K[0, 2] / sx, cam.K[1, 2] / sy)
    d0 = np.linalg.norm(P[fg.IRIS_R, :2] - P[fg.IRIS_L, :2])
    d1 = np.linalg.norm(u[fg.IRIS_R] - u[fg.IRIS_L])
    out = P.copy()
    out[:, :2] = u
    out[:, 2] *= d1 / d0 if d0 > 1e-6 else 1.0
    return out


# -- identity fit ----------------------------------------------------------------------------------------------------
def umeyama(A: np.ndarray, B: np.ndarray, w: np.ndarray | None = None):
    """Weighted similarity B ~= s R A + t (Umeyama 1991). Returns (s, R, t)."""
    w = np.ones(len(A)) if w is None else np.asarray(w, np.float64)
    W = w.sum() + 1e-12
    ma, mb = (w[:, None] * A).sum(0) / W, (w[:, None] * B).sum(0) / W
    a, b = A - ma, B - mb
    cov = (b * w[:, None]).T @ a / W
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    va = float((w * (a * a).sum(1)).sum() / W)
    s = float(np.trace(np.diag(D) @ S) / max(va, 1e-12))
    return s, R, mb - s * R @ ma


def umeyama_xy(A: np.ndarray, B: np.ndarray, w: np.ndarray | None = None, iters: int = 15):
    """Similarity B ~= s R A + t fitted to B's x/y only (B's z, e.g. MediaPipe's learned depth, is only the starting
    guess): alternate a 3D Umeyama with B's z replaced by the model's."""
    T = np.asarray(B, np.float64).copy()
    s = R = t = None
    for _ in range(iters):
        s, R, t = umeyama(A, T, w)
        T[:, 2] = (s * A @ R.T + t)[:, 2]
    return s, R, t


def angles_of(R: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) degrees of a face rotation (internal face frame -> image), signs as facegeom.pose_of."""
    ex, ey = R[:, 0], R[:, 1]
    yaw = math.degrees(math.atan2(-ex[2], math.hypot(ex[0], ex[1])))
    pitch = math.degrees(math.atan2(-ey[2], math.hypot(ey[0], ey[1])))
    roll = math.degrees(math.atan2(ex[1], ex[0]))
    return yaw, pitch, roll


def _internal(Q: np.ndarray) -> np.ndarray:
    """facegeom frontal (z toward the camera) <-> internal right-handed face frame (z into the scene)."""
    return np.asarray(Q, np.float64) * (1, 1, -1)


def _gauge(U: np.ndarray) -> np.ndarray:
    """Fix the shape's similarity gauge: nasion origin, eye axis = x, IPD (iris centres) = 1 (facegeom.frontalise)."""
    Q, _ = fg.frontalise(U)                      # U is image-like (z into the scene), as frontalise expects
    return _internal(Q)


def vertex_normals(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, T[:, k], n)
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)


@dataclasses.dataclass
class FitParams:
    iters: int = 12
    z_weight: float = 0.35         # MediaPipe's z is a learned prior: weight of its residual vs x/y
    iris_weight: float = 0.3       # iris points move with gaze
    lam: float = 1.5               # Laplacian regulariser toward the prior, in units of one full-weight frame
    ridge: float = 1e-3            # tiny absolute pull toward the prior (keeps the system positive definite)
    huber_k: float = 2.0           # Huber threshold in robust sigmas (frame x vertex residuals)
    frame_k: float = 2.5           # Huber threshold for a whole frame's median residual vs the visit's
    max_yaw: float = 50.0
    max_pitch: float = 40.0
    min_ipd_px: float = 8.0        # frames whose IPD is below this are not used
    good_ipd_px: float = 30.0      # full size weight from here (IPD 30 px ~ 120 px face)
    expr_region: float = 0.85      # region weight multiplier at full expression score: 1 - expr_region
    expr_frame: float = 0.4        # whole-frame weight multiplier at full score: 1 - expr_frame
    boot: int = 24                 # bootstrap resamples for the uncertainty (0 = off)
    boot_iters: int = 3
    seed: int = 0


BLEND_REGIONS = {
    "mouth": ("jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker", "mouthFunnel", "mouthStretchLeft",
              "mouthStretchRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthLeft", "mouthRight",
              "mouthFrownLeft", "mouthFrownRight", "cheekPuff"),
    "eyes": ("eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight"),
    "brows": ("browInnerUp", "browDownLeft", "browDownRight", "browOuterUpLeft", "browOuterUpRight"),
}
REGIONS = ("mouth", "eyes", "brows")


def expression_scores(Qs: np.ndarray, blends: Sequence[dict | None] | None = None) -> np.ndarray:
    """(F, 478, 3) frontalised meshes (IPD units) [+ blendshape dicts] -> (F, 3) scores in [0, 1] for the mouth/jaw,
    eyes and brows: how far each frame's face is from the visit's resting state. Used ONLY to down-weight frames and
    regions in the fit (apparent expression, description-only; never classified, never used for identity)."""
    F = len(Qs)
    out = np.zeros((F, 3))
    d = lambda a, b: np.linalg.norm(Qs[:, a, :2] - Qs[:, b, :2], axis=-1)       # noqa: E731
    mo = d(LIP_IN_TOP, LIP_IN_BOTTOM)
    mw = d(fg.MOUTH_R, fg.MOUTH_L)
    eo = (d(fg.EYE_R_TOP, EYE_R_BOT) + d(fg.EYE_L_TOP, EYE_L_BOT)) / 2
    br = (d(fg.BROW_R, fg.EYE_R_TOP) + d(fg.BROW_L, fg.EYE_L_TOP)) / 2
    med = lambda v: float(np.median(v))                                          # noqa: E731
    geo = np.stack([np.maximum(np.clip((mo - 0.05) / 0.2, 0, 1), np.clip(np.abs(mw - med(mw)) / 0.1, 0, 1)),
                    np.clip(np.abs(eo - med(eo)) / 0.06, 0, 1),
                    np.clip(np.abs(br - med(br)) / 0.06, 0, 1)], 1)
    for f in range(F):
        b = (blends[f] if blends is not None else None) or None
        if b:
            out[f] = [np.clip((max(b.get(k, 0.0) for k in BLEND_REGIONS[r]) - 0.15) / 0.45, 0, 1) for r in REGIONS]
            out[f, 0] = max(out[f, 0], geo[f, 0])           # an open mouth moves the chin whatever the scores say
        else:
            out[f] = geo[f]
    return out


def region_masks(Q: np.ndarray) -> np.ndarray:
    """(3, 478) bool: vertices moved by mouth/jaw, eye and brow expressions, from a frontal shape (IPD units)."""
    y = Q[:, 1]
    m = np.zeros((3, N_ALL), bool)
    m[0, :N_SURF] = y[:N_SURF] > y[fg.SUBNASALE]
    m[1, list(EYE_R + EYE_L)] = True
    m[1, N_SURF:] = True
    brow_y = float(np.mean(y[list(BROW_R + BROW_L)]))
    m[2, list(BROW_R + BROW_L)] = True
    m[2, :N_SURF] |= y[:N_SURF] < brow_y + 0.05
    return m


@functools.lru_cache(maxsize=None)
def _umbrella() -> np.ndarray:
    """Uniform umbrella operator (mean of neighbours - vertex) over the 468 surface vertices."""
    E = unique_edges(_tri_open())
    A = np.zeros((N_SURF, N_SURF))
    A[E[:, 0], E[:, 1]] = A[E[:, 1], E[:, 0]] = 1.0
    return A / A.sum(1, keepdims=True) - np.eye(N_SURF)


def taubin(Q: np.ndarray, n: int = 10, lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    """Taubin (lambda|mu) smoothing of the surface vertices: removes per-vertex noise without the shrinkage of plain
    Laplacian smoothing. Used to turn the visit's per-frame median into the regularisation prior."""
    V = np.asarray(Q, np.float64).copy()
    L = _umbrella()
    for _ in range(n):
        V[:N_SURF] += lam * (L @ V[:N_SURF])
        V[:N_SURF] += mu * (L @ V[:N_SURF])
    return V


@dataclasses.dataclass
class FitResult:
    S: np.ndarray                  # (478, 3) fitted shape, facegeom frontal frame (IPD = 1, z toward the camera)
    poses: list                    # per frame (s, R, t): image px ~= s R U + t with U = S * (1, 1, -1)
    w_frame: np.ndarray            # final per-frame weight (0 = not used)
    angles: np.ndarray             # (F, 3) yaw, pitch, roll
    resid_px: np.ndarray           # (F,) median 2D reprojection error of the fitted shape, px
    resid_ipd: np.ndarray          # (F,) the same / IPD px
    expr: np.ndarray               # (F, 3) expression scores used for down-weighting
    prior: str
    iters: int
    vertex_sd: np.ndarray | None = None    # (478,) bootstrap RMS displacement, IPD units
    boot_features: dict | None = None      # feature -> bootstrap sd
    boot_shapes: np.ndarray | None = None
    ms: float = 0.0


def _poses(U, X, W):
    out = []
    for f in range(len(X)):
        w = W[f]
        if w.sum() <= 1e-9:
            w = np.ones(len(w))
        out.append(umeyama(U, X[f], w))
    return out


def _solve_shape(H, b, M, U0, lam, ridge, LtL, iters: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Solve (blockdiag(H) + lam (L^T L x I3) + ridge I) U = b + (lam L^T L + ridge) M by conjugate gradient with a
    block-Jacobi preconditioner (the 3x3 vertex blocks), warm-started at U0. numpy only; ~ms for 478 vertices."""
    dg = np.diag(LtL)
    Pblk = H + (lam * dg + ridge)[:, None, None] * np.eye(3)[None]
    Pinv = np.linalg.inv(Pblk)

    def Aop(X):
        return np.einsum("nij,nj->ni", H, X) + lam * (LtL @ X) + ridge * X
    rhs = b + lam * (LtL @ M) + ridge * M
    X = U0.copy()
    r = rhs - Aop(X)
    z = np.einsum("nij,nj->ni", Pinv, r)
    pdir = z.copy()
    rz = float((r * z).sum())
    nb = float((rhs * rhs).sum()) + 1e-30
    for _ in range(iters):
        Ap = Aop(pdir)
        alpha = rz / float((pdir * Ap).sum())
        X += alpha * pdir
        r -= alpha * Ap
        if float((r * r).sum()) < tol * nb:
            break
        z = np.einsum("nij,nj->ni", Pinv, r)
        rz2 = float((r * z).sum())
        pdir = z + (rz2 / rz) * pdir
        rz = rz2
    return X


def _fit_core(X, wf, U, M, regions, expr, p: FitParams, iters: int, LtL):
    """Alternating similarity poses / shape. X (F, 478, 3) px, wf (F,) base frame weights. Returns (U, poses, wf_eff,
    W, iters run)."""
    F = len(X)
    A2 = np.array([1.0, 1.0, p.z_weight])
    ireg = np.ones(N_ALL)
    ireg[N_SURF:] = p.iris_weight
    T = topology(True)
    rw = np.ones((F, N_ALL))
    for r in range(3):
        rw *= 1.0 - p.expr_region * expr[:, r:r + 1] * regions[r][None, :]
    wfx = wf * (1.0 - p.expr_frame * expr.max(1))
    hub = np.ones((F, N_ALL))
    fhub = np.ones(F)
    poses, W, it = None, None, 0
    eye_v = list(EYE_R + EYE_L)
    for it in range(1, iters + 1):
        nrm = vertex_normals(U[:N_SURF], T)
        Wp = (wfx * fhub)[:, None] * rw * ireg[None, :]
        poses = _poses(U, X, Wp * hub)
        ang = np.array([angles_of(R) for _, R, _ in poses])
        gate = (np.abs(ang[:, 0]) <= p.max_yaw) & (np.abs(ang[:, 1]) <= p.max_pitch)
        vis = np.ones((F, N_ALL))
        for f, (s, R, t) in enumerate(poses):
            fac = np.clip(-(nrm @ R.T)[:, 2], 0.0, 1.0)            # normal toward the camera (internal z away)
            vis[f, :N_SURF] = 0.15 + 0.85 * fac
            vis[f, N_SURF:] = vis[f, eye_v].mean()
        # residuals in IPD units (z down-weighted)
        Y = np.stack([(X[f] - t) / s for f, (s, R, t) in enumerate(poses)])
        RU = np.stack([U @ R.T for _, R, _ in poses])
        res = np.sqrt((((RU - Y) ** 2) * A2).sum(-1))                # (F, N)
        base = (wfx * gate)[:, None] * rw * ireg[None, :] * vis
        used = base.sum(1) > 0
        if used.sum() == 0:
            break
        sig = 1.4826 * float(np.median(res[used])) + 1e-9
        hub = np.minimum(1.0, p.huber_k * sig / np.maximum(res, 1e-12))
        mf = np.median(res, 1)
        sf = float(np.median(mf[used])) + 1e-9
        fhub = np.minimum(1.0, p.frame_k * sf / np.maximum(mf, 1e-12))
        W = base * hub * fhub[:, None]
        # shape step: sum_f W_fv ||A (R_f U_v - y_fv)||^2 + lam ||L (U - M)||^2 + ridge ||U - M||^2
        G = np.stack([R.T * A2[None, :] for _, R, _ in poses])      # R^T diag(A2)
        H = np.einsum("fn,fij,fjk->nik", W, G, np.stack([R for _, R, _ in poses]))
        b = np.einsum("fn,fij,fnj->ni", W, G, Y)
        Un = _solve_shape(H, b, M, U, p.lam, p.ridge, LtL)
        Un = _gauge(Un)
        delta = float(np.abs(Un - U).max())
        U = Un
        if delta < 1e-4:
            break
    poses = _poses(U, X, W if W is not None else np.ones((F, N_ALL)))
    wf_eff = W.sum(1) / N_ALL if W is not None else np.zeros(F)
    return U, poses, wf_eff, W, it


def frame_base_weights(obs: Sequence[FrameObs], p: FitParams) -> np.ndarray:
    """Size (IPD px) x sharpness (vs the visit median) weights; 0 below min_ipd_px."""
    ipd = np.array([o.ipd_px for o in obs], np.float64)
    sh = np.array([o.sharp for o in obs], np.float64)
    ws = np.clip(ipd / p.good_ipd_px, 0, 1) ** 2
    ws[ipd < p.min_ipd_px] = 0
    msh = float(np.median(sh)) if len(sh) else 1.0
    return ws * np.clip(sh / (msh + 1e-9), 0.3, 1.0)


def fit_identity(X: np.ndarray, wf: np.ndarray, blends: Sequence[dict | None] | None = None,
                 prior: np.ndarray | None = None, params: FitParams | None = None) -> FitResult:
    """Joint fit of one shape + per-frame similarity poses. X: (F, 478, 3) landmarks in (undistorted) image px,
    z in px (smaller = nearer); wf: (F,) base frame weights (frame_base_weights). prior: canonical face (468 or 478
    points, facegeom frontal frame) or None (the visit's per-frame median shape after Taubin smoothing)."""
    p = params or FitParams()
    t0 = time.perf_counter()
    X = np.asarray(X, np.float64)
    F = len(X)
    Qs = np.stack([fg.frontalise(x)[0] for x in X])
    keep = wf > 0
    U0 = _internal(np.median(Qs[keep] if keep.any() else Qs, 0))
    U0 = _gauge(U0)
    if prior is not None:
        Mq = np.asarray(prior, np.float64)
        M = U0.copy()
        s, R, t = umeyama(_internal(Mq[:N_SURF]), U0[:N_SURF])
        M[:N_SURF] = s * _internal(Mq[:N_SURF]) @ R.T + t
        prior_kind = "canonical"
    else:
        M, prior_kind = _internal(taubin(_internal(U0))), "visit-median-smoothed"
    expr = expression_scores(Qs, blends)
    regions = region_masks(_internal(U0))
    LtL = _laplacian_sq()
    U, poses, wf_eff, W, it = _fit_core(X, wf, U0, M, regions, expr, p, p.iters, LtL)
    ang = np.array([angles_of(R) for _, R, _ in poses])
    rp = np.zeros(F)
    for f, (s, R, t) in enumerate(poses):
        pr = s * U[:N_SURF] @ R.T + t
        rp[f] = float(np.median(np.linalg.norm(pr[:, :2] - X[f, :N_SURF, :2], axis=1)))
    ipd = np.linalg.norm(X[:, fg.IRIS_R, :2] - X[:, fg.IRIS_L, :2], axis=1)
    res = FitResult(_internal(U), poses, wf_eff, ang, rp, rp / np.maximum(ipd, 1e-6), expr, prior_kind, it)
    used = np.flatnonzero(wf_eff > 1e-6)
    if p.boot and len(used) >= 4:
        rng = np.random.default_rng(p.seed)
        shapes, feats = [], []
        for _ in range(p.boot):
            ix = rng.choice(used, len(used), replace=True)
            Ub, *_ = _fit_core(X[ix], wf[ix], U, M, regions, expr[ix], p, p.boot_iters, LtL)
            shapes.append(_internal(Ub))
            feats.append(features(_internal(Ub)))
        Bs = np.stack(shapes)
        res.boot_shapes = Bs
        res.vertex_sd = np.sqrt(((Bs - Bs.mean(0)) ** 2).sum(-1).mean(0))
        res.boot_features = {k: round(float(np.std([f[k] for f in feats], ddof=1)), 4) for k in fg.FEATURES}
    res.ms = (time.perf_counter() - t0) * 1e3
    return res


def features(S: np.ndarray) -> dict[str, float]:
    """facegeom.features_of on a fitted shape (facegeom frontal frame, IPD units)."""
    return fg.features_of(_internal(S))


# -- two calibrated views ----------------------------------------------------------------------------------------------
def triangulate(cam_a, pa, size_a, cam_b, pb, size_b) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint triangulation of corresponding pixels in two calibrated cameras (geometry.Camera, world metres).
    Returns (X (N, 3), ray gap (N,) metres); NaN where the rays are parallel or meet behind a camera."""
    da, db = cam_a.rays(pa, size_a), cam_b.rays(pb, size_b)
    w0 = cam_a.C - cam_b.C
    b = (da * db).sum(1)
    d, e = da @ w0, db @ w0
    den = 1.0 - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        sa = (b * e - d) / den
        sb = (e - b * d) / den
    Pa = cam_a.C + sa[:, None] * da
    Pb = cam_b.C + sb[:, None] * db
    X = (Pa + Pb) / 2
    gap = np.linalg.norm(Pa - Pb, axis=1)
    bad = ~(den > 1e-9) | ~(sa > 0) | ~(sb > 0)
    X[bad] = np.nan
    gap[bad] = np.nan
    return X, gap


def project(cam, Xw: np.ndarray, frame_size) -> np.ndarray:
    """World points -> distorted pixels in a frame of frame_size."""
    import cv2
    Rcw = cam.R.T
    rv, _ = cv2.Rodrigues(Rcw)
    tv = -Rcw @ cam.C
    px, _ = cv2.projectPoints(np.asarray(Xw, np.float64).reshape(-1, 1, 3), rv, tv, cam.K, cam.D)
    return px.reshape(-1, 2) * (frame_size[0] / cam.size[0], frame_size[1] / cam.size[1])


def fit_metric_pair(U: np.ndarray, Xw: np.ndarray, w: np.ndarray, iters: int = 20) -> dict:
    """Triangulated landmarks (world m) vs the fitted shape U (internal frame, IPD units):
    Xw ~= s R diag(1, 1, k) U + t. s = metres per IPD, k = depth stretch of MediaPipe's relative z."""
    ok = np.isfinite(Xw).all(1) & (w > 0)
    A, B, ww = U[ok], Xw[ok], w[ok].astype(np.float64)
    k = 1.0
    s = R = t = None
    for _ in range(iters):
        Ak = A * (1, 1, k)
        s, R, t = umeyama(Ak, B, ww)
        Z = (B - t) @ R / s                                 # face-frame coordinates of the measured points
        zc, uc = Z[:, 2], A[:, 2]
        wm = ww / ww.sum()
        zc, uc = zc - (wm * zc).sum(), uc - (wm * uc).sum()
        kn = float((ww * zc * uc).sum() / max((ww * uc * uc).sum(), 1e-12))
        kn = float(np.clip(kn, 0.2, 5.0))
        r = np.linalg.norm(s * (A * (1, 1, kn)) @ R.T + t - B, axis=1)
        sig = 1.4826 * float(np.median(r)) + 1e-6
        ww = w[ok] * np.minimum(1.0, 2.0 * sig / np.maximum(r, 1e-9))
        if abs(kn - k) < 1e-4:
            k = kn
            break
        k = kn
    r = np.linalg.norm(s * (A * (1, 1, k)) @ R.T + t - B, axis=1)
    return {"s": s, "k": k, "R": R, "t": t, "resid_m_med": float(np.median(r)), "n": int(ok.sum())}


def multiview(U: np.ndarray, pairs: list[tuple[FrameObs, FrameObs]], cam_a, size_a, cam_b, size_b,
              max_gap_m: float = 0.03, min_pairs: int = 3, max_scale_rel_sem: float = 0.05,
              max_k_sd: float = 0.25) -> dict:
    """Metric scale and depth stretch of the fitted shape U (internal frame) from simultaneous frames of two
    calibrated cameras.

    scale  per pair: the face's distance Z from each camera (median of the triangulated landmarks) x the per-frame
           similarity scale of U in that camera's face-centred view (px per IPD, rotation-free) / focal length
           -> metres per IPD; median over both cameras and all pairs. Depth noise of the triangulation hardly matters.
    depth  per pair: Xw ~= s R diag(1, 1, k) U + t on the triangulated points; k = how much MediaPipe's relative depth
           must be stretched. Needs a baseline that resolves the face's ~4 cm relief: the expected per-point depth
           error Z^2 sigma_px / (f B) is reported, and k is only used when consistent between pairs.
    The ray gap (closest approach of the two rays of each landmark) checks sync/calibration; motion along the
    epipolar direction between the two exposures is invisible to it and goes straight into depth."""
    rows = []
    base = float(np.linalg.norm(cam_a.C - cam_b.C))
    for oa, ob in pairs:
        X, gap = triangulate(cam_a, oa.P[:, :2], size_a, cam_b, ob.P[:, :2], size_b)
        w = np.zeros(N_ALL)
        w[:N_SURF] = 1.0
        w[np.isnan(gap) | (gap > max_gap_m)] = 0
        if (w > 0).sum() < 50:
            rows.append({"idx": [oa.idx, ob.idx], "ok": False, "why": "too few rays meet",
                         "gap_mm_med": None if np.all(np.isnan(gap)) else round(float(np.nanmedian(gap)) * 1e3, 1)})
            continue
        ok = w > 0
        dd = np.linalg.norm(X[ok] - cam_a.C, axis=1)                    # weak-perspective scale ~ mean depth:
        lo, hi = np.percentile(dd, [10, 90])                             # 10 %-trimmed mean of the points
        ctr = X[ok][(dd >= lo) & (dd <= hi)].mean(0)
        ipd_lat = []
        for o, cam, size in ((oa, cam_a, size_a), (ob, cam_b, size_b)):
            Pu = undistort_landmarks(o.P, cam, size)
            sp, _, _ = umeyama_xy(U[:N_SURF], Pu[:N_SURF])                # px per IPD in the face-centred view
            f = float(cam.K[0, 0]) * size[0] / cam.size[0]
            ipd_lat.append(sp * float(np.linalg.norm(ctr - cam.C)) / f)
        m = fit_metric_pair(U, X, w)
        Xf = m["s"] * (U * (1, 1, m["k"])) @ m["R"].T + m["t"]
        ea = np.linalg.norm(project(cam_a, Xf[:N_SURF], size_a) - oa.P[:N_SURF, :2], axis=1)
        eb = np.linalg.norm(project(cam_b, Xf[:N_SURF], size_b) - ob.P[:N_SURF, :2], axis=1)
        dist = float(np.linalg.norm(ctr - cam_a.C))
        fa = float(cam_a.K[0, 0]) * size_a[0] / cam_a.size[0]
        view = (ctr - cam_a.C) / dist
        bperp = float(np.linalg.norm((cam_b.C - cam_a.C) - ((cam_b.C - cam_a.C) @ view) * view))
        rows.append({"idx": [oa.idx, ob.idx], "ok": True, "ipd_mm": round(float(np.mean(ipd_lat)) * 1e3, 2),
                     "ipd_mm_cams": [round(v * 1e3, 1) for v in ipd_lat],
                     "ipd_mm_3d": round(m["s"] * 1e3, 2), "depth_scale": round(m["k"], 3),
                     "gap_mm_med": round(float(np.nanmedian(gap[:N_SURF])) * 1e3, 1),
                     "fit_resid_mm_med": round(m["resid_m_med"] * 1e3, 1), "points": m["n"],
                     "reproj_px_med": [round(float(np.median(ea)), 2), round(float(np.median(eb)), 2)],
                     "ipd_px": [round(oa.ipd_px, 1), round(ob.ipd_px, 1)], "dist_m": round(dist, 2),
                     "depth_err_mm_per_px": round(dist * dist / (fa * max(bperp, 1e-6)) * 1e3, 1)})
    good = [r for r in rows if r["ok"]]
    out = {"pairs": rows, "n_pairs": len(good), "baseline_m": round(base, 3), "scale_used": False, "depth_used": False}
    if len(good) < min_pairs:
        out.update(used=False, why=f"only {len(good)} usable simultaneous pairs (need {min_pairs})")
        return out
    ipd = np.array([r["ipd_mm"] for r in good])
    k = np.array([r["depth_scale"] for r in good])
    mad = lambda v: 1.4826 * float(np.median(np.abs(v - np.median(v))))  # noqa: E731
    out.update(ipd_mm=round(float(np.median(ipd)), 2), ipd_mm_sd=round(mad(ipd), 2),
               ipd_mm_sem=round(1.2533 * mad(ipd) / math.sqrt(len(ipd)), 2),
               ipd_mm_3d_med=round(float(np.median([r["ipd_mm_3d"] for r in good])), 2),
               depth_scale=round(float(np.median(k)), 3), depth_scale_sd=round(mad(k), 3),
               depth_scale_sem=round(1.2533 * mad(k) / math.sqrt(len(k)), 3),
               gap_mm_med=round(float(np.median([r["gap_mm_med"] for r in good])), 1),
               dist_m_med=round(float(np.median([r["dist_m"] for r in good])), 2),
               depth_err_mm_per_px=round(float(np.median([r["depth_err_mm_per_px"] for r in good])), 1),
               reproj_px_med=[round(float(np.median([r["reproj_px_med"][i] for r in good])), 2) for i in (0, 1)])
    out["scale_used"] = bool(out["ipd_mm_sem"] <= max_scale_rel_sem * out["ipd_mm"])
    out["depth_used"] = bool(out["depth_scale_sd"] <= max_k_sd)
    out["used"] = out["scale_used"]
    why = []
    if not out["scale_used"]:
        why.append(f"metric scale inconsistent between pairs (sem {out['ipd_mm_sem']} mm)")
    if not out["depth_used"]:
        why.append(f"depth not resolved: stretch sd {out['depth_scale_sd']} between pairs > {max_k_sd} "
                   f"(expected depth error ~{out['depth_err_mm_per_px']} mm per px of landmark error at "
                   f"{out['dist_m_med']} m with a {out['baseline_m']} m baseline); MediaPipe's relative depth kept")
    if why:
        out["why"] = "; ".join(why)
    return out


def apply_depth_scale(S: np.ndarray, k: float) -> np.ndarray:
    """Stretch the fitted shape's depth (frontal z, about the nasion) by k."""
    S2 = np.asarray(S, np.float64).copy()
    S2[:, 2] *= k
    return S2


def metric_features(feats: dict, sd: dict | None, ipd_mm: float, ipd_mm_sd: float) -> dict:
    """IPD-unit features -> millimetres (prop and depth kinds), sd from the feature sd and the scale sd."""
    out = {}
    for k, (kind, _) in fg.FEATURES.items():
        if kind == "index" or feats.get(k) is None:
            continue
        v = feats[k] * ipd_mm
        e = math.hypot((sd or {}).get(k, 0.0) * ipd_mm, feats[k] * ipd_mm_sd)
        out[k] = {"mm": round(v, 1), "sd_mm": round(e, 1)}
    out["ipd"] = {"mm": round(ipd_mm, 1), "sd_mm": round(ipd_mm_sd, 1)}
    return out


# -- LODs: Loop subdivision, QEM decimation ------------------------------------------------------------------------------
def loop_subdivide(V: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One level of Loop subdivision (Warren's weights; boundary edges/vertices use the cubic B-spline rules, so the
    face outline stays a smooth curve). Triangles x 4; orientation is kept."""
    V = np.asarray(V, np.float64)
    T = np.asarray(T, np.int64)
    nv = len(V)
    he = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])      # half-edges, opposite = 3rd vertex
    opp = np.concatenate([T[:, 2], T[:, 0], T[:, 1]])
    key = np.sort(he, 1)
    E, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    ne = len(E)
    osum = np.zeros((ne, 3))
    np.add.at(osum, inv, V[opp])
    bnd = cnt == 1
    EV = np.where(bnd[:, None], 0.5 * (V[E[:, 0]] + V[E[:, 1]]),
                  0.375 * (V[E[:, 0]] + V[E[:, 1]]) + 0.125 * osum)
    nsum = np.zeros_like(V)
    deg = np.zeros(nv)
    np.add.at(nsum, E[:, 0], V[E[:, 1]])
    np.add.at(nsum, E[:, 1], V[E[:, 0]])
    np.add.at(deg, E[:, 0], 1)
    np.add.at(deg, E[:, 1], 1)
    beta = np.where(deg > 3, 3.0 / (8.0 * np.maximum(deg, 1)), 3.0 / 16.0)
    NV = (1 - deg * beta)[:, None] * V + beta[:, None] * nsum
    bE = E[bnd]
    if len(bE):
        bsum = np.zeros_like(V)
        bdeg = np.zeros(nv)
        np.add.at(bsum, bE[:, 0], V[bE[:, 1]])
        np.add.at(bsum, bE[:, 1], V[bE[:, 0]])
        np.add.at(bdeg, bE[:, 0], 1)
        np.add.at(bdeg, bE[:, 1], 1)
        bv = bdeg == 2
        NV[bv] = 0.75 * V[bv] + 0.125 * bsum[bv]
    unused = deg == 0
    NV[unused] = V[unused]
    nf = len(T)
    e01, e12, e20 = inv[:nf] + nv, inv[nf:2 * nf] + nv, inv[2 * nf:] + nv
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    T2 = np.concatenate([np.stack([a, e01, e20], 1), np.stack([e01, b, e12], 1),
                         np.stack([e20, e12, c], 1), np.stack([e01, e12, e20], 1)])
    return np.vstack([NV, EV]), T2


def decimate(V: np.ndarray, T: np.ndarray, target_tris: int, keep: Sequence[int] = (),
             boundary_weight: float = 50.0, min_cos: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy quadric-error (Garland-Heckbert) edge collapse onto an existing endpoint, so every surviving vertex is
    an original one (and the `keep` landmarks survive exactly). Boundary edges get perpendicular constraint planes;
    collapses that break the link condition or flip a triangle are refused.
    Returns (V', T', vid) with vid = original index of each vertex of V'."""
    V = np.asarray(V, np.float64)
    T = [list(map(int, t)) for t in T]
    nv = len(V)
    keep = set(int(k) for k in keep)
    alive = [True] * len(T)
    vf: list[set] = [set() for _ in range(nv)]
    for j, t in enumerate(T):
        for v in t:
            vf[v].add(j)
    Qm = np.zeros((nv, 4, 4))

    def plane(n, p):
        nn = np.linalg.norm(n)
        if nn < 1e-15:
            return np.zeros((4, 4))
        n = n / nn
        q = np.r_[n, -n @ p]
        return np.outer(q, q)
    for t in T:
        a, b, c = V[t[0]], V[t[1]], V[t[2]]
        n = np.cross(b - a, c - a)
        K = plane(n, a) * (0.5 * np.linalg.norm(n))
        for v in t:
            Qm[v] += K
    ecount: dict = {}
    for j, t in enumerate(T):
        for k in range(3):
            e = tuple(sorted((t[k], t[(k + 1) % 3])))
            ecount.setdefault(e, []).append(j)
    bverts = set()
    for e, js in ecount.items():
        if len(js) == 1:
            t = T[js[0]]
            a, b = V[e[0]], V[e[1]]
            c = V[[x for x in t if x not in e][0]]
            n = np.cross(b - a, c - a)
            perp = np.cross(n, b - a)
            K = plane(perp, a) * boundary_weight * float(np.dot(b - a, b - a))
            Qm[e[0]] += K
            Qm[e[1]] += K
            bverts |= set(e)

    def faces_of_edge(a, b):
        return [j for j in vf[a] & vf[b] if alive[j]]

    def is_boundary_edge(a, b):
        return len(faces_of_edge(a, b)) == 1

    def nbrs(v):
        return {x for j in vf[v] if alive[j] for x in T[j]} - {v}

    def cost(rm, to):
        q = Qm[rm] + Qm[to]
        h = np.r_[V[to], 1.0]
        return float(h @ q @ h)

    ver = [0] * nv
    heap: list = []

    def push(a, b):
        for rm, to in ((a, b), (b, a)):
            if rm in keep:
                continue
            if rm in bverts and not is_boundary_edge(a, b):
                continue
            heapq.heappush(heap, (cost(rm, to), rm, to, ver[rm], ver[to]))
    for e in ecount:
        push(*e)
    nt = len(T)
    while nt > target_tris and heap:
        c, rm, to, vr, vt = heapq.heappop(heap)
        if ver[rm] != vr or ver[to] != vt or not any(alive[j] for j in vf[rm]):
            continue
        shared = faces_of_edge(rm, to)
        if not shared:
            continue
        common = nbrs(rm) & nbrs(to)
        if len(common) != len(shared):                          # link condition
            continue
        ok = True
        for j in vf[rm]:
            if not alive[j] or j in shared:
                continue
            t = T[j]
            p0 = [V[x] for x in t]
            p1 = [V[to] if x == rm else V[x] for x in t]
            n0 = np.cross(p0[1] - p0[0], p0[2] - p0[0])
            n1 = np.cross(p1[1] - p1[0], p1[2] - p1[0])
            if n0 @ n1 <= min_cos * np.linalg.norm(n0) * np.linalg.norm(n1):
                ok = False
                break
        if not ok:
            continue
        for j in shared:
            alive[j] = False
            nt -= 1
        for j in list(vf[rm]):
            if alive[j]:
                T[j] = [to if x == rm else x for x in T[j]]
                vf[to].add(j)
        vf[rm] = set()
        Qm[to] += Qm[rm]
        if rm in bverts:
            bverts.add(to)
        ver[to] += 1
        ver[rm] += 1
        for n in nbrs(to):
            ver[n] += 1
        for n in nbrs(to):
            push(to, n)
            for m in nbrs(n):
                if m != to:
                    push(n, m)
    T2 = np.array([t for j, t in enumerate(T) if alive[j]], np.int64)
    vid = np.unique(T2)
    remap = -np.ones(nv, np.int64)
    remap[vid] = np.arange(len(vid))
    return V[vid], remap[T2], vid


def lods(S: np.ndarray, lod0_tris: int = 250, levels: int = 4) -> list[dict]:
    """Fitted shape (facegeom frontal frame) -> [{name, V, T, vid?}] LOD0 decimated, LOD1 native 898 triangles,
    LOD2/LOD3 Loop subdivision. Eyes and mouth are closed (as the canonical model)."""
    V = np.asarray(S, np.float64)[:N_SURF]
    T = topology(True)
    out = []
    V0, T0, vid = decimate(V, T, lod0_tris, keep=FEATURE_VERTS)
    out.append({"name": "LOD0", "V": V0, "T": T0, "vid": vid})
    out.append({"name": "LOD1", "V": V, "T": T})
    Vs, Ts = V, T
    for i in range(2, levels):
        Vs, Ts = loop_subdivide(Vs, Ts)
        out.append({"name": f"LOD{i}", "V": Vs, "T": Ts})
    return out


# -- glTF ----------------------------------------------------------------------------------------------------------------
def gltf_lods(levels: list[dict], unit_m: float | None = None, q: float = 4000.0,
              vertex_sd: np.ndarray | None = None, default: str = "LOD1") -> dict:
    """LODs (facegeom frontal frame, IPD units) -> one glTF 2.0 with one scene per LOD (scene = `default`),
    KHR_mesh_quantization int16 positions (/q), int8 normals, uint16/uint32 indices, one embedded buffer.
    unit_m: metres per IPD (two-view scale) -> the node scale makes the model metric; None keeps IPD units.
    vertex_sd (478,): bootstrap uncertainty as a custom _UNCERTAINTY attribute on LOD1 (uint8, x 1/1000 IPD)."""
    chunks, views, accs, meshes, nodes, scenes = [], [], [], [], [], []
    off = 0

    def add(buf: bytes, target: int | None, stride: int | None = None) -> int:
        nonlocal off
        pad = (-len(buf)) % 4
        buf += b"\0" * pad
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(buf) - pad}
        if stride:
            v["byteStride"] = stride
        if target:
            v["target"] = target
        chunks.append(buf)
        off += len(buf)
        views.append(v)
        return len(views) - 1
    scale = (unit_m if unit_m else 1.0) / q
    for lv in levels:
        Vg = np.asarray(lv["V"], np.float64) * (1, -1, 1)          # glTF: +y up, +z toward the viewer
        Tg = np.asarray(lv["T"], np.int64)[:, ::-1]                 # y flip mirrors: reverse the winding (CCW front)
        P16 = np.clip(np.round(Vg * q), -32767, 32767).astype("<i2")
        Nn = vertex_normals(Vg, Tg)
        N8 = np.clip(np.round(Nn * 127), -127, 127).astype("i1")
        pv = add(np.concatenate([P16, np.zeros((len(P16), 1), "<i2")], 1).tobytes(), 34962, 8)
        nv = add(np.concatenate([N8, np.zeros((len(N8), 1), "i1")], 1).tobytes(), 34962, 4)
        big = len(Vg) > 65535
        I = Tg.astype("<u4" if big else "<u2").ravel()
        iv = add(I.tobytes(), 34963)
        a0 = len(accs)
        accs += [{"bufferView": pv, "componentType": 5122, "count": len(P16), "type": "VEC3",
                  "min": P16.min(0).tolist(), "max": P16.max(0).tolist()},
                 {"bufferView": nv, "componentType": 5120, "normalized": True, "count": len(N8), "type": "VEC3"},
                 {"bufferView": iv, "componentType": 5125 if big else 5123, "count": len(I), "type": "SCALAR"}]
        attrs = {"POSITION": a0, "NORMAL": a0 + 1}
        if vertex_sd is not None and lv["name"] == "LOD1":
            sd8 = np.clip(np.round(np.asarray(vertex_sd)[:len(Vg)] * 1000), 0, 255).astype("u1")
            sv = add(sd8.tobytes(), 34962, None)
            accs.append({"bufferView": sv, "componentType": 5121, "count": len(sd8), "type": "SCALAR"})
            attrs["_UNCERTAINTY"] = len(accs) - 1
        meshes.append({"name": lv["name"], "primitives": [{"attributes": attrs, "indices": a0 + 2, "mode": 4}]})
        nodes.append({"name": lv["name"], "mesh": len(meshes) - 1, "scale": [scale] * 3})
        scenes.append({"name": lv["name"], "nodes": [len(nodes) - 1]})
    buf = b"".join(chunks)
    names = [lv["name"] for lv in levels]
    return {
        "asset": {"version": "2.0", "generator": VERSION},
        "extensionsUsed": ["KHR_mesh_quantization"], "extensionsRequired": ["KHR_mesh_quantization"],
        "scene": names.index(default) if default in names else 0, "scenes": scenes, "nodes": nodes, "meshes": meshes,
        "buffers": [{"byteLength": len(buf), "uri": "data:application/octet-stream;base64," + base64.b64encode(buf).decode()}],
        "bufferViews": views, "accessors": accs,
        "extras": {"units": "m" if unit_m else "IPD (inter-pupil distance = 1)",
                   "lods": {lv["name"]: int(len(lv["T"])) for lv in levels},
                   "uncertainty": "_UNCERTAINTY on LOD1 = bootstrap RMS displacement x 1/1000 IPD" if vertex_sd is not None else None,
                   "note": "fitted face shape; explanatory evidence only, never an identity decision"},
    }


def gltf_to_glb(g: dict) -> bytes:
    """Embedded-buffer glTF -> binary .glb (the buffer stored raw instead of base64)."""
    g = json.loads(json.dumps(g))
    raw = base64.b64decode(g["buffers"][0]["uri"].split(",", 1)[1])
    del g["buffers"][0]["uri"]
    js = json.dumps(g, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    raw += b"\0" * ((-len(raw)) % 4)
    total = 12 + 8 + len(js) + 8 + len(raw)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(raw), b"BIN\0") + raw)


def gltf_mesh(g: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of gltf_lods for one LOD (tests): (vertices (N, 3) in the file's units, glTF axes; faces (M, 3))."""
    raw = base64.b64decode(g["buffers"][0]["uri"].split(",", 1)[1])
    m = next(x for x in g["meshes"] if x["name"] == name)
    node = next(n for n in g["nodes"] if n["name"] == name)
    pa = g["accessors"][m["primitives"][0]["attributes"]["POSITION"]]
    ia = g["accessors"][m["primitives"][0]["indices"]]
    pv, iv = g["bufferViews"][pa["bufferView"]], g["bufferViews"][ia["bufferView"]]
    V = np.frombuffer(raw, "<i2", pa["count"] * 4, pv["byteOffset"]).reshape(-1, 4)[:, :3] * node["scale"][0]
    dt = "<u4" if ia["componentType"] == 5125 else "<u2"
    F = np.frombuffer(raw, dt, ia["count"], iv["byteOffset"]).reshape(-1, 3)
    return V, F


# -- per-visit driver --------------------------------------------------------------------------------------------------
def fit_visit(obs: Sequence[FrameObs], cams: dict | None = None, pairs: list[tuple[FrameObs, FrameObs]] | None = None,
              prior: np.ndarray | None = None, params: FitParams | None = None, mesh: bool = True) -> dict:
    """All landmark frames of one visit (any cameras) -> the facefit payload (json-serialisable) + `_fit` (the
    FitResult, for callers; strip it before storing). cams: {cam name: (geometry.Camera, frame_size)} to undistort
    the landmarks (and, with `pairs` of simultaneous frames of exactly two cameras, the metric two-view step)."""
    p = params or FitParams()
    base = {"version": VERSION, "explanatory": True,
            "note": "One face shape fitted over the visit; explanation only, identity was decided by the face embedding."}
    obs = list(obs)
    if not obs:
        return {**base, "n_frames": 0, "reason": "no_face"}
    cams = cams or {}
    X = np.stack([undistort_landmarks(o.P, *cams[o.cam]) if o.cam in cams else np.asarray(o.P, np.float64)
                  for o in obs])
    wf = frame_base_weights(obs, p)
    if not (wf > 0).any():
        return {**base, "n_frames": len(obs), "reason": "faces_too_small"}
    fit = fit_identity(X, wf, [o.blend for o in obs], prior, p)
    used = fit.w_frame > 1e-6
    S = fit.S
    feats = features(S)
    out = {**base, "n_frames": len(obs), "n_used": int(used.sum()), "prior": fit.prior, "iters": fit.iters,
           "cams": sorted({o.cam for o in obs}),
           "ipd_px_med": round(float(np.median([o.ipd_px for o, u in zip(obs, used) if u])), 1) if used.any() else None,
           "yaw_range": [round(float(fit.angles[used, 0].min()), 1), round(float(fit.angles[used, 0].max()), 1)] if used.any() else None,
           "reproj_px_med": round(float(np.median(fit.resid_px[used])), 2) if used.any() else None,
           "reproj_ipd_pct_med": round(100 * float(np.median(fit.resid_ipd[used])), 2) if used.any() else None,
           "apparent_expression": {
               "note": "apparent expression (description only): share of used frames whose mouth/eyes/brows moved "
                       "away from the visit's resting state; such frames and regions were down-weighted",
               **{r: round(float(np.mean(fit.expr[used, i] > 0.5)), 2) if used.any() else None
                  for i, r in enumerate(REGIONS)}},
           "fit_ms": round(fit.ms, 1)}
    mv = None
    if pairs and len(cams) >= 2:
        ca, cb = pairs[0][0].cam, pairs[0][1].cam
        mv = multiview(_internal(S), pairs, cams[ca][0], cams[ca][1], cams[cb][0], cams[cb][1])
        mv["cams"] = [ca, cb]
        out["multiview"] = {k: v for k, v in mv.items() if k != "pairs"} | {"pairs": mv["pairs"]}
        if mv.get("depth_used"):
            S = apply_depth_scale(S, mv["depth_scale"])
            feats = features(S)
    elif pairs is not None:
        out["multiview"] = {"used": False, "why": "two calibrated cameras needed"}
    sd = fit.boot_features or {}
    out["features"] = {k: {"value": feats[k], "sd": sd.get(k), "kind": fg.FEATURES[k][0]} for k in fg.FEATURES}
    if mv and mv.get("scale_used"):
        out["metric"] = metric_features(feats, sd, mv["ipd_mm"], max(mv["ipd_mm_sem"], 0.5))
        out["metric"]["note"] = ("millimetres from two-camera triangulation (face distance x angular size); as "
                                 "accurate as the rig calibration's scale" + ("" if mv.get("depth_used") else
                                 "; depth features keep MediaPipe's relative depth (not measured)"))
    if fit.vertex_sd is not None:
        out["vertex_sd_ipd"] = {"median": round(float(np.median(fit.vertex_sd[:N_SURF])), 4),
                                "p90": round(float(np.percentile(fit.vertex_sd[:N_SURF], 90)), 4)}
    if mesh:
        lv = lods(S)
        unit = (mv["ipd_mm"] / 1000.0) if mv and mv.get("scale_used") else None
        out["mesh"] = gltf_lods(lv, unit, vertex_sd=fit.vertex_sd)
    out["_fit"] = fit
    out["_S"] = S
    return out


def strip(payload: dict) -> dict:
    """The payload without the python objects / mesh (what goes into JSON storage)."""
    return {k: v for k, v in payload.items() if not k.startswith("_") and k != "mesh"}


# -- per-visit collector (same interface as facegeom.GeomCollector) ------------------------------------------------------
class FitCollector:
    """Fed by faceid.FaceSampler for every accepted face (offer(frame, o, w) like facegeom.GeomCollector), it runs the
    landmarker immediately on a crop around InspireFace's face box and keeps only the landmarks (478 x 3 floats,
    ~6 KB per frame; bounded by max_frames, lowest-weight frames dropped first), so no crops are buffered. result()
    fits the visit. cam/calib: the camera name and (geometry.Camera, frame_size) for lens-aware fitting, or None."""

    def __init__(self, landmarker: Callable, cam: str = "", calib: tuple | None = None, max_frames: int = 96,
                 max_ms: float = 4000.0, margin: float = 1.7, min_px: float = 24.0, prior: np.ndarray | None = None,
                 params: FitParams | None = None):
        self.lm, self.cam, self.calib = landmarker, cam, calib
        self.max_frames, self.max_ms, self.margin, self.min_px = max_frames, max_ms, margin, min_px
        self.prior, self.p = prior, params or FitParams()
        self.obs: list[tuple[float, FrameObs]] = []
        self.offered = self.missed = 0
        self.ms = 0.0

    def offer(self, frame: np.ndarray, o, w: float) -> None:
        if o.box is None or o.size < self.min_px or self.ms >= self.max_ms:
            return
        if len(self.obs) >= self.max_frames and w <= self.obs[-1][0]:
            return
        if self.calib and self.calib[1] is None:     # calibration given without a frame size: take this frame's
            self.calib = (self.calib[0], (frame.shape[1], frame.shape[0]))
        x1, y1, x2, y2 = o.box
        side = max(x2 - x1, y2 - y1) * self.margin
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        self.offered += 1
        t0 = time.perf_counter()
        try:
            fo = landmarks_in_window(frame, (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2), self.lm,
                                     cam=self.cam, idx=int(getattr(o, "t", 0) or 0), t=float(getattr(o, "t", 0) or 0))
        except Exception:
            log.exception("facefit landmarks failed")
            fo = None
        self.ms += (time.perf_counter() - t0) * 1e3
        if fo is None:
            self.missed += 1
            return
        self.obs.append((float(w), fo))
        self.obs.sort(key=lambda x: -x[0])
        del self.obs[self.max_frames:]

    def result(self, mesh: bool = True) -> dict:
        obs = sorted((fo for _, fo in self.obs), key=lambda fo: fo.t)
        cams = {self.cam: self.calib} if self.calib else None
        r = fit_visit(obs, cams=cams, prior=self.prior, params=self.p, mesh=mesh)
        r.update(offered=self.offered, missed=self.missed, landmark_ms=round(self.ms, 1))
        return {k: v for k, v in r.items() if not k.startswith("_")}

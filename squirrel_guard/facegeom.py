"""Face geometry + relative depth from a dense 3D face mesh (explanatory evidence; NEVER an identity decision).

What it does (per visit, on the few best near-frontal faces that faceid.FaceSampler already found):
  best face crops (|yaw| < 25 deg, |pitch| < 30 deg, >= geom_min_px) -> MediaPipe Face Landmarker (Apache-2.0, 478
  landmarks with a RELATIVE z, CPU) -> the mesh is frontalised (eye line = x, nasion->chin = y) and scaled by the
  inter-pupil distance (IPD = 1) -> per-face proportions and relative depths -> robust per-visit aggregate (median,
  spread = 1.4826 x MAD, n, face px + pose used) -> evidence images (landmark overlay, relative depth heatmap) and a
  small quantised glTF mesh for a rotatable 3D view.

POLICY (same rules as faceid.py)
  - Explanatory only: identity decisions stay with the 512-d embedding (facematch.decide). compare() shows how a
    visit's geometry differs from an enrolled person's, feature by feature, and says so in its output.
  - No ethnicity/race inference, no outside lookups. Geometry is never mapped to any group or category; it is only
    compared with the same person's own enrolled geometry. Expression stays description-only (faceid.apparent_of).
  - The depth is the model's RELATIVE shape estimate from one RGB image (strongly shaped by its learned face prior),
    not millimetres and not a depth sensor; 2D proportions from small surveillance faces are noisy. See
    docs/facegeom.md for the measured repeatability and which features are stable enough to show.

The module imports cv2/mediapipe lazily: compare()/merge() and the feature maths run with numpy only (the recorder host
bridge keeps a copy of compare/merge in deploy/recorder/bridge/facegeomcmp.py; tests keep the two in step).
"""
from __future__ import annotations

import base64
import dataclasses
import json
import logging
import math
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)
VERSION = "facegeom/1"

# MediaPipe 478-point topology (canonical face mesh; 468-477 = irises). "r"/"l" = the subject's right/left.
IRIS_R, IRIS_L = 468, 473
EYE_R_OUT, EYE_R_IN, EYE_L_IN, EYE_L_OUT = 33, 133, 362, 263
EYE_R_TOP, EYE_L_TOP = 159, 386
BROW_R, BROW_L = 105, 334
NASION, NOSE_TIP, SUBNASALE = 168, 1, 2
ALA_R, ALA_L = 129, 358
MOUTH_R, MOUTH_L, LIP_TOP, LIP_BOTTOM = 61, 291, 0, 17
CHEEK_R, CHEEK_L = 234, 454           # face outline at the cheekbones (zygion level)
CHEEKS_R, CHEEKS_L = 50, 280          # cheek surface below the eyes
JAW_R, JAW_L = 172, 397               # jaw angle region (gonion level)
CHIN, GLABELLA = 152, 9
MIRROR_PAIRS = ((EYE_R_OUT, EYE_L_OUT), (EYE_R_IN, EYE_L_IN), (EYE_R_TOP, EYE_L_TOP), (BROW_R, BROW_L),
                (ALA_R, ALA_L), (MOUTH_R, MOUTH_L), (CHEEK_R, CHEEK_L), (CHEEKS_R, CHEEKS_L), (JAW_R, JAW_L),
                (IRIS_R, IRIS_L))
MIDLINE = (10, 151, GLABELLA, NASION, 6, 197, 195, 5, 4, NOSE_TIP, SUBNASALE, LIP_TOP, LIP_BOTTOM, CHIN)

# feature -> (kind, label). "prop" = 2D proportion in the frontalised face plane / IPD; "depth" = relative depth / IPD
# (positive = toward the camera); "index" = dimensionless.
FEATURES = {
    "nose_width": ("prop", "Nose width (alae)"),
    "nose_length": ("prop", "Nose length (nasion to base)"),
    "mouth_width": ("prop", "Mouth width"),
    "eye_width_r": ("prop", "Right eye width"),
    "eye_width_l": ("prop", "Left eye width"),
    "eye_spacing": ("prop", "Inner eye spacing"),
    "cheek_width": ("prop", "Face width at cheekbones"),
    "jaw_width": ("prop", "Jaw width"),
    "face_height": ("prop", "Face height (brow to chin)"),
    "brow_eye": ("prop", "Brow to eye height"),
    "philtrum": ("prop", "Philtrum length"),
    "asymmetry": ("index", "Left/right asymmetry (0 = mirror symmetric)"),
    "nose_protrusion": ("depth", "Nose tip protrusion"),
    "eye_socket_depth": ("depth", "Eye socket depth"),
    "cheek_depth": ("depth", "Cheek contour set-back"),
    "jaw_depth": ("depth", "Jaw contour set-back"),
}
# Between-visit within-person spread floor (IPD units), used by compare(); set from scripts/facegeom_eval.py
# (docs/facegeom.md). A feature not listed falls back to DEFAULT_FLOOR.
WITHIN_SD = {
    "nose_width": 0.035, "nose_length": 0.04, "mouth_width": 0.05, "eye_width_r": 0.03, "eye_width_l": 0.03,
    "eye_spacing": 0.03, "cheek_width": 0.08, "jaw_width": 0.08, "face_height": 0.08, "brow_eye": 0.04,
    "philtrum": 0.03, "asymmetry": 0.02, "nose_protrusion": 0.04, "eye_socket_depth": 0.03, "cheek_depth": 0.06,
    "jaw_depth": 0.06,
}
DEFAULT_FLOOR = 0.05
# Features whose within-person repeatability was clearly better than the between-person spread on the evaluation
# clips (docs/facegeom.md). The others are still computed and stored but rendered as "noisy".
STABLE_FEATURES: tuple[str, ...] = ("eye_spacing", "nose_width", "nose_length", "face_height", "brow_eye",
                                    "nose_protrusion")
MIN_SHOW_PX = 48.0                     # below this median face size the panel says "too small to measure"


# -- geometry ----------------------------------------------------------------------------------------------------------
def frontalise(P: np.ndarray) -> tuple[np.ndarray, float]:
    """(478,3) mesh in crop pixels (x right, y down, z in x units, smaller = nearer) -> frontal coordinates centred
    at the nasion, scaled so IPD = 1: x = subject's right->left eye axis, y = down the face, z = toward the camera.
    Returns (Q, ipd_px)."""
    P = np.asarray(P, np.float64)
    er = P[[EYE_R_OUT, EYE_R_IN]].mean(0)
    el = P[[EYE_L_OUT, EYE_L_IN]].mean(0)
    ex = el - er
    ex /= np.linalg.norm(ex) + 1e-12
    down = P[CHIN] - P[NASION]
    ey = down - (down @ ex) * ex
    ey /= np.linalg.norm(ey) + 1e-12
    ez = np.cross(ex, ey)                              # x right, y down -> z into the scene
    R = np.stack([ex, ey, ez], 1)
    Q = (P - P[NASION]) @ R
    Q[:, 2] *= -1                                      # z toward the camera
    ipd = float(np.linalg.norm(P[IRIS_R] - P[IRIS_L])) if len(P) > IRIS_L else float(np.linalg.norm(el - er))
    if ipd < 1e-6:
        ipd = float(np.linalg.norm(el - er)) or 1.0
    return Q / ipd, ipd


def pose_of(P: np.ndarray) -> tuple[float, float]:
    """Approximate (yaw, pitch) in degrees of a mesh in image coordinates (for reporting/filtering)."""
    P = np.asarray(P, np.float64)
    ex = P[[EYE_L_OUT, EYE_L_IN]].mean(0) - P[[EYE_R_OUT, EYE_R_IN]].mean(0)
    yaw = math.degrees(math.atan2(-ex[2], math.hypot(ex[0], ex[1])))
    dn = P[CHIN] - P[NASION]
    pitch = math.degrees(math.atan2(-dn[2], math.hypot(dn[0], dn[1])))
    return yaw, pitch


def _d2(Q, a, b) -> float:
    return float(np.linalg.norm(Q[a, :2] - Q[b, :2]))


def features_of(P: np.ndarray) -> dict[str, float]:
    """Per-face features (IPD units) from one (478,3) mesh."""
    Q, _ = frontalise(P)
    z = Q[:, 2]
    x0 = float(np.median(Q[list(MIDLINE), 0]))
    asym = [np.linalg.norm([Q[a, 0] + Q[b, 0] - 2 * x0, Q[a, 1] - Q[b, 1], Q[a, 2] - Q[b, 2]])
            for a, b in MIRROR_PAIRS]
    f = {
        "nose_width": _d2(Q, ALA_R, ALA_L),
        "nose_length": _d2(Q, NASION, SUBNASALE),
        "mouth_width": _d2(Q, MOUTH_R, MOUTH_L),
        "eye_width_r": _d2(Q, EYE_R_OUT, EYE_R_IN),
        "eye_width_l": _d2(Q, EYE_L_IN, EYE_L_OUT),
        "eye_spacing": _d2(Q, EYE_R_IN, EYE_L_IN),
        "cheek_width": _d2(Q, CHEEK_R, CHEEK_L),
        "jaw_width": _d2(Q, JAW_R, JAW_L),
        "face_height": _d2(Q, GLABELLA, CHIN),
        "brow_eye": (_d2(Q, BROW_R, EYE_R_TOP) + _d2(Q, BROW_L, EYE_L_TOP)) / 2,
        "philtrum": _d2(Q, SUBNASALE, LIP_TOP),
        "asymmetry": float(np.mean(asym)),
        "nose_protrusion": float(z[NOSE_TIP] - (z[EYE_R_IN] + z[EYE_L_IN]) / 2),
        "eye_socket_depth": float(z[[BROW_R, BROW_L, CHEEKS_R, CHEEKS_L]].mean() - z[[IRIS_R, IRIS_L]].mean()),
        "cheek_depth": float(z[NASION] - (z[CHEEK_R] + z[CHEEK_L]) / 2),
        "jaw_depth": float(z[CHIN] - (z[JAW_R] + z[JAW_L]) / 2),
    }
    return {k: round(float(v), 4) for k, v in f.items()}


def robust(vals) -> dict:
    """median, spread (1.4826 x MAD; 0 for one value), sem (1.2533 x spread / sqrt n), n."""
    v = np.asarray([x for x in vals if x is not None and math.isfinite(x)], np.float64)
    if not len(v):
        return {"value": None, "spread": None, "sem": None, "n": 0}
    m = float(np.median(v))
    s = float(1.4826 * np.median(np.abs(v - m))) if len(v) > 1 else 0.0
    return {"value": round(m, 4), "spread": round(s, 4), "sem": round(1.2533 * s / math.sqrt(len(v)), 4), "n": int(len(v))}


def aggregate(per_face: list[dict], meta: list[dict]) -> dict:
    """Per-face feature dicts (+ their size/yaw/pitch meta) -> per-visit features with uncertainties."""
    feats = {k: robust([f.get(k) for f in per_face]) for k in FEATURES}
    sizes = [m["size"] for m in meta]
    return {"features": feats, "n_faces": len(per_face),
            "size_px": round(float(np.median(sizes)), 1) if sizes else None,
            "size_px_range": [round(float(min(sizes)), 1), round(float(max(sizes)), 1)] if sizes else None,
            "yaw_abs_med": round(float(np.median([abs(m["yaw"]) for m in meta])), 1) if meta else None,
            "pitch_abs_med": round(float(np.median([abs(m["pitch"]) for m in meta])), 1) if meta else None}


# -- comparison with an enrolled person (KEEP IN STEP with deploy/recorder/bridge/facegeomcmp.py) --------------------------
def merge(geoms: list[dict]) -> dict | None:
    """Several visits' facegeom (enrolled templates of one identity) -> one enrolled geometry: per feature the
    n-weighted mean of the visit medians, spread = pooled within-visit spread plus the between-visit spread."""
    geoms = [g for g in geoms if g and g.get("features")]
    if not geoms:
        return None
    out = {}
    for k in FEATURES:
        rows = [g["features"][k] for g in geoms if (g["features"].get(k) or {}).get("value") is not None]
        if not rows:
            continue
        n = [max(1, int(r.get("n") or 1)) for r in rows]
        N = sum(n)
        mu = sum(r["value"] * w for r, w in zip(rows, n)) / N
        within = sum((r.get("spread") or 0.0) ** 2 * w for r, w in zip(rows, n)) / N
        between = sum((r["value"] - mu) ** 2 * w for r, w in zip(rows, n)) / N
        sp = math.sqrt(within + between)
        out[k] = {"value": round(mu, 4), "spread": round(sp, 4), "sem": round(1.2533 * sp / math.sqrt(N), 4),
                  "n": N, "visits": len(rows)}
    return {"features": out, "visits": len(geoms), "version": VERSION}


def compare(visit: dict | None, enrolled: dict | None, within_sd: dict | None = None) -> dict | None:
    """Per-feature difference of a visit's geometry vs an identity's enrolled geometry.
    z = (visit - enrolled) / sqrt(sem_visit^2 + sem_enrolled^2 + floor^2), floor = measured between-visit spread of
    the same person (WITHIN_SD). EXPLANATORY ONLY: the identity decision is the face embedding's."""
    if not visit or not enrolled or not visit.get("features") or not enrolled.get("features"):
        return None
    floors = within_sd or WITHIN_SD
    rows = {}
    for k, (kind, label) in FEATURES.items():
        a, b = visit["features"].get(k) or {}, enrolled["features"].get(k) or {}
        if a.get("value") is None or b.get("value") is None:
            continue
        sd = math.sqrt((a.get("sem") or 0.0) ** 2 + (b.get("sem") or 0.0) ** 2 + floors.get(k, DEFAULT_FLOOR) ** 2)
        d = a["value"] - b["value"]
        z = d / sd
        rows[k] = {"visit": a["value"], "enrolled": b["value"], "diff": round(d, 4),
                   "rel_pct": round(100 * d / b["value"], 1) if abs(b["value"]) > 1e-3 and kind != "depth" else None,
                   "z": round(z, 2), "verdict": "consistent" if abs(z) < 2 else "differs somewhat" if abs(z) < 3
                   else "differs", "stable": k in STABLE_FEATURES}
    st = [r for k, r in rows.items() if r["stable"]]
    return {"explanatory": True,
            "note": "Shape comparison for explanation only; who this is was decided by the face embedding.",
            "features": rows, "n_stable": len(st), "n_stable_consistent": sum(r["verdict"] == "consistent" for r in st)}


# -- mesh model --------------------------------------------------------------------------------------------------------
class FaceMesher:
    """MediaPipe Face Landmarker on one face crop -> (478,3) landmarks in crop pixels (z in the same units as x,
    smaller = nearer) or None. Not thread-safe: one per worker (it lives on the FaceEngine)."""

    def __init__(self, model_path: str, min_conf: float = 0.3):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        self.mp = mp
        opt = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), num_faces=1,
                                           min_face_detection_confidence=min_conf, min_face_presence_confidence=min_conf)
        self.fl = vision.FaceLandmarker.create_from_options(opt)

    def __call__(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        import cv2
        h, w = crop_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        r = self.fl.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        if not r.face_landmarks:
            return None
        return np.array([(p.x * w, p.y * h, p.z * w) for p in r.face_landmarks[0]], np.float32)


# -- per-visit collector -----------------------------------------------------------------------------------------------
@dataclasses.dataclass
class GeomParams:
    keep: int = 8                  # best near-frontal faces kept (crops held until the end of the visit)
    max_yaw: float = 25.0
    max_pitch: float = 30.0
    min_px: float = 32.0
    crop_px: int = 256             # face crop side handed to the mesh model (face box x margin, resized)
    margin: float = 1.7


@dataclasses.dataclass(eq=False)
class _Cand:
    w: float
    crop: np.ndarray
    scale: float                   # crop px per frame px
    origin: tuple                  # crop (x0, y0) in frame px
    size: float
    yaw: float
    pitch: float
    lmk106: np.ndarray | None      # InspireFace dense landmarks in CROP pixels
    t: float = 0.0


class GeomCollector:
    """Keeps the best few near-frontal face crops of a visit (bounded: keep x crop_px^2 x 3 bytes) and turns them
    into the facegeom payload at the end. offer() is cheap (a crop copy); the mesh model runs only in result()."""

    def __init__(self, mesher: Callable[[np.ndarray], np.ndarray | None], params: GeomParams | None = None):
        self.mesher, self.p = mesher, params or GeomParams()
        self.cands: list[_Cand] = []
        self.offered = 0
        self.off_pose = 0

    def offer(self, frame: np.ndarray, o, w: float) -> None:
        """o: faceid.FaceObs (box in frame px, yaw/pitch, optional lmk106 in frame px); w: its faceid weight."""
        if o.box is None or o.size < self.p.min_px:
            return
        self.offered += 1
        if abs(o.yaw) > self.p.max_yaw or abs(o.pitch) > self.p.max_pitch:
            self.off_pose += 1
            return
        if len(self.cands) >= self.p.keep and w <= self.cands[-1].w:
            return
        import cv2
        x1, y1, x2, y2 = o.box
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * self.p.margin
        X0, Y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        S = int(round(side))
        H, W = frame.shape[:2]
        pad = [max(0, -Y0), max(0, Y0 + S - H), max(0, -X0), max(0, X0 + S - W)]
        sub = frame[max(0, Y0):min(H, Y0 + S), max(0, X0):min(W, X0 + S)]
        if sub.size == 0:
            return
        if any(pad):
            sub = cv2.copyMakeBorder(sub, *pad, cv2.BORDER_REPLICATE)
        sc = self.p.crop_px / S
        crop = cv2.resize(sub, (self.p.crop_px, self.p.crop_px),
                          interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)
        lm = getattr(o, "lmk106", None)
        lm = None if lm is None else (np.asarray(lm, np.float32) - (X0, Y0)) * sc
        self.cands.append(_Cand(w, crop, sc, (X0, Y0), float(o.size), float(o.yaw), float(o.pitch), lm,
                                float(getattr(o, "t", 0.0))))
        self.cands.sort(key=lambda c: -c.w)
        del self.cands[self.p.keep:]

    def result(self, images: bool = True) -> dict:
        base = {"version": VERSION, "explanatory": True, "offered": self.offered, "off_pose": self.off_pose,
                "params": {"max_yaw": self.p.max_yaw, "max_pitch": self.p.max_pitch, "min_px": self.p.min_px}}
        per, meta, meshes, used = [], [], [], []
        for c in self.cands:
            try:
                P = self.mesher(c.crop)
            except Exception:
                log.exception("face mesh failed")
                P = None
            if P is None or len(P) < 468:
                continue
            if len(P) < 478:            # no iris points: fall back to eye-corner centres for the IPD
                P = np.vstack([P, np.repeat(P[[EYE_R_OUT, EYE_R_IN]].mean(0, keepdims=True), 5, 0),
                               np.repeat(P[[EYE_L_OUT, EYE_L_IN]].mean(0, keepdims=True), 5, 0)])
            myaw, mpitch = pose_of(P)
            if abs(myaw) > self.p.max_yaw + 10:     # the mesh disagrees badly with InspireFace's pose: skip
                continue
            per.append(features_of(P))
            meta.append({"size": c.size, "yaw": c.yaw, "pitch": c.pitch, "mesh_yaw": round(myaw, 1), "t": c.t})
            meshes.append(frontalise(P)[0])
            used.append((c, P))
        if not per:
            return {**base, "n_faces": 0, "reason": "no_frontal_face" if not self.cands else "no_mesh"}
        out = {**base, **aggregate(per, meta), "faces": [{**m, **{"size": round(m["size"], 1), "yaw": round(m["yaw"], 1),
                                                               "pitch": round(m["pitch"], 1), "f": f}} for m, f in zip(meta, per)]}
        med = np.median(np.stack(meshes), 0)
        out["mesh"] = mesh_gltf(med)
        if images:
            c, P = used[0]
            out["overlay_jpg_b64"] = _b64(_jpeg(overlay_image(c.crop, P, c.lmk106)))
            out["depth_jpg_b64"] = _b64(_jpeg(depth_image(med)))
        return out


# -- evidence images + mesh --------------------------------------------------------------------------------------------
def _b64(b: bytes | None) -> str | None:
    return base64.b64encode(b).decode() if b else None


def _jpeg(img: np.ndarray | None, q: int = 80) -> bytes | None:
    import cv2
    if img is None:
        return None
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes() if ok else None


MEASURE_LINES = (("nose_width", ALA_R, ALA_L), ("mouth_width", MOUTH_R, MOUTH_L), ("eye_spacing", EYE_R_IN, EYE_L_IN),
                 ("cheek_width", CHEEK_R, CHEEK_L), ("jaw_width", JAW_R, JAW_L), ("nose_length", NASION, SUBNASALE),
                 ("face_height", GLABELLA, CHIN), ("ipd", IRIS_R, IRIS_L))


def overlay_image(crop: np.ndarray, P: np.ndarray, lmk106: np.ndarray | None = None) -> np.ndarray:
    """Best crop with the 478 mesh points (small, cyan), InspireFace's 106 points (yellow) and the measured spans."""
    import cv2
    img = crop.copy()
    for x, y, _ in P[:468]:
        cv2.circle(img, (int(round(x)), int(round(y))), 1, (230, 200, 40), -1, cv2.LINE_AA)
    if lmk106 is not None:
        for x, y in np.asarray(lmk106)[:, :2]:
            cv2.circle(img, (int(round(x)), int(round(y))), 2, (0, 220, 255), -1, cv2.LINE_AA)
    for name, a, b in MEASURE_LINES:
        if max(a, b) >= len(P):
            continue
        col = (60, 60, 255) if name == "ipd" else (80, 255, 80)
        cv2.line(img, tuple(int(v) for v in P[a, :2]), tuple(int(v) for v in P[b, :2]), col, 1, cv2.LINE_AA)
    cv2.putText(img, "mesh 478 (cyan)  106 (yellow)  IPD (red)", (4, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.33, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def triangles(Q: np.ndarray) -> np.ndarray:
    """Delaunay triangulation of the frontal x/y of the 468 surface points (no MediaPipe tables needed); long
    triangles across the open mouth/eyes are kept, which is fine for a shape preview."""
    import cv2
    xy = Q[:468, :2]
    lo, hi = xy.min(0) - 1, xy.max(0) + 1
    k = 1000.0
    sub = cv2.Subdiv2D((0, 0, int((hi[0] - lo[0]) * k) + 2, int((hi[1] - lo[1]) * k) + 2))
    pts = (xy - lo) * k
    index = {}
    for i, (x, y) in enumerate(pts):
        sub.insert((float(x), float(y)))
        index[(round(float(x), 1), round(float(y), 1))] = i
    tris = []
    for t in sub.getTriangleList():
        ids = []
        for j in range(3):
            key = (round(float(t[2 * j]), 1), round(float(t[2 * j + 1]), 1))
            if key not in index:
                break
            ids.append(index[key])
        if len(ids) == 3:
            tris.append(ids)
    T = np.array(tris, np.int32).reshape(-1, 3)
    if len(T):                                      # drop sliver triangles along the hull
        e = np.stack([np.linalg.norm(xy[T[:, a]] - xy[T[:, b]], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], 1)
        T = T[e.max(1) < 0.6]
    return T


def depth_image(Q: np.ndarray, side: int = 192) -> np.ndarray:
    """Relative depth heatmap of a frontal mesh (warm = toward the camera), painter-filled triangles + a legend."""
    import cv2
    T = triangles(Q)
    xy = Q[:468, :2]
    lo, hi = xy.min(0), xy.max(0)
    s = (side - 24) / float(max(hi - lo))
    off = (side - (hi - lo) * s) / 2
    px = (xy - lo) * s + off
    z = Q[:468, 2]
    lo_z, hi_z = np.percentile(z, [2, 99.5])
    zn = np.clip((z - lo_z) / (hi_z - lo_z + 1e-9), 0, 1)
    acc = np.zeros((side, side), np.float32)
    cnt = np.zeros((side, side), np.float32)
    for tri in T[np.argsort(z[T].mean(1))]:         # far first
        m = np.zeros((side, side), np.uint8)
        cv2.fillConvexPoly(m, np.round(px[tri]).astype(np.int32), 1, cv2.LINE_8)
        acc[m > 0] = float(zn[tri].mean())
        cnt[m > 0] = 1
    acc = cv2.GaussianBlur(acc, (5, 5), 0)
    img = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    img[cnt == 0] = (24, 24, 24)
    for i in range(side - 16):                      # legend: far -> near
        v = int(255 * i / (side - 17))
        cv2.line(img, (8 + i, side - 8), (8 + i, side - 4), tuple(int(c) for c in cv2.applyColorMap(
            np.uint8([[v]]), cv2.COLORMAP_INFERNO)[0, 0]), 1)
    cv2.putText(img, "far", (8, side - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, "near (relative)", (side - 92, side - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 220), 1,
                cv2.LINE_AA)
    return img


def mesh_gltf(Q: np.ndarray, q: float = 4000.0) -> dict:
    """Frontal mesh (IPD units) -> compact glTF 2.0 (KHR_mesh_quantization: int16 positions x 1/q, uint16 indices,
    one embedded base64 buffer; ~8-12 KB of JSON). glTF axes: +y up, +z toward the viewer."""
    V = Q[:468].astype(np.float64) * (1, -1, 1)
    P16 = np.clip(np.round(V * q), -32767, 32767).astype("<i2")
    T = triangles(Q)
    if len(T):                                      # counter-clockwise seen from +z
        a, b, c = V[T[:, 0], :2], V[T[:, 1], :2], V[T[:, 2], :2]
        cw = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) < 0
        T[cw] = T[cw][:, ::-1]
    I16 = T.astype("<u2").ravel()
    pb = np.concatenate([P16, np.zeros((len(P16), 1), "<i2")], 1).tobytes()   # 8-byte stride: 4-byte aligned
    buf = pb + I16.tobytes()
    return {
        "asset": {"version": "2.0", "generator": VERSION},
        "extensionsUsed": ["KHR_mesh_quantization"], "extensionsRequired": ["KHR_mesh_quantization"],
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "scale": [1 / q] * 3}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "buffers": [{"byteLength": len(buf), "uri": "data:application/octet-stream;base64," + _b64(buf)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pb), "byteStride": 8,
                         "target": 34962},
                        {"buffer": 0, "byteOffset": len(pb), "byteLength": I16.nbytes, "target": 34963}],
        "accessors": [{"bufferView": 0, "componentType": 5122, "count": len(P16), "type": "VEC3",
                       "min": P16.min(0).tolist(), "max": P16.max(0).tolist()},
                      {"bufferView": 1, "componentType": 5123, "count": len(I16), "type": "SCALAR"}],
    }


def gltf_arrays(g: dict) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of mesh_gltf (tests, the renderer): (vertices float (N,3) in IPD units, glTF axes; faces (M,3))."""
    raw = base64.b64decode(g["buffers"][0]["uri"].split(",", 1)[1])
    pa, ia = g["accessors"]
    pv, iv = g["bufferViews"][pa["bufferView"]], g["bufferViews"][ia["bufferView"]]
    st = pv.get("byteStride", 6) // 2
    V = np.frombuffer(raw, "<i2", pa["count"] * st, pv["byteOffset"]).reshape(-1, st)[:, :3] * g["nodes"][0]["scale"][0]
    F = np.frombuffer(raw, "<u2", ia["count"], iv["byteOffset"]).reshape(-1, 3)
    return V, F


# -- render helper for the persons UI ----------------------------------------------------------------------------------
def render_html(geom: dict | None, cmp: dict | None = None, img_urls: dict | None = None, show_all: bool = False,
                uid: str = "fg") -> str:
    """Self-contained HTML fragment (no external scripts): feature table with value +- spread (stable features
    first, noisy ones greyed or hidden), optional comparison with the enrolled person (explanatory), the overlay and
    depth images (img_urls {"overlay": url, "depth": url} or the inline base64) and a drag-to-rotate mesh view."""
    import html as H
    if not geom or not geom.get("n_faces"):
        return '<div class="facegeom"><p>No near-frontal face was measured on this visit.</p></div>'
    small = (geom.get("size_px") or 0) < MIN_SHOW_PX
    feats = geom.get("features") or {}
    crow = (cmp or {}).get("features") or {}
    rows = []
    for k in sorted(FEATURES, key=lambda k: (k not in STABLE_FEATURES, list(FEATURES).index(k))):
        f = feats.get(k) or {}
        if f.get("value") is None or (k not in STABLE_FEATURES and not show_all):
            continue
        kind, label = FEATURES[k]
        unit = "x IPD" if kind == "prop" else "x IPD (rel. depth)" if kind == "depth" else ""
        c = crow.get(k)
        ctd = f'<td>{c["enrolled"]:.3f}</td><td>{c["z"]:+.1f} ({H.escape(c["verdict"])})</td>' if c else \
            ("<td></td><td></td>" if cmp else "")
        rows.append(f'<tr class="{"" if k in STABLE_FEATURES else "noisy"}"><td>{H.escape(label)}</td>'
                    f'<td>{f["value"]:.3f} &plusmn; {f.get("spread") or 0:.3f}</td><td>{unit}</td>'
                    f'<td>{f.get("n")}</td>{ctd}</tr>')
    urls = img_urls or {}
    ov = urls.get("overlay") or (f'data:image/jpeg;base64,{geom["overlay_jpg_b64"]}' if geom.get("overlay_jpg_b64") else None)
    dp = urls.get("depth") or (f'data:image/jpeg;base64,{geom["depth_jpg_b64"]}' if geom.get("depth_jpg_b64") else None)
    imgs = "".join(f'<figure><img src="{H.escape(u)}" alt="{a}" width="192"><figcaption>{a}</figcaption></figure>'
                   for u, a in ((ov, "landmarks + measured spans"), (dp, "relative depth (not mm)")) if u)
    mesh = json.dumps(geom.get("mesh")) if geom.get("mesh") else "null"
    head = (f'<p class="muted">{geom["n_faces"]} near-frontal faces, median {geom.get("size_px")} px, '
            f'|yaw| {geom.get("yaw_abs_med")} deg. Proportions are divided by the inter-pupil distance (IPD); depth is '
            f'the mesh model\'s relative shape estimate, not millimetres.</p>')
    if small:
        head += '<p class="warn">Faces under %d px: these numbers are too noisy to compare.</p>' % MIN_SHOW_PX
    if cmp:
        head += (f'<p class="muted">Compared with the enrolled geometry: {cmp["n_stable_consistent"]} of '
                 f'{cmp["n_stable"]} stable features consistent. {H.escape(cmp["note"])}</p>')
    th = "<th>enrolled</th><th>z</th>" if cmp else ""
    return f"""<div class="facegeom" id="{uid}">{head}
<div class="fg-media">{imgs}<canvas id="{uid}-c" width="192" height="192" title="drag to rotate"></canvas></div>
<table><tr><th>feature</th><th>value</th><th>unit</th><th>faces</th>{th}</tr>{''.join(rows)}</table>
<style>#{uid} .noisy{{opacity:.55}} #{uid} .fg-media{{display:flex;gap:8px;flex-wrap:wrap}}
#{uid} canvas{{background:#181818;cursor:grab}} #{uid} figure{{margin:0}}</style>
<script>(function(){{var g={mesh};if(!g)return;var c=document.getElementById("{uid}-c"),x=c.getContext("2d");
var b=atob(g.buffers[0].uri.split(",")[1]),u=new Uint8Array(b.length);for(var i=0;i<b.length;i++)u[i]=b.charCodeAt(i);
var pa=g.accessors[0],ia=g.accessors[1],s=g.nodes[0].scale[0];
var P=new Int16Array(u.buffer,g.bufferViews[pa.bufferView].byteOffset,pa.count*4);
var I=new Uint16Array(u.buffer,g.bufferViews[ia.bufferView].byteOffset,ia.count);
var ry=0.5,rx=0.1,d=null;function draw(){{x.fillStyle="#181818";x.fillRect(0,0,192,192);
var cy=Math.cos(ry),sy=Math.sin(ry),cx=Math.cos(rx),sx=Math.sin(rx),V=[],T=[];
for(var i=0;i<pa.count;i++){{var a=P[4*i]*s,b2=-P[4*i+1]*s-0.6,z=P[4*i+2]*s,X=a*cy+z*sy,Z=-a*sy+z*cy,Y=b2*cx-Z*sx;Z=b2*sx+Z*cx;V.push([96+X*40,96+Y*40,Z]);}}
for(var j=0;j<I.length;j+=3){{var A=V[I[j]],B=V[I[j+1]],C=V[I[j+2]];var nz=(B[0]-A[0])*(C[1]-A[1])-(B[1]-A[1])*(C[0]-A[0]);
T.push([A[2]+B[2]+C[2],A,B,C,Math.min(1,Math.abs(nz)/30)]);}}T.sort(function(p,q){{return p[0]-q[0];}});
T.forEach(function(t){{var l=Math.round(60+150*t[4]);x.fillStyle="rgb("+l+","+Math.round(l*.85)+","+Math.round(l*.7)+")";
x.beginPath();x.moveTo(t[1][0],t[1][1]);x.lineTo(t[2][0],t[2][1]);x.lineTo(t[3][0],t[3][1]);x.fill();}});}}
c.onpointerdown=function(e){{d=[e.clientX,e.clientY,ry,rx];}};window.addEventListener("pointerup",function(){{d=null;}});
c.onpointermove=function(e){{if(!d)return;ry=d[2]+(e.clientX-d[0])/60;rx=d[3]+(e.clientY-d[1])/60;draw();}};draw();}})();</script>
</div>"""


def strip_images(geom: dict | None) -> dict | None:
    """facegeom without the inline images/mesh (what goes into face_meta)."""
    if not geom:
        return geom
    return {k: v for k, v in geom.items() if k not in ("overlay_jpg_b64", "depth_jpg_b64", "mesh")}


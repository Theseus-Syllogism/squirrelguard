"""Face identification for person events: InspireFace per frame + a FASR-inspired per-visit template.

PRIVACY RULES (the user's decisions; enforced here, in the bridge and in the persons UI)
  - Faces are only ever NAMED for enrolled, consenting people (household/known identities with an enrolled face
    template). There are no outside lookups; a template is compared with the local enrolled set only.
  - No ethnicity/race output: InspireFace's attribute network also predicts race. Attributes are OFF by default
    (face_attributes = false); when switched on, the race field is discarded inside this module the moment it is
    returned and is never stored, logged or sent.
  - Apparent age/gender are estimates. Facial expression is reported only as "apparent expression", as a
    description: it is never used for alerts or identity decisions.
  - Passive (single RGB camera) liveness must pass before a face match may mark someone household/known; a photo
    held up to the camera must not suppress alerts.
  - Templates of unknown people are kept with their sighting and follow the 90-day retention. The bridge may group
    them into UNNAMED identities (FACE_AUTO_TRACK, docs/faceid-v2.md); that never names anyone or makes anyone trusted.
  - InspireFace model packs (Pikachu/Megatron) follow InsightFace's terms: ACADEMIC/NON-COMMERCIAL use only. Keep
    them out of git and out of anything published; mount them into the container (/models/inspireface/<pack>).

PIPELINE (per analysed frame, called from kinematics.track()'s on_frame hook; frames are never buffered)
  person box (+ COCO head keypoints) -> square head window -> InspireFace detect (largest face in the window)
  -> 5-point alignment + 512-d recognition embedding (L2-normalised), face quality, RGB liveness, head pose
  (yaw/pitch/roll), optional apparent expression -> FaceObs (vector + a few numbers; the pixels are dropped
  except for the current best 1-3 evidence crops, 160 px JPEG).

AGGREGATION: FASR / FASR++ idea, our own implementation (no code from the authors was used or read)
  M. dos Santos, R. Laroca, J. C. R. Neves, D. Menotti, "Robust Face Super-Resolution and Recognition Through
  Multi-Feature Aggregation in Diffusion Models", arXiv:2607.05702, JBCS 2026, DOI 10.5753/jbcs.2026.5884 (FASR++);
  and the earlier FASR, arXiv:2408.15386 (SIBGRAPI 2024).
  The papers condition a diffusion super-resolution model on identity features AGGREGATED over several low-quality
  images of the same subject (e.g. consecutive video frames): each image is encoded separately to a 512-d
  recognition vector, the vectors are L2-normalised and averaged (FASR: plain mean, which the authors found
  better than max/sum on noisy low-resolution inputs); FASR++ merges them pairwise, each pair giving
  mean(a, b) + a small learned correction delta([a; b]), L2-normalised, and averages the pair outputs.
  We do NOT super-resolve or generate any pixels. We only use the aggregation idea to build one identity
  template per visit from many small, noisy surveillance faces:
    1. per-face weight w = quality x size x pose x sharpness x detection confidence
       (quality: InspireFace face-quality score; size: smoothstep of the face box side between min_px and
       good_px; pose: cos(yaw)^2 cos(pitch) from InspireFace's head pose, landmark yaw as a fallback; sharpness:
       variance of the Laplacian of the face crop at 112 px);
    2. keep the max_faces highest-weight faces (bounded memory);
    3. outlier rejection against a running consensus: start from the weighted mean of the best half, drop faces
       whose cosine to it is below max(floor, median - max(k * MAD, min_gap)) (a second person walking past, a
       mis-detection, a poster), re-estimate, twice;
    4. template = L2-normalise(sum_i w_i v_i) over the inliers  (FASR's mean, quality-weighted);
    5. optional learned refinement in the FASR++ form (PairwiseCombiner with a delta network, e.g. an ONNX MLP
       trained with a triplet loss on our own enrolled visits): see `PairwiseCombiner`. Not trained yet; the
       default is delta = 0, i.e. the weighted mean.
  Reported with the template: inlier count, rejected count, weighted quality, liveness summary, consistency (mean
  inlier cosine to the template: low means mixed/unclear faces), face size, |yaw|, evidence crops.

CPU BUDGET: at most `max_attempts` detector calls and `max_cpu_s` CPU seconds per event, only on frames where the
head is big enough (estimated from keypoints) and every `every`-th frame; after misses the sampler backs off
(every 2nd, 4th, 8th eligible frame) but always tries a head that has grown by 20 %, so the late, close-up face of
someone walking toward the camera is not lost to a budget spent on their back. Measured on the laptop (i7-14700HX,
Megatron pack, 320 px detector level): ~11 ms per head window without a face, ~45-60 ms with one (detection, quality,
liveness, expression, 512-d embedding); the engine needs ~1 GB RSS (Pikachu: ~0.7 GB, 3-4x faster, weaker).
"""
from __future__ import annotations

import base64
import dataclasses
import logging
import math
import time
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

log = logging.getLogger("squirrel_guard.faceid")

EXPRESSIONS = ("neutral", "happy", "sad", "surprise", "fear", "disgust", "anger")   # InspireFace emotion index order
AGE_BRACKETS = ((0, 2), (3, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 99))
AGGREGATOR_VERSION = "fasr-qwmean-v1"
EMB_DIM = 512


def l2n(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


@dataclasses.dataclass(eq=False)
class FaceObs:
    """One face on one frame. Only `crop` holds pixels, and only while the face is among the best few."""
    vec: np.ndarray                 # (512,) L2-normalised recognition embedding
    quality: float                  # InspireFace face quality 0..1 (low on small/soft surveillance faces)
    size: float                     # face box side in ORIGINAL frame pixels (min of w, h)
    yaw: float = 0.0                # degrees, 0 = frontal
    pitch: float = 0.0
    roll: float = 0.0
    det: float = 1.0                # detection confidence
    sharp: float = 100.0            # variance of Laplacian at 112 px
    liveness: float | None = None   # RGB passive liveness 0..1 (None = not computed)
    expression: int | None = None   # InspireFace emotion index (description only)
    age_bracket: int | None = None  # only with face_attributes = true
    gender: int | None = None       # 0 female, 1 male (apparent), only with face_attributes = true
    t: float = 0.0                  # frame index / time within the event
    box: tuple | None = None        # face box in frame pixels
    crop: bytes | None = None       # JPEG of the face (evidence), dropped for faces outside the best few
    lmk106: np.ndarray | None = None  # InspireFace dense landmarks (frame px), only with dense_landmarks (facegeom)


@dataclasses.dataclass
class AggParams:
    min_px: float = 24.0            # faces smaller than this are not used at all
    good_px: float = 80.0           # size weight saturates here
    min_quality: float = 0.0        # per-face floor on InspireFace quality (0 = keep all, weight handles it)
    max_faces: int = 96             # pool size (best by weight); 24 lost pose variety: 0.73 vs 0.89 genuine on a clip
    outlier_k: float = 3.0          # MAD multiplier
    outlier_floor: float = 0.20     # always drop faces with cosine < this to the consensus
    outlier_min_gap: float = 0.25   # never drop a face within this of the median cosine (same person, other pose:
                                    # a tight MAD alone rejected 5 of 24 genuine faces of one clip)
    live_thr: float = 0.5           # per-face liveness threshold
    live_frac: float = 0.6          # fraction of inlier faces (by weight) that must pass for liveness ok
    keep_crops: int = 3


def smoothstep(x: float, a: float, b: float) -> float:
    t = min(1.0, max(0.0, (x - a) / max(1e-6, b - a)))
    return t * t * (3 - 2 * t)


def face_weight(o: FaceObs, p: AggParams) -> float:
    """Quality x size x pose x sharpness x detection: how much this face should count in the template."""
    q = max(0.02, min(1.0, o.quality))
    s = max(0.05, smoothstep(o.size, p.min_px, p.good_px))
    yaw, pitch = min(abs(o.yaw), 89.0), min(abs(o.pitch), 89.0)
    pose = max(0.05, math.cos(math.radians(yaw)) ** 2 * math.cos(math.radians(pitch)))
    sharp = max(0.1, min(1.0, o.sharp / 100.0))
    return float(q * s * pose * sharp * max(0.1, min(1.0, o.det)))


class DeltaNet(Protocol):
    def __call__(self, pair: np.ndarray) -> np.ndarray: ...       # (2*512,) -> (512,)


class PairwiseCombiner:
    """FASR++-style learned aggregation interface (our implementation from the paper's description):
    combine(a, b) = L2N( (a + b) / 2 + eta * delta([a; b]) ); N vectors are taken in consecutive pairs (best first),
    an odd last vector is averaged in, and the pair outputs are averaged and normalised.
    With delta=None (the default; nothing is trained yet) this is the plain mean of pair means.
    A trained delta (e.g. a small ONNX MLP, triplet loss on our own enrolled visits) can be plugged in later; it
    must be validated with scripts/face_eval.py against the weighted mean before use."""

    def __init__(self, delta: DeltaNet | None = None, eta: float = 1.0):
        self.delta, self.eta = delta, eta

    def combine(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        m = (a + b) / 2
        if self.delta is not None:
            m = m + self.eta * np.asarray(self.delta(np.concatenate([a, b]).astype(np.float32)), np.float32)
        return l2n(m)

    def __call__(self, vecs: np.ndarray) -> np.ndarray:
        vecs = l2n(vecs)
        if len(vecs) == 1:
            return vecs[0]
        outs = [self.combine(vecs[i], vecs[i + 1]) for i in range(0, len(vecs) - 1, 2)]
        if len(vecs) % 2:
            outs.append(vecs[-1])
        return l2n(np.mean(outs, 0))


def consensus(V: np.ndarray, w: np.ndarray, p: AggParams) -> tuple[np.ndarray, np.ndarray]:
    """Inlier mask + weighted-mean centre after robust outlier rejection (two passes)."""
    n = len(V)
    order = np.argsort(-w)
    seed = order[:max(2, (n + 1) // 2)] if n > 2 else order
    c = l2n((V[seed] * w[seed, None]).sum(0))
    keep = np.ones(n, bool)
    for _ in range(2):
        sims = V @ c
        med = float(np.median(sims[keep]))
        mad = float(np.median(np.abs(sims[keep] - med))) + 1e-3
        keep = sims >= max(p.outlier_floor, med - max(p.outlier_k * 1.4826 * mad, p.outlier_min_gap))
        if not keep.any():                    # nothing agrees: keep the single best face
            keep = np.zeros(n, bool)
            keep[order[0]] = True
        c = l2n((V[keep] * w[keep, None]).sum(0))
    return keep, c


class FaceAggregator:
    """Bounded pool of the best faces of one visit -> one identity template (see the module docstring)."""

    def __init__(self, params: AggParams | None = None, combiner: Callable[[np.ndarray], np.ndarray] | None = None):
        self.p = params or AggParams()
        self.combiner = combiner
        self.pool: list[tuple[float, FaceObs]] = []
        self.seen = 0                 # faces offered
        self.too_small = 0
        self.low_quality = 0
        self.max_size = 0.0

    def add(self, o: FaceObs) -> bool:
        self.seen += 1
        self.max_size = max(self.max_size, o.size)
        if o.size < self.p.min_px:
            self.too_small += 1
            return False
        if o.quality < self.p.min_quality:
            self.low_quality += 1
            return False
        o.vec = l2n(o.vec)
        w = face_weight(o, self.p)
        self.pool.append((w, o))
        self.pool.sort(key=lambda t: -t[0])
        del self.pool[self.p.max_faces:]
        for i, (_, f) in enumerate(self.pool):   # pixels only for the best few
            if i >= self.p.keep_crops:
                f.crop = None
        return any(f is o for _, f in self.pool)

    def wants_crop(self, w: float) -> bool:
        """Would a face of weight w be among the best few (so its JPEG is worth encoding)?"""
        return len(self.pool) < self.p.keep_crops or w > self.pool[self.p.keep_crops - 1][0]

    def result(self) -> dict:
        """JSON-ready summary. vec is None when no usable face was seen (reason says why)."""
        base = {"aggregator": AGGREGATOR_VERSION, "faces_seen": self.seen, "max_size_px": round(self.max_size, 1)}
        if not self.pool:
            reason = "no_face" if self.seen == 0 else "too_small" if self.too_small >= self.low_quality else "low_quality"
            return {**base, "vec": None, "n_faces": 0, "reason": reason}
        w = np.array([t[0] for t in self.pool], np.float32)
        faces = [t[1] for t in self.pool]
        V = np.stack([f.vec for f in faces])
        keep, c = consensus(V, w, self.p)
        if self.combiner is not None:
            idx = np.where(keep)[0]
            c = l2n(self.combiner(V[idx[np.argsort(-w[idx])]]))
        sims = V @ c
        wi = w[keep]
        fi = [f for f, k in zip(faces, keep) if k]
        lv = [(f.liveness, ww) for f, ww in zip(fi, wi) if f.liveness is not None and f.liveness >= 0]
        if lv:
            lw = sum(ww for _, ww in lv) + 1e-12
            frac = sum(ww for l, ww in lv if l >= self.p.live_thr) / lw
            live = {"score": round(float(sum(l * ww for l, ww in lv) / lw), 3), "pass_frac": round(float(frac), 3),
                    "frames": len(lv), "ok": bool(frac >= self.p.live_frac)}
        else:
            live = {"score": None, "pass_frac": None, "frames": 0, "ok": False}
        crops = [f.crop for j, f in enumerate(faces[:self.p.keep_crops]) if f.crop and keep[j]]
        out = {**base, "vec": [round(float(x), 5) for x in c], "n_faces": int(keep.sum()),
               "n_rejected": int((~keep).sum()),
               "quality": round(float(np.average([f.quality for f in fi], weights=wi)), 3),
               "quality_max": round(float(max(f.quality for f in fi)), 3),
               "weight_sum": round(float(wi.sum()), 3),
               "consistency": round(float(sims[keep].mean()), 3),
               "size_px": round(float(np.median([f.size for f in fi])), 1),
               "yaw_abs_med": round(float(np.median([abs(f.yaw) for f in fi])), 1),
               "liveness": live,
               "best_crop_b64": base64.b64encode(crops[0]).decode() if crops else None,
               "crops_b64": [base64.b64encode(x).decode() for x in crops],
               "faces": [{"t": round(float(f.t), 2), "size": round(f.size, 1), "q": round(f.quality, 3),
                          "yaw": round(f.yaw, 1), "pitch": round(f.pitch, 1), "live": None if f.liveness is None
                          else round(f.liveness, 3), "w": round(float(ww), 4), "sim": round(float(s), 3),
                          "inlier": bool(k)} for f, ww, s, k in zip(faces, w, sims, keep)],
               "apparent": apparent_of(fi, wi)}
        return out


def apparent_of(faces: list[FaceObs], w: np.ndarray) -> dict:
    """Weighted votes -> apparent expression (description only) and, when computed, apparent age/gender.
    Never contains anything about race/ethnicity."""
    out: dict = {}
    votes: dict = {}
    for f, ww in zip(faces, w):
        if f.expression is not None and 0 <= f.expression < len(EXPRESSIONS):
            votes[EXPRESSIONS[f.expression]] = votes.get(EXPRESSIONS[f.expression], 0.0) + float(ww)
    if votes:
        best = max(votes, key=votes.get)
        out["expression"] = best
        out["expression_share"] = round(votes[best] / (sum(votes.values()) + 1e-12), 2)
        out["expression_note"] = "apparent expression (description only; never used for alerts or identity)"
    ages = [(AGE_BRACKETS[f.age_bracket], ww) for f, ww in zip(faces, w)
            if f.age_bracket is not None and 0 <= f.age_bracket < len(AGE_BRACKETS)]
    if ages:
        av: dict = {}
        for b, ww in ages:
            av[b] = av.get(b, 0.0) + float(ww)
        b = max(av, key=av.get)
        out["age"] = {"low": b[0], "high": b[1], "note": "apparent age (estimate)"}
    gs = [(f.gender, ww) for f, ww in zip(faces, w) if f.gender in (0, 1)]
    if gs:
        m = sum(ww for g, ww in gs if g == 1) / (sum(ww for _, ww in gs) + 1e-12)
        out["gender"] = {"value": "male" if m >= 0.5 else "female", "conf": round(float(max(m, 1 - m)), 2),
                         "note": "apparent gender (estimate)"}
    return out


# ---------------------------------------------------------------------------------------------------------------
# geometry helpers (no InspireFace needed)

def head_window(box, kp=None, frame_shape=None, min_side: float = 48.0) -> tuple[int, int, int, int] | None:
    """Square window around the head of a person box. With COCO keypoints (nose, eyes, ears >= 0.3 confidence) it
    is centred on them and sized from their spread; otherwise the top of the box."""
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    if kp is not None:
        kp = np.asarray(kp)
        hk = kp[:5][kp[:5, 2] >= 0.3]
        if len(hk) >= 2:
            cx, cy = hk[:, 0].mean(), hk[:, 1].mean()
            span = float(max(np.ptp(hk[:, 0]), np.ptp(hk[:, 1])))
            side = max(3.2 * span, 0.18 * bh, min_side)
            return _clip_square(cx, cy, side, frame_shape)
    side = max(0.75 * bw, min(0.35 * bh, 1.2 * bw), min_side)
    return _clip_square((x1 + x2) / 2, y1 + side * 0.45, side, frame_shape)


def _clip_square(cx, cy, side, frame_shape):
    X1, Y1, X2, Y2 = cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2
    if frame_shape is not None:
        H, W = frame_shape[:2]
        X1, Y1, X2, Y2 = max(0.0, X1), max(0.0, Y1), min(float(W), X2), min(float(H), Y2)
    if X2 - X1 < 16 or Y2 - Y1 < 16:
        return None
    return int(X1), int(Y1), int(math.ceil(X2)), int(math.ceil(Y2))


def head_size_px(box, kp=None) -> float:
    """Rough expected face size (px) from keypoints (eye/ear spread) or the box height; used to skip frames."""
    if kp is not None:
        kp = np.asarray(kp)
        hk = kp[:5][kp[:5, 2] >= 0.3]
        if len(hk) >= 2:
            return float(max(np.ptp(hk[:, 0]), np.ptp(hk[:, 1]))) * 1.6
    return 0.12 * float(box[3] - box[1])


def landmark_yaw(pts5: np.ndarray) -> float:
    """Yaw (deg) from 5 landmarks (left eye, right eye, nose, mouth corners): nose offset from the eye midpoint
    relative to the eye distance. Rough, only a fallback when the head pose is missing."""
    le, re, nose = pts5[0], pts5[1], pts5[2]
    d = float(np.linalg.norm(re - le)) + 1e-6
    off = float(nose[0] - (le[0] + re[0]) / 2) / d
    return float(np.degrees(np.arcsin(np.clip(off * 1.6, -1, 1))))


def sharpness(gray_face: np.ndarray) -> float:
    if gray_face.size == 0:
        return 0.0
    g = cv2.resize(gray_face, (112, 112), interpolation=cv2.INTER_AREA if gray_face.shape[0] > 112 else cv2.INTER_LINEAR)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def face_jpeg(frame: np.ndarray, box, side: int = 160) -> bytes | None:
    x1, y1, x2, y2 = box
    m = 0.25 * max(x2 - x1, y2 - y1)
    H, W = frame.shape[:2]
    X1, Y1, X2, Y2 = int(max(0, x1 - m)), int(max(0, y1 - m)), int(min(W, x2 + m)), int(min(H, y2 + m))
    if X2 - X1 < 8 or Y2 - Y1 < 8:
        return None
    c = frame[Y1:Y2, X1:X2]
    s = side / max(c.shape[:2])
    c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    ok, jpg = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpg.tobytes() if ok else None


# ---------------------------------------------------------------------------------------------------------------
# InspireFace engine

_LAUNCHED: str | None = None


def _launch(pack: str) -> str:
    """Load an InspireFace model pack once per process. `pack` is a file path (preferred in containers) or a pack
    name (Pikachu / Megatron), which InspireFace downloads to ~/.inspireface on first use."""
    global _LAUNCHED
    import inspireface as isf
    if _LAUNCHED is not None:
        if _LAUNCHED != pack:
            raise RuntimeError(f"InspireFace already launched with {_LAUNCHED}; one pack per process")
        return Path(pack).name
    try:
        isf.set_logging_level(isf.HF_LOG_ERROR)
    except Exception:
        pass
    p = Path(pack)
    if p.is_file():
        isf.launch(p.name, resource_path=str(p))
    else:
        isf.launch(pack)
    _LAUNCHED = pack
    return p.name


class FaceEngine:
    """InspireFace session: detect the face inside a head window, then quality, liveness, pose, expression and the
    512-d embedding. Not thread-safe: one engine per worker thread."""

    def __init__(self, model_pack: str = "Megatron", detect_level: int = 320, liveness: bool = True,
                 expression: bool = True, attributes: bool = False, det_conf: float = 0.5):
        import inspireface as isf
        self.isf = isf
        self.pack = _launch(model_pack)
        opt = isf.HF_ENABLE_FACE_RECOGNITION | isf.HF_ENABLE_QUALITY | isf.HF_ENABLE_FACE_POSE
        if liveness:
            opt |= isf.HF_ENABLE_LIVENESS
        if expression:
            opt |= isf.HF_ENABLE_FACE_EMOTION
        if attributes:     # apparent age/gender; the same network's race head is discarded below
            opt |= isf.HF_ENABLE_FACE_ATTRIBUTE
        self.opt = opt
        self.liveness, self.expression, self.attributes = liveness, expression, attributes
        self.sess = isf.InspireFaceSession(opt, isf.HF_DETECT_MODE_ALWAYS_DETECT, max_detect_num=4,
                                           detect_pixel_level=detect_level)
        self.sess.set_detection_confidence_threshold(det_conf)
        self.sess.set_filter_minimum_face_pixel_size(0)

    def faces(self, img: np.ndarray, offset=(0, 0), scale: float = 1.0, t: float = 0.0, want_crop=None,
              frame: np.ndarray | None = None, largest_only: bool = True) -> list[FaceObs]:
        """Faces in img (a head window cut from `frame` at `offset`, upscaled by `scale`). Sizes and boxes are
        reported in frame pixels."""
        fs = self.sess.face_detection(img)
        if not fs:
            return []
        if largest_only:
            fs = [max(fs, key=lambda f: (f.location[2] - f.location[0]) * (f.location[3] - f.location[1])
                      * f.detection_confidence)]
        ex = self.sess.face_pipeline(img, fs, self.opt)
        out = []
        for f, e in zip(fs, ex):
            vec = self.sess.face_feature_extract(img, f)
            x1, y1, x2, y2 = f.location
            g = cv2.cvtColor(img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)], cv2.COLOR_BGR2GRAY) \
                if y2 > y1 and x2 > x1 else np.zeros((0, 0), np.uint8)
            fb = (offset[0] + x1 / scale, offset[1] + y1 / scale, offset[0] + x2 / scale, offset[1] + y2 / scale)
            yaw = float(f.yaw)
            if not math.isfinite(yaw):
                try:
                    yaw = landmark_yaw(np.asarray(self.sess.get_face_five_key_points(f)))
                except Exception:
                    yaw = 0.0
            o = FaceObs(vec=np.asarray(vec, np.float32), quality=float(e.quality_confidence),
                        size=float(min(fb[2] - fb[0], fb[3] - fb[1])), yaw=yaw, pitch=float(f.pitch),
                        roll=float(f.roll), det=float(f.detection_confidence), sharp=sharpness(g),
                        liveness=float(e.rgb_liveness_confidence) if self.liveness else None,
                        expression=int(e.emotion) if self.expression and e.emotion >= 0 else None,
                        age_bracket=int(e.age_bracket) if self.attributes and e.age_bracket >= 0 else None,
                        gender=int(e.gender) if self.attributes and e.gender in (0, 1) else None,
                        t=t, box=fb)
            e.race = -1          # policy: never keep race/ethnicity
            if frame is not None and (want_crop is None or want_crop(o)):
                o.crop = face_jpeg(frame, [int(v) for v in fb])
            if getattr(self, "dense_landmarks", False):     # facegeom overlay (set by profiler.facegeom_setup)
                o.lmk106 = self._dense(f, offset, scale)
            out.append(o)
        return out


    def _dense(self, f, offset, scale) -> np.ndarray | None:
        try:
            lm = np.asarray(self.sess.get_face_dense_landmark(f), np.float32).reshape(-1, 2)
        except Exception:
            return None
        return lm / scale + np.asarray(offset, np.float32)


class FaceSampler:
    """Per-event face collector for the on_frame hook: bounded CPU, never buffers frames.
    sampler.on_frame(frame, pose_or_None); sampler.on_box(frame, box) for detector boxes; sampler.result()."""

    def __init__(self, engine, params: AggParams | None = None, every: int = 1, max_attempts: int = 150,
                 max_cpu_s: float = 8.0, upscale_to: int = 0, combiner=None, backoff: bool = True):
        self.engine = engine
        self.agg = FaceAggregator(params, combiner)
        self.every, self.max_attempts, self.max_cpu_s, self.upscale_to = max(1, every), max_attempts, max_cpu_s, upscale_to
        self.i = 0
        self.attempts = 0
        self.cpu_s = 0.0
        self.skipped_small = 0
        self.skipped_backoff = 0
        self.backoff = backoff
        self.miss = 0               # consecutive detector calls without a usable face
        self.k = 0                  # eligible frames seen
        self.big = 0.0              # largest head estimate tried so far
        self.geom = None            # facegeom.GeomCollector (profiler.facegeom_attach), sees every accepted face
        self.fit = None             # facefit.FitCollector (profiler.facefit_attach): one fitted face per visit

    def done(self) -> bool:
        return self.attempts >= self.max_attempts or self.cpu_s >= self.max_cpu_s

    def on_frame(self, fr: np.ndarray, p) -> None:
        i = self.i
        self.i += 1
        if p is None or i % self.every or self.done():
            return
        self.on_box(fr, p.box, getattr(p, "kp", None), t=i)

    def on_box(self, fr: np.ndarray, box, kp=None, t: float = 0.0) -> None:
        if self.done():
            return
        if head_size_px(box, kp) < self.agg.p.min_px * 0.6:
            self.skipped_small += 1
            self.agg.max_size = max(self.agg.max_size, head_size_px(box, kp))
            return
        # back off while faces are not found (back turned, far away), but always try a head that has grown: a
        # visitor walking toward the camera shows the best face at the END, which a fixed budget would miss
        hs = head_size_px(box, kp)
        self.k += 1
        step = min(8, 2 ** (self.miss // 3))
        if self.backoff and hs <= self.big * 1.2 and self.k % step:
            self.skipped_backoff += 1
            return
        self.big = max(self.big, hs)
        win = head_window(box, kp, fr.shape)
        if win is None:
            return
        X1, Y1, X2, Y2 = win
        img = fr[Y1:Y2, X1:X2]
        s = 1.0
        if self.upscale_to and max(img.shape[:2]) < self.upscale_to:   # detector minimum size; no new detail
            s = self.upscale_to / max(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        img = np.ascontiguousarray(img)
        t0 = time.process_time()
        self.attempts += 1
        try:
            obs = self.engine.faces(img, offset=(X1, Y1), scale=s, t=t, frame=fr,
                                    want_crop=lambda o: self.agg.wants_crop(face_weight(o, self.agg.p)))
        except Exception:
            log.exception("face analysis failed")
            obs = []
        self.cpu_s += time.process_time() - t0
        used = False
        for o in obs:
            used |= self.agg.add(o)
            if self.geom is not None and o.size >= self.agg.p.min_px:
                self.geom.offer(fr, o, face_weight(o, self.agg.p))
            if self.fit is not None and o.size >= self.agg.p.min_px:
                self.fit.offer(fr, o, face_weight(o, self.agg.p))
        self.miss = 0 if used else self.miss + 1

    def result(self) -> dict:
        r = self.agg.result()
        r.update({"attempts": self.attempts, "cpu_s": round(self.cpu_s, 2), "skipped_small": self.skipped_small,
                  "skipped_backoff": self.skipped_backoff,
                  "model": getattr(self.engine, "pack", None)})
        if r.get("vec") is None and self.attempts == 0 and self.skipped_small:
            r["reason"] = "too_small"
        if self.geom is not None:
            try:
                r["facegeom"] = self.geom.result()
            except Exception:
                log.exception("face geometry failed")
        if self.fit is not None:
            try:
                from . import facefit
                ff = self.fit.result()
                g = ff.pop("mesh", None)
                if g:                               # all LODs as one binary glTF for the evidence folder
                    ff["glb_b64"] = base64.b64encode(facefit.gltf_to_glb(g)).decode()
                r["facefit"] = ff
            except Exception:
                log.exception("face fit failed")
        return r

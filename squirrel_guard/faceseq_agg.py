"""Combine a face sequence into one identity fingerprint, and decide conservatively (docs/faceseq.md).

Baseline: quality-weighted mean of AdaFace embeddings. Weight per frame = AdaFace feature norm (its built-in quality
proxy, low on blurry/tiny/off-angle faces) x a pose factor x a size factor; optional external quality (OFIQ unified
score, eDifFIQA) multiplies in when available. Learned alternative: CAFace (caface.py) when its backbone is set up.

Rules (privacy + error control):
  - faces below MIN_FACE_PX camera pixels are never used for identity: logged as "too far";
  - a name is only proposed when the combined fingerprint clears the threshold AND at least MIN_AGREE individual
    usable frames clear `single_thr` on their own (one lucky frame or the average alone is not enough);
  - the decision record keeps the frames used, their sizes and weights (evidence).
"""
from __future__ import annotations

import numpy as np

MIN_FACE_PX = 12.0
MIN_AGREE = 2


def frame_weights(norms: np.ndarray, face_px: np.ndarray, yaw: np.ndarray, extra: np.ndarray | None = None,
                  norm_floor: float = 15.0, norm_scale: float = 4.0) -> np.ndarray:
    q = np.clip((norms - norm_floor) / norm_scale, 0.0, None) ** 2          # AdaFace norm: steep, as in its margin
    pose = np.clip(1.0 - np.abs(yaw) / 0.6, 0.05, 1.0)                       # profile faces count much less
    size = np.clip((face_px - MIN_FACE_PX) / 24.0, 0.0, 1.0)                # 12 px -> 0, 36+ px -> 1
    w = q * pose * size
    if extra is not None:
        w = w * np.clip(extra, 0.0, None)
    return w


def combine(E: np.ndarray, w: np.ndarray) -> np.ndarray | None:
    if w.sum() <= 0:
        return None
    v = (E * w[:, None]).sum(0)
    return v / (np.linalg.norm(v) + 1e-9)


def decide(E: np.ndarray, norms: np.ndarray, face_px: np.ndarray, yaw: np.ndarray, gallery: dict[str, np.ndarray],
           thr: float = 0.30, single_thr: float | None = None, extra: np.ndarray | None = None) -> dict:
    """gallery: {name: L2-normalised template}. Returns the decision record (name or None, with evidence)."""
    single_thr = thr if single_thr is None else single_thr
    usable = face_px >= MIN_FACE_PX
    rec = {"frames": int(len(E)), "usable": int(usable.sum()), "too_far": int((~usable).sum()),
           "face_px_max": round(float(face_px.max()), 1) if len(face_px) else 0.0}
    if not usable.any():
        rec.update(name=None, reason="too far: no face >= %d px" % MIN_FACE_PX)
        return rec
    w = frame_weights(norms, face_px, yaw, extra)
    w[~usable] = 0
    t = combine(E, w)
    if t is None:
        rec.update(name=None, reason="no usable-quality face")
        return rec
    scores = {n: float(t @ g) for n, g in gallery.items()}
    best = max(scores, key=scores.get) if scores else None
    singles = (E[usable] @ gallery[best]) if best else np.array([])
    agree = int((singles >= single_thr).sum())
    rec.update(scores={k: round(v, 3) for k, v in scores.items()}, best=best,
               combined=round(scores.get(best, 0.0), 3) if best else None, agree_frames=agree,
               weights_top=sorted([round(float(x), 3) for x in w if x > 0], reverse=True)[:8])
    if best and scores[best] >= thr and agree >= MIN_AGREE:
        rec.update(name=best, reason="combined %.2f >= %.2f and %d frames agree" % (scores[best], thr, agree))
    else:
        rec.update(name=None, reason="below threshold or too few agreeing frames")
    return rec

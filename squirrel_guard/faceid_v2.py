"""Face identification v2 (OPT-IN; faceid.py is unchanged and stays the default).

Adds, on top of squirrel_guard.faceid (InspireFace detection, quality, liveness, pose and the visit aggregator):
  - AdaFace (quality-adaptive margin, Kim et al., CVPR 2022) IR-50 / IR-101 ONNX embeddings computed on the SAME
    detection, from the standard ArcFace 5-point aligned 112x112 crop (InspireFace's five key points);
  - score fusion of the InspireFace and AdaFace cosines after per-model impostor normalisation;
  - a multi-template gallery (one template per pose bucket x colour/IR modality instead of one mean);
  - an open-set threshold calibrated from impostor scores (target FAR) instead of a fixed cosine;
  - best-K face selection for the visit template;
  - body re-ID / stature side evidence, reported separately and only used when the face is unusable.
What actually measured better is in docs/faceid-v2.md (scripts/faceid_eval.py); everything here is off unless the
profiler config asks for it (`face_v2 = true`, see the doc).

PRIVACY RULES are those of faceid.py: names only for enrolled consenting people, no race/ethnicity output, no
outside lookups, face geometry (facefit/facegeom) is never an identity input, liveness still gates household.
No super-resolution / face restoration is used anywhere (it invents identity detail).

LICENCES: AdaFace / CVLFace code MIT; the weights follow their training data (WebFace4M/12M: non-commercial
research only) -> local use only, never committed, published or baked into an image, like the InspireFace packs.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import cv2
import numpy as np

from .faceid import AggParams, FaceAggregator, FaceEngine, FaceObs, consensus, face_weight, l2n

V2_VERSION = "faceid-v2-1"

# ArcFace / InsightFace 5-point reference for a 112x112 crop (left eye, right eye, nose, mouth left, mouth right;
# "left" = image left). InspireFace's get_face_five_key_points uses the same order (checked on stock frames).
ARCFACE_REF = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def align112(img: np.ndarray, pts5) -> np.ndarray | None:
    """Similarity transform of the 5 points onto ARCFACE_REF; 112x112 BGR crop (bilinear: small faces are
    interpolated, never restored)."""
    pts = np.asarray(pts5, np.float32).reshape(5, 2)
    if not np.isfinite(pts).all():
        return None
    M, _ = cv2.estimateAffinePartial2D(pts, ARCFACE_REF, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def ir_score(bgr: np.ndarray) -> float:
    """0 for a greyscale (night IR) image, larger for colour: mean |channel difference| on the face."""
    if bgr is None or bgr.size == 0 or bgr.ndim < 3:
        return 0.0
    f = bgr.astype(np.int16)
    return float((np.abs(f[..., 0] - f[..., 1]).mean() + np.abs(f[..., 1] - f[..., 2]).mean()) / 2)


def is_ir(bgr: np.ndarray, thr: float = 2.0) -> bool:
    return ir_score(bgr) < thr


class AdaFace:
    """AdaFace ONNX (scripts/export_adaface.py). embed(list of 112x112 BGR crops) -> (L2-normalised (N,512),
    feature norms (N,)). The norm is AdaFace's built-in quality proxy (low on blurry/tiny faces)."""

    def __init__(self, path: str | Path, threads: int = 2, flip_tta: bool = False, backend: str = "ort"):
        self.name = Path(path).stem
        self.flip_tta = flip_tta
        if backend == "trt":                 # RTX GPU: TensorRT fp16 engine (squirrel_guard/trt.py, docs/trt.md)
            from .trt import TRTModel
            eng = TRTModel(path, fp16=True, max_batch=64)

            class _S:
                def run(self, _o, feed):
                    return eng(feed["images"])
            self.sess = _S()
            return
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.set_denormal_as_zero", "1")   # denormal activations: 2-3x slower
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def embed(self, crops: list[np.ndarray], batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
        if not crops:
            return np.zeros((0, 512), np.float32), np.zeros(0, np.float32)
        out = []
        for i in range(0, len(crops), batch):
            x = np.stack([c[..., ::-1] for c in crops[i:i + batch]]).astype(np.float32)   # BGR -> RGB
            x = (x / 255.0 - 0.5) / 0.5
            x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
            f = self.sess.run(None, {"images": x})[0]
            if self.flip_tta:
                f = f + self.sess.run(None, {"images": np.ascontiguousarray(x[..., ::-1])})[0]
            out.append(f)
        F = np.concatenate(out).astype(np.float32)
        n = np.linalg.norm(F, axis=1)
        return F / (n[:, None] + 1e-12), n


class FaceEngineV2(FaceEngine):
    """FaceEngine plus, per face: `pts5` (frame px), `aligned` (112x112 BGR, ArcFace alignment from the head
    window) and `ir` (colour score of the crop, ~0 = night IR). InspireFace work is unchanged; the extra cost is one
    warpAffine per face. AdaFace runs later, in batch, on `aligned` (see AdaFaceSampler)."""

    def faces(self, img, offset=(0, 0), scale: float = 1.0, *a, **k):
        self._img, self._extra = img, []
        try:
            out = super().faces(img, offset, scale, *a, **k)
        finally:
            self._img = None
        for o, (pts, al, irs, lmk) in zip(out, self._extra):
            o.pts5, o.aligned, o.ir = pts, al, irs
            if lmk is not None:
                o.lmk106 = lmk
        return out

    @property
    def dense_landmarks(self):          # faceid.FaceEngine.faces calls _dense() per face when this is true
        return True

    @dense_landmarks.setter
    def dense_landmarks(self, v):
        self._want_dense = bool(v)

    def _dense(self, f, offset, scale):
        img = self._img
        pts = al = None
        irs = 0.0
        try:
            p = np.asarray(self.sess.get_face_five_key_points(f), np.float32).reshape(5, 2)
            al = align112(img, p)
            pts = p / scale + np.asarray(offset, np.float32)
            irs = ir_score(al) if al is not None else 0.0
        except Exception:
            pass
        lmk = super()._dense(f, offset, scale) if getattr(self, "_want_dense", False) else None
        self._extra.append((pts, al, irs, lmk))
        return lmk


# ---------------------------------------------------------------------------------------------------------------
# templates

def pose_bucket(yaw: float) -> str:
    return "frontal" if abs(yaw) < 30 else "side"


def modality(ir: float | None, thr: float = 2.0) -> str:
    return "ir" if ir is not None and ir < thr else "colour"


def best_k(faces: list[FaceObs], vecs: np.ndarray, k: int, p: AggParams | None = None) -> np.ndarray | None:
    """Template from the k highest-weight faces only (faceid.face_weight), with the same outlier rejection."""
    p = p or AggParams()
    if not len(faces):
        return None
    w = np.array([face_weight(f, p) for f in faces], np.float32)
    idx = np.argsort(-w)[:k]
    V = l2n(vecs[idx])
    _, c = consensus(V, w[idx], p)
    return c


def aggregate(faces: list[FaceObs], vecs: np.ndarray, p: AggParams | None = None, norm_weight: np.ndarray | None = None,
              max_faces: int | None = None) -> np.ndarray | None:
    """The production aggregator (FaceAggregator) on another model's vectors: same per-face weights, pool and
    outlier rejection; optionally multiplied by an AdaFace feature-norm weight."""
    p = dataclasses.replace(p or AggParams(), keep_crops=0, **({"max_faces": max_faces} if max_faces else {}))
    a = FaceAggregator(p)
    for i, f in enumerate(faces):
        g = dataclasses.replace(f, vec=vecs[i], crop=None)
        if norm_weight is not None:
            g.quality = max(0.02, min(1.0, f.quality)) * float(norm_weight[i])
        a.add(g)
    r = a.result()
    return None if r["vec"] is None else np.asarray(r["vec"], np.float32)


def multi_templates(faces: list[FaceObs], vecs: np.ndarray, min_faces: int = 3, p: AggParams | None = None) -> dict:
    """{(pose bucket, modality): template} for every bucket with >= min_faces faces, plus ('all', 'all')."""
    out = {}
    t = aggregate(faces, vecs, p)
    if t is not None:
        out[("all", "all")] = t
    keys = [(pose_bucket(f.yaw), modality(getattr(f, "ir", None))) for f in faces]
    for key in set(keys):
        idx = [i for i, k in enumerate(keys) if k == key]
        if len(idx) >= min_faces:
            t = aggregate([faces[i] for i in idx], vecs[idx], p)
            if t is not None:
                out[key] = t
    return out


@dataclasses.dataclass
class Gallery:
    """Enrolled templates per identity (several per identity: per visit, per pose bucket and modality).
    score(probe) = best cosine over that identity's templates (max fusion)."""
    templates: dict = dataclasses.field(default_factory=dict)      # identity -> list[(tag, vec)]

    def add(self, ident, vec, tag=("all", "all")):
        self.templates.setdefault(ident, []).append((tag, l2n(np.asarray(vec, np.float32))))

    def scores(self, probe) -> dict:
        q = l2n(np.asarray(probe, np.float32))
        return {i: max(float(v @ q) for _, v in ts) for i, ts in self.templates.items() if ts}


# ---------------------------------------------------------------------------------------------------------------
# calibration and fusion

@dataclasses.dataclass
class Calibration:
    """Per-model impostor statistics -> z-normalised scores and a threshold at a target false-accept rate."""
    mean: float
    std: float
    thr_by_far: dict            # {far: raw-score threshold}

    @classmethod
    def from_impostors(cls, scores, fars=(1e-2, 1e-3)) -> "Calibration":
        s = np.sort(np.asarray(scores, np.float64))
        thr = {f: float(np.quantile(s, 1 - f)) if len(s) else 1.0 for f in fars}
        return cls(float(s.mean()), float(s.std() + 1e-6), thr)

    def z(self, s):
        return (np.asarray(s, np.float64) - self.mean) / self.std


def fuse(scores: dict, cals: dict, weights: dict | None = None) -> float:
    """Weighted mean of impostor-z-normalised scores of several face models ({model: cosine})."""
    weights = weights or {k: 1.0 for k in scores}
    zs = [weights[k] * float(cals[k].z(v)) for k, v in scores.items() if v is not None]
    ws = [weights[k] for k, v in scores.items() if v is not None]
    return sum(zs) / max(1e-9, sum(ws)) if zs else float("nan")


def side_evidence(face_usable: bool, reid_cos: float | None, height_diff_cm: float | None,
                  reid_cal: Calibration | None = None) -> dict:
    """Body re-ID + stature as WEAK side evidence, only when the face is unusable. Never names anyone by itself:
    the result is a 'possible' hint for the review UI, reported separately from the face decision."""
    if face_usable or reid_cos is None:
        return {"used": False}
    z = float(reid_cal.z(reid_cos)) if reid_cal else None
    h_ok = None if height_diff_cm is None else abs(height_diff_cm) <= 6.0
    return {"used": True, "reid_cos": round(float(reid_cos), 3), "reid_z": None if z is None else round(z, 2),
            "stature_consistent": h_ok, "kind": "possible (body, weak)"}


def visit_vectors(obs: list[FaceObs], ada: AdaFace | None, min_px: float = 24.0) -> dict:
    """Per-visit summary for the v2 path: production InspireFace template plus AdaFace template (same faces)."""
    faces = [o for o in obs if o.size >= min_px and getattr(o, "aligned", None) is not None]
    out: dict = {"version": V2_VERSION, "n": len(faces)}
    if not faces:
        return out
    out["isf"] = aggregate(faces, np.stack([l2n(f.vec) for f in faces]))
    if ada is not None:
        V, n = ada.embed([f.aligned for f in faces])
        out["ada"] = aggregate(faces, V)
        out["ada_norm_med"] = float(np.median(n))
    out["ir_frac"] = float(np.mean([modality(getattr(f, "ir", None)) == "ir" for f in faces]))
    return out


def _cos(a, b) -> float:
    return float(l2n(a) @ l2n(b)) if a is not None and b is not None else math.nan


# ---------------------------------------------------------------------------------------------------------------
# profiler hook (opt-in: [profiler] face_v2_adaface = "/models/adaface/adaface_ir101_webface12m.onnx")

from .faceid import FaceSampler  # noqa: E402


class FaceSamplerV2(FaceSampler):
    """FaceSampler whose result() also carries `v2`: the AdaFace template of the SAME pooled inlier faces, built
    with the production aggregator (weights, outlier rejection). The InspireFace `vec` and every other field are
    unchanged, so the bridge keeps working as before until it is taught to compare `v2.vec` with v2 templates."""

    def __init__(self, engine, *a, ada: AdaFace | None = None, **k):
        super().__init__(engine, *a, **k)
        self.ada = ada

    def result(self) -> dict:
        r = super().result()
        if self.ada is None or r.get("vec") is None:
            return r
        faces = [f for _, f in self.agg.pool if getattr(f, "aligned", None) is not None]
        if not faces:
            return r
        try:
            V, n = self.ada.embed([f.aligned for f in faces])
            t = aggregate(faces, V, self.agg.p)
        except Exception:
            return r
        if t is not None:
            r["v2"] = {"version": V2_VERSION, "model": self.ada.name, "vec": [round(float(x), 5) for x in t],
                       "n_faces": len(faces), "ada_norm_med": round(float(np.median(n)), 2),
                       "ir_frac": round(float(np.mean([modality(getattr(f, "ir", None)) == "ir" for f in faces])), 2)}
        for _, f in self.agg.pool:      # crops are no longer needed
            f.aligned = None
        return r

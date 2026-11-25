"""Person re-ID v2: newer appearance embedders (CLIP-ReID, SOLIDER, OSNet-AIN) behind reid.Embedder's interface.

OFF by default. The profiler keeps using reid.Embedder (OSNet x0.25) unless its config says otherwise:

    [profiler]
    reid_backend = "osnet"          # default: reid_model (OSNet x0.25 MSMT17, 512-d)
    # reid_backend = "v2"           # use reid_v2_model for the sighting `embedding` instead (NOT comparable with
    # reid_v2_model = "/models/clipreid_vitb16_msmt17.int8.onnx"   #   the stored OSNet vectors: see docs/reid-gait.md)
    # reid_shadow_model = "/models/clipreid_vitb16_msmt17.int8.onnx"  # dual-embedding period: ALSO compute this one
                                    # and send it as `embedding_v2` (the bridge ignores it until it has a column)

Embeddings from different models live in different spaces (different sizes, too): the bridge's identity centroids,
the 0.70 / 0.55 thresholds and every stored `sightings.embedding` belong to OSNet. Switching needs re-embedding or a
dual-embedding period (docs/reid-gait.md, "Migration").

Preprocessing comes from the ONNX metadata written by scripts/export_reid_v2.py (sg_input_h, sg_input_w, sg_mean,
sg_std); a file without it falls back to PRESETS by file name. Output: L2-normalised float32 vectors, compared by
dot product (cosine), exactly like reid.Embedder.

Weights: trained on MSMT17 (research dataset, academic use). LOCAL use only: never publish the models or derived
embeddings. Code licences: CLIP-ReID MIT, SOLIDER-REID MIT, torchreid MIT. Sources and checksums: docs/reid-gait.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
PRESETS = {   # file-name stem (without .int8) -> (input height, width, (mean, std)); used only without metadata
    "clipreid_vitb16_msmt17": (256, 128, HALF),
    "solider_swin_t_msmt17": (384, 128, HALF),
    "solider_swin_s_msmt17": (384, 128, HALF),
    "osnet_ain_x1_0_msmt17": (256, 128, IMAGENET),
    "osnet_x0_25_msmt17": (256, 128, IMAGENET),
}


def _preset(path: str) -> tuple:
    stem = Path(path).name.split(".")[0]
    if stem not in PRESETS:
        raise ValueError(f"{path}: no sg_* metadata and no preset for '{stem}' (export with scripts/export_reid_v2.py)")
    return PRESETS[stem]


class Embedder:
    """Same interface as reid.Embedder: __call__(crops) -> (N, D), one(crops) -> (D,). Crops are BGR uint8."""

    def __init__(self, model_path: str, threads: int = 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        meta = self.sess.get_modelmeta().custom_metadata_map
        if "sg_input_h" in meta:
            h, w = int(meta["sg_input_h"]), int(meta["sg_input_w"])
            mean, std = json.loads(meta["sg_mean"]), json.loads(meta["sg_std"])
            self.name = meta.get("sg_model", Path(model_path).stem)
        else:
            h, w, (mean, std) = _preset(model_path)
            self.name = Path(model_path).name.split(".")[0]
        self.hw = (h, w)
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        self.dim = int(self.sess.get_outputs()[0].shape[-1]) if isinstance(self.sess.get_outputs()[0].shape[-1], int) else None

    def preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = self.hw
        x = cv2.resize(crop_bgr, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32) / 255.0
        return ((x - self.mean) / self.std).transpose(2, 0, 1)

    def __call__(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """crops -> (N, D) L2-normalised fingerprints."""
        out = self.sess.run(None, {self.inp: np.stack([self.preprocess(c) for c in crops_bgr])})[0].astype(np.float32)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)

    def one(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Several crops of one sighting -> one averaged, re-normalised fingerprint."""
        v = self(crops_bgr).mean(0)
        return v / (np.linalg.norm(v) + 1e-12)


def load_embedder(pc: dict, threads: int = 4):
    """The profiler's sighting embedder. reid_backend "osnet" (default) keeps reid.Embedder on reid_model."""
    backend = str(pc.get("reid_backend", "osnet")).lower()
    if backend == "osnet":
        from .reid import Embedder as OsnetEmbedder
        return OsnetEmbedder(pc["reid_model"], threads=threads)
    if backend == "v2":
        if not pc.get("reid_v2_model"):
            raise ValueError("reid_backend = 'v2' needs reid_v2_model")
        return Embedder(pc["reid_v2_model"], threads=threads)
    raise ValueError(f"unknown reid_backend {backend!r} (osnet | v2)")


def load_shadow(pc: dict, threads: int = 4) -> Embedder | None:
    """Optional second embedder for a dual-embedding period (reid_shadow_model); None when not configured."""
    return Embedder(pc["reid_shadow_model"], threads=threads) if pc.get("reid_shadow_model") else None

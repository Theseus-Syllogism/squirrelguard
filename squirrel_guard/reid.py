"""Person re-identification fingerprint: OSNet x0.25 (MSMT17) via ONNX -> L2-normalised 512-d vector.

Same person (similar clothing) -> high cosine similarity. It mostly encodes appearance/clothing, not the face.
"""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Embedder:
    def __init__(self, model_path: str, threads: int = 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """crops -> (N, 512) L2-normalised fingerprints."""
        batch = []
        for c in crops_bgr:
            x = cv2.resize(c, (128, 256), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32) / 255.0
            batch.append(((x - MEAN) / STD).transpose(2, 0, 1))
        out = self.sess.run(None, {self.inp: np.stack(batch)})[0]
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)

    def one(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Several crops of one sighting -> one averaged, re-normalised fingerprint."""
        v = self(crops_bgr).mean(0)
        return v / (np.linalg.norm(v) + 1e-12)

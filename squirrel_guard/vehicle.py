"""Vehicle appearance fingerprint: DINOv2 ViT-S/14 (ONNX, 384-d, L2-normalised).

No vehicle re-ID weights were available, so this uses DINOv2's general instance-level features. It separates
the same car from different cars, but similar models in the same colour can score close: plates win when present.
"""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
VEHICLES = {"car": "car", "taxi": "car", "limousine": "car", "van": "car", "ambulance": "car",
            "land_vehicle": "car", "vehicle": "car", "truck": "truck", "bus": "bus", "motorcycle": "motorcycle"}


def vehicle_label(raw: str) -> str | None:
    return VEHICLES.get(raw.lower().replace(" ", "_"))


class VehicleEmbedder:
    def __init__(self, model_path: str, threads: int = 4, size: int = 224):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.size = size

    def _prep(self, c: np.ndarray) -> np.ndarray:
        h, w = c.shape[:2]  # letterbox to a square so the car's proportions survive
        s = self.size / max(h, w)
        r = cv2.resize(c, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        canvas = np.full((self.size, self.size, 3), 124, np.uint8)
        y0, x0 = (self.size - r.shape[0]) // 2, (self.size - r.shape[1]) // 2
        canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
        x = canvas[:, :, ::-1].astype(np.float32) / 255.0
        return ((x - MEAN) / STD).transpose(2, 0, 1)

    def __call__(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        out = self.sess.run(None, {self.inp: np.stack([self._prep(c) for c in crops_bgr])})[0]
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)

    def one(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        v = self(crops_bgr).mean(0)
        return v / (np.linalg.norm(v) + 1e-12)

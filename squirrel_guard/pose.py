"""Body keypoints: YOLOv8s-pose (COCO, 17 keypoints) via ONNX.

Keypoint order (COCO): 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips,
13/14 knees, 15/16 ankles (left/right). Each keypoint is (x, y, confidence) in the input image's pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 5, 6, 9, 10, 11, 12, 13, 14, 15, 16


@dataclass
class Pose:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    kp: np.ndarray                          # (17, 3): x, y, conf


class PoseEstimator:
    def __init__(self, model_path: str, conf: float = 0.35, threads: int = 4, backend: str = "ort"):
        self.conf = conf
        if backend == "trt":                 # RTX GPU: TensorRT fp16 engine (squirrel_guard/trt.py, docs/trt.md)
            from .trt import TRTModel
            eng = TRTModel(model_path, fp16=True, max_batch=1)

            class _S:
                def run(self, _o, feed):
                    return eng(next(iter(feed.values())))
            self.sess, self.inp, self.size = _S(), "images", 640
            return
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.size = int(self.sess.get_inputs()[0].shape[2])

    def __call__(self, img: np.ndarray) -> list[Pose]:
        h, w = img.shape[:2]
        s = self.size / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        dy, dx = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.inp: x})[0][0].T          # (8400, 56)
        out = out[out[:, 4] >= self.conf]
        if not len(out):
            return []
        cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1)
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), out[:, 4].tolist(), self.conf, 0.5)
        res = []
        for i in np.array(keep).reshape(-1):
            r = out[i]
            kp = r[5:].reshape(17, 3).copy()
            kp[:, 0] = (kp[:, 0] - dx) / s
            kp[:, 1] = (kp[:, 1] - dy) / s
            x1, y1 = (r[0] - r[2] / 2 - dx) / s, (r[1] - r[3] / 2 - dy) / s
            x2, y2 = (r[0] + r[2] / 2 - dx) / s, (r[1] + r[3] / 2 - dy) / s
            res.append(Pose((max(0., x1), max(0., y1), min(w - 1., x2), min(h - 1., y2)), float(r[4]), kp))
        return res

    def in_region(self, img: np.ndarray, box, grow: float = 2.0) -> list[Pose]:
        """Run on a square window around `box` (grown) so a small, distant person gets enough pixels."""
        H, W = img.shape[:2]
        x1, y1, x2, y2 = box
        side = max(x2 - x1, y2 - y1) * grow
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        X1, Y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        X2, Y2 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
        if X2 - X1 < 16 or Y2 - Y1 < 16:
            return []
        res = self(img[Y1:Y2, X1:X2])
        for p in res:
            p.kp[:, 0] += X1
            p.kp[:, 1] += Y1
            b = p.box
            p.box = (b[0] + X1, b[1] + Y1, b[2] + X1, b[3] + Y1)
        return res

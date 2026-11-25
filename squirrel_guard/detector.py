"""YOLOv8 ONNX detector (Ultralytics export). Works with any class set; names come from the model metadata."""
from __future__ import annotations

import ast
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels


class Detector:
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45, threads: int = 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.names: dict[int, str] = ast.literal_eval(meta["names"]) if "names" in meta else {}
        self.conf = conf
        self.iou = iou

    def _letterbox(self, img: np.ndarray):
        h, w = img.shape[:2]
        r = self.size / max(h, w)
        nw, nh = round(w * r), round(h * r)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, r, left, top

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        img, r, dx, dy = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, swapRB=True)
        out = self.sess.run(None, {self.input_name: blob})[0][0]  # (4 + nc, N)
        out = out.T
        scores = out[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        keep = conf >= self.conf
        if not keep.any():
            return []
        xywh, cls, conf = out[keep, :4], cls[keep], conf[keep]
        x1 = (xywh[:, 0] - xywh[:, 2] / 2 - dx) / r
        y1 = (xywh[:, 1] - xywh[:, 3] / 2 - dy) / r
        bw, bh = xywh[:, 2] / r, xywh[:, 3] / r
        boxes = np.stack([x1, y1, bw, bh], 1)
        idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), conf.tolist(), cls.tolist(), self.conf, self.iou)
        H, W = frame.shape[:2]
        dets = []
        for i in np.array(idx).flatten():
            x, y, w, h = boxes[i]
            dets.append(Detection(
                label=self.names.get(int(cls[i]), str(int(cls[i]))),
                score=float(conf[i]),
                box=(max(0, int(x)), max(0, int(y)), min(W, int(x + w)), min(H, int(y + h))),
            ))
        return sorted(dets, key=lambda d: -d.score)

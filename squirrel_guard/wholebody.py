"""Whole-body 2D keypoints (133 = body 17 + feet 6 + face 68 + hands 2 x 21) with RTMW (RTMPose family, Apache-2.0).

  wb = WholeBody("models/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122/end2end.onnx")
  kp = wb(frame_bgr, box)          # (133, 3): x, y (frame pixels), confidence 0..1 (calibrated per model)

Top-down: the person box (from the YOLO pose detector already in the pipeline) is padded 1.25x, warped to the model's
aspect (3:4) and size, normalised (ImageNet mean/std, RGB) and decoded from SimCC (per-keypoint 1D x/y distributions,
2x sub-pixel bins: argmax / 2 -> crop pixels -> inverse affine -> frame pixels; score = the smaller of the two peaks).
The two RTMW exports report peaks on different scales (x-l 384x288: raw ~0.5-6; l-m 256x192: ~0-1), so the score is
mapped to a 0..1 confidence per model, matched by quantiles on the same frames (x-l raw 1.0 -> 0, 3.5 -> 1; l-m 0.13 ->
0, 0.55 -> 1; positions of the two models agree to 1.1 % of the box height). raw=True returns the raw peaks.
Why this over the 17-point YOLO pose: heels and toes (gait, ground contact, foot angle), hands (what is held, wrist
rotation) and 68 face points, from a model trained on 14 datasets (COCO-WholeBody, Halpe, UBody, ...).

Infrared: the Wyze cameras switch to grayscale IR at night. Grayscale frames are detected (channels equal) and fed as
3-channel grey; clahe=True adds contrast-limited histogram equalisation on the crop. Measured on daylight frames
converted to grey: 96-98 % of joints within 3 % of body height of the colour result, CLAHE no better (off by
default); real night IR (blooming, low contrast) is still to be measured: docs/rigfit.md.

Keypoint index groups (COCO-WholeBody order): BODY 0-16 (COCO), FEET 17-22 (L big toe, L small toe, L heel, R big
toe, R small toe, R heel), FACE 23-90, LEFT_HAND 91-111, RIGHT_HAND 112-132.
"""
from __future__ import annotations

import numpy as np

BODY = list(range(0, 17))
FEET = list(range(17, 23))
FACE = list(range(23, 91))
LHAND = list(range(91, 112))
RHAND = list(range(112, 133))
L_BIGTOE, L_SMALLTOE, L_HEEL, R_BIGTOE, R_SMALLTOE, R_HEEL = 17, 18, 19, 20, 21, 22
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


def is_grayscale(img: np.ndarray, tol: float = 2.0) -> bool:
    """IR night frames: the three channels are (nearly) identical."""
    if img.ndim == 2:
        return True
    s = img[::8, ::8].astype(np.int16)
    return float(np.abs(s[..., 0] - s[..., 1]).mean() + np.abs(s[..., 1] - s[..., 2]).mean()) < tol


def _affine(box, out_w: int, out_h: int, pad: float = 1.25):
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * pad, (y2 - y1) * pad
    aspect = out_w / out_h
    if w > h * aspect:
        h = w / aspect
    else:
        w = h * aspect
    # crop [cx - w/2, cx + w/2] x [cy - h/2, cy + h/2] -> [0, out_w] x [0, out_h]
    s = out_w / w
    return np.array([[s, 0, out_w / 2 - s * cx], [0, s, out_h / 2 - s * cy]], np.float32), s


class WholeBody:
    SCORE_RANGE = {384: (1.0, 3.5), 256: (0.13, 0.55)}   # model input height -> raw peak mapped to 0 and 1

    def __init__(self, model_path: str, threads: int = 4, providers: list | None = None, ir_mode: str = "auto",
                 clahe: bool = False, raw: bool = False):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(model_path, so, providers=providers or ["CPUExecutionProvider"])
        shp = self.sess.get_inputs()[0].shape          # [batch, 3, H, W]
        self.h, self.w = int(shp[2]), int(shp[3])
        self.ir_mode = ir_mode
        self.clahe = clahe
        self.raw = raw
        self.score_range = self.SCORE_RANGE.get(self.h, (1.0, 3.5))
        self._clahe = None

    def _prep(self, frame, box, ir: bool):
        import cv2
        M, _ = _affine(box, self.w, self.h)
        crop = cv2.warpAffine(frame, M, (self.w, self.h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        if ir:
            g = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if self.clahe:
                if self._clahe is None:
                    self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                g = self._clahe.apply(g)
            crop = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        rgb = crop[:, :, ::-1].astype(np.float32)
        return ((rgb - MEAN) / STD).transpose(2, 0, 1), M

    def __call__(self, frame: np.ndarray, box) -> np.ndarray:
        return self.batch(frame, [box])[0]

    def batch(self, frame: np.ndarray, boxes) -> list[np.ndarray]:
        if not len(boxes):
            return []
        ir = self.ir_mode == "on" or (self.ir_mode == "auto" and is_grayscale(frame))
        if frame.ndim == 2:
            import cv2
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        xs, Ms = zip(*(self._prep(frame, b, ir) for b in boxes))
        sx, sy = self.sess.run(None, {self.sess.get_inputs()[0].name: np.stack(xs).astype(np.float32)})
        out = []
        for k, M in enumerate(Ms):
            px = sx[k].argmax(-1) / 2.0                 # SimCC split ratio 2
            py = sy[k].argmax(-1) / 2.0
            sc = np.minimum(sx[k].max(-1), sy[k].max(-1))
            if not self.raw:
                lo, hi = self.score_range
                sc = np.clip((sc - lo) / (hi - lo), 0.0, 1.0)
            Mi = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
            pts = np.c_[px, py, np.ones_like(px)] @ Mi.T
            out.append(np.c_[pts, sc].astype(np.float32))
        return out

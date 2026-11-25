"""Learned gait signature: YOLOv8-seg silhouettes -> OpenGait GaitBase (trained on GREW) via ONNX -> 4096-d vector.

Pipeline for ONE tracked person:
  1. Silhouette per frame: YOLOv8s-seg (COCO class 0) runs on a padded square window around the given box; the
     person instance that best overlaps the box is kept and its mask (sigmoid > 0.5, as Ultralytics does) is
     taken at the window's resolution.
  2. OpenGait size normalisation (datasets/pretreatment.py, imgs2pickle, the CASIA-B/OU-MVLP branch): crop the
     rows that contain the mask, resize to height 64 keeping the aspect ratio (INTER_CUBIC), centre a 64-wide
     window on the column-sum median (zero-pad when it runs off an edge) -> 64x64. GREW itself ships silhouettes
     already resized to 64x64 (its pretreatment branch just loads them), so this is the closest reproducible
     match. Then the GREW test transform (configs/gaitbase/gaitbase_da_grew.yaml: BaseSilCuttingTransform):
     cut int(64 // 64) * 10 = 10 columns off each side -> 64x44, divide by 255.
  3. GaitBase (ResNet9 per frame, max over frames, 16 horizontal strips, per-strip FC) -> (256 ch, 16 parts):
     OpenGait's inference 'embeddings' (pre-BNNeck), which its GREW evaluator ranks with mean per-part Euclidean
     distance. Here the 256x16 is flattened and L2-normalised; compare with distance() = cosine distance.

Frame order does not matter to the model (set pooling), so the low event sampling rate (~8 fps) is fine, but it
needs at least ~10 good silhouettes covering a full stride. Limits: silhouettes from a home camera are much
noisier than the dataset's (shadows, bags, occlusion by cars or fences, partial bodies at the frame edge);
the GREW model is view-tolerant but not view-invariant, and clothing/carried objects change the silhouette.
The vector is a WEAK identity hint for review, never a decision on its own. Research-only weights (OpenGait is
academic-use only; GREW is a research dataset): this is for a private, non-commercial system.
Export: scripts/export_gaitbase.py.
"""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

SIL_H, SIL_W = 64, 44
MAX_FRAMES = 720          # OpenGait evaluator frames_all_limit for GREW
EDGE_MARGIN = 0.015       # a body ending this close (fraction of frame height) to the top/bottom is treated as cut off
MIN_ASPECT = 1.3          # mask height / width below this = truncated or not a walking body; frame dropped


def _iou(a, b) -> float:
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def normalise_silhouette(mask: np.ndarray, size: int = 64) -> np.ndarray | None:
    """Binary/0-255 mask (any size) -> (64, 44) uint8 silhouette, following OpenGait's pretreatment + cutting."""
    img = (mask > 0).astype(np.uint8) * 255 if mask.dtype == bool or mask.max() <= 1 else mask.astype(np.uint8)
    if img.sum() <= 10000:                       # pretreatment.py: "has no data"
        return None
    y_sum = img.sum(axis=1)
    y_top = (y_sum != 0).argmax(axis=0)
    y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top: y_btm + 1, :]
    ratio = img.shape[1] / img.shape[0]
    img = cv2.resize(img, (int(size * ratio), size), interpolation=cv2.INTER_CUBIC)
    x_csum = img.sum(axis=0).astype(np.int64).cumsum()
    nz = np.nonzero(x_csum > img.sum() / 2)[0]
    if not len(nz) or nz[0] == 0:                # pretreatment.py: "has no center" (also rejects idx 0)
        return None
    x_center = int(nz[0])
    half = size // 2
    left, right = x_center - half, x_center + half
    if left <= 0 or right >= img.shape[1]:
        left += half
        right += half
        pad = np.zeros((img.shape[0], half), np.uint8)
        img = np.concatenate([pad, img, pad], axis=1)
    sil = img[:, left:right]
    cut = int(sil.shape[-1] // 64) * 10          # BaseSilCuttingTransform (cutting=None)
    return np.ascontiguousarray(sil[:, cut:-cut]) if cut else sil


class GaitEmbedder:
    def __init__(self, gait_model: str, seg_model: str, threads: int = 4, seg_conf: float = 0.25,
                 grow: float = 1.25):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.gait = ort.InferenceSession(gait_model, so, providers=["CPUExecutionProvider"])
        self.seg = ort.InferenceSession(seg_model, so, providers=["CPUExecutionProvider"])
        self.seg_inp = self.seg.get_inputs()[0].name
        self.size = int(self.seg.get_inputs()[0].shape[2])
        self.seg_conf, self.grow = seg_conf, grow

    # --- segmentation ----------------------------------------------------------------------------------------
    def _mask(self, img: np.ndarray, box) -> np.ndarray | None:
        """Person mask (uint8 0/255, same size as img) of the instance best overlapping `box` (img coords)."""
        h, w = img.shape[:2]
        s = self.size / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        dy, dx = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        pred, proto = self.seg.run(None, {self.seg_inp: x})
        pred = pred[0].T                                           # (N, 4 + 80 + 32)
        pred = pred[pred[:, 4] >= self.seg_conf]                   # class 0 = person
        if not len(pred):
            return None
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        keep = cv2.dnn.NMSBoxes(np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1).tolist(), pred[:, 4].tolist(),
                                self.seg_conf, 0.5)
        pred = pred[np.array(keep).reshape(-1)]
        boxes = np.stack([(pred[:, 0] - pred[:, 2] / 2 - dx) / s, (pred[:, 1] - pred[:, 3] / 2 - dy) / s,
                          (pred[:, 0] + pred[:, 2] / 2 - dx) / s, (pred[:, 1] + pred[:, 3] / 2 - dy) / s], 1)
        ious = [_iou(b, box) for b in boxes]
        i = int(np.argmax(ious))
        if ious[i] < 0.3:
            return None
        _, c, ph, pw = proto.shape
        m = (pred[i, 84:84 + c] @ proto[0].reshape(c, -1)).reshape(ph, pw)   # logits; sigmoid>0.5 <=> logit>0
        r = ph / self.size
        m = m[int(dy * r):int(round((dy + nh) * r)), int(dx * r):int(round((dx + nw) * r))]
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        x1, y1, x2, y2 = boxes[i]                                  # Ultralytics crop_mask: zero outside the box
        out = np.zeros((h, w), np.uint8)
        X1, Y1, X2, Y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, np.ceil(x2))), int(min(h, np.ceil(y2)))
        out[Y1:Y2, X1:X2] = (m[Y1:Y2, X1:X2] > 0).astype(np.uint8) * 255
        return out

    def silhouettes(self, frames: list[np.ndarray], boxes: list[tuple[float, float, float, float] | None]) -> np.ndarray:
        """Frames + this person's box per frame (None = not seen) -> (T, 64, 44) uint8 silhouettes (T <= len)."""
        sils = []
        for fr, box in zip(frames, boxes):
            if box is None:
                continue
            H, W = fr.shape[:2]
            x1, y1, x2, y2 = box
            edge = max(3, int(EDGE_MARGIN * H))
            if y2 - y1 < 48 or y1 < edge or y2 > H - 1 - edge:     # too small, or (probably) cut off by the frame edge
                continue
            side = max(x2 - x1, y2 - y1) * self.grow
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            X1, Y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            X2, Y2 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
            m = self._mask(fr[Y1:Y2, X1:X2], (x1 - X1, y1 - Y1, x2 - X1, y2 - Y1))
            if m is None:
                continue
            ys, xs = np.nonzero(m)
            if not len(ys):
                continue
            if (Y1 + ys.min() < edge or Y1 + ys.max() > H - 1 - edge   # body runs off the frame top/bottom
                    or (ys.max() - ys.min()) < MIN_ASPECT * (xs.max() - xs.min())):   # not a whole standing body
                continue
            sil = normalise_silhouette(m)
            if sil is not None:
                sils.append(sil)
        return np.stack(sils) if sils else np.zeros((0, SIL_H, SIL_W), np.uint8)

    # --- gait model -------------------------------------------------------------------------------------------
    def part_features(self, sils: np.ndarray) -> np.ndarray:
        """(T, 64, 44) uint8 silhouettes -> raw (256, 16) GaitBase part features (OpenGait's 'embeddings')."""
        x = (sils[:MAX_FRAMES].astype(np.float32) / 255.0)[None]
        return self.gait.run(None, {"sils": x})[0][0]

    def embed(self, frames, boxes, min_frames: int = 10) -> np.ndarray | None:
        """-> (4096,) float32 L2-normalised gait vector, or None with fewer than min_frames silhouettes."""
        sils = self.silhouettes(frames, boxes)
        if len(sils) < min_frames:
            return None
        v = self.part_features(sils).reshape(-1).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance (0 = identical, 2 = opposite) between two embed() vectors."""
        return float(1.0 - np.dot(a, b))

"""Reading a vehicle visit: number plate, true colour, and how it moved. Runs in the profiler (gpuserver).

  PlateReader(models).read(crops)     -> {"text", "conf", "char_conf", "reads", "agree"} or None
  body_colour(crop, frame)            -> {"name", "confidence", "night", "rgb", "note"}
  motion(path_data, frame_size)       -> {"direction", "stopped_s", "moved_px", "note"}

Plates: open-image-models YOLOv9-t 640 plate detector + fast-plate-ocr cct-s-v2 "global" reader (both MIT-licensed
code; models pinned by SHA-256 in deploy/gpuserver/squirrel-guard/Dockerfile). Each crop gives one reading with a
probability per character slot; readings are combined slot by slot (log-probabilities summed), so a character that
one frame reads badly is outvoted by the others. The result keeps every character's confidence.

Colour: infrared night images have no colour; they are recognised (almost no saturation across the frame) and the
colour is reported as unknown instead of guessed. In daylight the frame is white-balanced (grey-world) before the
car body is classified, so a low sun no longer turns a silver car red.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

PLATE_DET = "yolo-v9-t-640-license-plates-end2end.onnx"
PLATE_OCR = "cct_s_v2_global.onnx"
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
PAD = "_"
MIN_PLATE_PX = 18                                                # plate box height below this is unreadable
MIN_CONF = 0.5                                                   # below this the reading is kept as a candidate only


class PlateReader:
    def __init__(self, models: str | Path, threads: int = 2, det_score: float = 0.35):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        m = Path(models)
        self.det = ort.InferenceSession(str(m / PLATE_DET), so, providers=["CPUExecutionProvider"])
        self.ocr = ort.InferenceSession(str(m / PLATE_OCR), so, providers=["CPUExecutionProvider"])
        self.det_score = det_score

    def find(self, img: np.ndarray) -> list[tuple[float, tuple[int, int, int, int]]]:
        h, w = img.shape[:2]
        r = min(640 / h, 640 / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw, dh = (640 - nw) / 2, (640 - nh) / 2
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
        canvas = np.full((640, 640, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = []
        for row in self.det.run(None, {"images": x})[0]:
            if row[6] >= self.det_score:
                b = (int((row[1] - dw) / r), int((row[2] - dh) / r), int((row[3] - dw) / r), int((row[4] - dh) / r))
                out.append((float(row[6]), b))
        return sorted(out, reverse=True)

    def probs(self, plate_bgr: np.ndarray) -> np.ndarray:
        """(10, 37) character probabilities for one plate image."""
        rgb = cv2.cvtColor(cv2.resize(plate_bgr, (128, 64), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        p = self.ocr.run(None, {"input": rgb[None].astype(np.uint8)})[0][0]
        if p.min() < 0 or abs(p.sum(-1).mean() - 1) > 0.05:      # logits: make probabilities
            e = np.exp(p - p.max(-1, keepdims=True))
            p = e / e.sum(-1, keepdims=True)
        return p

    def read(self, crops: list[np.ndarray], max_reads: int = 8) -> dict | None:
        """Combine the plate readings of several vehicle crops (largest first)."""
        logp, n, texts = None, 0, []
        for crop in sorted(crops, key=lambda c: -c.shape[0] * c.shape[1])[:max_reads * 2]:
            found = self.find(crop)
            if not found:
                continue
            score, (x1, y1, x2, y2) = found[0]
            if y2 - y1 < MIN_PLATE_PX:
                continue
            mx, my = int(0.06 * (x2 - x1)), int(0.10 * (y2 - y1))
            sub = crop[max(0, y1 - my):y2 + my, max(0, x1 - mx):x2 + mx]
            if sub.size == 0:
                continue
            p = self.probs(sub)
            texts.append(decode(p)[0])
            lp = np.log(np.clip(p, 1e-6, 1)) * score
            logp = lp if logp is None else logp + lp
            n += 1
            if n >= max_reads:
                break
        if logp is None:
            return None
        e = np.exp(logp - logp.max(-1, keepdims=True))
        text, conf = decode(e / e.sum(-1, keepdims=True))
        if not text:
            return None
        agree = sum(t == text for t in texts) / len(texts)
        gm = float(np.prod(conf) ** (1 / len(conf)))
        readable = gm >= MIN_CONF and len(text) >= 3 and (n == 1 or agree >= 0.5)
        return {"text": text if readable else None, "candidate": text, "readable": readable,
                "char_conf": [round(c, 3) for c in conf], "conf": round(gm, 3), "reads": n, "agree": round(agree, 2), "each": texts}


def decode(p: np.ndarray) -> tuple[str, list[float]]:
    idx = p.argmax(-1)
    chars, conf = [], []
    for slot, k in enumerate(idx):
        ch = ALPHABET[k]
        if ch == PAD:
            continue
        chars.append(ch)
        conf.append(float(p[slot, k]))
    return "".join(chars), conf


# ------------------------------------------------------------------------------------------------- colour
def is_night(frame: np.ndarray) -> bool:
    """Infrared/greyscale image: hardly any saturation anywhere, channels nearly equal."""
    small = cv2.resize(frame, (160, 90)) if frame.shape[1] > 160 else frame
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    b, g, r = [small[:, :, i].astype(np.float32) for i in range(3)]
    chroma = np.mean(np.abs(b - g) + np.abs(g - r) + np.abs(r - b)) / 3
    return float(np.percentile(hsv[:, :, 1], 90)) < 30 and chroma < 6


def white_gains(frame: np.ndarray, box=None) -> np.ndarray | None:
    """Per-channel gains from the near-neutral surfaces AROUND the vehicle (road, walls, pavement). Grey-world over
    the whole frame fails when one car fills it (a red car would be pushed toward blue), so the car is masked out and
    only low-saturation pixels count. None when too little neutral background is visible: then no correction."""
    f = frame.astype(np.float32)
    mask = np.ones(frame.shape[:2], bool)
    if box is not None:
        x1, y1, x2, y2 = box
        mask[max(0, y1):y2, max(0, x1):x2] = False
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    neutral = mask & (hsv[:, :, 1] < 64) & (hsv[:, :, 2] > 50) & (hsv[:, :, 2] < 245)
    if neutral.sum() < 0.03 * mask.size:
        return None
    means = f[neutral].mean(0) + 1e-6
    return np.clip(means.mean() / means, 0.75, 1.33)


def body_colour(crop: np.ndarray, frame: np.ndarray | None = None, box=None) -> dict:
    """crop: the vehicle; frame + box (x1, y1, x2, y2 of the crop in the frame): for night detection and white
    balance from the surroundings."""
    ref = frame if frame is not None else crop
    if is_night(ref):
        return {"name": None, "confidence": 0.0, "night": True,
                "note": "night/infrared or black-and-white image: colour cannot be judged"}
    gains = white_gains(frame, box) if frame is not None else None
    img = np.clip(crop.astype(np.float32) * gains, 0, 255).astype(np.uint8) if gains is not None else crop
    h, w = img.shape[:2]
    body = img[int(0.30 * h):int(0.80 * h), int(0.08 * w):int(0.92 * w)]  # skip roof/windows and wheels/road
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    H, S, V = hsv[:, 0] * 2, hsv[:, 1] / 255, hsv[:, 2] / 255
    keep = (V > 0.08) & ~((V > 0.92) & (S < 0.10))                 # drop deep shadow and glare
    if keep.sum() < 50:
        return {"name": None, "confidence": 0.0, "night": False, "note": "too little of the body is visible"}
    H, S, V = H[keep], S[keep], V[keep]
    names = np.empty(len(H), object)
    grey = (S < 0.20) | (V < 0.22) | ((V < 0.35) & (S < 0.40))   # dark paint: reflections tint it, it is still black
    names[grey & (V >= 0.72)] = "white"
    names[grey & (V >= 0.40) & (V < 0.72)] = "grey"
    names[grey & (V < 0.40)] = "black"
    c = ~grey
    hue = H[c]
    cn = np.where((hue < 15) | (hue >= 340), "red", np.where(hue < 40, "orange", np.where(hue < 68, "yellow",
                  np.where(hue < 165, "green", np.where(hue < 255, "blue", np.where(hue < 300, "purple", "red"))))))
    beige = (hue >= 20) & (hue < 60) & (S[c] < 0.42) & (V[c] >= 0.50)
    cn = np.where(beige, "beige", cn)
    cn = np.where((cn == "orange") & (V[c] < 0.50), "brown", cn)
    cn = np.where((cn == "yellow") & (V[c] < 0.45), "brown", cn)
    names[c] = cn
    vals, counts = np.unique(names.astype(str), return_counts=True)
    order = np.argsort(-counts)
    top, share = vals[order[0]], counts[order[0]] / counts.sum()
    second = vals[order[1]] if len(order) > 1 and counts[order[1]] / counts.sum() > 0.25 else None
    med = np.median(img.reshape(-1, 3), 0)[::-1]
    note = "" if share >= 0.5 else f"mixed: also {second}" if second else "mixed colours"
    return {"name": str(top), "confidence": round(float(share), 2), "night": False, "second": second,
            "balanced": gains is not None,
            "rgb": [int(x) for x in med], "note": note}


# ------------------------------------------------------------------------------------------------- motion
def motion(path_data: list | None, stationary_px: float = 0.01) -> dict | None:
    """Frigate path_data: [[[x, y], t], ...] with x, y as fractions of the frame. Direction of travel across the image
    and how long it stood still (steps under 1 % of the frame per second count as standing)."""
    if not path_data or len(path_data) < 2:
        return None
    pts = np.array([p[0] for p in path_data], float)
    ts = np.array([p[1] for p in path_data], float)
    d = pts[-1] - pts[0]
    stopped = 0.0
    for i in range(1, len(pts)):
        dt = ts[i] - ts[i - 1]
        if dt > 0 and np.linalg.norm(pts[i] - pts[i - 1]) / dt < stationary_px:
            stopped += dt
    if np.linalg.norm(d) < 0.05:
        direction = "stayed in place"
    elif abs(d[0]) >= abs(d[1]):
        direction = "left to right" if d[0] > 0 else "right to left"
    else:
        direction = "toward the camera" if d[1] > 0 else "away from the camera"
    return {"direction": direction, "stopped_s": round(stopped, 1), "seconds": round(float(ts[-1] - ts[0]), 1),
            "moved": round(float(np.linalg.norm(d)), 3), "start": [round(float(x), 3) for x in pts[0]],
            "end": [round(float(x), 3) for x in pts[-1]]}

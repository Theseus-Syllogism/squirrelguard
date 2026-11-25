"""Draw the detection box, label + score and a timestamp onto an event snapshot (for push and email)."""
from __future__ import annotations

import datetime as dt

import cv2
import numpy as np

COLORS = {"squirrel": (0, 165, 255), "person": (255, 120, 0)}  # BGR, same as the clip renderer


def annotate(jpeg: bytes, label: str, score: float, when: float, box_abs=None, box_rel=None,
             min_width: int = 1280) -> bytes | None:
    """box_abs = [x1, y1, x2, y2] in snapshot pixels; box_rel = [x, y, w, h] as fractions (Frigate API)."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if box_abs is None and box_rel:
        x, y, bw, bh = box_rel
        box_abs = [x * w, y * h, (x + bw) * w, (y + bh) * h]
    # enlarge small detect-stream snapshots so text stays legible in mail clients (no new detail, just size)
    s = max(1.0, min_width / w)
    if s > 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
    lab = (label or "object").lower()
    c = COLORS.get(lab, (80, 220, 80))
    font, th = cv2.FONT_HERSHEY_SIMPLEX, max(2, round(w / 640))
    fs = 0.5 * th
    if box_abs:
        x1, y1, x2, y2 = (int(v * s) for v in box_abs)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, th + 1)
        txt = f"{lab} {score:.0%}"
        (tw, tth), base = cv2.getTextSize(txt, font, fs, th)
        ty = y1 - 6 if y1 - tth - 12 > 0 else y1 + tth + 12
        cv2.rectangle(img, (x1, ty - tth - 6), (x1 + tw + 10, ty + base), c, -1)
        cv2.putText(img, txt, (x1 + 5, ty - 2), font, fs, (255, 255, 255), th, cv2.LINE_AA)
    stamp = dt.datetime.fromtimestamp(when).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    (tw, tth), base = cv2.getTextSize(stamp, font, fs, th)
    x0, y0 = w - tw - 20, h - 14
    cv2.rectangle(img, (x0 - 10, y0 - tth - 10), (w - 6, h - 4), (0, 0, 0), -1)
    cv2.putText(img, stamp, (x0, y0), font, fs, (255, 255, 255), th, cv2.LINE_AA)
    ok, out = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out.tobytes() if ok else None

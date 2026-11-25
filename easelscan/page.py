"""Easel page geometry and image cleanup: frames from the camera -> one straightened, evenly lit page.

Only the page crop leaves this module: the full camera frame (room, people) is never stored.
Calibration (calib.json): {"corners_upright": [[x,y] TL, TR, BR, BL], "rotate": "ccw90" | "none", "out_px": [w, h]}
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

SOI, EOI = b"\xff\xd8", b"\xff\xd9"


def split_jpegs(blob: bytes) -> list[np.ndarray]:
    """Decode a stream of concatenated JPEG frames (ffmpeg -f mjpeg output)."""
    frames, i = [], 0
    while True:
        a = blob.find(SOI, i)
        if a < 0:
            break
        b = blob.find(EOI, a + 2)
        if b < 0:
            break
        img = cv2.imdecode(np.frombuffer(blob[a:b + 2], np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
        i = b + 2
    return frames


def load_calib(path: str | Path) -> dict:
    c = json.loads(Path(path).read_text())
    c["corners_upright"] = np.float32(c["corners_upright"])
    return c


def upright(frame: np.ndarray, calib: dict) -> np.ndarray:
    rot = str(calib.get("rotate", "ccw90"))
    if "counter" in rot or rot.startswith("ccw"):
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rot.startswith("cw"):
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def straighten(frame: np.ndarray, calib: dict, scale: float = 1.0) -> np.ndarray:
    """Perspective-correct the page. scale > 1 resamples the page larger (interpolation only)."""
    tl, tr, br, bl = calib["corners_upright"]
    w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)) * scale))
    h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)) * scale))
    M = cv2.getPerspectiveTransform(calib["corners_upright"], np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return cv2.warpPerspective(upright(frame, calib), M, (w, h), flags=cv2.INTER_CUBIC)


def average(frames: list[np.ndarray]) -> np.ndarray:
    """Median of a few frames of the still page: removes sensor noise and JPEG blocking, keeps the strokes."""
    return np.median(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)


def clean(page_bgr: np.ndarray, gamma: float = 1.6) -> np.ndarray:
    """Flatten uneven lighting (divide by a blurred background) and deepen the strokes. Returns greyscale."""
    g = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = max(31, (min(g.shape) // 15) | 1)
    bg = cv2.medianBlur(cv2.GaussianBlur(g, (0, 0), 3).astype(np.uint8), k).astype(np.float32) + 1
    return np.clip((g / bg) ** gamma * 245, 0, 255).astype(np.uint8)


def ink_mask(clean_page: np.ndarray) -> np.ndarray:
    m = (clean_page < 150).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def occluded(page_bgr: np.ndarray, frames: list[np.ndarray] | None = None) -> float:
    """Share of the page covered by something that is not paper+ink (a person, a hand): large dark or
    coloured blobs, or motion between the frames. 0 = clear."""
    hsv = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2HSV)
    dark = (hsv[..., 2] < 70).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))    # strokes are thin; bodies are not
    share = float(dark.mean())
    if frames is not None and len(frames) >= 2:
        a = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.int16)
        b = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY).astype(np.int16)
        share = max(share, float((np.abs(a - b) > 40).mean()) * 4)
    return share


def changed(prev_clean: np.ndarray | None, cur_clean: np.ndarray, min_share: float = 0.004) -> tuple[bool, float]:
    """Has the writing changed? Compare ink masks, tolerant of a pixel or two of camera shift."""
    if prev_clean is None:
        return True, 1.0
    if prev_clean.shape != cur_clean.shape:
        prev_clean = cv2.resize(prev_clean, cur_clean.shape[::-1])
    a, b = ink_mask(prev_clean), ink_mask(cur_clean)
    k = np.ones((5, 5), np.uint8)
    new = (b & (1 - cv2.dilate(a, k))).mean()
    gone = (a & (1 - cv2.dilate(b, k))).mean()
    share = float(new + gone)
    return share >= min_share, share

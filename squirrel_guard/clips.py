"""Burn detection boxes into a Frigate event clip and re-encode it small enough for a phone notification.

Frigate's saved clips are plain video; boxes only exist in its live UI and snapshots. We re-run our detector over
the clip (every Nth output frame, boxes held in between), draw them, and encode H.264 with ffmpeg.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .detector import Detector

log = logging.getLogger(__name__)

# Open Images class names -> the labels we show. Everything else is ignored.
ALIASES = {"man": "person", "woman": "person", "boy": "person", "girl": "person", "person": "person"}
COLORS = {"squirrel": (0, 165, 255), "person": (255, 120, 0)}  # BGR; others get a default


def _label(raw: str) -> str:
    l = raw.lower()
    return ALIASES.get(l, l)


def render(clip_mp4: bytes, detector: Detector, keep: set[str], out_width: int = 960, out_fps: float = 12,
           detect_every: int = 2, max_seconds: float = 30, crf: int = 28, max_bytes: int = 9_500_000) -> bytes | None:
    """Frames are streamed into ffmpeg as they are drawn (no whole-clip buffer)."""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.mp4"
        src.write_bytes(clip_mp4)
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            log.warning("cannot open clip")
            return None
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, round(src_fps / out_fps))
        limit = int(max_seconds * src_fps)
        dst = Path(td) / "out.mp4"
        enc = None
        boxes, i, k, W, H = [], 0, 0, 0, 0
        while i < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                if k % detect_every == 0:
                    boxes = [d for d in detector(frame) if _label(d.label) in keep]
                h, w = frame.shape[:2]
                scale = out_width / w
                out = cv2.resize(frame, (out_width, int(round(h * scale / 2)) * 2), interpolation=cv2.INTER_AREA)
                th = max(1, round(2 * out_width / 960))
                for d in boxes:
                    lab = _label(d.label)
                    x1, y1, x2, y2 = (int(v * scale) for v in d.box)
                    c = COLORS.get(lab, (80, 220, 80))
                    cv2.rectangle(out, (x1, y1), (x2, y2), c, th)
                    txt = f"{lab} {d.score:.0%}"
                    (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(out, (x1, max(0, y1 - tht - 8)), (x1 + tw + 6, y1), c, -1)
                    cv2.putText(out, txt, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                if enc is None:
                    H, W = out.shape[:2]
                    enc = subprocess.Popen(
                        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                         "-s", f"{W}x{H}", "-r", str(out_fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                         "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
                        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                enc.stdin.write(out.tobytes())
                k += 1
            i += 1
        cap.release()
        if enc is None:
            log.warning("clip had no frames")
            return None
        enc.stdin.close()
        if enc.wait() != 0:
            log.warning("ffmpeg failed: %s", enc.stderr.read().decode(errors="replace")[-300:])
            return None
        data = dst.read_bytes()
        q = crf
        while len(data) > max_bytes and q < crf + 15:  # shrink by re-encoding the rendered file
            q += 5
            smaller = Path(td) / f"out{q}.mp4"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(dst), "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", str(q), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            str(smaller)], check=False)
            if smaller.exists():
                data = smaller.read_bytes()
        if len(data) > max_bytes:
            log.warning("clip still %.1f MB after re-encoding", len(data) / 1e6)
            return None
        log.info("clip rendered: %d frames, %dx%d, %.1f MB (crf %d)", k, W, H, len(data) / 1e6, q)
        return data

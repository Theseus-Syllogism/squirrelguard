"""RTSP frame source: ffmpeg decodes the stream and hands us raw BGR frames at a fixed rate."""
from __future__ import annotations

import logging
import subprocess
import time
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)


def frames(url: str, width: int, height: int, fps: float, ffmpeg: str = "ffmpeg") -> Iterator[np.ndarray]:
    """Yield HxWx3 uint8 BGR frames forever, reconnecting when the stream drops."""
    frame_bytes = width * height * 3
    backoff = 2.0
    while True:
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp", "-i", url,
            "-an", "-vf", f"fps={fps},scale={width}:{height}",
            "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        got_any = False
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                got_any = True
                backoff = 2.0
                yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
        finally:
            proc.kill()
            err = proc.stderr.read().decode(errors="replace").strip()
            proc.wait()
        log.warning("stream ended%s; reconnecting in %.0fs", f": {err[-300:]}" if err else "", backoff)
        time.sleep(backoff)
        if not got_any:
            backoff = min(backoff * 2, 60.0)

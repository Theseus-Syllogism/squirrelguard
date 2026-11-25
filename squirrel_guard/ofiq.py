"""OFIQ (BSI reference implementation of ISO/IEC 29794-5, MIT; built in ~/src/ofiq by its own build) as an optional,
explainable per-frame face-quality source for faceseq_agg (extra weight) and for evidence records.

On surveillance faces OFIQ's unified score (MagFace-based) is ~0-2 for everything, so it is NOT used; the component
measures are: Sharpness, HeadPoseYaw/Pitch/RollFrontal, IlluminationUniformity, NoCompressionArtifacts,
InterEyeDistance. Crops are upscaled (bicubic, for OFIQ's face detector only) before scoring. ~0.45 s per crop with
4 threads: offline / evidence use, not the live path.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

OFIQ_DIR = Path(os.environ.get("OFIQ_DIR", Path.home() / "src/ofiq"))
USE = ("Sharpness", "HeadPoseYawFrontal", "HeadPosePitchFrontal", "IlluminationUniformity", "NoCompressionArtifacts")


def score(crops_bgr: list[np.ndarray], threads: int = 4, up_to: int = 400) -> list[dict]:
    """Per crop: {measure: scalar 0-100 or None} for the USE measures, plus 'ok' and 'quality' (geometric mean of
    the available USE scalars / 100, 0 when OFIQ found no face)."""
    with tempfile.TemporaryDirectory() as d:
        for i, c in enumerate(crops_bgr):
            z = max(1.0, up_to / max(c.shape[:2]))
            cv2.imwrite(f"{d}/{i:05d}.png", cv2.resize(c, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC))
        out = Path(d) / "scores.json"
        subprocess.run(["python3", str(OFIQ_DIR / "score_dir.py"), d, "--out", str(out), "--threads", str(threads),
                        "--det-conf", "0.15", "-q"], check=True, capture_output=True)
        res = {Path(r["file"]).stem: r for r in json.loads(out.read_text())["results"]}
    rows = []
    for i in range(len(crops_bgr)):
        r = res.get(f"{i:05d}", {})
        m = {k: (r.get("measures", {}).get(k) or {}).get("scalar") for k in USE}
        vals = [v / 100 for v in m.values() if v is not None]
        rows.append({**m, "ok": bool(r.get("ok")),
                     "quality": float(np.exp(np.mean(np.log(np.clip(vals, 0.01, 1))))) if r.get("ok") and vals else 0.0})
    return rows

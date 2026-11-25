#!/usr/bin/env python
"""Whole-body keypoints for every camera of a multi-camera clip set (input to scripts/rigfit.py).

  .venv/bin/python scripts/rigfit_keypoints.py OUT.npz --cam v4=V4.mp4 --cam frontyard=FY.mp4 [--fps 8] [--max-s 32]

Per camera: frames decoded at --fps (kinematics.decode, the same sampling as scripts/multicam_demo.py), the one person
picked per frame by the YOLO pose detector (multicam_demo.single_person), then RTMW 133-point whole-body keypoints in
that box (squirrel_guard.wholebody, IR/grayscale handled there). Writes OUT.npz: for each camera NAME,
NAME_kp (frames, 133, 3: x, y, raw SimCC score; NaN where no person), NAME_box (frames, 4), NAME_size (W, H),
NAME_fps, NAME_ir (frames, bool: grayscale frame).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard import kinematics, rigpipe  # noqa: E402
from squirrel_guard.pose import PoseEstimator  # noqa: E402
from squirrel_guard.wholebody import WholeBody  # noqa: E402

RTMW = ROOT / "models/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122/end2end.onnx"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cam", action="append", required=True, help="NAME=CLIP")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--max-s", type=float, default=32.0)
    ap.add_argument("--model", default=str(RTMW))
    ap.add_argument("--ir-mode", default="auto", choices=["auto", "on", "off"])
    a = ap.parse_args()
    pose = PoseEstimator(str(ROOT / "models/yolov8s-pose.onnx"))
    wb = WholeBody(a.model, ir_mode=a.ir_mode)
    res = {}
    for spec in a.cam:
        name, clip = spec.split("=", 1)
        t0 = time.time()
        frames, fps = kinematics.decode(Path(clip).read_bytes(), a.fps, a.max_s)
        kpd = rigpipe.extract(frames, pose, wb)
        res[f"{name}_kp"], res[f"{name}_box"] = kpd["kp"], kpd["box"]
        res[f"{name}_size"], res[f"{name}_fps"], res[f"{name}_ir"] = np.array(kpd["size"]), np.array(fps), kpd["ir"]
        res[f"{name}_calibrated"] = np.array(kpd["calibrated"])
        irs, kps = kpd["ir"], kpd["kp"]
        n_p = int(np.isfinite(res[f"{name}_box"][:, 0]).sum())
        print(f"{name}: {len(kps)} frames @ {fps:.2f} fps, person in {n_p}, IR frames {int(np.sum(irs))}, "
              f"{time.time() - t0:.0f} s")
    np.savez_compressed(a.out, **res)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

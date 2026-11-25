#!/usr/bin/env python
"""Benchmark + agreement check: AdaFace and YOLOv8s-pose on onnxruntime CPU (live path) vs TensorRT fp16 (RTX 4070).

  ~/src/splat/.venv/bin/python scripts/trt_bench.py      (needs onnxruntime in that venv too)
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squirrel_guard.faceid_v2 import AdaFace  # noqa: E402
from squirrel_guard.pose import PoseEstimator  # noqa: E402

M = Path.home() / ".cache/squirrel-guard/models"
crops = list(np.load("data/face/1794339326.27506-2muu3e/crops.npy"))
crops = (crops * 4)[:128]
res = {}
cpu, gpu = AdaFace(M / "adaface_ir101_webface12m.onnx", threads=4), AdaFace(M / "adaface_ir101_webface12m.onnx", backend="trt")
for name, m in (("cpu", cpu), ("trt", gpu)):
    m.embed(crops[:8])
    t = time.time()
    E, N = m.embed(crops, batch=64)
    res[f"adaface_{name}_ms_per_face"] = round((time.time() - t) / len(crops) * 1000, 2)
    res[f"_E_{name}"], res[f"_N_{name}"] = E, N
cos = (res.pop("_E_cpu") * res.pop("_E_trt")).sum(1)
Nc, Ng = res.pop("_N_cpu"), res.pop("_N_trt")
res["adaface_cos_cpu_vs_trt_min"] = round(float(cos.min()), 5)
res["adaface_norm_rel_diff_max"] = round(float(np.abs(Nc - Ng).max() / Nc.mean()), 5)

cap = cv2.VideoCapture("data/face/1794339326.27506-2muu3e.mp4")
frames = []
while len(frames) < 40:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
pc, pg = PoseEstimator(str(M / "yolov8s-pose.onnx"), threads=4), PoseEstimator(str(M / "yolov8s-pose.onnx"), backend="trt")
for name, p in (("cpu", pc), ("trt", pg)):
    p(frames[0])
    t = time.time()
    out = [p(f) for f in frames]
    res[f"pose_{name}_ms_per_frame"] = round((time.time() - t) / len(frames) * 1000, 1)
    res[f"_P_{name}"] = out


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


P1, P2 = res.pop("_P_cpu"), res.pop("_P_trt")
ious, kpd, cnt = [], [], 0
for a, b in zip(P1, P2):
    cnt += len(a) != len(b)
    for p in a:
        if b:
            q = max(b, key=lambda q: iou(p.box, q.box))
            ious.append(iou(p.box, q.box))
            vis = (p.kp[:, 2] > 0.5) & (q.kp[:, 2] > 0.5)
            if vis.any():
                kpd.append(float(np.linalg.norm(p.kp[vis, :2] - q.kp[vis, :2], axis=1).mean()))
res["pose_box_iou_min"] = round(float(min(ious)), 4) if ious else None
res["pose_kp_mean_px_diff"] = round(float(np.mean(kpd)), 3) if kpd else None
res["pose_frames_with_count_mismatch"] = cnt
print(json.dumps(res, indent=1))

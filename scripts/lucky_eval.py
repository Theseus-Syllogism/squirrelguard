#!/usr/bin/env python
"""Self-test of the lucky-imaging face composite (squirrel_guard/lucky.py) with real ground truth: shrink a clip
(simulated distance), build the composite from the small faces, and compare it with the TRUE full-resolution face
pixels of the same frame. Baseline = the best single small frame upscaled with bicubic.

  .venv/bin/python scripts/lucky_eval.py data/face/EVENT.mp4 --scale 0.25 --out data/face/EVENT/lucky
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squirrel_guard import lucky  # noqa: E402
from squirrel_guard.faceid_v2 import AdaFace  # noqa: E402
from squirrel_guard.faceseq import FaceSeq  # noqa: E402

MODELS = Path.home() / ".cache/squirrel-guard/models"


def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if m == 0 else float(10 * np.log10(255 ** 2 / m))


def ssim(a, b):
    a, b = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float64) for x in (a, b))
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = cv2.GaussianBlur(a, (7, 7), 1.5), cv2.GaussianBlur(b, (7, 7), 1.5)
    sa = cv2.GaussianBlur(a * a, (7, 7), 1.5) - ma ** 2
    sb = cv2.GaussianBlur(b * b, (7, 7), 1.5) - mb ** 2
    sab = cv2.GaussianBlur(a * b, (7, 7), 1.5) - ma * mb
    return float(np.mean(((2 * ma * mb + C1) * (2 * sab + C2)) / ((ma ** 2 + mb ** 2 + C1) * (sa + sb + C2))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--up", type=int, default=2, help="composite output scale")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    full, small = [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        full.append(f)
        small.append(cv2.resize(f, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA))
    fs = FaceSeq(str(MODELS / "yolov8s-pose.onnx"))
    start = None
    for i, f in enumerate(small):
        ps = fs.pose(f)
        if ps:
            start = (i, max(ps, key=lambda p: (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])).box)
            break
    obs = fs.track(small[start[0]:], fps, start[1])
    for o in obs:
        o.frame += start[0]
    obs = [o for o in obs if o.face_px >= 8]
    _, N = AdaFace(str(MODELS / "adaface_ir101_webface12m.onnx"), threads=4).embed([o.crop112 for o in obs])
    res = lucky.composite(small, obs, N, scale=a.up)
    X0, Y0, side, sc = res["canvas"]
    # truth: the same canvas in the full-resolution frame, resampled to the composite size
    F = full[res["reference_frame"]]
    k = 1 / a.scale
    M = np.array([[sc / k, 0, -X0 * sc], [0, sc / k, -Y0 * sc]], np.float64)
    truth = cv2.warpAffine(cv2.GaussianBlur(F, (0, 0), 0.35 * k / sc + 1e-3) if k / sc > 1 else F, M,
                           (side * sc, side * sc), flags=cv2.INTER_AREA)
    rows = {}
    ref_o = next(o for o in obs if o.frame == res["reference_frame"])
    cx, cy = (ref_o.pts5.mean(0) - np.array([X0, Y0])) * sc
    mask = np.zeros(truth.shape[:2], np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy - 0.05 * ref_o.face_px * sc)),
                (int(0.42 * ref_o.face_px * sc), int(0.55 * ref_o.face_px * sc)), 0, 0, 360, 1, -1)
    m = mask.astype(bool)
    for name in ("reference_bicubic", "stack_mean", "composite"):
        d = (res[name].astype(np.float64) - truth.astype(np.float64))[m]
        rows[name] = {"psnr_all": round(psnr(res[name], truth), 2), "ssim_all": round(ssim(res[name], truth), 4),
                      "psnr_face": round(float(10 * np.log10(255 ** 2 / max(np.mean(d ** 2), 1e-9))), 2)}
    u = res["uncertainty"]
    um = cv2.applyColorMap(np.clip(u * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    big = lambda x: cv2.resize(x, (256, 256), interpolation=cv2.INTER_NEAREST)
    sheet = np.hstack([big(res["reference_bicubic"]), big(res["stack_mean"]), big(lucky.stamp(res["composite"])),
                       big(truth), big(um)])
    cv2.imwrite(str(out / f"sheet_s{a.scale}.png"), sheet)
    rep = {"consistency_db": res["consistency_db"], "face_uncertainty": res["face_uncertainty"],
           "suspect": res["suspect"], "scale": a.scale, "face_px_ref": round(float(ref_o.face_px), 1), "frames_used": len(res["frames_used"]),
           "hole_frac": res["hole_frac"], "metrics_vs_truth": rows}
    (out / f"report_s{a.scale}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()

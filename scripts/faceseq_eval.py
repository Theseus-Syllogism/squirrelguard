#!/usr/bin/env python
"""Run the small-face front end (squirrel_guard/faceseq.py) on a clip and report every face observation.

  .venv/bin/python scripts/faceseq_eval.py data/face/EVENT.mp4 --out data/face/EVENT
    [--pose ~/.cache/.../yolov8s-pose.onnx] [--adaface ~/.cache/squirrel-guard/models/adaface_ir101_webface12m.onnx]

Writes <out>/obs.json (per frame: face px, YuNet score, sharpness, yaw proxy, AdaFace norm), <out>/crops.png
(aligned 112x112 crops in time order, labelled with face px) and <out>/emb.npy (AdaFace embeddings, L2-normalised).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squirrel_guard.faceid_v2 import AdaFace  # noqa: E402
from squirrel_guard.faceseq import FaceSeq  # noqa: E402

MODELS = Path.home() / ".cache/squirrel-guard/models"


def read(clip: Path, step: int, scale: float = 1.0):
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            if scale != 1.0:      # simulated distance: antialiased shrink of the whole frame
                f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(f)
        i += 1
    return frames, fps / step


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pose", default=str(MODELS / "yolov8s-pose.onnx"))
    ap.add_argument("--adaface", default=str(MODELS / "adaface_ir101_webface12m.onnx"))
    ap.add_argument("--step", type=int, default=2, help="use every Nth frame")
    ap.add_argument("--scale", type=float, default=1.0, help="shrink frames (simulated distance x 1/scale)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, fps = read(Path(a.clip), a.step, a.scale)
    fs = FaceSeq(a.pose)
    start = None
    for i, f in enumerate(frames):                      # first frame with a person: biggest one
        H, W = f.shape[:2]
        ps = fs.pose(f)
        if ps:
            start = (i, max(ps, key=lambda p: (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])).box)
            break
    if start is None:
        print("no person found")
        return
    obs = fs.track(frames[start[0]:], fps, start[1])
    for o in obs:
        o.frame += start[0]
    if not obs:
        print("person found, but no face in any frame")
        return
    ada = AdaFace(a.adaface, threads=4)
    E, N = ada.embed([o.crop112 for o in obs])
    np.save(out / "emb.npy", E)
    np.save(out / "crops.npy", np.stack([o.crop112 for o in obs]))      # aligned BGR 112x112 (camera pixels)
    rows = [{"frame": o.frame, "t": round(o.frame / fps, 2), "face_px": round(o.face_px, 1),
             "yunet": round(o.det_score, 3), "sharp": round(o.sharpness, 1), "yaw_proxy": round(o.yaw_proxy, 2),
             "ada_norm": round(float(n), 2)} for o, n in zip(obs, N)]
    (out / "obs.json").write_text(json.dumps(rows, indent=1))
    tiles = []
    for o, n in zip(obs, N):
        t = o.crop112.copy()
        cv2.putText(t, f"{o.face_px:.0f}px n{n:.0f}", (2, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        tiles.append(t)
    cols = 10
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
    cv2.imwrite(str(out / "crops.png"), sheet)
    fp = np.array([o.face_px for o in obs])
    print(f"{len(frames)} frames, {len(obs)} faces; face px min/med/max {fp.min():.0f}/{np.median(fp):.0f}/"
          f"{fp.max():.0f}; AdaFace norm min/med/max {N.min():.1f}/{np.median(N):.1f}/{N.max():.1f}")


if __name__ == "__main__":
    main()

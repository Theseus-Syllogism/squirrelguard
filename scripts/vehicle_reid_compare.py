"""Offline comparison: PP-ShiTuV2 general recognition vs DINOv2 ViT-S/14 as the vehicle appearance fingerprint.

Nothing in production changes: this only prints numbers. Production matching stays on DINOv2 (squirrel_guard/vehicle.py).

Usage: .venv/bin/python scripts/vehicle_reid_compare.py [--fps 3] clip.mp4 ...   (or a directory of crops named
       <vehicle-id>_<anything>.jpg, e.g. exported vehicle sightings, with --crops DIR)
  Clips: vehicles are detected (models/yolov8s-oiv7.onnx) and linked frame-to-frame by box overlap into tracklets; one
  tracklet = one vehicle identity. Positive pairs = two crops of the same tracklet at least --gap seconds apart (the
  view changes as the car moves); negatives = crops of different tracklets. This is easier than real re-ID (same
  lighting, same minute), so it ranks the embeddings rather than predicting production accuracy.
Reported per model: mean cosine same/different, ROC AUC, rank-1 (query crop vs all crops of other tracklets plus the
same tracklet's crops >= gap away), the "different" cosine at the 1 % false-match point, and ms per crop (4 threads).

Models: models/dinov2_vits14_224.onnx (scripts/export_model.py) and models/ppshituv2_general_rec.onnx
(scripts/export_pulc.py --models shitu; PP-ShiTuV2 PPLCNetV2_base, Apache-2.0, 224x224 plain resize, ImageNet norm).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from squirrel_guard.detector import Detector  # noqa: E402
from squirrel_guard.vehicle import VehicleEmbedder, vehicle_label  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class ShiTu:
    def __init__(self, path: str, threads: int = 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, crops: list) -> np.ndarray:
        x = np.stack([((cv2.resize(c, (224, 224))[:, :, ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
                      for c in crops]).astype(np.float32)
        out = np.concatenate([self.sess.run(None, {self.inp: x[i:i + 16]})[0] for i in range(0, len(x), 16)])
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def tracklets(path: Path, det: Detector, fps: float, min_side: int) -> list[list[tuple[float, np.ndarray]]]:
    cap = cv2.VideoCapture(str(path))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25
    step, i, live, done = max(1, round(vfps / fps)), 0, [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            if fr.shape[1] > 1920:
                fr = cv2.resize(fr, (1920, int(fr.shape[0] * 1920 / fr.shape[1])))
            t = i / vfps
            boxes = [d.box for d in det(fr) if vehicle_label(d.label)
                     and min(d.box[2] - d.box[0], d.box[3] - d.box[1]) >= min_side]
            nxt = []
            for b in boxes:
                best = max(live, key=lambda tr: iou(tr["box"], b), default=None)
                if best is not None and iou(best["box"], b) >= 0.3 and best not in nxt:
                    best["box"] = b
                    best["crops"].append((t, fr[b[1]:b[3], b[0]:b[2]].copy()))
                    nxt.append(best)
                else:
                    nxt.append({"box": b, "crops": [(t, fr[b[1]:b[3], b[0]:b[2]].copy())]})
            done += [tr for tr in live if tr not in nxt]
            live = nxt
        i += 1
    cap.release()
    return [tr["crops"] for tr in done + live if len(tr["crops"]) >= 3]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    s = np.r_[pos, neg]
    r = s.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(E: np.ndarray, ids: np.ndarray, ts: np.ndarray, gap: float) -> dict:
    S = E @ E.T
    n = len(E)
    same = ids[:, None] == ids[None]
    far = np.abs(ts[:, None] - ts[None]) >= gap
    iu = np.triu_indices(n, 1)
    pos = S[iu][(same & far)[iu]]
    neg = S[iu][(~same)[iu]]
    hits = tot = 0
    for q in range(n):
        cand = (~same[q]) | (same[q] & far[q])
        cand[q] = False
        if not (same[q] & far[q]).any() or cand.sum() < 2:
            continue
        j = np.flatnonzero(cand)[S[q, cand].argmax()]
        hits += ids[j] == ids[q]
        tot += 1
    return {"pairs_same": len(pos), "pairs_diff": len(neg), "cos_same": round(float(pos.mean()), 3),
            "cos_diff": round(float(neg.mean()), 3), "auc": round(auc(pos, neg), 4),
            "diff_p99": round(float(np.quantile(neg, 0.99)), 3),
            "same_below_diff_p99": round(float((pos < np.quantile(neg, 0.99)).mean()), 3),
            "rank1": round(hits / max(tot, 1), 3), "queries": tot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--crops", help="directory of <id>_*.jpg crops instead of clips")
    ap.add_argument("--fps", type=float, default=3)
    ap.add_argument("--gap", type=float, default=1.5, help="seconds between crops of a positive pair")
    ap.add_argument("--min-side", type=int, default=64)
    ap.add_argument("--max-per-track", type=int, default=12)
    a = ap.parse_args()
    crops, ids, ts = [], [], []
    if a.crops:
        for k, f in enumerate(sorted(Path(a.crops).glob("*.jpg"))):
            crops.append(cv2.imread(str(f)))
            ids.append(f.stem.split("_")[0])
            ts.append(k * 1000.0)   # different sightings: always "far apart"
    det = Detector(str(ROOT / "models" / "yolov8s-oiv7.onnx"), conf=0.35, threads=4)
    for c in a.clips:
        trs = tracklets(Path(c).expanduser(), det, a.fps, a.min_side)
        print(f"{Path(c).name}: {len(trs)} tracklets, {sum(len(t) for t in trs)} crops")
        for k, tr in enumerate(trs):
            sel = np.linspace(0, len(tr) - 1, min(len(tr), a.max_per_track)).round().astype(int)
            for j in sorted(set(sel)):
                crops.append(tr[j][1])
                ids.append(f"{Path(c).stem}#{k}")
                ts.append(tr[j][0])
    ids_a, ts_a = np.array(ids), np.array(ts)
    print(f"total: {len(crops)} crops, {len(set(ids))} vehicles")
    models = {"dinov2_vits14 (production)": VehicleEmbedder(str(ROOT / "models" / "dinov2_vits14_224.onnx"), threads=4),
              "ppshituv2_general_rec": ShiTu(str(ROOT / "models" / "ppshituv2_general_rec.onnx"), threads=4)}
    for name, m in models.items():
        t = time.perf_counter()
        E = np.concatenate([m(crops[i:i + 16]) for i in range(0, len(crops), 16)])
        ms = (time.perf_counter() - t) * 1000 / len(crops)
        print(f"{name:28s} dim {E.shape[1]:4d} {ms:6.1f} ms/crop  {evaluate(E, ids_a, ts_a, a.gap)}")


if __name__ == "__main__":
    main()

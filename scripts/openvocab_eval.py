"""Run the open-vocabulary object pass on clips exactly as the profiler does, print what it finds and how long it took.

Usage: .venv/bin/python scripts/openvocab_eval.py --model models/yolov8s-worldv2-sg.onnx clip.mp4 [clip.mp4 ...]
         [--pose models/yolov8s-pose.onnx] [--threads 4] [--every 2] [--min-score 0.25] [--sheet DIR]
         [--bench]   per-frame latency at 1 and 4 threads on a 1080p frame instead
Per clip: pose track at 8 fps (kinematics.decode/track), object pass on every --every-th analysed frame around the
tracked person (whole frame when nobody is tracked), ObjectAggregator summary. --sheet DIR writes <clip>.jpg per clip: the
best-scoring crop of every label (green box = carried, yellow = near, red = scene; white = the tracked person), for
eyeballing what each hit really is.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squirrel_guard.kinematics import decode, track  # noqa: E402
from squirrel_guard.openvocab import ObjectAggregator, OpenVocabDetector, describe, relation  # noqa: E402
from squirrel_guard.pose import PoseEstimator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COL = {"carried": (0, 220, 0), "near": (0, 220, 220), "scene": (0, 0, 230)}


def bench(model: str, frame: np.ndarray, n: int = 30) -> None:
    for th in (1, 4):
        det = OpenVocabDetector(model, threads=th)
        for _ in range(3):
            det.detect(frame)
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            det.detect(frame)
            ts.append(time.perf_counter() - t)
        print(f"{Path(model).name} {det.size}px threads={th}: median {np.median(ts) * 1000:.0f} ms, "
              f"p90 {np.percentile(ts, 90) * 1000:.0f} ms per frame")


def run(clip: Path, det: OpenVocabDetector, pose: PoseEstimator, every: int, sheet: str | None) -> None:
    frames, fps = decode(clip.read_bytes(), 8.0, 40.0)
    agg = ObjectAggregator(min_frames=2)
    st = {"i": 0, "t": 0.0, "n": 0, "raw": []}
    best: dict = {}

    def on_frame(fr, p):
        i = st["i"]
        st["i"] += 1
        if i % every:
            return
        t = time.perf_counter()
        dets = det.detect_around(fr, p.box) if p is not None else det.detect(fr)
        st["t"] += time.perf_counter() - t
        st["n"] += 1
        agg.add(dets, p, t=round(i / fps, 2), size=(fr.shape[1], fr.shape[0]))
        st["raw"].extend((round(i / fps, 1), d["label"], d["score"], relation(d["box"], p)) for d in dets)
        if sheet is not None:   # keep the best crop per label: object box + person box with context
            for d in dets:
                if d["score"] > best.get(d["label"], (0,))[0]:
                    best[d["label"]] = (d["score"], crop(fr, d, p, f"{d['label']} {d['score']:.2f} {relation(d['box'], p)}"
                                                        f" t={i / fps:.1f}"))

    t0 = time.perf_counter()
    trk, size = track(frames, pose, on_frame=on_frame)
    tot = time.perf_counter() - t0
    tracked = sum(p is not None for p in trk)
    print(f"\n== {clip.name}: {len(trk)} frames @ {fps:.1f} fps, person on {tracked}, object pass on {st['n']} frames: "
          f"{st['t']:.1f} s ({st['t'] / max(1, st['n']) * 1000:.0f} ms/frame; whole clip incl. pose {tot:.1f} s)")
    raw: dict = {}
    for _, lab, sc, rel in st["raw"]:
        r = raw.setdefault(lab, [0, 0.0, set()])
        r[0] += 1
        r[1] = max(r[1], sc)
        r[2].add(rel)
    print("  raw hits:", ", ".join(f"{k} x{v[0]} max {v[1]:.2f} {sorted(v[2])}" for k, v in
                                   sorted(raw.items(), key=lambda kv: -kv[1][0])) or "none")
    s = agg.summary()
    print("  summary:", describe(s) or "nothing (after min_frames)")
    for o in s:
        print(f"    {o['label']:<14} max {o['score']:.2f} frames {o['frames']}/{o['sampled']} carried "
              f"{o['carried_frames']} ({o['carried_fraction']:.0%}) -> {o['relation']}  prompts {o['prompts']}")
    if sheet is not None and best:
        tiles = [im for _, im in sorted(best.values(), key=lambda v: -v[0])]
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
        out = Path(sheet) / f"{clip.stem}.jpg"
        cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        print("  crops:", out)


def crop(fr, d, p, text, side=320):
    """Square tile around the object box (2x, at least 160 px) with the person box, labelled."""
    x1, y1, x2, y2 = d["box"]
    s = max(160., 2 * max(x2 - x1, y2 - y1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X1, Y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    X2, Y2 = int(min(fr.shape[1], cx + s / 2)), int(min(fr.shape[0], cy + s / 2))
    im = fr.copy()
    if p is not None:
        cv2.rectangle(im, tuple(map(int, p.box[:2])), tuple(map(int, p.box[2:])), (255, 255, 255), 2)
    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), COL[relation(d["box"], p)], 2)
    im = cv2.resize(im[Y1:Y2, X1:X2], (side, side))
    cv2.rectangle(im, (0, 0), (side, 22), (0, 0, 0), -1)
    cv2.putText(im, text, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--model", default=str(ROOT / "models" / "yolov8s-worldv2-sg.onnx"))
    ap.add_argument("--pose", default=str(ROOT / "models" / "yolov8s-pose.onnx"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--min-score", type=float, default=0.25)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    if a.bench:
        frame = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), np.uint8)
        if a.clips:
            cap = cv2.VideoCapture(a.clips[0])
            ok, fr = cap.read()
            frame = fr if ok else frame
        bench(a.model, frame)
        return
    det = OpenVocabDetector(a.model, min_score=a.min_score, threads=a.threads)
    print("vocabulary:", ", ".join(det.vocabulary))
    pose = PoseEstimator(a.pose, threads=a.threads)
    if a.sheet:
        Path(a.sheet).mkdir(parents=True, exist_ok=True)
    for c in a.clips:
        run(Path(c), det, pose, a.every, a.sheet)

if __name__ == "__main__":
    main()

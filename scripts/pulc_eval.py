"""PULC attribute models on test footage: CPU latency per crop and the visit summaries the profiler would send.

Usage: .venv/bin/python scripts/pulc_eval.py [--frames 12] [--max-crops 6] [--save-crops DIR] clip_or_image ...
  Each argument is a video (frames sampled evenly, like the profiler) or an image / a directory of images (one visit).
  People are found with the profiler's detector (models/yolov8s-oiv7.onnx); vehicles too when --vehicles is given.
  --latency also times one crop at batch 1 with 1 and 4 intra-op threads (median of 50 runs after warm-up).
  Person crops pass the profiler's full-body gate (upright, not cut by the frame edge) unless --no-gate.
  --ir simulates night IR footage (grey, darker, sensor noise) to see how the attributes degrade.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from squirrel_guard.detector import Detector  # noqa: E402
from squirrel_guard.profiler import PEOPLE, frames_from_clip, full_body  # noqa: E402
from squirrel_guard.pulc import PersonAttributes, VehicleAttributes, describe_person  # noqa: E402
from squirrel_guard.vehicle import vehicle_label  # noqa: E402

MODELS = ROOT / "models"


def frames_of(p: Path, n: int) -> list:
    if p.is_dir():
        return [cv2.imread(str(f)) for f in sorted(p.iterdir()) if f.suffix.lower() in (".jpg", ".png", ".jpeg")]
    if p.suffix.lower() in (".jpg", ".png", ".jpeg"):
        return [cv2.imread(str(p))]
    return frames_from_clip(p.read_bytes(), n)


def latency(model_path: str, cls, wh: tuple) -> dict:
    crop = np.random.default_rng(0).integers(0, 255, (wh[1] * 2, wh[0] * 2, 3), np.uint8)
    out = {}
    for th in (1, 4):
        m = cls(model_path, threads=th)
        for _ in range(5):
            m.raw([crop])
        ts = []
        for _ in range(50):
            t = time.perf_counter()
            m.raw([crop])
            ts.append((time.perf_counter() - t) * 1000)
        out[f"{th}_thread_ms"] = round(statistics.median(ts), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--max-crops", type=int, default=6)
    ap.add_argument("--min-height", type=int, default=64)
    ap.add_argument("--vehicles", action="store_true")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--save-crops")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ir", action="store_true", help="greyscale + gamma 1.8 + noise, like the Wyze at night")
    ap.add_argument("--no-gate", action="store_true", help="keep crops the profiler's full-body gate would drop")
    ap.add_argument("--min-aspect", type=float, default=1.5)
    a = ap.parse_args()
    pm, vm = str(MODELS / "pulc_person_attribute.onnx"), str(MODELS / "pulc_vehicle_attribute.onnx")
    if a.latency:
        print("latency person  (crop -> probs, batch 1):", latency(pm, PersonAttributes, (192, 256)))
        print("latency vehicle (crop -> probs, batch 1):", latency(vm, VehicleAttributes, (256, 192)))
    det = Detector(str(MODELS / "yolov8s-oiv7.onnx"), conf=0.3, threads=4)
    pa, va = PersonAttributes(pm, threads=4), VehicleAttributes(vm, threads=4)
    for inp in a.inputs:
        p = Path(inp).expanduser()
        frames = [f for f in frames_of(p, a.frames) if f is not None]
        if a.ir:
            rng = np.random.default_rng(0)
            g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255 for f in frames]
            frames = [cv2.cvtColor(np.clip((x ** 1.8) * 255 + rng.normal(0, 6, x.shape), 0, 255).astype(np.uint8),
                                   cv2.COLOR_GRAY2BGR) for x in g]
        pcrops, vcrops = [], []
        for fr in frames:
            ds = det(fr)
            ppl = [d for d in ds if d.label.lower() in PEOPLE]
            if ppl:
                d = max(ppl, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
                x1, y1, x2, y2 = d.box
                if y2 - y1 >= a.min_height and (a.no_gate or full_body(d.box, fr.shape, a.min_aspect)):
                    pcrops.append((d.score * (y2 - y1), fr[y1:y2, x1:x2]))
            if a.vehicles:
                vs = [d for d in ds if vehicle_label(d.label)]
                if vs:
                    d = max(vs, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
                    x1, y1, x2, y2 = d.box
                    if min(x2 - x1, y2 - y1) >= a.min_height:
                        vcrops.append(((x2 - x1) * (y2 - y1), fr[y1:y2, x1:x2]))
        print(f"\n== {p.name}{' (IR sim)' if a.ir else ''}: {len(frames)} frames, {len(pcrops)} person crops, {len(vcrops)} vehicle crops")
        if pcrops:
            per = pa.probs([c for _, c in pcrops])
            for (w, c), pr in zip(pcrops, per):
                o = max(("Front", "Side", "Back"), key=pr.get)
                top = sorted(((k, v) for k, v in pr.items() if k not in ("Front", "Side", "Back")), key=lambda kv: -kv[1])[:5]
                print(f"   crop {c.shape[1]}x{c.shape[0]} {o:5s} " + ", ".join(f"{k} {v:.2f}" for k, v in top))
            v = pa.visit([c for _, c in pcrops], [w for w, _ in pcrops], a.max_crops)
            print("   VISIT", describe_person(v))
            print("   ", json.dumps({k: v[k] for k in ("orientation", "views", "apparent", "tags", "spread")}))
            if a.json:
                print(json.dumps(v))
            if a.save_crops:
                d = Path(a.save_crops)
                d.mkdir(parents=True, exist_ok=True)
                for i, (_, c) in enumerate(pcrops):
                    cv2.imwrite(str(d / f"{p.stem}_{i}.jpg"), c)
        if vcrops:
            v = va.visit([c for _, c in vcrops], [w for w, _ in vcrops], a.max_crops)
            print("   VEHICLE", json.dumps(v))


if __name__ == "__main__":
    main()

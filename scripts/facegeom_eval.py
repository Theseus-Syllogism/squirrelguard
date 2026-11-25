"""Face geometry repeatability: per-feature CV within a person (between visits/segments) vs between persons.

Usage (a Python with mediapipe + inspireface + onnxruntime, e.g. the profiler image's versions):
  python scripts/facegeom_eval.py --model face_landmarker.task \
      --seg suit=faces/suit_199623.mp4@0-4 --seg suit=faces/suit_199623.mp4@4-8 ... [--json out.json] [--images DIR]
  A clip spec is PATH[@START-END] in seconds; relative paths are looked up under ~/.cache/squirrel-guard-test/.

Per segment ("visit"): the pose track at --fps, FaceSampler (no budget) with a GeomCollector that keeps EVERY
near-frontal face, so per-face statistics are available; the visit value is the production aggregate (median of the
--keep best faces by faceid weight). Reported per feature:
  - within-person CV% between visits (sd of visit values / mean, pooled over persons with >= 2 visits)
  - between-person CV% (sd of person means / grand mean)
  - ratio between/within (> ~2 = the feature separates these people better than it wobbles; "stable")
  - per-face CV% inside a visit by face size bin (how noisy one frame is)
Depth features can be near zero/negative, so they use the sd in IPD units instead of CV (marked "sd").
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_eval import parse_spec  # noqa: E402
from squirrel_guard import facegeom as fg  # noqa: E402
from squirrel_guard.faceid import AggParams, FaceEngine, FaceSampler  # noqa: E402
from squirrel_guard.kinematics import decode, track  # noqa: E402
from squirrel_guard.pose import PoseEstimator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run_segment(spec, engine, pose, mesher, args):
    path, t0, t1 = parse_spec(spec)
    frames, efps = decode(path.read_bytes(), args.fps, 1e9)
    s = FaceSampler(engine, AggParams(min_px=24), max_attempts=10 ** 6, max_cpu_s=1e9, backoff=False)
    s.geom = fg.GeomCollector(mesher, fg.GeomParams(keep=10 ** 6, min_px=args.min_px))

    def window(gen):
        for i, fr in enumerate(gen):
            t = i / efps
            if t < t0:
                continue
            if t > t1:
                break
            if args.width and fr.shape[1] > args.width:
                fr = cv2.resize(fr, (args.width, round(fr.shape[0] * args.width / fr.shape[1])), interpolation=cv2.INTER_AREA)
            yield fr

    track(window(frames), pose, on_frame=s.on_frame)
    return s.result().get("facegeom") or {}


def visit_value(g, keep):
    faces = (g.get("faces") or [])[:keep]
    if not faces:
        return None, None
    agg = fg.aggregate([f["f"] for f in faces], faces)
    return agg, faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", action="append", default=[], help="LABEL=CLIP[@T0-T1]; same LABEL = same person")
    ap.add_argument("--model", required=True, help="MediaPipe face_landmarker.task")
    ap.add_argument("--pack", default=str(Path.home() / ".inspireface/ms/models/tunmxy--InspireFace/snapshots/master/Megatron"))
    ap.add_argument("--pose", default=str(ROOT / "models/yolov8s-pose.onnx"))
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--width", type=int, default=2560)
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--min-px", type=float, default=32)
    ap.add_argument("--chunk", action="store_true",
                    help="also split each segment's faces (in time order) into pseudo-visits of --keep faces")
    ap.add_argument("--json")
    ap.add_argument("--images")
    args = ap.parse_args()
    pose = PoseEstimator(args.pose, threads=4)
    engine = FaceEngine(args.pack, detect_level=320)
    engine.dense_landmarks = True
    mesher = fg.FaceMesher(args.model)
    if args.images:
        Path(args.images).mkdir(parents=True, exist_ok=True)

    visits = []
    for k, sp in enumerate(args.seg):
        label, spec = sp.split("=", 1)
        g = run_segment(spec, engine, pose, mesher, args)
        agg, faces = visit_value(g, args.keep)
        print(f"SEG {k:2d} {label:6s} {spec:36s} offered {g.get('offered')} off-pose {g.get('off_pose')} "
              f"meshed {g.get('n_faces')} px {g.get('size_px')} {g.get('reason') or ''}", flush=True)
        if args.images and g.get("n_faces"):
            for key, name in (("overlay_jpg_b64", "overlay"), ("depth_jpg_b64", "depth")):
                if g.get(key):
                    Path(args.images, f"{k:02d}_{label}_{name}.jpg").write_bytes(__import__("base64").b64decode(g[key]))
            Path(args.images, f"{k:02d}_{label}_mesh.gltf").write_text(json.dumps(g["mesh"]))
            Path(args.images, f"{k:02d}_{label}.html").write_text(
                "<!doctype html><meta charset=utf-8><body style='background:#222;color:#ddd;font:13px sans-serif'>"
                + fg.render_html(g, show_all=True))
        visits.append({"label": label, "spec": spec, "agg": agg, "all": g.get("faces") or [],
                       "summary": {k2: g.get(k2) for k2 in ("offered", "off_pose", "n_faces", "size_px", "reason")}})

    if args.chunk:          # pseudo-visits: consecutive runs of --keep faces (different moments, poses, distances)
        out = []
        for v in visits:
            fs = sorted(v["all"], key=lambda x: x.get("t", 0))
            for i in range(0, len(fs) - args.keep + 1, args.keep):
                ch = fs[i:i + args.keep]
                out.append({**v, "agg": fg.aggregate([x["f"] for x in ch], ch), "all": ch})
        visits = out
    good = [v for v in visits if v["agg"]]
    persons = sorted({v["label"] for v in good})
    report = {"features": {}, "size_bins": {}, "visits": [{"label": v["label"], "spec": v["spec"], **v["summary"],
                                                          "values": {k: (v["agg"]["features"][k]["value"] if v["agg"] else None)
                                                                     for k in fg.FEATURES}} for v in visits]}
    print(f"\n{'feature':18s} {'mean':>6s} {'within':>8s} {'between':>8s} {'ratio':>6s} {'face-CV <48 / 48-96 / >=96 px':>32s}")
    for f, (kind, _) in fg.FEATURES.items():
        rel = kind != "depth"
        by_p = {p: [v["agg"]["features"][f]["value"] for v in good if v["label"] == p
                    and v["agg"]["features"][f]["value"] is not None] for p in persons}
        allv = [x for xs in by_p.values() for x in xs]
        if not allv:
            continue
        mu = float(np.mean(allv))
        wv = [(np.var(xs, ddof=1), len(xs) - 1) for xs in by_p.values() if len(xs) >= 2]
        within = float(np.sqrt(sum(a * b for a, b in wv) / max(1, sum(b for _, b in wv)))) if wv else None
        means = [np.mean(xs) for xs in by_p.values() if xs]
        between = float(np.std(means, ddof=1)) if len(means) >= 2 else None
        bins = {}
        for name, lo, hi in (("<48", 0, 48), ("48-96", 48, 96), (">=96", 96, 1e9)):
            cvs = []
            for v in good:
                xs = [x["f"][f] for x in v["all"] if lo <= x["size"] < hi]
                if len(xs) >= 3:
                    sd = float(np.std(xs, ddof=1))
                    cvs.append(100 * sd / abs(np.mean(xs)) if rel else sd)
            bins[name] = round(float(np.median(cvs)), 2) if cvs else None
        conv = (lambda x: None if x is None else round(100 * x / abs(mu), 2)) if rel else \
            (lambda x: None if x is None else round(x, 4))
        ratio = round(between / within, 2) if within and between else None
        report["features"][f] = {"mean": round(mu, 4), "unit": "cv%" if rel else "sd", "within": conv(within),
                                 "between": conv(between), "within_sd": None if within is None else round(within, 4),
                                 "ratio": ratio, "face_level": bins}
        print(f"{f:18s} {mu:6.3f} {str(conv(within)):>8s} {str(conv(between)):>8s} {str(ratio):>6s} "
              f"{str(bins['<48']):>10s} {str(bins['48-96']):>10s} {str(bins['>=96']):>10s}  ({'CV%' if rel else 'sd'})")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

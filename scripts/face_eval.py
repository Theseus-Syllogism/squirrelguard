"""Face identification evaluation: enrol from one clip (or part of one), test on others, as the profiler would.

Usage:
  .venv/bin/python scripts/face_eval.py --enroll real/walk.mp4@0-10.4 \
      --test real/walk.mp4@10.4-99 --impostor openvocab/street_21782.mp4 [...] \
      [--pack Megatron|Pikachu|/path/to/pack] [--fps 8] [--max-s 40] [--detect-level 320] [--upscale 0] [--json out.json]
      [--crops DIR]      writes the evidence crops of every template (contact sheets) for eyeballing
  A clip spec is PATH[@START-END] in seconds; relative paths are looked up under ~/.cache/squirrel-guard-test/.

For every clip segment: the pose track at --fps (kinematics.decode/track, exactly as motion_of does), FaceSampler on
the on_frame hook, per-face observations kept for the report (the profiler keeps only the bounded pool).
Reported:
  - per face: size px, quality, |yaw|, liveness, cosine to the ENROLLED template (per-frame scores)
  - per segment: the aggregated visit template's cosine to the enrolled template (aggregated score),
    for genuine (same person) and impostor (other people) segments; the mean of the per-frame scores for comparison
  - quality / liveness / genuine similarity by face size (a proxy for distance to the camera)
  - CPU time per detector attempt and per event
Thresholds for the bridge (FACE_MATCH_SIM) are read off these numbers; see docs/faceid.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squirrel_guard.faceid import AggParams, FaceAggregator, FaceEngine, FaceSampler, l2n  # noqa: E402
from squirrel_guard.kinematics import decode, track  # noqa: E402
from squirrel_guard.pose import PoseEstimator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TESTDATA = Path.home() / ".cache/squirrel-guard-test"


class Recorder:
    """Engine wrapper that keeps a copy of every face observation (evaluation only)."""

    def __init__(self, engine):
        self.engine, self.obs, self.times = engine, [], []
        self.pack = engine.pack

    def faces(self, *a, **k):
        t = time.perf_counter()
        out = self.engine.faces(*a, **k)
        self.times.append(time.perf_counter() - t)
        self.obs.extend(out)
        return out


def parse_spec(spec: str):
    path, _, rng = spec.partition("@")
    p = Path(path).expanduser()
    if not p.is_absolute() and not p.exists():
        p = TESTDATA / path
    t0, t1 = (float(x) for x in rng.split("-")) if rng else (0.0, 1e9)
    return p, t0, t1


def run_segment(spec, engine, pose, args):
    path, t0, t1 = parse_spec(spec)
    frames, efps = decode(path.read_bytes(), args.fps, 1e9)
    params = AggParams(min_px=args.min_px, max_faces=args.max_faces)
    rec = Recorder(engine)
    s = FaceSampler(rec, params, every=args.every, max_attempts=10**6 if args.no_budget else args.max_attempts,
                    max_cpu_s=1e9 if args.no_budget else args.max_cpu_s, upscale_to=args.upscale,
                    backoff=not args.no_budget)

    def window(gen):        # only the requested time range, at most max_s seconds of it
        for i, fr in enumerate(gen):
            t = i / efps
            if t < t0:
                continue
            if t > t1 or t - t0 > args.max_s:
                break
            if args.width and fr.shape[1] > args.width:     # e.g. 4K stock -> the Wyze v4's 2560x1440
                fr = cv2.resize(fr, (args.width, round(fr.shape[0] * args.width / fr.shape[1])), interpolation=cv2.INTER_AREA)
            yield fr

    wall = time.perf_counter()
    cpu = time.process_time()
    trk, _ = track(window(frames), pose, on_frame=s.on_frame)
    res = s.result()
    res["segment_cpu_s"] = round(time.process_time() - cpu, 2)   # decode + pose + faces
    res["frames"] = len(trk)
    res["tracked"] = sum(p is not None for p in trk)
    res["wall_s"] = round(time.perf_counter() - wall, 2)
    res["face_ms_median"] = round(1000 * float(np.median(rec.times)), 1) if rec.times else None
    res["fps_eff"] = round(efps, 2)
    # aggregation variants on the same faces (ablation): pool size, weighting, outlier rejection
    usable = [o for o in rec.obs if o.size >= params.min_px]
    res["variants"] = {}
    if usable:
        for name, kw, w in (("qw_pool24", {"max_faces": 24}, True),
                            ("qw_all", {"max_faces": 10**6}, True),
                            ("qw_all_norej", {"max_faces": 10**6, "outlier_floor": -1.0, "outlier_k": 1e9}, True),
                            ("mean_all_rej", {"max_faces": 10**6}, False)):
            a = FaceAggregator(AggParams(min_px=args.min_px, keep_crops=0, **kw))
            for o in usable:
                a.add(o)
            if not w:          # FASR baseline weights (uniform) but with our outlier rejection
                a.pool = [(1.0, f) for _, f in a.pool]
            res["variants"][name] = a.result()["vec"]
        res["variants"]["mean_all"] = [float(x) for x in l2n(np.mean([l2n(o.vec) for o in usable], 0))]
    for o in rec.obs:
        o.t = t0 + o.t / efps
    return res, rec.obs


def size_bins(rows):
    bins = [(0, 24), (24, 32), (32, 48), (48, 64), (64, 96), (96, 1e9)]
    out = []
    for a, b in bins:
        r = [x for x in rows if a <= x["size"] < b]
        if not r:
            continue
        g = [x["sim"] for x in r if x["kind"] == "genuine"]
        im = [x["sim"] for x in r if x["kind"] == "impostor"]
        lv = [x["live"] for x in r if x["live"] is not None]
        out.append({"size": f"{a:.0f}-{b:.0f}" if b < 1e8 else f">={a:.0f}", "n": len(r),
                    "quality": round(float(np.mean([x["q"] for x in r])), 3),
                    "liveness": round(float(np.mean(lv)), 3) if lv else None,
                    "genuine_sim": round(float(np.mean(g)), 3) if g else None, "n_gen": len(g),
                    "impostor_sim": round(float(np.mean(im)), 3) if im else None, "n_imp": len(im)})
    return out


def sheet(res, path):
    import base64
    ims = [cv2.imdecode(np.frombuffer(base64.b64decode(c), np.uint8), cv2.IMREAD_COLOR) for c in res.get("crops_b64") or []]
    if ims:
        h = 160
        ims = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h)) for i in ims]
        cv2.imwrite(str(path), np.hstack(ims))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", action="append", default=[], help="LABEL=CLIP[@T0-T1]; same LABEL = same person")
    ap.add_argument("--enroll", action="append", default=[], help="shortcut: --seg me=SPEC (first segment)")
    ap.add_argument("--test", action="append", default=[], help="shortcut: --seg me=SPEC")
    ap.add_argument("--impostor", action="append", default=[], help="shortcut: --seg <clip name>=SPEC")
    ap.add_argument("--pack", default="Megatron")
    ap.add_argument("--pose", default=str(ROOT / "models/yolov8s-pose.onnx"))
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--max-s", type=float, default=40)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--max-attempts", type=int, default=150)
    ap.add_argument("--max-cpu-s", type=float, default=8)
    ap.add_argument("--max-faces", type=int, default=96)
    ap.add_argument("--min-px", type=float, default=24)
    ap.add_argument("--detect-level", type=int, default=320)
    ap.add_argument("--upscale", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--width", type=int, default=2560, help="downscale wider frames to this width (0 = native)")
    ap.add_argument("--no-budget", action="store_true", help="analyse every frame (per-frame statistics)")
    ap.add_argument("--json")
    ap.add_argument("--crops")
    args = ap.parse_args()
    segs = [s.split("=", 1) for s in args.seg]
    segs += [("me", s) for s in args.enroll + args.test]
    segs += [(Path(s.split("@")[0]).stem, s) for s in args.impostor]
    pose = PoseEstimator(args.pose, threads=args.threads)
    engine = FaceEngine(args.pack, detect_level=args.detect_level)
    if args.crops:
        Path(args.crops).mkdir(parents=True, exist_ok=True)

    results = []
    for k, (label, spec) in enumerate(segs):
        r, obs = run_segment(spec, engine, pose, args)
        lv = r.get("liveness") or {}
        print(f"SEG {k:2d} {label:10s} {spec:42s} faces {r['n_faces']:2d}/{r['faces_seen']:3d} rej {r.get('n_rejected')} "
              f"q {r.get('quality')} live {lv.get('score')}/{lv.get('ok')} cons {r.get('consistency')} "
              f"size {r.get('size_px')} (max {r.get('max_size_px')}) |yaw| {r.get('yaw_abs_med')} {r.get('reason') or ''} "
              f"cpu {r['cpu_s']}s/{r['attempts']} att {r['face_ms_median']}ms/att tracked {r['tracked']}/{r['frames']} "
              f"expr {(r.get('apparent') or {}).get('expression')}", flush=True)
        if args.crops:
            sheet(r, Path(args.crops) / f"{k:02d}_{label}.jpg")
        results.append((label, spec, r, obs))

    usable = [(i, x) for i, x in enumerate(results) if x[2].get("vec") is not None]
    rows, pairs = [], []
    for i, (la, sa, ra, oa) in usable:          # i = enrolment segment, j = probe segment
        E = np.asarray(ra["vec"], np.float32)
        for j, (lb, sb, rb, ob) in usable:
            if i == j:
                continue
            kind = "genuine" if la == lb else "impostor"
            if kind == "genuine" and Path(sa.split("@")[0]) == Path(sb.split("@")[0]) and "@" not in sa + sb:
                continue
            pf = [float(np.dot(l2n(o.vec), E)) for o in ob]
            for o, sim in zip(ob, pf):
                rows.append({"kind": kind, "enroll": i, "probe": j, "t": round(o.t, 2), "size": o.size, "q": o.quality,
                             "yaw": abs(o.yaw), "live": o.liveness, "sim": sim})
            plain = float(np.dot(l2n(np.mean([l2n(o.vec) for o in ob], 0)), E))
            best = float(np.dot(l2n(max(ob, key=lambda o: o.quality).vec), E))   # single best-quality frame
            va = ra.get("variants") or {}     # same variant on both sides (enrolment and probe)
            var = {k: round(float(np.dot(np.asarray(v, np.float32), np.asarray(va[k], np.float32))), 3)
                   for k, v in (rb.get("variants") or {}).items() if v is not None and va.get(k) is not None}
            pairs.append({"kind": kind, "enroll": i, "probe": j, "agg": round(float(np.dot(rb["vec"], E)), 3), **var,
                          "plain": round(plain, 3), "best_frame": round(best, 3),
                          "frame_mean": round(float(np.mean(pf)), 3), "frame_min": round(float(np.min(pf)), 3),
                          "frame_max": round(float(np.max(pf)), 3), "n_faces": rb["n_faces"]})
    print("\nPAIRS (enrol segment template vs probe segment; agg = quality-weighted outlier-rejected template, "
          "plain = unweighted mean of all faces, best_frame = single best-quality face, frame_* = per-face cosines)")
    for p in pairs:
        print(f"  {p['kind']:8s} {p['enroll']:2d} -> {p['probe']:2d}: agg {p['agg']:.3f} plain {p['plain']:.3f} "
              f"best_frame {p['best_frame']:.3f} frames mean {p['frame_mean']:.3f} [{p['frame_min']:.3f}, {p['frame_max']:.3f}] "
              f"(n {p['n_faces']})")
    summ = {}
    for kind in ("genuine", "impostor"):
        ps = [p for p in pairs if p["kind"] == kind]
        fr = [r["sim"] for r in rows if r["kind"] == kind]
        if not ps:
            continue
        summ[kind] = {m: {"mean": round(float(np.mean([p[m] for p in ps])), 3),
                          "min": round(float(np.min([p[m] for p in ps])), 3),
                          "max": round(float(np.max([p[m] for p in ps])), 3)} for m in ("agg", "plain", "best_frame", "qw_pool24", "qw_all",
                                                                       "qw_all_norej", "mean_all_rej", "mean_all")
                       if all(m in p for p in ps)}
        summ[kind]["per_frame"] = {"n": len(fr), "mean": round(float(np.mean(fr)), 3),
                                   "p10": round(float(np.percentile(fr, 10)), 3),
                                   "p90": round(float(np.percentile(fr, 90)), 3),
                                   "min": round(float(np.min(fr)), 3), "max": round(float(np.max(fr)), 3)}
        summ[kind]["pairs"] = len(ps)
        print(f"\n{kind.upper()} ({len(ps)} pairs): " + json.dumps(summ[kind]))
    print("\nBY FACE SIZE (distance proxy; sims are per face vs the other segments' templates)")
    for b in size_bins(rows):
        print("  ", b)
    print("\nSEGMENT INDEX: " + ", ".join(f"{k}={l}:{s}" for k, (l, s, _, _) in enumerate(results)))
    import resource
    print(f"\npeak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB (pose + InspireFace + decode)")
    if args.json:
        Path(args.json).write_text(json.dumps({"segments": [{"label": l, "spec": s, **{k: v for k, v in r.items()
                                                if k not in ("vec", "crops_b64", "best_crop_b64", "variants")}}
                                                for l, s, r, _ in results],
                                               "pairs": pairs, "summary": summ, "faces": rows,
                                               "by_size": size_bins(rows), "args": vars(args)}, indent=1, default=float))


if __name__ == "__main__":
    main()

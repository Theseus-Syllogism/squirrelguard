#!/usr/bin/env python
"""Measure multi-camera fusion (squirrel_guard.fusion) on real clips: fused vs single-camera rig fits, held-out
reprojection, stature, gait availability, and the cost/quality of the rig frame rate. Optionally write the fused
replay with the SAME code the live profiler uses (fusion.build_replay + replay_assets).

  .venv/bin/python scripts/fusion_eval.py --work W --cam v4=V4.mp4,calib/v4.json --cam testcam=TC.mp4,calib/testcam.json
      [--base-fps 25] [--fps 8 12 25] [--max-s 30] [--start v4=1.5] [--json OUT.json]
      [--write-replay deploy/recorder/replay/demo --name fused0246 --replay-rig-fps 12 --event testcam=<id>]

Keypoints are extracted ONCE per camera at --base-fps on a common time grid (cached in W/<cam>_<fps>.npz) and
subsampled for lower rates (nearest grid instant), so every rate sees the same detections.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard import fusion, rig, rigpipe  # noqa: E402
from squirrel_guard.geometry import Camera  # noqa: E402
from squirrel_guard.pose import Pose  # noqa: E402


def load_or_extract(work: Path, name, clip, cam, fps, max_s, start_s, pose_model, rig_model, threads):
    f = work / f"{name}_{fps:g}_{max_s:g}_{start_s:g}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=False)
        picks = [None if not np.isfinite(d["pbox"][i, 0]) else Pose(tuple(d["pbox"][i]), float(d["pscore"][i]), d["pkp"][i])
                 for i in range(len(d["pbox"]))]
        return {"name": name, "cam": cam, "size": tuple(int(v) for v in d["size"]), "kp": d["kp"], "box": d["box"],
                "picks": picks, "seconds": float(d["seconds"]), "frames_keypointed": int(d["fk"]),
                "frames_person": int(d["fp"])}
    from squirrel_guard.pose import PoseEstimator
    from squirrel_guard.wholebody import WholeBody
    pose = PoseEstimator(pose_model, threads=threads[0])
    wb = WholeBody(rig_model, threads=threads[1])
    v = fusion.extract_view(Path(clip).read_bytes(), name, cam, pose, wb, fps, max_s, start_s=start_s)
    pb = np.array([p.box if p is not None else [np.nan] * 4 for p in v["picks"]], np.float32)
    ps = np.array([p.score if p is not None else np.nan for p in v["picks"]], np.float32)
    pk = np.array([p.kp if p is not None else np.full((17, 3), np.nan) for p in v["picks"]], np.float32)
    np.savez_compressed(f, kp=v["kp"], box=v["box"], size=np.array(v["size"]), pbox=pb, pscore=ps, pkp=pk,
                        seconds=v["seconds"], fk=v["frames_keypointed"], fp=v["frames_person"])
    return v


def subsample(v: dict, base: float, fps: float) -> dict:
    n = int(np.floor((len(v["kp"]) - 1) * fps / base)) + 1
    idx = np.clip(np.round(np.arange(n) * base / fps).astype(int), 0, len(v["kp"]) - 1)
    return {**v, "kp": v["kp"][idx], "box": v["box"][idx], "picks": [v["picks"][i] for i in idx]}


def gait_short(g: dict) -> dict:
    return {k: g.get(k) for k in ("walking_frames", "steps_used", "cadence_spm", "step_time_s_cv", "step_length_m_median",
                                  "heel_strikes", "knee_rom_deg", "knee_symmetry") if k in g}


def single_init(v: dict, shift: float, fps: float, n: int):
    """What the profiler starts a one-camera rig from: that camera's own replay path (trajectory.estimate on its
    track: feet on the floor, else head plane), on the primary's time base."""
    from squirrel_guard import trajectory
    tr = trajectory.estimate(v["picks"], fps, v["size"], v["cam"], heights_px=True, head_height=1.70)
    if tr is None:
        return None
    xy = rigpipe.init_from_replay(tr, len(v["picks"]), fps)
    return fusion.shift_series(xy, shift)[:n]


def evaluate(views: list[dict], fps: float, iters, max_gap: float, max_err: float = 0.06) -> dict:
    t0 = time.time()
    fz = fusion.fuse_views(views, fps, iters=iters, max_gap_m=max_gap, max_err_box=max_err)
    out = {"fps": fps, "ok": fz["ok"], "reason": fz.get("reason"), "cameras": fz["report"],
           "fuse_s": round(time.time() - t0, 1)}
    if not fz["ok"]:
        return out
    res = fz["res"]
    al = {v["name"]: v for v in fz["aligned"]}
    # every camera ALSO aligned (dropped ones too, by their best found shift if any) for held-out checks
    out["fused"] = {"stature_m": round(res["stature_m"], 3), "fit_s": res["fit_s"],
                    "frames_seen": int(res["seen"].sum()), "frames": len(res["seen"]),
                    "error": {nm: fusion.view_error(res, al[nm]) for nm in al},
                    "gait": gait_short(rig.gait_metrics(res, 1.0 / fps))}
    out["single"] = {}
    names = list(al)
    byname = {v["name"]: v for v in views}
    for nm in names:
        t1 = time.time()
        r1 = rig.fit([al[nm]], len(al[nm]["kp"]), 1.0 / fps, init_xy=single_init(byname[nm], fz["shifts"][nm], fps,
                                                                                  len(al[nm]["kp"])),
                     stature0=1.70, iters=iters, verbose=False)
        out["single"][nm] = {"stature_m": round(r1["stature_m"], 3), "fit_s": round(time.time() - t1, 1),
                             "frames_seen": int(r1["seen"].sum()),
                             "train_error": fusion.view_error(r1, al[nm]),
                             "heldout_error": {o: fusion.view_error(r1, al[o]) for o in names if o != nm},
                             "gait": gait_short(rig.gait_metrics(r1, 1.0 / fps))}
    if len(names) >= 3:
        out["leave_one_out"] = {}
        for nm in names:
            rest = [al[o] for o in names if o != nm]
            r2 = rig.fit(rest, len(rest[0]["kp"]), 1.0 / fps, init_xy=None, stature0=fz["stature0"], iters=iters,
                         verbose=False)
            out["leave_one_out"][nm] = {"fit_on": [v["name"] for v in rest], "stature_m": round(r2["stature_m"], 3),
                                        "heldout_error": fusion.view_error(r2, al[nm])}
    out["_fz"] = fz
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--cam", action="append", required=True, help="NAME=CLIP,CALIB.json")
    ap.add_argument("--start", action="append", default=[], help="NAME=SECONDS: this clip starts that much later "
                    "than the window (Frigate recordings coverage)")
    ap.add_argument("--base-fps", type=float, default=25.0)
    ap.add_argument("--fps", type=float, nargs="+", default=[12.0])
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--iters", type=int, nargs=2, default=[300, 900])
    ap.add_argument("--max-gap", type=float, default=0.15)
    ap.add_argument("--max-err", type=float, default=0.06, help="rig error limit per view (box heights); diagnostic: "
                    "raise it to see a rejected fusion anyway")
    ap.add_argument("--threads", type=int, nargs=2, default=[4, 2], help="YOLO pose threads, RTMW threads")
    ap.add_argument("--torch-threads", type=int, default=2, help="rig fit threads (the profiler uses rig_threads = 2)")
    ap.add_argument("--pose-model", default=str(ROOT / "models/yolov8s-pose.onnx"))
    ap.add_argument("--rig-model", default=str(ROOT / "models/rtmw/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122/end2end.onnx"))
    ap.add_argument("--seg-model", default=str(ROOT / "models/yolov8s-seg-320.onnx"))
    ap.add_argument("--json")
    ap.add_argument("--write-replay", help="directory for <name>.json + assets (fused replay, live code path)")
    ap.add_argument("--name", default="fused")
    ap.add_argument("--replay-rig-fps", type=float, default=12.0)
    ap.add_argument("--event", action="append", default=[], help="NAME=FRIGATE_EVENT_ID (members of the replay)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.torch_threads)

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    starts = {k: float(v) for k, v in (s.split("=", 1) for s in a.start)}
    clips, base_views = {}, []
    for spec in a.cam:
        name, rest = spec.split("=", 1)
        clip, calib = rest.split(",")
        cam = Camera.load(calib)
        clips[name] = clip
        v = load_or_extract(work, name, clip, cam, a.base_fps, a.max_s, -starts.get(name, 0.0), a.pose_model,
                            a.rig_model, a.threads)
        print(f"{name}: {len(v['kp'])} grid frames @ {a.base_fps:g} fps, {v['frames_keypointed']} keypointed, person in "
              f"{v['frames_person']}, {v['seconds']} s ({1000 * v['seconds'] / max(1, v['frames_keypointed']):.0f} ms/frame)")
        base_views.append(v)
    report = {"base_fps": a.base_fps, "cameras": {v["name"]: {"keypoint_s": v["seconds"],
                                                               "frames_keypointed": v["frames_keypointed"],
                                                               "ms_per_frame": round(1000 * v["seconds"] /
                                                                                     max(1, v["frames_keypointed"]))}
                                                   for v in base_views}, "rates": []}
    keep_fz = None
    for f in a.fps:
        vs = [subsample(v, a.base_fps, f) for v in base_views]
        r = evaluate(vs, f, tuple(a.iters), a.max_gap, a.max_err)
        fz = r.pop("_fz", None)
        if abs(f - a.replay_rig_fps) < 1e-6:
            keep_fz = (fz, vs)
        report["rates"].append(r)
        print(f"\n== {f:g} fps: {'FUSED' if r['ok'] else 'not fused: ' + str(r['reason'])} ({r['fuse_s']} s)")
        for nm, c in r["cameras"].items():
            print(f"  {nm}: {json.dumps(c)}")
        if r["ok"]:
            fu = r["fused"]
            print(f"  fused: stature {fu['stature_m']} m, seen {fu['frames_seen']}/{fu['frames']}, fit {fu['fit_s']} s, "
                  "error " + ", ".join(f"{k} {e['median_err_box']} ({e['median_err_px']} px)" for k, e in fu["error"].items()))
            print(f"         gait {json.dumps(fu['gait'])}")
            for nm, s in r["single"].items():
                print(f"  {nm} alone: stature {s['stature_m']} m, train {s['train_error']['median_err_box']}, held-out "
                      + ", ".join(f"{o} {e['median_err_box']} ({e['median_err_px']} px)" for o, e in s["heldout_error"].items())
                      + f"; fit {s['fit_s']} s; gait {json.dumps(s['gait'])}")
            for nm, s in (r.get("leave_one_out") or {}).items():
                print(f"  fit on {'+'.join(s['fit_on'])}: stature {s['stature_m']}, held-out {nm} "
                      f"{s['heldout_error']['median_err_box']} ({s['heldout_error']['median_err_px']} px)")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, default=float))
    if a.write_replay and keep_fz and keep_fz[0] and keep_fz[0]["ok"]:
        from squirrel_guard import kinematics, replay_assets  # noqa: F401
        fz, vs = keep_fz
        f = a.replay_rig_fps
        step = fusion.replay_step(f)
        prim = next(v for v in vs if v["name"] == fz["primary"])
        coll = replay_assets.Collector(every=2, max_cutouts=120)
        smp = fusion.ClipSampler(Path(clips[prim["name"]]).read_bytes(), f, len(prim["kp"]),
                                 -starts.get(prim["name"], 0.0))
        nxt = 0
        for fr in smp:                                   # cut-outs/background from the cached picks: no inference
            for k in smp.targets:
                if k % step:
                    continue
                while nxt < k:                           # replay frames before the clip starts
                    coll.n += 1
                    nxt += step
                coll.on_frame(fr, prim["picks"][k])
                nxt = k + step
        ev = dict(e.split("=", 1) for e in a.event)
        members = [{"camera": c, "frigate_event": e} for c, e in ev.items()]
        tr = fusion.build_replay(fz, vs, f, step, members=members)
        if tr is None:
            sys.exit("no trajectory")
        if fz["primary"] in ev:
            tr["event"] = ev[fz["primary"]]
        tr["events_by_camera"] = ev
        tr["source"] = (f"fused from {', '.join(fz['report'])} (squirrel_guard.fusion, live code path): one rig over "
                        f"{', '.join(v['name'] for v in fz['aligned'])}")
        masker = replay_assets.PersonMasker(a.seg_model, threads=4)
        rp, atlas, bg = replay_assets.build(coll, prim["cam"], tr, masker, fps=f / step)
        out = Path(a.write_replay)
        replay_assets.attach(tr, rp, atlas, bg, out, a.name)
        (out / f"{a.name}.json").write_text(json.dumps(tr, separators=(",", ":")))
        print(f"wrote {out / (a.name + '.json')}: {tr['fusion']['cameras']}, {tr['rig']['frames_fitted']} rig frames")


if __name__ == "__main__":
    main()

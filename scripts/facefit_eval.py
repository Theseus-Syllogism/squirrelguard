#!/usr/bin/env python3
"""Evaluate squirrel_guard/facefit.py (one fitted face per visit) against facegeom's per-frame aggregate.

Usage (a Python with mediapipe + onnxruntime + opencv; see docs/facefit.md):
  python scripts/facefit_eval.py [--model face_landmarker.task | legacy] [--out ~/.cache/squirrel-guard-test/facefit]

Default data: the consenting owner's two-camera walk (~/.cache/squirrel-guard-test/multicam/demo1803/{v4,frontyard}.mp4
and rig0/calib/*.json). Writes facefit_eval.json + LOD glTF/GLB files to --out (outside git; never commit them).
Steps:
  1. per camera and per image mode (colour, gray = grayscale-converted, irsim = grayscale + sensor noise matched to a
     real Wyze IR frame), every decoded frame: YOLOv8-pose -> head window -> 2-pass face mesh -> FrameObs (cached);
  2. repeatability: split the visit's frames in halves (time halves, odd/even, 20 random halves) and compare the two
     halves' features/shapes, facefit vs facegeom's production aggregate (median of the best 8 near-frontal faces)
     and vs the median of all near-frontal faces; and v4-only vs frontyard-only (cross-camera);
  3. two views: verify the clip offset with face-landmark ray gaps, triangulate simultaneous frames, metric scale
     and depth stretch; reprojection errors;
  4. IR: detection rate, landmark shift vs colour, fit residual and feature change; the real IR frame;
  5. LODs (triangles, bytes) and runtime per frame.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard import facefit as ff  # noqa: E402
from squirrel_guard import facegeom as fg  # noqa: E402
from squirrel_guard import kinematics  # noqa: E402
from squirrel_guard.faceid import head_window  # noqa: E402
from squirrel_guard.geometry import Camera  # noqa: E402
from squirrel_guard.pose import PoseEstimator  # noqa: E402

DATA = Path.home() / ".cache/squirrel-guard-test/multicam/demo1803"
PROP = [k for k, (kind, _) in fg.FEATURES.items() if kind == "prop"]
DEPTH = [k for k, (kind, _) in fg.FEATURES.items() if kind == "depth"]


# -- per-frame extraction ------------------------------------------------------------------------------------------
def noise_sigma(gray: np.ndarray) -> float:
    """Sensor noise estimate (Immerkaer 1996: mean |3x3 Laplacian-difference| on the smoother half of the image,
    excluding clipped pixels), grey levels."""
    g = gray.astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(g, -1, k))[1:-1, 1:-1]
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))[1:-1, 1:-1]
    m = (mag < np.percentile(mag, 50)) & (g[1:-1, 1:-1] > 10) & (g[1:-1, 1:-1] < 245)
    return float(np.sqrt(np.pi / 2) * r[m].mean() / 6)


# mode -> landmarks_in_window(ir=): gray = plain grayscale, grayclahe = always CLAHE, grayauto = plain + CLAHE retry
MODES = {"colour": "auto", "gray": "gray", "grayclahe": "force", "grayauto": "auto", "irsim": "auto"}


def to_mode(fr: np.ndarray, mode: str, rng) -> np.ndarray:
    """colour: as recorded; gray/grayclahe: grayscale in 3 equal channels (what the Wyze IR stream delivers);
    irsim: grayscale + a rough night-IR look (contrast x 0.6, soft ISP denoise blur, mild noise). The Wyze IR
    snapshot available is strongly denoised (noise ~0.2 grey levels), so irsim is a stress test, not a sensor model."""
    if mode == "colour":
        return fr
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    if mode == "irsim":
        f = g.astype(np.float32)
        f = (f - f.mean()) * 0.6 + f.mean() + 20
        f = cv2.GaussianBlur(f, (0, 0), 1.2) + rng.normal(0, 2.0, f.shape)
        g = np.clip(f, 0, 255).astype(np.uint8)
    return cv2.merge([g, g, g])


def pick_person(poses):
    """The pose whose head keypoints are most confident (the pose model also fires on a static object)."""
    best, bs = None, 0.0
    for p in poses:
        s = float(p.kp[:5, 2].sum())
        if s > bs:
            best, bs = p, s
    return best if bs >= 1.0 else None


def extract(clip: Path, cam: str, mode: str, pe, lm, fps: float, t_window=None) -> dict:
    frames, efps = kinematics.decode(clip.read_bytes(), fps, 40)
    ir = MODES[mode]
    rng = np.random.default_rng(7)
    obs, headvis, t_pose, t_lm, size = [], [], [], [], None
    for i, fr in enumerate(frames):
        size = (fr.shape[1], fr.shape[0])
        if t_window and not (t_window[0] <= i / efps <= t_window[1]):
            continue
        img = to_mode(fr, mode, rng)
        t0 = time.perf_counter()
        p = pick_person(pe(img))
        t_pose.append((time.perf_counter() - t0) * 1e3)
        if p is None:
            continue
        if p.kp[0, 2] >= 0.5 and p.kp[1, 2] >= 0.5 and p.kp[2, 2] >= 0.5:
            headvis.append(i)
        win = head_window(p.box, p.kp, img.shape)
        if win is None:
            continue
        o = ff.landmarks_in_window(img, win, lm, ir=ir, cam=cam, idx=i, t=i / efps)
        if o is not None:
            t_lm.append(o.ms)
            obs.append(o)
    return {"obs": obs, "fps": efps, "size": size, "head_visible": headvis, "pose_ms": t_pose, "lm_ms": t_lm}


# -- old method (facegeom production aggregate) ----------------------------------------------------------------------
def old_select(obs, keep=8, max_yaw=25.0, max_pitch=30.0):
    """facegeom.GeomCollector's choice: near-frontal faces, best `keep` by size x pose x sharpness weight."""
    c = [o for o in obs if abs(o.yaw) <= max_yaw and abs(o.pitch) <= max_pitch]
    if not c:
        return []
    msh = np.median([o.sharp for o in c])

    def w(o):
        return (min(1.0, o.ipd_px / 30.0) * math.cos(math.radians(o.yaw)) ** 2 * math.cos(math.radians(o.pitch))
                * min(1.0, o.sharp / (msh + 1e-9)))
    return sorted(c, key=w, reverse=True)[:keep] if keep else c


def old_result(obs, keep=8) -> dict | None:
    sel = old_select(obs, keep)
    if not sel:
        return None
    per = [fg.features_of(o.P) for o in sel]
    agg = fg.aggregate(per, [{"size": o.ipd_px, "yaw": o.yaw, "pitch": o.pitch} for o in sel])
    med = np.median(np.stack([fg.frontalise(o.P)[0] for o in sel]), 0)
    return {"features": {k: v["value"] for k, v in agg["features"].items()}, "S": med, "n": len(sel)}


def new_result(obs, cams, boot=0, prior=None) -> dict | None:
    p = ff.FitParams(boot=boot)
    r = ff.fit_visit(obs, cams=cams, params=p, mesh=False, prior=prior)
    if r.get("reason"):
        return None
    return {"features": {k: v["value"] for k, v in r["features"].items()}, "S": r["_S"], "n": r["n_used"],
            "payload": r}


def shape_rms(A, B) -> float:
    """RMS 3D distance (IPD units) of two frontal shapes over the surface, after a rigid+scale alignment."""
    s, R, t = ff.umeyama(A[:ff.N_SURF], B[:ff.N_SURF])
    d = (s * A[:ff.N_SURF] @ R.T + t) - B[:ff.N_SURF]
    return float(np.sqrt((d ** 2).sum(1).mean()))


def splits(n, rng, n_random=20):
    idx = np.arange(n)
    out = {"halves": (idx[: n // 2], idx[n // 2:]), "odd_even": (idx[0::2], idx[1::2])}
    for r in range(n_random):
        p = rng.permutation(n)
        out[f"random{r}"] = (np.sort(p[: n // 2]), np.sort(p[n // 2:]))
    return out


def compare_pair(a, b) -> dict:
    d = {}
    for k in PROP:
        if a["features"].get(k) is not None and b["features"].get(k) is not None:
            d[k] = 100 * abs(a["features"][k] - b["features"][k]) / (abs(a["features"][k] + b["features"][k]) / 2 + 1e-9)
    for k in DEPTH:
        if a["features"].get(k) is not None and b["features"].get(k) is not None:
            d[k] = abs(a["features"][k] - b["features"][k])
    d["shape_rms"] = shape_rms(a["S"], b["S"])
    return d


def repeatability(obs, cams, rng, methods) -> dict:
    """Per method: per split type, per feature |A - B| (% for proportions, IPD for depth) + shape RMS."""
    obs = sorted(obs, key=lambda o: (o.t, o.cam))
    sp = splits(len(obs), rng)
    out = {m: {} for m in methods}
    for name, (ia, ib) in sp.items():
        A, B = [obs[i] for i in ia], [obs[i] for i in ib]
        for m, fn in methods.items():
            ra, rb = fn(A, cams), fn(B, cams)
            if ra is None or rb is None:
                continue
            out[m][name] = compare_pair(ra, rb)
    summ = {}
    for m, rows in out.items():
        keys = sorted({k for r in rows.values() for k in r})
        rnd = [r for n, r in rows.items() if n.startswith("random")]
        summ[m] = {k: {"halves": round(rows.get("halves", {}).get(k, float("nan")), 4),
                       "odd_even": round(rows.get("odd_even", {}).get(k, float("nan")), 4),
                       "random_mean": round(float(np.mean([r[k] for r in rnd if k in r])), 4) if rnd else None}
                   for k in keys}
        st = [k for k in fg.STABLE_FEATURES if k in PROP and k in summ[m]]
        summ[m]["_stable_props_pct"] = {t: round(float(np.mean([summ[m][k][t] for k in st])), 3)
                                        for t in ("halves", "odd_even", "random_mean")}
        allp = [k for k in PROP if k in summ[m]]
        summ[m]["_all_props_pct"] = {t: round(float(np.mean([summ[m][k][t] for k in allp])), 3)
                                     for t in ("halves", "odd_even", "random_mean")}
    return summ


# -- main ----------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4", default=str(DATA / "v4.mp4"))
    ap.add_argument("--frontyard", default=str(DATA / "frontyard.mp4"))
    ap.add_argument("--calib", default=str(DATA / "rig0/calib"))
    ap.add_argument("--ir-frame", default=str(Path.home() / ".cache/squirrel-guard-test/multicam/20261106-1752/frontyard.jpg"))
    ap.add_argument("--model", default="legacy", help="face_landmarker.task, or 'legacy' (mediapipe<=0.10.21 face_mesh)")
    ap.add_argument("--pose", default=str(ROOT / "models/yolov8s-pose.onnx"))
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--shift", type=int, default=-15, help="expected frontyard frame shift vs v4 (verified by a scan)")
    ap.add_argument("--modes", default="colour,gray,grayclahe,grayauto,irsim")
    ap.add_argument("--mv-fps", type=float, default=25.0, help="decode rate for the two-view pairs (sync error <= 1/2 frame)")
    ap.add_argument("--mv-t0", type=float, default=9.5, help="v4 time window of the two-view pairs (s)")
    ap.add_argument("--mv-t1", type=float, default=15.0)
    ap.add_argument("--out", default=str(Path.home() / ".cache/squirrel-guard-test/facefit"))
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--boot", type=int, default=24)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cams = {n: Camera.load(Path(a.calib) / f"{n}.json") for n in ("v4", "frontyard")}
    clips = {"v4": Path(a.v4), "frontyard": Path(a.frontyard)}
    rng = np.random.default_rng(1)
    R = {"model": a.model, "fps": a.fps}

    ir_img = cv2.imread(a.ir_frame) if Path(a.ir_frame).is_file() else None
    sig_ir = noise_sigma(cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)) if ir_img is not None else None
    data, pe, lm = {}, None, None

    def models():
        nonlocal pe, lm
        if pe is None:
            pe, lm = PoseEstimator(a.pose), ff.make_landmarker(None if a.model == "legacy" else a.model)
        return pe, lm
    jobs = [(cam, mode, a.fps, None) for mode in a.modes.split(",") for cam in clips]
    jobs += [(cam, "mv25", a.mv_fps, (a.mv_t0 + (a.shift / a.fps if cam == "frontyard" else 0) - 1.0,
                                      a.mv_t1 + (a.shift / a.fps if cam == "frontyard" else 0) + 1.0))
             for cam in clips]
    for cam, mode, fps, win in jobs:
        clip = clips[cam]
        cache = out / f"obs_{cam}_{mode}.pkl"
        if cache.is_file() and not a.recompute:
            data[(cam, mode)] = pickle.loads(cache.read_bytes())
            continue
        t0 = time.perf_counter()
        d = extract(clip, cam, "colour" if mode == "mv25" else mode, *models(), fps, win)
        d["wall_s"] = time.perf_counter() - t0
        cache.write_bytes(pickle.dumps(d))
        data[(cam, mode)] = d
        print(f"{cam}/{mode}: {len(d['obs'])} faces, {d['wall_s']:.1f} s", flush=True)
    sizes = {cam: data[(cam, "colour")]["size"] for cam in clips}
    calib = {cam: (cams[cam], sizes[cam]) for cam in clips}

    # -- runtime --------------------------------------------------------------------------------------------------
    d = data[("v4", "colour")]
    R["runtime"] = {"pose_ms_med": round(float(np.median(d["pose_ms"])), 1),
                    "landmarks_2pass_ms_med": round(float(np.median(d["lm_ms"])), 1),
                    "ir_frame_noise_sigma": round(sig_ir, 2) if sig_ir is not None else None,
                    "colour_frame_noise_sigma": round(noise_sigma(cv2.cvtColor(
                        next(iter(kinematics.decode(clips["v4"].read_bytes(), a.fps, 2)[0])), cv2.COLOR_BGR2GRAY)), 2)}

    obs = {cam: data[(cam, "colour")]["obs"] for cam in clips}
    for cam in clips:
        o = obs[cam]
        R.setdefault("frames", {})[cam] = {
            "decoded_fps": round(data[(cam, "colour")]["fps"], 3), "faces": len(o),
            "head_visible_frames": len(data[(cam, "colour")]["head_visible"]),
            "ipd_px": [round(float(np.min([x.ipd_px for x in o])), 1), round(float(np.median([x.ipd_px for x in o])), 1),
                       round(float(np.max([x.ipd_px for x in o])), 1)] if o else None,
            "t_range": [round(min(x.t for x in o), 2), round(max(x.t for x in o), 2)] if o else None}

    # -- main fit (both cameras, single-view stage) with timing -------------------------------------------------------
    allobs = sorted(obs["v4"] + obs["frontyard"], key=lambda o: o.t)
    t0 = time.perf_counter()
    ff.fit_visit(allobs, cams=calib, params=ff.FitParams(boot=0), mesh=False)
    fit_ms_noboot = (time.perf_counter() - t0) * 1e3

    # -- two views: verify the offset with landmark ray gaps, then pairs ---------------------------------------------
    mvobs = {cam: data[(cam, "mv25")]["obs"] for cam in clips}
    mvfps = data[("v4", "mv25")]["fps"]
    by = {cam: {o.idx: o for o in mvobs[cam]} for cam in clips}
    k0 = int(round(a.shift * mvfps / a.fps))
    scan = []
    for k in range(k0 - 8, k0 + 9):
        gaps = []
        for i, oa in by["v4"].items():
            ob = by["frontyard"].get(i + k)
            if ob is None:
                continue
            _, g = ff.triangulate(cams["v4"], oa.P[:ff.N_SURF, :2], sizes["v4"], cams["frontyard"],
                                  ob.P[:ff.N_SURF, :2], sizes["frontyard"])
            if np.isfinite(g).any():
                gaps.append(float(np.nanmedian(g)))
        if gaps:
            scan.append((k, len(gaps), float(np.median(gaps))))
    best = min((s_ for s_ in scan if s_[1] >= 5), key=lambda s_: s_[2], default=None)
    R["offset_scan"] = {"fps": round(mvfps, 3), "expected_shift_frames": k0, "expected_shift_s": round(a.shift / a.fps, 3),
                        "best_shift_frames": best[0] if best else None,
                        "best_shift_s": round(best[0] / mvfps, 3) if best else None,
                        "best_gap_mm": round(best[2] * 1e3, 1) if best else None, "pairs_at_best": best[1] if best else 0,
                        "scan": [(k, n, round(g * 1e3, 1)) for k, n, g in scan]}
    shift = best[0] if best else k0
    pairs = [(oa, by["frontyard"][i + shift]) for i, oa in sorted(by["v4"].items()) if (i + shift) in by["frontyard"]]
    R["offset_scan"]["shift_used"] = shift
    t0 = time.perf_counter()
    full = ff.fit_visit(allobs, cams=calib, pairs=pairs, params=ff.FitParams(boot=a.boot), mesh=True)
    fit_ms_boot = (time.perf_counter() - t0) * 1e3
    R["fit"] = {k: v for k, v in ff.strip(full).items()}
    R["runtime"].update(fit_ms_no_bootstrap=round(fit_ms_noboot, 1), fit_ms_with_bootstrap_and_lods=round(fit_ms_boot, 1),
                        frames_in_fit=len(allobs), fit_ms_per_frame=round(fit_ms_noboot / max(1, len(allobs)), 2))
    fit = full["_fit"]
    # reprojection per camera
    rp = {}
    for cam in clips:
        m = np.array([o.cam == cam for o in allobs]) & (fit.w_frame > 1e-6)
        if m.any():
            rp[cam] = {"frames": int(m.sum()), "px_med": round(float(np.median(fit.resid_px[m])), 2),
                       "px_p90": round(float(np.percentile(fit.resid_px[m], 90)), 2),
                       "ipd_pct_med": round(100 * float(np.median(fit.resid_ipd[m])), 2)}
    R["reprojection"] = rp
    # old vs new values on the whole visit
    old = old_result(allobs)
    R["values"] = {k: {"facefit": full["features"][k]["value"], "facefit_sd": full["features"][k]["sd"],
                       "facegeom_aggregate": old["features"].get(k) if old else None} for k in fg.FEATURES}
    R["values_n"] = {"facefit_frames": full["n_used"], "facegeom_faces": old["n"] if old else 0}

    # LOD files
    g = full["mesh"]
    lod_info = {}
    for lv in ff.lods(full["_S"]):
        one = ff.gltf_lods([lv], (full["multiview"]["ipd_mm"] / 1000) if full.get("multiview", {}).get("used") else None,
                           default=lv["name"])
        js = json.dumps(one, separators=(",", ":")).encode()
        glb = ff.gltf_to_glb(one)
        (out / f"face_{lv['name']}.glb").write_bytes(glb)
        lod_info[lv["name"]] = {"triangles": int(len(lv["T"])), "vertices": int(len(lv["V"])),
                                "gltf_json_bytes": len(js), "glb_bytes": len(glb)}
    (out / "face_lods.gltf").write_text(json.dumps(g, separators=(",", ":")))
    R["lods"] = lod_info
    R["lods_combined"] = {"gltf_json_bytes": len(json.dumps(g, separators=(",", ":"))),
                          "glb_bytes": len(ff.gltf_to_glb(g))}
    t0 = time.perf_counter()
    ff.lods(full["_S"])
    R["runtime"]["lods_ms"] = round((time.perf_counter() - t0) * 1e3, 1)

    # -- repeatability -------------------------------------------------------------------------------------------------
    methods = {"facegeom_best8": lambda o, c: old_result(o, 8),
               "facegeom_all_frontal": lambda o, c: old_result(o, 0),
               "facefit": lambda o, c: new_result(o, c)}
    R["repeat"] = {}
    for label, sel in (("v4", obs["v4"]), ("frontyard", obs["frontyard"]), ("both", allobs)):
        if len(sel) >= 8:
            R["repeat"][label] = repeatability(sel, calib, rng, methods)
    # cross-camera: v4-only vs frontyard-only
    cc = {}
    for m, fn in methods.items():
        ra, rb = fn(obs["v4"], calib), fn(obs["frontyard"], calib)
        if ra and rb:
            cc[m] = {k: round(v, 4) for k, v in compare_pair(ra, rb).items()}
            cc[m]["_stable_props_pct"] = round(float(np.mean([cc[m][k] for k in fg.STABLE_FEATURES if k in PROP])), 3)
            cc[m]["_all_props_pct"] = round(float(np.mean([cc[m][k] for k in PROP])), 3)
    R["cross_camera"] = cc

    # -- IR / grayscale ------------------------------------------------------------------------------------------------
    ir = {}
    for cam in clips:
        col = {o.idx: o for o in data[(cam, "colour")]["obs"]}
        hv = set(data[(cam, "colour")]["head_visible"])
        col_fit = new_result(list(col.values()), calib)
        for mode in a.modes.split(","):
            dm = data.get((cam, mode))
            if dm is None:
                continue
            mo = {o.idx: o for o in dm["obs"]}
            common = sorted(set(mo) & set(col))
            shift_ipd = [float(np.median(np.linalg.norm(mo[i].P[:ff.N_SURF, :2] - col[i].P[:ff.N_SURF, :2], axis=1))
                               / col[i].ipd_px) for i in common]
            rr = new_result(list(mo.values()), calib)
            row = {"faces": len(mo), "faces_where_head_visible": len(set(mo) & hv), "head_visible": len(hv),
                   "detection_rate": round(len(set(mo) & hv) / max(1, len(hv)), 3),
                   "landmark_shift_vs_colour_ipd_pct_med": round(100 * float(np.median(shift_ipd)), 2) if shift_ipd else None}
            if rr:
                pl = rr["payload"]
                row["fit_reproj_ipd_pct_med"] = pl["reproj_ipd_pct_med"]
                if col_fit:
                    row["features_vs_colour"] = {k: round(v, 4) for k, v in compare_pair(rr, col_fit).items()}
            ir[f"{cam}/{mode}"] = row
    if ir_img is not None:
        pe, lm = models()
        ps = pe(ir_img)
        p = pick_person(ps)
        faces = 0
        if p is not None:
            win = head_window(p.box, p.kp, ir_img.shape)
            faces = int(win is not None and ff.landmarks_in_window(ir_img, win, lm) is not None)
        # whole-frame sweep: the mesh model on overlapping windows must not invent faces in an empty IR scene
        H, W = ir_img.shape[:2]
        fp = 0
        nwin = 0
        for side in (H // 2, H // 3):
            for y in range(0, H - side + 1, side // 2):
                for x in range(0, W - side + 1, side // 2):
                    nwin += 1
                    fp += ff.landmarks_in_window(ir_img, (x, y, x + side, y + side), lm, refine=False, zooms=(1.0,)) is not None
        R["ir_real_frame"] = {"grayscale_detected": ff.is_grayscale(ir_img), "persons": len(ps),
                              "persons_with_head": int(p is not None), "faces": faces,
                              "sweep_windows": nwin, "sweep_false_faces": int(fp)}
    R["ir"] = ir

    (out / "facefit_eval.json").write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps({k: R[k] for k in ("frames", "offset_scan", "reprojection", "lods", "runtime", "ir",
                                        "ir_real_frame", "cross_camera") if k in R}, indent=1, default=float))
    print(json.dumps(R["fit"].get("multiview", {}) | {"pairs": None}, indent=1, default=float))
    print(json.dumps(R["repeat"], indent=0, default=float)[:6000])


if __name__ == "__main__":
    main()

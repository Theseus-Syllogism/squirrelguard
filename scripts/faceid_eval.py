"""Face identification v2 evaluation: InspireFace (production) vs AdaFace vs fusion, gallery and template variants.

Usage (data under ~/.cache/squirrel-guard-test/faceid-v2, never in git):
  .venv/bin/python scripts/faceid_eval.py visits   # sighting clips + stock stranger clips -> faces with the PRODUCTION
                                                   # sampler/budget (faceid.FaceSampler) on FaceEngineV2
  .venv/bin/python scripts/faceid_eval.py celeba [--ids 326 --per-id 8]   # public genuine/impostor pairs at
                                                   # controlled face sizes (downscale + JPEG), plus greyscale
  .venv/bin/python scripts/faceid_eval.py embed    # AdaFace IR-50 / IR-101 on every aligned crop
  .venv/bin/python scripts/faceid_eval.py report [--json out.json]

Data and labels
  - Owner (identity #1, "test"): sightings confirmed by the system's face decision (tier A: match_kind face_template
    / face_meta.match = match) and tier B (face_vec cosine >= 0.40 to the A mean, >= 3 faces: chosen BY THE
    INSPIREFACE MODEL, so it favours the baseline), plus labels.csv of the re-ID labelling sheet if it exists.
    Tier "R": identity #1 by body re-ID only (match_kind match/merged) with a face: an independent but noisy label;
    the v4 desk sightings #47-#74 are kept apart (unknown whether the owner, see docs/reid-gait.md).
  - Strangers in video: Pixabay stock clips (reid-gait/stock + faces/, one uploader = one person, first/second half
    = two visits). Out of domain (bright, sharp).
  - Public pairs: CelebA (non-commercial research licence; one test shard of the flwrlabs/celeba mirror),
    downscaled so the face is 24/32/48/64 px, JPEG q=75, then the production detector + alignment on the small
    image; enrolment image at native size (~90 px). Greyscale = a NIR stand-in, not real IR.
Stats: TAR at FAR (thresholds from impostor scores), 90 % bootstrap CI over identities (or visits), AUC.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard.faceid import AggParams, FaceSampler, face_weight, l2n  # noqa: E402
from squirrel_guard.faceid_v2 import (AdaFace, Calibration, FaceEngineV2, aggregate, best_k,  # noqa: E402
                                      modality, multi_templates, pose_bucket, ir_score)

TEST = Path.home() / ".cache/squirrel-guard-test"
DATA = TEST / "faceid-v2"
MODELS = {"ada50": ROOT / "models/adaface_ir50_webface4m.onnx", "ada101": ROOT / "models/adaface_ir101_webface12m.onnx"}
STOCK_DUP = {"suit_199623", "model_140828", "road_76621"}     # same clips as reid-gait/stock px199623/px140828/px76621
STOCK_EXCLUDE = {"px426_0"}
RNG = np.random.default_rng(0)


# --- records --------------------------------------------------------------------------------------------------
def rec_of(o, extra=None) -> dict:
    r = {"isf": l2n(o.vec).astype(np.float32), "q": o.quality, "size": o.size, "yaw": o.yaw, "pitch": o.pitch,
         "det": o.det, "sharp": o.sharp, "live": o.liveness, "ir": getattr(o, "ir", None), "t": o.t,
         "aligned": getattr(o, "aligned", None)}
    r.update(extra or {})
    return r


class Rec:
    def __init__(self, engine):
        self.engine, self.obs, self.pack = engine, [], engine.pack

    def faces(self, *a, **k):
        out = self.engine.faces(*a, **k)
        self.obs.extend(out)
        return out


def cmd_visits(a) -> None:
    from squirrel_guard.kinematics import decode, track
    from squirrel_guard.pose import PoseEstimator
    pose = PoseEstimator(str(ROOT / "models/yolov8s-pose.onnx"), threads=a.threads)
    eng = FaceEngineV2("Megatron")
    out_p = DATA / "visits.pkl"
    done = pickle.loads(out_p.read_bytes()) if out_p.exists() else {}
    R = json.loads((DATA / "sightings.json").read_text())
    jobs = [(f"s{r['id']}", DATA / "clips" / f"sg_{r['frigate_event']}.mp4", 0, 1e9) for r in R]
    stock = sorted((TEST / "reid-gait/stock").glob("*.mp4")) + \
        [p for p in sorted((TEST / "faces").glob("*.mp4")) if p.stem not in STOCK_DUP]
    for p in stock:
        cap = cv2.VideoCapture(str(p))
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1.0, cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        for h in (0, 1):
            if f"{p.stem}_{h}" not in STOCK_EXCLUDE:
                jobs.append((f"x:{p.stem}_{h}", p, h * dur / 2, (h + 1) * dur / 2))
    for key, path, t0, t1 in jobs:
        if key in done or not path.exists():
            continue
        frames, efps = decode(path.read_bytes(), 8.0, 1e9)
        rec = Rec(eng)
        s = FaceSampler(rec, AggParams())          # production defaults: 150 attempts, 8 CPU s, backoff

        def window(gen):
            for i, fr in enumerate(gen):
                t = i / efps
                if t < t0:
                    continue
                if t > t1 or t - t0 > 40:
                    break
                if fr.shape[1] > 2560:
                    fr = cv2.resize(fr, (2560, round(fr.shape[0] * 2560 / fr.shape[1])), interpolation=cv2.INTER_AREA)
                yield fr
        t = time.time()
        track(window(frames), pose, on_frame=s.on_frame)
        done[key] = {"faces": [rec_of(o) for o in rec.obs], "prod": s.result().get("vec")}
        print(key, len(rec.obs), "faces", round(time.time() - t, 1), "s", flush=True)
        out_p.write_bytes(pickle.dumps(done))


def cmd_celeba(a) -> None:
    import pyarrow.parquet as pq
    eng = FaceEngineV2("Megatron")
    t = pq.read_table(DATA / "celeba/test0.parquet", columns=["image", "celeb_id"])
    ids = t.column("celeb_id").to_pylist()
    imgs = t.column("image").to_pylist()
    by = {}
    for i, c in enumerate(ids):
        by.setdefault(c, []).append(i)
    chosen = sorted(c for c, v in by.items() if len(v) >= 4)[:a.ids]
    out = []
    sizes = (24, 32, 48, 64)

    def detect(img):
        H, W = img.shape[:2]
        pad = max(H, W) // 2                       # context around the face, like a head window
        big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        try:
            fs = eng.faces(np.ascontiguousarray(big), offset=(-pad, -pad))
        except Exception:                          # rare InspireFace pipeline errors count as a miss
            return None
        return fs[0] if fs else None

    t0 = time.time()
    for n, c in enumerate(chosen):
        for i in by[c][:a.per_id]:
            im = cv2.imdecode(np.frombuffer(imgs[i]["bytes"], np.uint8), cv2.IMREAD_COLOR)
            o = detect(im)
            if o is None:
                continue
            out.append(rec_of(o, {"id": c, "img": i, "cond": "native", "target": 0}))
            for s in sizes:
                f = s / max(8.0, o.size)
                small = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                small = cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])[1], cv2.IMREAD_COLOR)
                conds = [("colour", small)]
                if s in (32, 48, 64):
                    g = cv2.cvtColor(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                    conds.append(("grey", g))
                for cond, img in conds:
                    p = detect(img)
                    out.append(rec_of(p, {"id": c, "img": i, "cond": cond, "target": s}) if p is not None else
                               {"id": c, "img": i, "cond": cond, "target": s, "miss": True})
        if n % 25 == 0:
            print(n, len(chosen), len(out), round(time.time() - t0), "s", flush=True)
    (DATA / "celeba.pkl").write_bytes(pickle.dumps(out))
    print("celeba", len(out), "records")


def cmd_embed(a) -> None:
    for name in ("visits", "celeba"):
        p = DATA / f"{name}.pkl"
        if not p.exists():
            continue
        D = pickle.loads(p.read_bytes())
        recs = [r for v in D.values() for r in v["faces"]] if name == "visits" else D
        recs = [r for r in recs if r.get("aligned") is not None]
        for m, path in MODELS.items():
            e = AdaFace(path, threads=a.threads)
            t = time.time()
            todo = [r for r in recs if m not in r]
            V, nrm = e.embed([r["aligned"] for r in todo])
            for r, v, nn in zip(todo, V, nrm):
                r[m], r[m + "_norm"] = v, float(nn)
            print(name, m, len(recs), "crops", round(1000 * (time.time() - t) / max(1, len(recs)), 1), "ms/crop",
                  flush=True)
        p.write_bytes(pickle.dumps(D))


# --- statistics -----------------------------------------------------------------------------------------------
def tar_at_far(gen, imp, far) -> tuple[float, float]:
    imp = np.sort(np.asarray(imp))
    if not len(imp) or not len(gen):
        return float("nan"), float("nan")
    thr = imp[min(len(imp) - 1, int(np.ceil((1 - far) * len(imp))) - 1)] if far * len(imp) >= 1 else imp[-1]
    return float((np.asarray(gen) > thr).mean()), float(thr)


def auc(pos, neg) -> float:
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None]
    return float((p > n).mean() + 0.5 * (p == n).mean()) if p.size and n.size else float("nan")


def boot(fn, groups: list, n=300) -> tuple[float, float]:
    """90 % CI of fn(resampled group list)."""
    vals = []
    for _ in range(n):
        g = [groups[i] for i in RNG.integers(0, len(groups), len(groups))]
        v = fn(g)
        if v == v:
            vals.append(v)
    return (float(np.percentile(vals, 5)), float(np.percentile(vals, 95))) if vals else (float("nan"),) * 2


def fmt(x, d=3):
    return "-" if x is None or x != x else f"{x:.{d}f}"


def ci(x):
    return f"{fmt(x[0], 2)}-{fmt(x[1], 2)}"


# --- CelebA ----------------------------------------------------------------------------------------------------
def celeba_report(models, fused_pairs) -> list[str]:
    D = pickle.loads((DATA / "celeba.pkl").read_bytes())
    native = {}
    for r in D:
        if r["cond"] == "native" and "ada50" in r:
            native.setdefault(r["id"], []).append(r)
    L = ["### CelebA, single image: enrol 1 native image (~90 px), probe 1 other low-res image",
         "",
         "Genuine = same identity, different photo; impostor = every other enrolled identity. TAR at FAR 1 % and "
         "0.1 % (thresholds from the impostor scores of that row); 90 % CI by resampling identities.",
         "",
         "| condition | probes (detected / tried) | model | TAR@1% | CI | TAR@0.1% | CI | AUC |", "|---|---|---|---|---|---|---|---|"]
    ids = sorted(native)
    gal = {c: native[c][0] for c in ids}
    conds = [("colour", 24), ("colour", 32), ("colour", 48), ("colour", 64), ("grey", 32), ("grey", 48), ("grey", 64)]
    cal_store = {}
    for cond, s in conds:
        tried = [r for r in D if r["cond"] == cond and r["target"] == s and r["id"] in gal and r["img"] != gal[r["id"]]["img"]]
        probes = [r for r in tried if not r.get("miss") and "ada50" in r]
        if not probes:
            continue
        G = {m: np.stack([gal[c][m] for c in ids]) for m in models}
        idx = {c: k for k, c in enumerate(ids)}
        S = {m: np.stack([p[m] for p in probes]) @ G[m].T for m in models}      # (P, I)
        # fusion: z-normalise each model with its impostor stats in this table, average
        imask = np.ones_like(S[models[0]], bool)
        for k, p in enumerate(probes):
            imask[k, idx[p["id"]]] = False
        for name, parts in fused_pairs.items():
            zs = []
            for m in parts:
                imp = S[m][imask]
                zs.append((S[m] - imp.mean()) / imp.std())
            S[name] = np.mean(zs, 0)
        per_id = {}
        for k, p in enumerate(probes):
            per_id.setdefault(p["id"], []).append(k)
        groups = list(per_id.values())
        for m in list(models) + list(fused_pairs):
            def stat(gs, far, m=m):
                ks = [k for g in gs for k in g]
                gen = [S[m][k, idx[probes[k]["id"]]] for k in ks]
                imp = S[m][ks][imask[ks]]
                return tar_at_far(gen, imp, far)[0]
            ks = list(range(len(probes)))
            gen = [S[m][k, idx[probes[k]["id"]]] for k in ks]
            imp = S[m][imask]
            if cond == "colour" and s == 48:
                cal_store[m] = imp
            t1, t01 = stat(groups, 1e-2), stat(groups, 1e-3)
            L.append(f"| {cond} {s}px | {len(probes)} / {len(tried)} | {m} | {fmt(t1)} | {ci(boot(lambda g: stat(g, 1e-2), groups))} "
                     f"| {fmt(t01)} | {ci(boot(lambda g: stat(g, 1e-3), groups))} | {fmt(auc(gen, imp))} |")
    return L


def celeba_templates(models) -> list[str]:
    """Visit-like templates: 4 low-res probes of one identity (48/32/24 px mixed) -> one template vs the native
    gallery. Aggregation variants: production weighted mean, best-K, plain mean, AdaFace norm weighting."""
    D = pickle.loads((DATA / "celeba.pkl").read_bytes())
    native, low = {}, {}
    for r in D:
        if r.get("miss") or "ada50" not in r:
            continue
        if r["cond"] == "native":
            native.setdefault(r["id"], []).append(r)
        elif r["cond"] == "colour" and r["target"] in (24, 32, 48):
            low.setdefault(r["id"], []).append(r)
    ids = [c for c in sorted(native) if len(native[c]) >= 1 and len([r for r in low.get(c, []) if r["img"] != native[c][0]["img"]]) >= 3]
    from types import SimpleNamespace as NS

    def obs(r):
        return NS(quality=r["q"], size=r["size"], yaw=r["yaw"], pitch=r["pitch"], det=r["det"], sharp=r["sharp"],
                  liveness=r["live"], expression=None, age_bracket=None, gender=None, ir=r["ir"], crop=None, t=0)
    L = ["", "### CelebA, visit-like templates: probe = all low-res (24/32/48 px) images of one identity, other photos",
         "", "| model | aggregation | TAR@1% | CI | AUC |", "|---|---|---|---|---|"]
    from squirrel_guard.faceid import FaceObs
    for m in models:
        G = np.stack([native[c][0][m] for c in ids])
        for agg in ("prod", "mean", "best3", "best1", "normw"):
            T = []
            for c in ids:
                rs = [r for r in low[c] if r["img"] != native[c][0]["img"]]
                fo = [FaceObs(vec=r[m], **{k: v for k, v in vars(obs(r)).items() if k in FaceObs.__dataclass_fields__})
                      for r in rs]
                V = np.stack([r[m] for r in rs])
                if agg == "prod":
                    t = aggregate(fo, V)
                elif agg == "mean":
                    t = l2n(V.mean(0))
                elif agg.startswith("best"):
                    t = best_k(fo, V, int(agg[4:]))
                else:
                    nw = np.array([r.get(m + "_norm", 1.0) for r in rs]) if m != "isf" else None
                    if nw is None:
                        continue
                    t = aggregate(fo, V, norm_weight=nw / nw.max())
                T.append(t if t is not None else l2n(V.mean(0)))
            if len(T) != len(ids):
                continue
            S = np.stack(T) @ G.T
            n = len(ids)
            groups = list(range(n))

            def stat(gs):
                gen = [S[i, i] for i in gs]
                imp = [S[i, j] for i in gs for j in range(n) if j != i]
                return tar_at_far(gen, imp, 1e-2)[0]
            gen = np.diag(S)
            imp = S[~np.eye(n, dtype=bool)]
            L.append(f"| {m} | {agg} | {fmt(stat(groups))} | {ci(boot(stat, groups, 200))} | {fmt(auc(gen, imp))} |")
    L.append(f"\n{len(ids)} identities.")
    return L


# --- in-domain (camera) ---------------------------------------------------------------------------------------
def owner_labels(R) -> dict:
    out = {}
    R = [{**r, "id": str(r["id"])} for r in R]
    fa = {r["id"] for r in R if r["match_kind"] in ("face_template", "face")
          or (((r.get("face_meta") or {}).get("match")) or {}).get("status") == "match"}
    for i in fa:
        out[i] = "A"
    fv = {r["id"]: l2n(np.asarray(r["face_vec"], np.float32)) for r in R if r.get("face_vec")}
    ref = [fv[i] for i in fa if i in fv]
    t = l2n(np.mean(ref, 0)) if ref else None
    for r in R:
        i = r["id"]
        if i in out:
            continue
        if t is not None and i in fv and (r.get("face_meta") or {}).get("n_faces", 0) >= 3 and float(fv[i] @ t) >= 0.40:
            out[i] = "B"
        elif str(r["identity_id"]) == "1" and r["match_kind"] in ("match", "merged"):
            out[i] = "D" if 47 <= int(i) <= 74 else "R"
    p = TEST / "reid-gait/labels.csv"
    if p.exists():
        for row in csv.DictReader(p.open()):
            lab = (row.get("label") or "").strip().lower()
            if lab == "owner":
                out[str(row["sighting_id"])] = "U"
    return out


def visit_obs(faces, m):
    from squirrel_guard.faceid import FaceObs
    fo = [FaceObs(vec=f[m], quality=f["q"], size=f["size"], yaw=f["yaw"], pitch=f["pitch"], det=f["det"],
                  sharp=f["sharp"], liveness=f["live"]) for f in faces]
    for o, f in zip(fo, faces):
        o.ir = f["ir"]
    return fo


def camera_report(models, cal_imp) -> list[str]:
    V = pickle.loads((DATA / "visits.pkl").read_bytes())
    R = {str(r["id"]): r for r in json.loads((DATA / "sightings.json").read_text())}
    lab = owner_labels(list(R.values()))
    ts = {k: np.datetime64(R[k]["ts_start"][:19]) for k in R}
    min_px = 24.0

    def usable(key):
        return [f for f in V[key]["faces"] if f["size"] >= min_px and "ada50" in f]
    owner = {k[1:]: usable(k) for k in V if k.startswith("s") and lab.get(k[1:]) in ("A", "B", "U", "R")}
    owner = {k: v for k, v in owner.items() if len(v) >= 3}
    strangers = {k: usable(k) for k in V if k.startswith("x:")}
    strangers = {k: v for k, v in strangers.items() if len(v) >= 3}
    desk = {k[1:]: usable(k) for k in V if k.startswith("s") and lab.get(k[1:]) == "D"}
    desk = {k: v for k, v in desk.items() if len(v) >= 3}
    L = ["", "## Camera data (owner = identity #1, leave-one-visit-out)", "",
         f"Visits with >= 3 usable faces (>= {min_px:.0f} px): owner {len(owner)} "
         f"(tiers {', '.join(f'{t}={sum(lab[k] == t for k in owner)}' for t in 'ABUR')}), stock strangers {len(strangers)}, "
         f"v4 desk sightings (#47-#74, label unknown) {len(desk)}. All owner visits are from 2026-11-06 15:33-20:14.",
         "",
         "Gallery for each owner probe = the other owner visits more than 15 min away (simultaneous frontyard/v4 "
         "sightings of one walk are not counted as separate evidence). Impostor = stock stranger visit vs that "
         "gallery. Score = best template cosine.", ""]
    variants = [("single mean (prod)", "single"), ("per-visit max", "visit"), ("multi-template (pose x IR)", "multi")]
    L += ["| model | gallery | owner probes | genuine median | genuine min | stranger max | AUC | CI | TAR @ stranger-max | TAR @ CelebA FAR 1% |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    tables = {}
    for m in models:
        T = {}      # key -> (prod template, multi templates)
        for k, fs in list(owner.items()) + list(strangers.items()) + list(desk.items()):
            fo = visit_obs(fs, m)
            Vm = np.stack([f[m] for f in fs])
            T[k] = (aggregate(fo, Vm), multi_templates(fo, Vm))
        for vname, mode in variants:
            gen, gid = [], []
            imp_by_probe = {}
            for k in owner:
                gal = [j for j in owner if j != k and abs((ts[j] - ts[k]).astype("timedelta64[m]").astype(int)) > 15]
                if not gal:
                    continue
                def score(t_probe, gal=gal):
                    if mode == "single":
                        g = l2n(np.mean([T[j][0] for j in gal], 0))
                        return float(t_probe @ g)
                    if mode == "visit":
                        return max(float(t_probe @ T[j][0]) for j in gal)
                    return max(float(t_probe @ v) for j in gal for v in T[j][1].values())
                gen.append(score(T[k][0]))
                gid.append(k)
                imp_by_probe[k] = [score(T[x][0]) for x in strangers]
            imp = [s for v in imp_by_probe.values() for s in v]
            tables[(m, mode)] = (gen, imp, gid)
            groups = list(range(len(gen)))
            a_ci = boot(lambda g: auc([gen[i] for i in g], [s for i in g for s in imp_by_probe[gid[i]]]), groups)
            smax = max(imp) if imp else float("nan")
            thr = np.quantile(cal_imp[m], 0.99) if m in cal_imp else float("nan")
            L.append(f"| {m} | {vname} | {len(gen)} | {fmt(float(np.median(gen)))} | {fmt(min(gen))} | {fmt(smax)} | "
                     f"{fmt(auc(gen, imp))} | {ci(a_ci)} | {fmt(float(np.mean(np.asarray(gen) > smax)))} | "
                     f"{fmt(float(np.mean(np.asarray(gen) > thr)))} (thr {fmt(thr, 2)}) |")
    # per-face: size / IR breakdown against the full owner gallery (leave own visit out)
    L += ["", "### Per face, by size and colour/IR (face vs the owner's other-visit mean template)", "",
          "| model | bucket | owner faces | genuine median | stranger faces | stranger p99 | per-face AUC |",
          "|---|---|---|---|---|---|---|"]
    for m in models:
        allt = {k: aggregate(visit_obs(fs, m), np.stack([f[m] for f in fs])) for k, fs in owner.items()}
        rows = []
        for k, fs in owner.items():
            g = l2n(np.mean([allt[j] for j in owner if j != k and abs((ts[j] - ts[k]).astype("timedelta64[m]").astype(int)) > 15] or [allt[k]], 0))
            rows += [("g", f, float(f[m] @ g)) for f in fs]
        g_all = l2n(np.mean(list(allt.values()), 0))
        rows += [("i", f, float(f[m] @ g_all)) for fs in strangers.values() for f in fs]
        for name, sel in (("24-40 px", lambda f: f["size"] < 40), ("40-64 px", lambda f: 40 <= f["size"] < 64),
                          ("64-96 px", lambda f: 64 <= f["size"] < 96), (">= 96 px", lambda f: f["size"] >= 96),
                          ("IR (grey)", lambda f: modality(f["ir"]) == "ir"), ("colour", lambda f: modality(f["ir"]) != "ir"),
                          ("|yaw| >= 45", lambda f: abs(f["yaw"]) >= 45)):
            gg = [s for t, f, s in rows if t == "g" and sel(f)]
            ii = [s for t, f, s in rows if t == "i" and sel(f)]
            L.append(f"| {m} | {name} | {len(gg)} | {fmt(float(np.median(gg)) if gg else None)} | {len(ii)} | "
                     f"{fmt(float(np.quantile(ii, .99)) if ii else None)} | {fmt(auc(gg, ii)) if gg and ii else '-'} |")
    # desk sightings: descriptive only
    L += ["", "### v4 desk sightings #47-#74 (label unknown): cosine of their visit template to the owner mean", "",
          "| model | n | median | min | max |", "|---|---|---|---|---|"]
    for m in models:
        ot = l2n(np.mean([aggregate(visit_obs(fs, m), np.stack([f[m] for f in fs])) for fs in owner.values()], 0))
        d = [float(aggregate(visit_obs(fs, m), np.stack([f[m] for f in fs])) @ ot) for fs in desk.values()]
        if d:
            L.append(f"| {m} | {len(d)} | {fmt(float(np.median(d)))} | {fmt(min(d))} | {fmt(max(d))} |")
    # temporal: best-K vs production aggregation, leave-one-out single-mean gallery
    L += ["", "### Visit template: production aggregation vs best-K (single-mean gallery)", "",
          "| model | template | genuine median | stranger max | AUC |", "|---|---|---|---|---|"]
    for m in models:
        for agg in ("prod", "best1", "best3", "best5", "best10", "normw"):
            if agg == "normw" and m == "isf":
                continue
            T = {}
            for k, fs in list(owner.items()) + list(strangers.items()):
                fo = visit_obs(fs, m)
                Vm = np.stack([f[m] for f in fs])
                if agg == "prod":
                    T[k] = aggregate(fo, Vm)
                elif agg == "normw":
                    nw = np.array([f[m + "_norm"] for f in fs])
                    T[k] = aggregate(fo, Vm, norm_weight=nw / nw.max())
                else:
                    T[k] = best_k(fo, Vm, int(agg[4:]))
            gen, imp = [], []
            for k in owner:
                gal = [j for j in owner if j != k and abs((ts[j] - ts[k]).astype("timedelta64[m]").astype(int)) > 15]
                if not gal:
                    continue
                g = l2n(np.mean([T[j] for j in gal], 0))
                gen.append(float(T[k] @ g))
                imp += [float(T[x] @ g) for x in strangers]
            L.append(f"| {m} | {agg} | {fmt(float(np.median(gen)))} | {fmt(max(imp))} | {fmt(auc(gen, imp))} |")
    return L, tables, owner, strangers, lab


def fusion_camera(tables, models, cal_imp) -> list[str]:
    L = ["", "### Fusion on camera data (z-normalised with CelebA 48 px impostor statistics)", "",
         "| fusion | gallery | genuine min | stranger max | AUC | TAR @ stranger-max |", "|---|---|---|---|---|---|"]
    for parts in (("isf", "ada101"), ("isf", "ada50"), ("isf", "ada50", "ada101")):
        for mode in ("single", "visit", "multi"):
            if not all((m, mode) in tables and m in cal_imp for m in parts):
                continue
            cal = {m: Calibration.from_impostors(cal_imp[m]) for m in parts}
            gen = np.mean([cal[m].z(tables[(m, mode)][0]) for m in parts], 0)
            imp = np.mean([cal[m].z(tables[(m, mode)][1]) for m in parts], 0)
            L.append(f"| {'+'.join(parts)} | {mode} | {fmt(float(gen.min()), 2)} | {fmt(float(imp.max()), 2)} | "
                     f"{fmt(auc(gen, imp))} | {fmt(float(np.mean(gen > imp.max())))} |")
    return L


def body_side(owner, strangers, lab) -> list[str]:
    """OSNet (production body re-ID) for owner sightings WITHOUT a usable face: descriptive only."""
    R = {str(r["id"]): r for r in json.loads((DATA / "sightings.json").read_text())}
    emb = json.loads((TEST / "reid-gait/embeddings.json").read_text()) if (TEST / "reid-gait/embeddings.json").exists() else {}
    L = ["", "### Body side evidence (OSNet x0.25, production) when the face is unusable", ""]
    prod = next((v for k, v in emb.get("models", emb).items() if "prod" in k), None) if isinstance(emb, dict) else None
    if not prod or "visit" not in prod:
        return L + ["embeddings.json from reid_gait_eval.py not found / unexpected layout; skipped."]
    vis, stock = prod["visit"], prod.get("stock", {})
    face_ok = set(owner)
    lab_owner = [k for k, t in lab.items() if t in ("A", "B", "U")]
    noface = [k for k in lab_owner if k not in face_ok and k in vis]
    withf = [k for k in lab_owner if k in face_ok and k in vis]
    if not withf:
        return L + ["no owner sightings with both a usable face and a body embedding."]
    c = l2n(np.mean([l2n(np.asarray(vis[k])) for k in withf], 0))
    gen = [float(l2n(np.asarray(vis[k])) @ c) for k in noface]
    imp = [float(l2n(np.asarray(v)) @ c) for v in stock.values()]
    L.append(f"Owner sightings (A/B/U) without a usable face but with a body embedding: {len(noface)}; body cosine to the "
             f"owner's face-confirmed centroid: {', '.join(fmt(g, 2) for g in gen) or '-'}; stock strangers: n={len(imp)}, "
             f"max {fmt(max(imp) if imp else None, 2)}. Stature: no per-sighting stature from the rig is stored in this "
             "snapshot (height_cm empty), so it could not be evaluated.")
    return L


def cmd_report(a) -> None:
    models = ["isf", "ada50", "ada101"]
    L = ["# faceid v2 evaluation", ""]
    cal_imp = {}
    if (DATA / "celeba.pkl").exists():
        L += celeba_report(models, {"isf+ada101": ("isf", "ada101"), "isf+ada50": ("isf", "ada50")})
        L += celeba_templates(models)
        D = pickle.loads((DATA / "celeba.pkl").read_bytes())
        nat = {}
        for r in D:
            if r["cond"] == "native" and "ada50" in r:
                nat.setdefault(r["id"], r)
        pr = [r for r in D if r["cond"] == "colour" and r["target"] == 48 and not r.get("miss") and "ada50" in r]
        for m in models:
            G = np.stack([nat[c][m] for c in nat])
            ids = list(nat)
            S = np.stack([r[m] for r in pr]) @ G.T
            cal_imp[m] = np.array([S[i, j] for i, r in enumerate(pr) for j, c in enumerate(ids) if c != r["id"]])
    if (DATA / "visits.pkl").exists():
        CL, tables, owner, strangers, lab = camera_report(models, cal_imp)
        L += CL
        L += fusion_camera(tables, models, cal_imp)
        L += body_side(owner, strangers, lab)
    txt = "\n".join(L)
    print(txt)
    (DATA / "report.md").write_text(txt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["visits", "celeba", "embed", "report"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ids", type=int, default=326)
    ap.add_argument("--per-id", type=int, default=8)
    a = ap.parse_args()
    {"visits": cmd_visits, "celeba": cmd_celeba, "embed": cmd_embed, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()

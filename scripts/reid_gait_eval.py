"""Re-ID and gait signatures on this household's own sightings: old (OSNet x0.25, GaitBase) vs candidates.

All person data stays OUTSIDE the repo, in DATA (default ~/.cache/squirrel-guard-test/reid-gait). Steps:

  fetch    read-only: sightings table (recorder host, via the persons-ui container), person crops (recorder host), event clips (Frigate
           on gpuserver, only events still in its ~3-day retention). Temp files on gpuserver are sg_-prefixed and removed.
  crops    profiler-style visit crops: 6 frames spread over each clip, YOLOv8-OIV7, largest person >= 64 px.
           Also crops of the stranger clips in DATA/stock (two "visits" per clip: first and second half).
  embed    every re-ID model given with --model NAME=PATH (default: the production OSNet + whatever v2 ONNX files
           exist in models/) on the stored best crop and on the visit crops -> DATA/embeddings.json
  gait     8 fps tracks with the profiler's pose model + kinematics.track, GaitBase on the cut silhouettes (same as
           the profiler) and a pose-angle signature from the 17 keypoints -> DATA/gait.pkl
  report   labels (below) + metrics -> markdown tables on stdout, DATA/report.json
  sheet    labelling contact sheet (DATA/labelling/index.html) + labels.csv template for the user
  bench    CPU latency per crop / per gait sequence, --threads (default 4), nothing else running please

Labels (never guessed from body re-ID):
  A  face decision: match_kind face_template / face, or face_meta.match.status == "match"  -> owner
  B  face evidence: the visit's face template (face_vec) has cosine >= 0.40 (FACE_MATCH_SIM) to the mean template
     of the A visits, with >= 3 faces. Same face model, but WITHOUT the liveness/count gates -> owner (weaker)
  U  DATA/labels.csv filled in by the user (sighting_id,label: owner | other-<tag> | unsure); overrides A/B
  S  stranger clips in DATA/stock (Pixabay, one clip per uploader; each clip = one non-household person)
Different-person statistics come only from U "other-*" labels and from S. S is out of domain (bright, sharp,
outdoor stock footage), so owner-vs-S similarities are an optimistic stand-in for a real stranger at the door.

Usage: .venv/bin/python scripts/reid_gait_eval.py {fetch,crops,embed,gait,report,sheet,bench} [--data DIR]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODELS = ROOT / "models"
DATA = Path.home() / ".cache/squirrel-guard-test/reid-gait"
FACE_MATCH_SIM = 0.40
STOCK_EXCLUDE = {"px426_0"}            # first half of that clip is an umbrella, no person

SQL = """SELECT id, frigate_event, camera, ts_start, ts_end, identity_id, match_kind, match_sim, possible_id,
 possible_sim, reviewed, crops, crop_file, embedding, gait_vec, face_vec, face_meta, face, attributes FROM sightings
 ORDER BY ts_start"""


# --- fetch (read-only) ----------------------------------------------------------------------------------------
def cmd_fetch(a) -> None:
    d = a.data
    d.mkdir(parents=True, exist_ok=True)
    py = ("import os, json, psycopg\nfrom psycopg.rows import dict_row\n"
          "with psycopg.connect(os.environ['PG_URL'], row_factory=dict_row) as c:\n"
          "    c.read_only = True\n"
          f"    rows = c.execute('''{SQL}''').fetchall()\n"
          "for r in rows:\n"
          "    r['ts_start'] = r['ts_start'].isoformat(); r['ts_end'] = r['ts_end'].isoformat() if r['ts_end'] else None\n"
          "print(json.dumps(rows))\n")
    if a.print_query:
        print(py)
        return
    pr = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "db-host",
                         "cd /srv/camstack && docker compose exec -T persons-ui python -"],
                        input=py, capture_output=True, text=True)
    if pr.returncode != 0:
        raise SystemExit(f"recorder host query failed ({pr.returncode}): {pr.stderr.strip()[-300:]}")
    out = pr.stdout
    (d / "sightings.json").write_text(out)
    R = json.loads(out)
    print(f"{len(R)} sightings")
    (d / "crops").mkdir(exist_ok=True)
    names = "\n".join(r["crop_file"] for r in R if r["crop_file"]) + "\n"
    tar = subprocess.run(["ssh", "db-host", "cd /srv/camstack/snapshots/persons && tar cf - -T -"],
                         input=names.encode(), capture_output=True, check=True).stdout
    subprocess.run(["tar", "xf", "-", "-C", str(d / "crops")], input=tar, check=True)
    if a.no_clips:
        return
    (d / "clips").mkdir(exist_ok=True)
    evs = [r["frigate_event"] for r in R if not (d / "clips" / f"sg_{r['frigate_event']}.mp4").exists()]
    evs = [e for e in evs if all(ch.isalnum() or ch in ".-" for ch in e)]
    if evs:
        sh = ("for e in " + " ".join(evs) + "; do\n"
              "  c=$(docker exec frigate curl -s -o /tmp/sg_$e.mp4 -w '%{http_code}' "
              "http://127.0.0.1:5000/api/events/$e/clip.mp4)\n"
              "  docker cp frigate:/tmp/sg_$e.mp4 /tmp/sg_$e.mp4 >/dev/null 2>&1; docker exec frigate rm -f /tmp/sg_$e.mp4\n"
              "  [ \"$c\" = 200 ] || rm -f /tmp/sg_$e.mp4; echo \"$e $c\"\ndone\n")
        log = subprocess.run(["ssh", "gpuserver", "bash -s"], input=sh, capture_output=True, text=True).stdout
        print(sum(" 200" in ln for ln in log.splitlines()), "of", len(evs), "clips available")
        subprocess.run(["scp", "-q", "gpuserver:/tmp/sg_*.mp4", str(d / "clips")])
        subprocess.run(["ssh", "gpuserver", "rm -f /tmp/sg_*.mp4"])
        for f in (d / "clips").glob("*.mp4"):
            if f.stat().st_size < 2048:
                f.unlink()


# --- crops ----------------------------------------------------------------------------------------------------
def largest_person_crops(det, frames) -> list:
    from squirrel_guard.profiler import PEOPLE
    out = []
    for fr in frames:
        ds = [d for d in det(fr) if d.label.lower() in PEOPLE]
        if not ds:
            continue
        d = max(ds, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
        x1, y1, x2, y2 = d.box
        if y2 - y1 >= 64:
            out.append(fr[y1:y2, x1:x2])
    return out


def cmd_crops(a) -> None:
    from squirrel_guard.detector import Detector
    from squirrel_guard.profiler import frames_from_clip
    d = a.data
    det = Detector(str(MODELS / "yolov8s-oiv7.onnx"), conf=0.3, threads=a.threads)
    for sub, items in (("crops_visit", [(str(r["id"]), d / "clips" / f"sg_{r['frigate_event']}.mp4", None)
                                        for r in load_sightings(d)]),
                       ("crops_stock", [(f"{p.stem}_{h}", p, h) for p in sorted((d / "stock").glob("*.mp4"))
                                        for h in (0, 1)])):
        out = d / sub
        out.mkdir(exist_ok=True)
        idx = {}
        for key, clip, half in items:
            if not clip.exists():
                continue
            if half is None:
                frames = frames_from_clip(clip.read_bytes(), 6)
            else:
                cap = cv2.VideoCapture(str(clip))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frames = []
                for i in range(6):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int((half + (i + 0.5) / 6) * total / 2))
                    ok, fr = cap.read()
                    if ok:
                        frames.append(fr)
                cap.release()
            names = []
            for k, c in enumerate(largest_person_crops(det, frames)):
                n = f"{key}_{k}.jpg"
                cv2.imwrite(str(out / n), c, [cv2.IMWRITE_JPEG_QUALITY, 95])
                names.append(n)
            idx[key] = names
        (out / "index.json").write_text(json.dumps(idx))
        print(sub, len(idx), "visits,", sum(map(len, idx.values())), "crops")


# --- embed ----------------------------------------------------------------------------------------------------
def default_models() -> dict:
    m = {"osnet_x0_25 (prod)": MODELS / "osnet_x0_25_msmt17.onnx"}
    for stem in ("osnet_ain_x1_0_msmt17", "clipreid_vitb16_msmt17", "solider_swin_t_msmt17", "solider_swin_s_msmt17"):
        for suf in ("", ".int8"):
            p = MODELS / f"{stem}{suf}.onnx"
            if p.exists():
                m[stem + suf] = p
    return m


def cmd_embed(a) -> None:
    from squirrel_guard import reid_v2
    d = a.data
    models = dict(kv.split("=", 1) for kv in a.model) if a.model else default_models()
    R = load_sightings(d)
    best = {r["id"]: cv2.imread(str(d / "crops" / r["crop_file"])) for r in R if r["crop_file"]}
    vidx = json.loads((d / "crops_visit/index.json").read_text())
    visit = {int(k): [cv2.imread(str(d / "crops_visit" / n)) for n in v] for k, v in vidx.items() if v}
    sidx = json.loads((d / "crops_stock/index.json").read_text()) if (d / "crops_stock/index.json").exists() else {}
    stock = {k: [cv2.imread(str(d / "crops_stock" / n)) for n in v] for k, v in sidx.items()
             if v and k not in STOCK_EXCLUDE and (d / "stock" / (k.rsplit("_", 1)[0] + ".mp4")).exists()}
    res = {"models": {}}
    for name, path in models.items():
        e = reid_v2.Embedder(str(path), threads=a.threads)
        l2 = lambda v: v / (np.linalg.norm(v) + 1e-12)
        res["models"][name] = {
            "path": str(path), "mb": Path(path).stat().st_size / 1e6,
            "best": {i: e([c])[0].tolist() for i, c in best.items()},
            "visit": {i: e.one(cs).tolist() for i, cs in visit.items()},
            "stock": {k: e.one(cs).tolist() for k, cs in stock.items()},
            "stock_best": {k: l2(e([cs[len(cs) // 2]])[0]).tolist() for k, cs in stock.items()}}
        print(name, "done", flush=True)
    (d / "embeddings.json").write_text(json.dumps(res))


# --- gait -----------------------------------------------------------------------------------------------------
def _ang(a, b, c) -> float:
    v1, v2 = a - b, c - b
    return math.degrees(math.acos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1)))


def pose_signature(track: list, min_frames: int = 10, conf: float = 0.4) -> np.ndarray | None:
    """Hand-built skeleton signature from 17-keypoint frames: percentiles (10/50/90) of left/right-sorted knee
    angles, thigh and shank angles from vertical, ankle spread, arm swing (all / box height or in degrees), torso
    lean and the torso : leg length ratio. None with fewer than min_frames frames with clear legs and torso."""
    rows = []
    for t in track:
        if t is None:
            continue
        kp = np.asarray(t["kp"], np.float32)
        if (kp[[5, 6, 11, 12, 13, 14, 15, 16], 2] < conf).any():
            continue
        bh = t["box"][3] - t["box"][1]
        P = kp[:, :2]
        knees = sorted([_ang(P[11], P[13], P[15]), _ang(P[12], P[14], P[16])])
        vert = lambda u, v: math.degrees(math.atan2(abs(v[0] - u[0]), v[1] - u[1] + 1e-6))
        thighs = sorted([vert(P[11], P[13]), vert(P[12], P[14])])
        shanks = sorted([vert(P[13], P[15]), vert(P[14], P[16])])
        sh, hp, an = P[[5, 6]].mean(0), P[[11, 12]].mean(0), P[[15, 16]].mean(0)
        lean = math.degrees(math.atan2(abs(sh[0] - hp[0]), hp[1] - sh[1] + 1e-6))
        ratio = np.linalg.norm(sh - hp) / (np.linalg.norm(hp - an) + 1e-6)
        spread = abs(P[15, 0] - P[16, 0]) / bh
        arms = sorted([abs(P[9, 0] - P[5, 0]) / bh if kp[9, 2] >= conf else np.nan,
                       abs(P[10, 0] - P[6, 0]) / bh if kp[10, 2] >= conf else np.nan])
        rows.append(knees + thighs + shanks + [spread, lean, ratio] + arms)
    if len(rows) < min_frames:
        return None
    X = np.asarray(rows, np.float64)
    f = np.nanpercentile(X, [10, 50, 90], axis=0).T.reshape(-1)
    return np.nan_to_num(f, nan=0.0).astype(np.float32)


def cmd_gait(a) -> None:
    from squirrel_guard.gait_embed import GaitEmbedder
    from squirrel_guard.kinematics import decode, track
    from squirrel_guard.pose import PoseEstimator
    d = a.data
    pose = PoseEstimator(str(MODELS / "yolov8s-pose.onnx"), threads=a.threads)
    ge = GaitEmbedder(str(MODELS / "gaitbase_grew.onnx"), str(MODELS / "yolov8s-seg-320.onnx"), threads=a.threads)
    jobs = [(str(r["id"]), d / "clips" / f"sg_{r['frigate_event']}.mp4") for r in load_sightings(d)]
    jobs += [(p.stem, p) for p in sorted((d / "stock").glob("*.mp4"))]
    out = pickle.loads((d / "gait.pkl").read_bytes()) if (d / "gait.pkl").exists() and not a.redo else {}
    for key, p in jobs:
        if not p.exists() or key in out:
            continue
        frames, fps = decode(p.read_bytes(), fps=8.0, max_s=40.0)
        sils = []

        def on_frame(fr, q):
            if q is not None and len(sils) < 720:
                s = ge.silhouettes([fr], [q.box])
                if len(s):
                    sils.append(s[0])
        trk, size = track(frames, pose, on_frame=on_frame)
        vec = None
        if len(sils) >= 10:
            v = ge.part_features(np.stack(sils)).reshape(-1).astype(np.float32)
            vec = (v / (np.linalg.norm(v) + 1e-12)).tolist()
        tr = [None if q is None else {"box": list(map(float, q.box)), "kp": q.kp.tolist()} for q in trk]
        out[key] = {"fps": fps, "n_frames": len(trk), "n_sils": len(sils), "gaitbase": vec, "track": tr}
        print(key, len(trk), "frames", len(sils), "silhouettes", "vec" if vec else "-", flush=True)
        (d / "gait.pkl").write_bytes(pickle.dumps(out))


# --- labels + metrics -----------------------------------------------------------------------------------------
def load_sightings(d: Path) -> list:
    return json.loads((d / "sightings.json").read_text())


def labels(d: Path, R: list) -> dict:
    """sighting id -> (label, tier). label 'owner' or 'other-<tag>'; tiers A, B, U (see module doc)."""
    out = {}
    fa = {r["id"] for r in R if r["match_kind"] in ("face_template", "face")
          or (((r.get("face_meta") or {}).get("match")) or {}).get("status") == "match"}
    for i in fa:
        out[i] = ("owner", "A")
    fv = {r["id"]: np.asarray(r["face_vec"], np.float32) for r in R if r.get("face_vec")}
    ref = [fv[i] / np.linalg.norm(fv[i]) for i in fa if i in fv]
    if ref:
        t = np.mean(ref, 0)
        t /= np.linalg.norm(t)
        for r in R:
            i = r["id"]
            if i in fv and i not in out and (r.get("face_meta") or {}).get("n_faces", 0) >= 3 \
                    and float(fv[i] @ t) / np.linalg.norm(fv[i]) >= FACE_MATCH_SIM:
                out[i] = ("owner", "B")
    p = d / "labels.csv"
    if p.exists():
        for row in csv.DictReader(p.open()):
            lab = (row.get("label") or "").strip().lower()
            if lab and lab != "unsure":
                out[int(row["sighting_id"])] = (lab, "U")
            elif lab == "unsure":
                out.pop(int(row["sighting_id"]), None)
    return out


def auc(pos, neg) -> float | None:
    if not len(pos) or not len(neg):
        return None
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def evaluate(emb: dict, R: dict, lab: dict, strangers: dict, tiers: set) -> dict:
    """emb: sighting id -> unit vector; strangers: key -> unit vector (each a different person, by clip).
    Positives = pairs of sightings with the same label; negatives = different labels + every labelled sighting vs
    every stranger. Retrieval: each owner sighting queries {other labelled sightings} + strangers."""
    ids = sorted(i for i, (l, t) in lab.items() if t in tiers and i in emb)
    if len(ids) < 2:
        return {"n": len(ids)}
    ts = {i: dt.datetime.fromisoformat(R[i]["ts_start"]) for i in ids}
    same, diff = [], []
    split = {"same_cam": [], "diff_cam": [], "lt30min": [], "ge30min": []}
    for a_ in range(len(ids)):
        for b_ in range(a_ + 1, len(ids)):
            i, j = ids[a_], ids[b_]
            c = float(emb[i] @ emb[j])
            if lab[i][0] == lab[j][0]:
                same.append(c)
                if lab[i][0] == "owner":
                    split["same_cam" if R[i]["camera"] == R[j]["camera"] else "diff_cam"].append(c)
                    split["lt30min" if abs((ts[i] - ts[j]).total_seconds()) < 1800 else "ge30min"].append(c)
            else:
                diff.append(c)
    owner = [i for i in ids if lab[i][0] == "owner"]
    st = list(strangers.values())
    vs_s = [float(emb[i] @ s) for i in owner for s in st]
    neg = diff + vs_s
    r1, aps = [], []
    for i in owner:
        g = [(float(emb[i] @ emb[j]), lab[j][0] == "owner") for j in ids if j != i] + [(float(emb[i] @ s), False) for s in st]
        if not any(h for _, h in g):
            continue
        g.sort(key=lambda x: -x[0])
        hit = np.array([h for _, h in g], float)
        r1.append(hit[0])
        aps.append(float((np.cumsum(hit) / np.arange(1, len(hit) + 1))[hit == 1].mean()))
    oo = [c for c in same]
    thr = float(max(neg)) if neg else None
    # 90 % interval of the owner-vs-stranger AUC by resampling SIGHTINGS (owner) and stranger visits, not pairs:
    # pairs from the same few sightings are not independent
    rng = np.random.default_rng(0)
    O = np.array([[float(emb[i] @ emb[j]) for j in owner] for i in owner])
    S = np.array([[float(emb[i] @ s) for s in st] for i in owner]) if st else None
    boots = []
    for _ in range(500 if S is not None and len(owner) >= 3 else 0):
        oi = rng.integers(0, len(owner), len(owner))
        si = rng.integers(0, S.shape[1], S.shape[1])
        sub = O[np.ix_(oi, oi)]
        pos = sub[np.triu_indices(len(oi), 1)][oi[np.triu_indices(len(oi), 1)[0]] != oi[np.triu_indices(len(oi), 1)[1]]]
        a_b = auc(pos, S[np.ix_(oi, si)].ravel()) if len(pos) else None
        if a_b is not None:
            boots.append(a_b)
    q = lambda x, p: round(float(np.percentile(x, p)), 3) if len(x) else None
    return {"n": len(ids), "n_owner": len(owner), "n_pos_pairs": len(same), "n_neg_pairs": len(neg),
            "pos_median": q(oo, 50), "pos_p10": q(oo, 10), **{k: q(v, 50) for k, v in split.items()},
            **{f"n_{k}": len(v) for k, v in split.items()},
            "neg_median": q(neg, 50), "neg_max": round(thr, 3) if thr is not None else None,
            "auc": auc(oo, neg), "auc_ci90": [round(float(np.percentile(boots, 5)), 2), round(float(np.percentile(boots, 95)), 2)] if boots else None,
            "frac_ge_070": float((np.asarray(oo) >= 0.70).mean()) if oo else None,
            "frac_ge_055": float((np.asarray(oo) >= 0.55).mean()) if oo else None,
            "rank1": float(np.mean(r1)) if r1 else None, "mAP": float(np.mean(aps)) if aps else None,
            "tpr_at_zero_fp": float((np.asarray(oo) > thr).mean()) if thr is not None and oo else None}


def descriptive_unknowns(emb: dict, R: dict, lab: dict) -> dict:
    """Mean cosine of each unlabelled identity's sightings to the labelled-owner centroid (for review only)."""
    owner = [emb[i] for i, (l, _) in lab.items() if l == "owner" and i in emb]
    if not owner:
        return {}
    c = np.mean(owner, 0)
    c /= np.linalg.norm(c)
    out: dict = {}
    for i, r in R.items():
        if i in lab or i not in emb:
            continue
        out.setdefault(r["identity_id"], []).append(float(emb[i] @ c))
    return {k: round(float(np.mean(v)), 3) for k, v in sorted(out.items(), key=lambda kv: kv[0] or 0)}


def fmt(v) -> str:
    if isinstance(v, list):
        return "-".join(f"{x:.2f}" for x in v)
    return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def table(rows: list, cols: list) -> str:
    out = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(k)) if k != "model" else r[k] for _, k in cols) + " |")
    return "\n".join(out)


COLS = [("model", "model"), ("owner pairs", "n_pos_pairs"), ("same-person cos median", "pos_median"),
        ("p10", "pos_p10"), ("same cam", "same_cam"), ("other cam", "diff_cam"), ("<30 min apart", "lt30min"),
        (">=30 min apart", "ge30min"), ("diff-person cos median", "neg_median"), ("max", "neg_max"),
        ("AUC", "auc"), ("AUC 90% CI", "auc_ci90"), ("rank-1", "rank1"), ("mAP", "mAP"), ("TPR at 0 FP", "tpr_at_zero_fp")]


def cmd_report(a) -> None:
    d = a.data
    Rl = load_sightings(d)
    R = {r["id"]: r for r in Rl}
    lab = labels(d, Rl)
    n_t = {t: sum(1 for v in lab.values() if v[1] == t) for t in "ABU"}
    others = sorted({l for l, _ in lab.values() if l != "owner"})
    print(f"sightings {len(R)}; labelled owner A={n_t['A']} B={n_t['B']} user={n_t['U']}; "
          f"other people (user labels): {others or 'none'}")
    rep = {"labels": {str(k): v for k, v in lab.items()}, "reid": {}, "gait": {}}
    E = json.loads((d / "embeddings.json").read_text())
    for tiers, title in (({"A", "U"}, "A (+U)"), ({"A", "B", "U"}, "A+B (+U)")):
        for mode in ("visit", "best"):
            rows = []
            for name, m in E["models"].items():
                emb = {int(k): np.asarray(v) for k, v in m["best"].items()}
                if mode == "visit":
                    emb.update({int(k): np.asarray(v) for k, v in m["visit"].items()})
                st = {k: np.asarray(v) for k, v in m["stock" if mode == "visit" else "stock_best"].items()}
                r = evaluate(emb, R, lab, st, tiers)
                r["model"] = name
                r["unknown_identities_to_owner"] = descriptive_unknowns(emb, R, lab)
                rows.append(r)
            rep["reid"][f"{title}/{mode}"] = rows
            print(f"\n### Re-ID, labels {title}, {'visit (6-crop mean, like the profiler)' if mode == 'visit' else 'stored best crop only'}"
                  f" - {rows[0].get('n_owner')} owner sightings, {len(E['models'][rows[0]['model']]['stock'])} stranger visits\n")
            print(table(rows, COLS))
    gp = d / "gait.pkl"
    if gp.exists():
        G = pickle.loads(gp.read_bytes())
        G = {k: v for k, v in G.items() if k.isdigit() or (d / "stock" / f"{k}.mp4").exists()}
        sig = {k: pose_signature(v["track"]) for k, v in G.items()}
        ok = {k: s for k, s in sig.items() if s is not None}
        if ok:
            M = np.stack(list(ok.values()))
            mu, sd = M.mean(0), M.std(0) + 1e-6
            ok = {k: (s - mu) / sd for k, s in ok.items()}
            ok = {k: s / (np.linalg.norm(s) + 1e-12) for k, s in ok.items()}
        for gname, vecs in (("GaitBase GREW (prod)", {k: np.asarray(v["gaitbase"]) for k, v in G.items() if v["gaitbase"]}),
                            ("pose-angle signature", ok)):
            emb = {int(k): v for k, v in vecs.items() if k.isdigit()}
            st = {k: v for k, v in vecs.items() if not k.isdigit() and (d / "stock" / f"{k}.mp4").exists()}
            rows = []
            for tiers, title in (({"A", "U"}, "A (+U)"), ({"A", "B", "U"}, "A+B (+U)")):
                r = evaluate(emb, R, lab, st, tiers)
                r["model"] = f"{gname}, labels {title}"
                r["n_strangers"] = len(st)
                rows.append(r)
            rep["gait"][gname] = rows
            print(f"\n### Gait: {gname} ({len(emb)} household sequences, {len(st)} stranger sequences)\n")
            print(table(rows, COLS))
    (d / "report.json").write_text(json.dumps(rep, indent=1, default=str))


# --- labelling sheet ------------------------------------------------------------------------------------------
def cmd_sheet(a) -> None:
    d = a.data
    R = load_sightings(d)
    out = d / "labelling"
    (out / "img").mkdir(parents=True, exist_ok=True)
    groups: dict = {}
    for r in R:
        groups.setdefault(r["identity_id"], []).append(r)
    parts = ["<!doctype html><meta charset=utf-8><title>Sighting labels</title><style>body{font:14px sans-serif;"
             "margin:16px;background:#fff;color:#111}h2{margin:24px 0 6px}.g{display:flex;flex-wrap:wrap;gap:6px}"
             "figure{margin:0;width:128px}img{height:160px;max-width:128px;object-fit:contain;background:#eee}"
             "figcaption{font-size:11px}</style>",
             "<h1>Label these sightings</h1><p>Local file, never upload. For each sighting fill <code>label</code> in "
             "<code>labels.csv</code>: <b>owner</b>, <b>other-a</b>, <b>other-b</b>, ... (same tag = same person), "
             "or <b>unsure</b>. Labelling a whole identity row-by-row is fine. Scores are deliberately not shown so they "
             "do not steer the labels. Sightings are grouped by the identity the bridge filed them under.</p>"]
    rows = []
    for iid in sorted(groups, key=lambda x: (x is None, x)):
        rs = groups[iid]
        parts.append(f"<h2>identity {iid} ({len(rs)} sightings)</h2><div class=g>")
        for r in rs:
            if r["crop_file"] and (d / "crops" / r["crop_file"]).exists():
                shutil.copy(d / "crops" / r["crop_file"], out / "img" / f"{r['id']}.jpg")
            t = r["ts_start"][5:16].replace("T", " ")
            parts.append(f"<figure><img src='img/{r['id']}.jpg' loading=lazy><figcaption><b>#{r['id']}</b> "
                         f"{html.escape(r['camera'])}<br>{t}</figcaption></figure>")
            rows.append({"sighting_id": r["id"], "identity_id": r["identity_id"], "camera": r["camera"],
                         "ts_start": r["ts_start"][:19], "label": ""})
        parts.append("</div>")
    (out / "index.html").write_text("\n".join(parts))
    tmpl = d / "labels_template.csv"
    with tmpl.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(out / "index.html", tmpl, "(copy to labels.csv and fill in 'label')")


# --- bench ----------------------------------------------------------------------------------------------------
def cmd_bench(a) -> None:
    from squirrel_guard import reid_v2
    d = a.data
    crops = [cv2.imread(str(p)) for p in sorted((d / "crops_visit").glob("*.jpg"))[:30]]
    print(f"threads={a.threads}; median of 20 runs after 3 warm-up\n\n| model | MB | ms / crop | ms / 6-crop visit |\n|---|---|---|---|")
    models = dict(kv.split("=", 1) for kv in a.model) if a.model else default_models()
    for name, path in models.items():
        e = reid_v2.Embedder(str(path), threads=a.threads)
        for c in crops[:3]:
            e([c])
        t1, t6 = [], []
        for k in range(20):
            t = time.perf_counter(); e([crops[k % len(crops)]]); t1.append(time.perf_counter() - t)
            t = time.perf_counter(); e.one(crops[k % 20: k % 20 + 6]); t6.append(time.perf_counter() - t)
        print(f"| {name} | {Path(path).stat().st_size / 1e6:.1f} | {1000 * np.median(t1):.1f} | {1000 * np.median(t6):.1f} |")
    if (d / "gait.pkl").exists():
        import onnxruntime as ort
        G = pickle.loads((d / "gait.pkl").read_bytes())
        so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
        s = ort.InferenceSession(str(MODELS / "gaitbase_grew.onnx"), so, providers=["CPUExecutionProvider"])
        x = np.random.rand(1, 60, 64, 44).astype(np.float32)
        s.run(None, {"sils": x})
        ts = []
        for _ in range(10):
            t = time.perf_counter(); s.run(None, {"sils": x}); ts.append(time.perf_counter() - t)
        trk = next(v["track"] for v in G.values() if v["gaitbase"])
        t = time.perf_counter()
        for _ in range(20):
            pose_signature(trk)
        print(f"\nGaitBase, 60 silhouettes: {1000 * np.median(ts):.1f} ms (the silhouettes themselves: YOLOv8s-seg-320, "
              f"~30 ms/frame on gpuserver); pose-angle signature from an existing track: {50 * (time.perf_counter() - t):.2f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["fetch", "crops", "embed", "gait", "report", "sheet", "bench"])
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--model", action="append", help="NAME=PATH (embed/bench; repeatable)")
    ap.add_argument("--redo", action="store_true", help="gait: recompute everything")
    ap.add_argument("--no-clips", action="store_true", help="fetch: only the table and the crops")
    ap.add_argument("--print-query", action="store_true", help="fetch: print the remote read-only snippet, run nothing")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()

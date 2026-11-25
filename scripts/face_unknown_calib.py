"""Calibrate the auto-tracking threshold for UNKNOWN faces (facematch.FACE_UNKNOWN_SIM_V2 / FACE_UNKNOWN_SIM).

Unknown tracking compares one visit template with the visit templates of earlier unnamed identities (best of the
last N), so the relevant scores are VISIT-vs-VISIT, not visit-vs-enrolled. A false merge files a stranger under
someone else's unnamed identity (it never names anyone), so the target is a very low impostor rate.

Sources (local only, never in git; see scripts/faceid_eval.py for how they were built):
  1. CelebA visit-like templates: each identity's low-res photos (24-64 px, colour and grey) split into two disjoint
     halves -> two "visits" (production aggregator); genuine = halves of one identity, impostor = halves of two.
  2. Stock stranger clips (visits.pkl x:<uploader>_0/_1): genuine = the two halves of one uploader, impostor =
     different uploaders.
  3. Camera sightings exported from the recorder host (--sightings JSON: id, identity_id, kind, match_kind, ts, v2, isf,
     meta): owner visits (identity 1 with an enrolled-face match) more than 15 min apart = genuine; owner vs stock
     strangers = impostor. Other unnamed identities are unlabelled and only described.
Usage: .venv/bin/python scripts/face_unknown_calib.py --sightings sightings_faces.json [--model ada101|isf]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy/recorder/bridge"))
from squirrel_guard.faceid import AggParams, FaceAggregator, FaceObs, l2n  # noqa: E402
from squirrel_guard.faceid_v2 import aggregate  # noqa: E402
import facematch  # noqa: E402

DATA = Path.home() / ".cache/squirrel-guard-test/faceid-v2"
RNG = np.random.default_rng(11)


def obs(r, m):
    return FaceObs(vec=r[m], quality=r["q"], size=r["size"], yaw=r["yaw"], pitch=r["pitch"], det=r["det"],
                   sharp=r["sharp"], liveness=r.get("live"))


def visit(recs, m):
    """(template, gate meta) with the production aggregator; meta from the InspireFace pool like the profiler."""
    recs = [r for r in recs if r["size"] >= 24 and m in r]
    if len(recs) < 1:
        return None, None
    a = FaceAggregator(AggParams(keep_crops=0))
    for r in recs:
        a.add(obs(r, "isf"))
    meta = a.result()
    t = aggregate([obs(r, m) for r in recs], np.stack([r[m] for r in recs])) if m != "isf" else \
        (np.asarray(meta["vec"], np.float32) if meta.get("vec") else None)
    return t, meta


def far_table(gen, imp, label):
    gen, imp = np.asarray(gen), np.sort(np.asarray(imp))
    print(f"\n{label}: genuine n={len(gen)} median {np.median(gen):.3f} p5 {np.percentile(gen, 5):.3f} min {gen.min():.3f};"
          f" impostor n={len(imp)} p99 {np.percentile(imp, 99):.3f} p99.9 {np.percentile(imp, 99.9):.3f}"
          f" p99.99 {np.percentile(imp, 99.99):.3f} max {imp.max():.3f}")
    for thr in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
        print(f"  thr {thr:.2f}: false-merge rate {np.mean(imp >= thr):.2e} ({int((imp >= thr).sum())}/{len(imp)}),"
              f" genuine pass {np.mean(gen >= thr):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sightings")
    ap.add_argument("--model", default="ada101")
    a = ap.parse_args()
    m = a.model
    # 1. CelebA
    D = pickle.loads((DATA / "celeba.pkl").read_bytes())
    by = {}
    for r in D:
        if r["cond"] != "native" and m in r:
            by.setdefault(r["id"], []).append(r)
    T = []
    for i, rs in by.items():
        if len(rs) < 6:
            continue
        idx = RNG.permutation(len(rs))
        h1, h2 = [rs[j] for j in idx[: len(rs) // 2]], [rs[j] for j in idx[len(rs) // 2:]]
        t1, m1 = visit(h1, m)
        t2, m2 = visit(h2, m)
        if t1 is not None and t2 is not None:
            T.append((i, l2n(t1), l2n(t2), m1, m2))
    A = np.stack([t[1] for t in T])
    B = np.stack([t[2] for t in T])
    S = A @ B.T
    n = len(T)
    gen = np.diag(S)
    imp = np.concatenate([S[~np.eye(n, dtype=bool)], (A @ A.T)[np.triu_indices(n, 1)], (B @ B.T)[np.triu_indices(n, 1)]])
    far_table(gen, imp, f"CelebA visit-vs-visit ({n} identities, {m})")
    # 2. stock strangers
    V = pickle.loads((DATA / "visits.pkl").read_bytes())
    xs = {}
    for k, v in V.items():
        if k.startswith("x:"):
            t, meta = visit(v["faces"], m)
            if t is not None and meta.get("n_faces", 0) >= 3:
                xs[k] = l2n(t)
    keys = sorted(xs)
    g2, i2 = [], []
    for x in range(len(keys)):
        for y in range(x + 1, len(keys)):
            s = float(xs[keys[x]] @ xs[keys[y]])
            (g2 if keys[x].rsplit("_", 1)[0] == keys[y].rsplit("_", 1)[0] else i2).append(s)
    if g2 and i2:
        far_table(g2, i2, f"Stock stranger clips ({len(keys)} visits, {m})")
    # 3. camera
    if a.sightings:
        R = json.loads(Path(a.sightings).read_text())
        col = "v2" if m.startswith("ada") else "isf"
        rows = [r for r in R if r.get(col) and r.get("meta")]
        usable = [r for r in rows if not facematch.gate_reasons(r["meta"])]
        print(f"\ncamera sightings with a {col} template: {len(rows)}; passing every gate: {len(usable)}"
              f" (gate failures: {dict(sorted(_count([g for r in rows for g in facematch.gate_reasons(r['meta'])]).items()))})")
        owner = [r for r in rows if r["identity_id"] == 1 and (r["match_kind"] == "face_template" or
                 ((r["meta"].get("match") or {}).get("status") == "match"))]
        gen = [float(l2n(p[col]) @ l2n(q[col])) for i, p in enumerate(owner) for q in owner[i + 1:]
               if abs(p["ts"] - q["ts"]) > 900]
        gate_gen = [float(l2n(p[col]) @ l2n(q[col])) for i, p in enumerate(owner) for q in owner[i + 1:]
                    if abs(p["ts"] - q["ts"]) > 900 and not facematch.gate_reasons(p["meta"])
                    and not facematch.gate_reasons(q["meta"])]
        imp = [float(l2n(p[col]) @ x) for p in owner for x in xs.values()] if col == "v2" and m == "ada101" else []
        if gen and imp:
            far_table(gen, imp, f"Camera owner visit pairs (>15 min apart) vs stock strangers ({len(owner)} owner visits)")
        if gate_gen:
            print(f"  owner pairs where both visits pass the gates: n={len(gate_gen)} median {np.median(gate_gen):.3f}"
                  f" min {min(gate_gen):.3f}")
        ot = l2n(np.mean([l2n(r[col]) for r in owner], 0)) if owner else None
        others = {}
        for r in rows:
            if r["identity_id"] != 1 and ot is not None:
                others.setdefault(r["identity_id"], []).append(float(l2n(r[col]) @ ot))
        for iid, ss in sorted(others.items()):
            print(f"  unlabelled identity #{iid}: {len(ss)} visits, cosine to the owner mean: median {np.median(ss):.3f}"
                  f" max {max(ss):.3f}")


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


if __name__ == "__main__":
    main()

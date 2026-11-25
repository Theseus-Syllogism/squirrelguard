"""Correlation strength between one visit and earlier sightings, for the replay's investigation pane.

Pure numpy (the recorder host has no AVX; plain numpy works). One number per edge, built the same way as the bridge's
readable scores (deploy/recorder/bridge/explain.py): start from the re-ID appearance probability and nudge the odds with
the other evidence. The result is a RANKING aid with an "estimated" probability, never a decision: the bridge's
matching logic is untouched and nothing here names anyone.

Evidence per edge (each listed with its value, so the UI can say why two visits are linked):
  reid      cosine of the OSNet re-ID embeddings (sightings.embedding; appearance, mostly clothing)
  face      cosine of the two visits' face templates (sightings.face_vec) when both have one: correlation only,
            no name (names come only from enrolled household/known templates, docs/faceid.md)
  gait      GaitBase signature distance (1 - cosine of sightings.gait_vec), a weak hint
  identity  both sightings are filed under the same identity (the bridge's or the user's grouping)
"""
from __future__ import annotations

import math
import os

import numpy as np

E = os.environ
MATCH_SIM = float(E.get("PERSON_MATCH_SIM", "0.70"))        # same defaults as bridge/explain.py
POSSIBLE_SIM = float(E.get("PERSON_POSSIBLE_SIM", "0.55"))
GAIT_SIG_SAME = float(E.get("GAIT_SIG_SAME", "0.05"))
GAIT_SIG_DIFF = float(E.get("GAIT_SIG_DIFF", "0.15"))
FACE_MATCH_SIM = float(E.get("FACE_MATCH_SIM", "0.40"))     # bridge/facematch.py: same-person face cosine
FACE_WEAK_SIM = 0.30
FACE_DIFF_SIM = 0.15


def band_of(p: float) -> str:
    return "strong" if p >= 0.9 else "likely" if p >= 0.7 else "possible" if p >= 0.4 else "unlikely"


def pct(p: float) -> int:
    return min(99, max(1, round(p * 100)))


def appearance_prob(cos: float) -> float:
    """Re-ID cosine -> rough P(same person); same logistic as explain.appearance_prob (MATCH_SIM -> 70 %,
    POSSIBLE_SIM -> 40 %)."""
    lo, hi = math.log(0.4 / 0.6), math.log(0.7 / 0.3)
    slope = (hi - lo) / max(1e-6, MATCH_SIM - POSSIBLE_SIM)
    return 1 / (1 + math.exp(-(slope * (cos - MATCH_SIM) + hi)))


def l2n(v) -> np.ndarray | None:
    if v is None:
        return None
    a = np.asarray(v, np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else None


def cosines(q, rows: list) -> list:
    """cosine of q to each row (None where a row or q is missing / of another length)."""
    qn = l2n(q)
    out = [None] * len(rows)
    if qn is None:
        return out
    idx = [i for i, r in enumerate(rows) if r is not None and len(r) == len(qn)]
    if not idx:
        return out
    m = np.asarray([rows[i] for i in idx], np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
    for i, c in zip(idx, m @ qn):
        out[i] = float(c)
    return out


def strength(reid_cos=None, face_cos=None, gait_dist=None, same_identity=False, reviewed=False) -> dict:
    """-> {p, pct, band, edges: [{kind, value, effect, text}]}; edges strongest-first by effect."""
    edges = []
    if reid_cos is not None:
        p = appearance_prob(reid_cos)
        edges.append({"kind": "reid", "value": round(reid_cos, 3), "effect": "base",
                      "text": f"appearance (re-ID) cosine {reid_cos:.2f}"})
    else:
        p = 0.2
    p = min(max(p, 1e-4), 1 - 1e-4)
    odds = p / (1 - p)
    if face_cos is not None:
        if face_cos >= FACE_MATCH_SIM:
            odds *= 4.0; eff = "raises"
        elif face_cos >= FACE_WEAK_SIM:
            odds *= 1.5; eff = "raises"
        elif face_cos < FACE_DIFF_SIM:
            odds *= 0.5; eff = "lowers"
        else:
            eff = "neutral"
        edges.append({"kind": "face", "value": round(face_cos, 3), "effect": eff,
                      "text": f"face templates cosine {face_cos:.2f} (same person usually >= {FACE_MATCH_SIM:.2f})"})
    if gait_dist is not None:
        if gait_dist <= GAIT_SIG_SAME:
            odds *= 1.5; eff = "raises"
        elif gait_dist >= GAIT_SIG_DIFF:
            odds *= 0.6; eff = "lowers"
        else:
            eff = "neutral"
        edges.append({"kind": "gait", "value": round(gait_dist, 3), "effect": eff,
                      "text": f"gait signature distance {gait_dist:.3f} (weak hint)"})
    if same_identity:
        odds *= 3.0 if reviewed else 2.0
        edges.append({"kind": "identity", "value": None, "effect": "raises",
                      "text": "filed under the same identity" + (" (reviewed)" if reviewed else "")})
    p = odds / (1 + odds)
    return {"p": round(p, 4), "pct": pct(p), "band": band_of(p), "edges": edges}


def rank(visit: dict, cands: list, top_k: int = 12) -> list:
    """visit/cands: dicts with id, identity_id, embedding, face_vec, gait_vec (optional), reviewed.
    -> cands (copied) with 'reid_cos', 'face_cos', 'gait_dist' and 'strength', best first, at most top_k."""
    if not cands:
        return []
    rc = cosines(visit.get("embedding"), [c.get("embedding") for c in cands])
    fc = cosines(visit.get("face_vec"), [c.get("face_vec") for c in cands])
    gc = cosines(visit.get("gait_vec"), [c.get("gait_vec") for c in cands])
    out = []
    for c, r, f, g in zip(cands, rc, fc, gc):
        same = visit.get("identity_id") is not None and c.get("identity_id") == visit.get("identity_id")
        s = strength(r, f, None if g is None else 1.0 - g, same, bool(c.get("reviewed")))
        d = {k: v for k, v in c.items() if k not in ("embedding", "face_vec", "gait_vec")}
        d.update({"reid_cos": None if r is None else round(r, 3), "face_cos": None if f is None else round(f, 3),
                  "gait_dist": None if g is None else round(1.0 - g, 3), "strength": s})
        out.append(d)
    out.sort(key=lambda d: d["strength"]["p"], reverse=True)
    return out[:top_k]


def vehicle_strength(dt_s: float, together_before: int, window_s: float = 600.0) -> dict:
    """A vehicle seen near the visit: closer in time and seen with this person before = stronger (co-occurrence,
    not identity)."""
    near = max(0.0, 1.0 - abs(dt_s) / window_s)
    p = min(0.95, 0.2 + 0.35 * near + 0.1 * min(4, together_before))
    edges = [{"kind": "time", "value": round(dt_s), "effect": "base",
              "text": f"seen {abs(dt_s) / 60:.0f} min {'before' if dt_s < 0 else 'after'} this visit" if abs(dt_s) >= 60
              else "seen within a minute of this visit"}]
    if together_before:
        edges.append({"kind": "cooccur", "value": together_before, "effect": "raises",
                      "text": f"seen near this identity {together_before} time{'s' if together_before != 1 else ''} before"})
    return {"p": round(p, 4), "pct": pct(p), "band": band_of(p), "edges": edges}

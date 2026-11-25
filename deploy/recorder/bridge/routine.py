"""Routine model: what is normal at this camera, and how surprising a new person visit is.

Learned from the stored sightings (last 90 days), numpy only (the recorder host has no AVX). Four independent parts, each
scored in BITS of surprise relative to a "no idea" baseline, so they add up and read the same way:

  hour   von Mises kernel density over time of day (bandwidth ~1.3 h) mixed with max(2 %, 3/(n+3)) uniform:
         bits = -log2(24 * p(hour)); 0 = as common as average, 4 = 16x rarer than average.
  route  entry side -> exit side of the frame (left/right/top/bottom/middle), Dirichlet(0.25) counts:
         bits = -log2(25 * p(route)).
  stay   log duration and stillness vs the median/MAD of past visits; only LONGER/STILLER than usual counts:
         bits = -log2(one-sided tail probability) of the robust z.
  path   route SHAPE: each path resampled to 16 points (Frigate image coordinates, no calibration needed) and
         compressed with PCA (a linear autoencoder) fitted on past paths; reconstruction error vs past errors,
         one-sided robust z -> bits as for stay. Spreads are floored and each part is capped at 12 bits, so a
         history of near-identical visits cannot turn a small deviation into an enormous score.

A visit is UNUSUAL when there are >= MIN_HISTORY past visits, the total is above a self-calibrated threshold
(the larger of MIN_BITS and the 98th percentile of the past visits' own totals, so about 1 in 50 ordinary visits
would trip it) and at least one part is >= 4 bits. Every part comes with a sentence for the alert/why page.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np

MIN_HISTORY = 30
MIN_BITS = 8.0
PART_MIN_BITS = 4.0
KAPPA = 8.0              # von Mises concentration for time of day (~1.3 h)
UNIFORM_FLOOR = 0.02     # uniform share of the hour density: max(floor, 3 / (n + 3)) -> fades in as history grows
ALPHA = 0.25             # Dirichlet pseudo-count per route
PART_CAP = 12.0          # one part can say at most "4000x rarer than usual"
SIDES = ("left", "right", "top", "bottom", "middle")
PATH_N = 16


def side(pt) -> str:
    if not pt:
        return "middle"
    x, y = float(pt[0]), float(pt[1])
    d = {"left": x, "right": 1 - x, "top": y, "bottom": 1 - y}
    s = min(d, key=d.get)
    return s if d[s] < 0.15 else "middle"


def _tail_bits(z: float) -> float:
    """One-sided surprise of a robust z (only z > 0 counts): -log2 P(Z >= z) - 1 (so z = 0 gives 0 bits)."""
    if not math.isfinite(z) or z <= 0:
        return 0.0
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return max(0.0, -math.log2(max(p, 1e-12)) - 1.0)


def _robust(vals, floor: float):
    """(median, robust sd) with the sd floored: identical past visits must not make every small deviation huge."""
    v = np.asarray([x for x in vals if x is not None and math.isfinite(x)], float)
    if len(v) < 5:
        return None
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    return med, max(mad, floor)


def _hour(ts: dt.datetime, tz) -> float:
    t = ts.astimezone(tz)
    return t.hour + t.minute / 60 + t.second / 3600


def _path_vec(b: dict):
    p = b.get("path")
    if not p or len(p) != PATH_N:
        return None
    return np.asarray(p, float).reshape(-1)


class Routine:
    def __init__(self, history: list[dict], tz):
        """history: [{"ts": aware datetime, "behavior": {...}}] of past person visits (excluding the one scored)."""
        self.tz, self.n = tz, len(history)
        self.hours = np.array([_hour(h["ts"], tz) for h in history], float)
        self.routes: dict[str, int] = {}
        for h in history:
            b = h.get("behavior") or {}
            r = f"{side(b.get('entry'))}->{side(b.get('exit'))}"
            self.routes[r] = self.routes.get(r, 0) + 1
        bs = [h.get("behavior") or {} for h in history]
        self.dur = _robust([math.log1p(b["duration_s"]) for b in bs if b.get("duration_s") is not None], 0.3)
        self.still = _robust([math.log1p(b.get("motionless_frames") or 0) for b in bs], 0.7)
        X = [v for v in (_path_vec(b) for b in bs) if v is not None]
        self.pca = None
        if len(X) >= 10:
            X = np.asarray(X)
            mu = X.mean(0)
            U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
            k = int(min(4, len(X) - 1, np.searchsorted(np.cumsum(S ** 2) / max((S ** 2).sum(), 1e-12), 0.9) + 1))
            V = Vt[:k]
            err = np.linalg.norm((X - mu) - (X - mu) @ V.T @ V, axis=1)
            self.pca = (mu, V, _robust(list(err), 0.08))   # 0.08 image units over 16 points
        self.threshold = MIN_BITS
        if self.n >= MIN_HISTORY:
            own = [self._parts(h["ts"], h.get("behavior") or {}, fit=True) for h in history]
            self.threshold = max(MIN_BITS, float(np.percentile([sum(p["bits"] for p in o) for o in own], 98)))

    # -- parts -------------------------------------------------------------------------------------------------
    def _p_hour(self, h: float) -> float:
        if not len(self.hours):
            return 1 / 24
        ang = (h - self.hours) * 2 * math.pi / 24
        dens = np.exp(KAPPA * (np.cos(ang) - 1)).mean()          # unnormalised von Mises mixture, peak 1
        norm = 1 / (2 * math.pi * np.i0(KAPPA) * math.exp(-KAPPA)) * (2 * math.pi / 24)  # -> density per hour
        mix = max(UNIFORM_FLOOR, 3.0 / (len(self.hours) + 3))
        return (1 - mix) * float(dens) * norm + mix / 24

    def _parts(self, ts, b: dict, fit: bool = False) -> list[dict]:
        out = []
        h = _hour(ts, self.tz)
        ph = self._p_hour(h)
        near = int((np.abs(((self.hours - h + 12) % 24) - 12) <= 1).sum()) if len(self.hours) else 0
        out.append({"code": "hour", "bits": max(0.0, -math.log2(24 * ph)),
                    "text": f"{ts.astimezone(self.tz):%H:%M}: {near} of {self.n} past visits were within an hour of this time"})
        r = f"{side(b.get('entry'))}->{side(b.get('exit'))}"
        c = self.routes.get(r, 0)
        pr = (c + ALPHA) / (self.n + ALPHA * len(SIDES) ** 2)
        out.append({"code": "route", "bits": max(0.0, -math.log2(len(SIDES) ** 2 * pr)),
                    "text": f"came in {r.split('->')[0]}, left {r.split('->')[1]}: {c} of {self.n} past visits did"})
        if self.dur and b.get("duration_s") is not None:
            z = (math.log1p(b["duration_s"]) - self.dur[0]) / self.dur[1]
            out.append({"code": "stay", "bits": _tail_bits(z),
                        "text": f"stayed {b['duration_s']:.0f} s (typical {math.expm1(self.dur[0]):.0f} s)"})
        if self.still and b.get("motionless_frames") is not None:
            z = (math.log1p(b["motionless_frames"] or 0) - self.still[0]) / self.still[1]
            out.append({"code": "still", "bits": _tail_bits(z),
                        "text": f"stood still for {b['motionless_frames']} frames (typical {math.expm1(self.still[0]):.0f})"})
        v = _path_vec(b)
        if self.pca and v is not None and self.pca[2]:
            mu, V, (med, mad) = self.pca
            e = float(np.linalg.norm((v - mu) - (v - mu) @ V.T @ V))
            out.append({"code": "path", "bits": _tail_bits((e - med) / mad),
                        "text": f"path shape differs from usual routes (error {e:.2f} vs typical {med:.2f})"})
        for p in out:
            p["bits"] = round(min(PART_CAP, p["bits"]), 2)
        return out

    def score(self, ts, b: dict) -> dict:
        parts = self._parts(ts, b)
        total = round(sum(p["bits"] for p in parts), 2)
        ready = self.n >= MIN_HISTORY
        top = max(parts, key=lambda p: p["bits"]) if parts else None
        unusual = bool(ready and total >= self.threshold and top and top["bits"] >= PART_MIN_BITS)
        reasons = [p["text"] for p in sorted(parts, key=lambda p: -p["bits"]) if p["bits"] >= 2.0]
        return {"bits": total, "threshold": round(self.threshold, 2), "history": self.n, "ready": ready,
                "unusual": unusual, "parts": parts, "reasons": reasons,
                "note": None if ready else f"learning: {self.n} of {MIN_HISTORY} visits needed before alerts"}

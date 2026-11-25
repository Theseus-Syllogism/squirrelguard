"""Face geometry comparison for the bridge (numpy-free copy of squirrel_guard/facegeom.py merge/compare).

KEEP IN STEP with squirrel_guard/facegeom.py (tests/test_facegeom.py checks that both give the same results).
Explanatory only: who a face belongs to is decided by facematch.decide on the 512-d embedding, never here.
"""
from __future__ import annotations

import math

VERSION = "facegeom/1"

# feature -> (kind, label). "prop" = 2D proportion in the frontalised face plane / IPD; "depth" = relative depth / IPD
# (positive = toward the camera); "index" = dimensionless.
FEATURES = {
    "nose_width": ("prop", "Nose width (alae)"),
    "nose_length": ("prop", "Nose length (nasion to base)"),
    "mouth_width": ("prop", "Mouth width"),
    "eye_width_r": ("prop", "Right eye width"),
    "eye_width_l": ("prop", "Left eye width"),
    "eye_spacing": ("prop", "Inner eye spacing"),
    "cheek_width": ("prop", "Face width at cheekbones"),
    "jaw_width": ("prop", "Jaw width"),
    "face_height": ("prop", "Face height (brow to chin)"),
    "brow_eye": ("prop", "Brow to eye height"),
    "philtrum": ("prop", "Philtrum length"),
    "asymmetry": ("index", "Left/right asymmetry (0 = mirror symmetric)"),
    "nose_protrusion": ("depth", "Nose tip protrusion"),
    "eye_socket_depth": ("depth", "Eye socket depth"),
    "cheek_depth": ("depth", "Cheek contour set-back"),
    "jaw_depth": ("depth", "Jaw contour set-back"),
}
# Between-visit within-person spread floor (IPD units), used by compare(); set from scripts/facegeom_eval.py
# (docs/facegeom.md). A feature not listed falls back to DEFAULT_FLOOR.
WITHIN_SD = {
    "nose_width": 0.035, "nose_length": 0.04, "mouth_width": 0.05, "eye_width_r": 0.03, "eye_width_l": 0.03,
    "eye_spacing": 0.03, "cheek_width": 0.08, "jaw_width": 0.08, "face_height": 0.08, "brow_eye": 0.04,
    "philtrum": 0.03, "asymmetry": 0.02, "nose_protrusion": 0.04, "eye_socket_depth": 0.03, "cheek_depth": 0.06,
    "jaw_depth": 0.06,
}
DEFAULT_FLOOR = 0.05
# Features whose within-person repeatability was clearly better than the between-person spread on the evaluation
# clips (docs/facegeom.md). The others are still computed and stored but rendered as "noisy".
STABLE_FEATURES: tuple[str, ...] = ("eye_spacing", "nose_width", "nose_length", "face_height", "brow_eye",
                                    "nose_protrusion")


def merge(geoms: list[dict]) -> dict | None:
    """Several visits' facegeom (enrolled templates of one identity) -> one enrolled geometry: per feature the
    n-weighted mean of the visit medians, spread = pooled within-visit spread plus the between-visit spread."""
    geoms = [g for g in geoms if g and g.get("features")]
    if not geoms:
        return None
    out = {}
    for k in FEATURES:
        rows = [g["features"][k] for g in geoms if (g["features"].get(k) or {}).get("value") is not None]
        if not rows:
            continue
        n = [max(1, int(r.get("n") or 1)) for r in rows]
        N = sum(n)
        mu = sum(r["value"] * w for r, w in zip(rows, n)) / N
        within = sum((r.get("spread") or 0.0) ** 2 * w for r, w in zip(rows, n)) / N
        between = sum((r["value"] - mu) ** 2 * w for r, w in zip(rows, n)) / N
        sp = math.sqrt(within + between)
        out[k] = {"value": round(mu, 4), "spread": round(sp, 4), "sem": round(1.2533 * sp / math.sqrt(N), 4),
                  "n": N, "visits": len(rows)}
    return {"features": out, "visits": len(geoms), "version": VERSION}


def compare(visit: dict | None, enrolled: dict | None, within_sd: dict | None = None) -> dict | None:
    """Per-feature difference of a visit's geometry vs an identity's enrolled geometry.
    z = (visit - enrolled) / sqrt(sem_visit^2 + sem_enrolled^2 + floor^2), floor = measured between-visit spread of
    the same person (WITHIN_SD). EXPLANATORY ONLY: the identity decision is the face embedding's."""
    if not visit or not enrolled or not visit.get("features") or not enrolled.get("features"):
        return None
    floors = within_sd or WITHIN_SD
    rows = {}
    for k, (kind, label) in FEATURES.items():
        a, b = visit["features"].get(k) or {}, enrolled["features"].get(k) or {}
        if a.get("value") is None or b.get("value") is None:
            continue
        sd = math.sqrt((a.get("sem") or 0.0) ** 2 + (b.get("sem") or 0.0) ** 2 + floors.get(k, DEFAULT_FLOOR) ** 2)
        d = a["value"] - b["value"]
        z = d / sd
        rows[k] = {"visit": a["value"], "enrolled": b["value"], "diff": round(d, 4),
                   "rel_pct": round(100 * d / b["value"], 1) if abs(b["value"]) > 1e-3 and kind != "depth" else None,
                   "z": round(z, 2), "verdict": "consistent" if abs(z) < 2 else "differs somewhat" if abs(z) < 3
                   else "differs", "stable": k in STABLE_FEATURES}
    st = [r for k, r in rows.items() if r["stable"]]
    return {"explanatory": True,
            "note": "Shape comparison for explanation only; who this is was decided by the face embedding.",
            "features": rows, "n_stable": len(st), "n_stable_consistent": sum(r["verdict"] == "consistent" for r in st)}

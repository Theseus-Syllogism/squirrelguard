"""Face template matching against ENROLLED household/known identities (numpy only: the recorder host has no AVX).

Input: the profiler's per-visit face template ("faceid": 512-d vector aggregated over the visit's best faces, see
squirrel_guard/faceid.py) and the enrolled templates (table face_templates, only for identities of kind
household/known whose face was enrolled from the persons UI with consent).

Decision (decide()):
  - no usable template                    -> status "none"   (FACE_TOO_SMALL when the faces were too small)
  - best enrolled identity >= FACE_MATCH_SIM and >= FACE_MARGIN above the next identity:
      gates: liveness ok, >= FACE_MIN_FACES inlier faces, weighted quality >= FACE_MIN_QUALITY,
             consistency >= FACE_MIN_CONSISTENCY
      all pass                            -> status "match"  (FACE_TEMPLATE_MATCH): decides the identity, outranks re-ID
      liveness fails                      -> status "liveness_fail" (FACE_LIVENESS_FAIL): NEVER marks household/known
      other gate fails                    -> status "weak" (FACE_TOO_FEW / FACE_LOW_QUALITY / FACE_INCONSISTENT)
  - otherwise                             -> status "no_match" (FACE_NO_MATCH: a face was seen, not an enrolled one)
Nobody is ever named from a face unless they are enrolled. Apparent expression is not an input here.

Auto-tracking of UNKNOWN faces (decide_unknown(), FACE_AUTO_TRACK, default on): when the enrolled decision is not a
match and the template passes every gate above, it is compared with the recent visit templates of UNNAMED unknown
identities (kind 'unknown', no label). >= FACE_UNKNOWN_SIM_V2 (AdaFace; FACE_UNKNOWN_SIM for InspireFace) and
FACE_MARGIN above the next identity -> that unnamed identity ("face_unknown"); otherwise the face belongs to a new
unnamed identity. It never names anyone and never makes anyone trusted: only the persons UI (the user) labels or
enrols. Thresholds: scripts/face_unknown_calib.py (2026-11-08): AdaFace visit-vs-visit impostors max 0.323 over
197,820 CelebA pairs and 0.163 on camera (owner vs stock strangers); owner visit pairs pass 0.40 in 82 %.
"""
from __future__ import annotations

import os

import numpy as np

E = os.environ
# Calibrated with scripts/face_eval.py (Megatron pack, 2026-11-06): see docs/faceid.md for the numbers.
FACE_MATCH_SIM = float(E.get("FACE_MATCH_SIM", "0.40"))
# AdaFace IR-101 (squirrel_guard/faceid_v2.py, docs/faceid-v2.md): its own score scale. Interim 0.30 from the 2026-11-07
# evaluation (weakest owner visit 0.40, strongest stand-in stranger 0.12); recalibrate once real strangers are labelled.
FACE_MATCH_SIM_V2 = float(E.get("FACE_MATCH_SIM_V2", "0.30"))
FACE_V2 = E.get("FACE_V2", "1") not in ("0", "false", "no")
FACE_MARGIN = float(E.get("FACE_MARGIN", "0.08"))
FACE_MIN_FACES = int(E.get("FACE_MIN_FACES", "3"))
FACE_MIN_QUALITY = float(E.get("FACE_MIN_QUALITY", "0.10"))
FACE_MIN_CONSISTENCY = float(E.get("FACE_MIN_CONSISTENCY", "0.35"))
FACE_MIN_PX = float(E.get("FACE_MIN_PX", "32"))
ENROLL_KINDS = ("household", "known")
FACE_AUTO_TRACK = E.get("FACE_AUTO_TRACK", "1") not in ("0", "false", "no")
FACE_UNKNOWN_SIM_V2 = float(E.get("FACE_UNKNOWN_SIM_V2", "0.40"))
FACE_UNKNOWN_SIM = float(E.get("FACE_UNKNOWN_SIM", "0.50"))          # InspireFace fallback (CelebA impostor max 0.42)
FACE_UNKNOWN_SPLIT_V2 = float(E.get("FACE_UNKNOWN_SPLIT_V2", "0.20"))  # below: the face clearly is not that identity
FACE_UNKNOWN_SPLIT = float(E.get("FACE_UNKNOWN_SPLIT", "0.25"))
FACE_UNKNOWN_LAST_N = int(E.get("FACE_UNKNOWN_LAST_N", "8"))          # visit templates kept per unnamed identity


def l2n(v) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def gate_reasons(meta: dict) -> list[str]:
    """Why a template may NOT decide an identity (empty list = usable)."""
    out = []
    if not (meta.get("liveness") or {}).get("ok"):
        out.append("FACE_LIVENESS_FAIL")
    if int(meta.get("n_faces") or 0) < FACE_MIN_FACES:
        out.append("FACE_TOO_FEW")
    if float(meta.get("quality") or 0) < FACE_MIN_QUALITY:
        out.append("FACE_LOW_QUALITY")
    if float(meta.get("size_px") or 0) < FACE_MIN_PX:
        out.append("FACE_TOO_SMALL")
    if float(meta.get("consistency") or 0) < FACE_MIN_CONSISTENCY:
        out.append("FACE_INCONSISTENT")
    return out


def decide(vec, meta: dict, enrolled: list[dict], thr: float | None = None) -> dict:
    """vec: visit template (or None); meta: the rest of the faceid message; enrolled: dicts with template_id,
    identity_id, label, kind, vec, model. Returns {status, identity, label, sim, second_sim, template, reasons}."""
    meta = meta or {}
    thr = FACE_MATCH_SIM if thr is None else thr
    res = {"status": "none", "identity": None, "label": None, "sim": None, "second_sim": None, "template": None,
           "reasons": [], "thresholds": {"sim": thr, "margin": FACE_MARGIN, "min_faces": FACE_MIN_FACES,
                                         "min_quality": FACE_MIN_QUALITY}}
    if vec is None:
        if meta.get("reason") == "too_small" or (meta.get("max_size_px") or 0) and meta.get("max_size_px") < FACE_MIN_PX:
            res["reasons"] = ["FACE_TOO_SMALL"]
        return res
    v = l2n(vec)
    model = meta.get("model")
    best: dict = {}                                  # identity -> (sim, template row)
    for t in enrolled:
        if t.get("kind") not in ENROLL_KINDS:
            continue                                  # only household/known identities can be named by face
        if model and t.get("model") and t["model"] != model:
            continue                                  # embeddings of different model packs are not comparable
        s = float(np.dot(v, l2n(t["vec"])))
        if t["identity_id"] not in best or s > best[t["identity_id"]][0]:
            best[t["identity_id"]] = (s, t)
    gates = gate_reasons(meta)
    ranked = sorted(best.values(), key=lambda x: -x[0])
    if ranked:
        s1, t1 = ranked[0]
        s2 = ranked[1][0] if len(ranked) > 1 else None
        res.update(sim=round(s1, 4), second_sim=None if s2 is None else round(s2, 4), template=t1.get("template_id"))
        if s1 >= thr and (s2 is None or s1 - s2 >= FACE_MARGIN):
            res.update(identity=t1["identity_id"], label=t1.get("label"))
            if not gates:
                res.update(status="match", reasons=["FACE_TEMPLATE_MATCH"])
            elif "FACE_LIVENESS_FAIL" in gates:
                res.update(status="liveness_fail", identity=None, candidate=t1["identity_id"], reasons=gates)
            else:
                res.update(status="weak", identity=None, candidate=t1["identity_id"], reasons=gates)
            return res
    res["status"] = "no_match"
    res["reasons"] = ["FACE_NO_MATCH"] + [g for g in gates if g in ("FACE_LIVENESS_FAIL", "FACE_TOO_SMALL")]
    return res


def face_prob(sim: float, thr: float | None = None) -> float:
    """Template cosine -> displayed probability for a gated match: 90 % at the threshold, up to 99 %."""
    return min(0.99, 0.90 + max(0.0, sim - (FACE_MATCH_SIM if thr is None else thr)) * 0.5)


def decide_any(fid: dict, enrolled: list[dict]) -> dict:
    """AdaFace (fid["v2"]) against AdaFace templates when the visit has one AND an enrolled identity has a template of
    that model; otherwise InspireFace as before. The other engine's verdict is kept as "shadow" for comparison."""
    v2 = fid.get("v2") if FACE_V2 else None
    base = decide(fid.get("vec"), fid, enrolled)
    base["engine"] = fid.get("model") or "inspireface"
    if not v2 or v2.get("vec") is None or not any(t.get("model") == v2.get("model") and t.get("kind") in ENROLL_KINDS
                                                  for t in enrolled):
        return base
    d2 = decide(v2["vec"], {**fid, "model": v2.get("model")}, enrolled, thr=FACE_MATCH_SIM_V2)
    d2["engine"] = v2.get("model")
    d2["shadow"] = {k: base.get(k) for k in ("engine", "status", "identity", "sim", "second_sim")}
    return d2


def better(new: dict | None, old: dict | None) -> bool:
    """Is a new visit template better evidence than the stored one? (more inlier faces x quality)"""
    if not new or new.get("vec") is None:
        return False
    if not old or old.get("n_faces") is None:
        return True
    return float(new.get("weight_sum") or 0) > float(old.get("weight_sum") or 0) or \
        int(new.get("n_faces") or 0) > int(old.get("n_faces") or 0) + 2


def usable(fid: dict | None) -> bool:
    """A visit template that may be used for auto-tracking: same gates as an enrolled match."""
    return bool(fid) and fid.get("vec") is not None and not gate_reasons(fid)


def decide_unknown(fid: dict, gallery: list[dict]) -> dict:
    """The visit's template vs the recent visit templates of UNNAMED unknown identities.
    gallery: dicts {identity_id, vec, model} (several per identity; built by the bridge from stored sightings and
    occupant faces of kind 'unknown' identities without a label). AdaFace (fid["v2"]) when the gallery has templates
    of that model, else InspireFace. Returns {status: match | new | unusable, identity (match only), sim,
    second_sim, engine, reasons, thresholds, sims (per identity; for the bridge's split rule, not stored)}.
    Never returns a label: unknown identities are never named here."""
    gates = gate_reasons(fid or {})
    res = {"status": "unusable", "identity": None, "sim": None, "second_sim": None, "engine": None,
           "reasons": gates or ["FACE_NONE"], "sims": {}}
    if not fid or fid.get("vec") is None or gates:
        return res
    v2 = fid.get("v2") if FACE_V2 else None
    if v2 and v2.get("vec") is not None and any(t.get("model") == v2.get("model") for t in gallery):
        vec, model, thr, split = v2["vec"], v2.get("model"), FACE_UNKNOWN_SIM_V2, FACE_UNKNOWN_SPLIT_V2
    else:
        vec, model, thr, split = fid["vec"], fid.get("model"), FACE_UNKNOWN_SIM, FACE_UNKNOWN_SPLIT
    q = l2n(vec)
    best: dict = {}
    for t in gallery:
        if t.get("model") != model or t.get("vec") is None:
            continue
        s = float(np.dot(q, l2n(t["vec"])))
        if s > best.get(t["identity_id"], -2.0):
            best[t["identity_id"]] = s
    ranked = sorted(best.items(), key=lambda x: -x[1])
    res.update(engine=model or "inspireface", sims=best, status="new", reasons=["FACE_UNKNOWN_NEW"],
               thresholds={"sim": thr, "margin": FACE_MARGIN, "split": split})
    if ranked:
        (i1, s1) = ranked[0]
        s2 = ranked[1][1] if len(ranked) > 1 else None
        res.update(sim=round(s1, 4), second_sim=None if s2 is None else round(s2, 4))
        if s1 >= thr and (s2 is None or s1 - s2 >= FACE_MARGIN):
            res.update(status="match", identity=i1, reasons=["FACE_UNKNOWN_MATCH"])
        elif s1 >= thr:
            res["reasons"] = ["FACE_UNKNOWN_AMBIGUOUS"]       # two unnamed identities fit: start a new one
    return res


def face_disagrees(dec: dict | None, identity_id) -> bool:
    """Does a usable face clearly NOT belong to this (unnamed) identity, which has usable faces of its own?"""
    if not dec or dec.get("status") not in ("new", "match") or identity_id not in (dec.get("sims") or {}):
        return False
    return dec["sims"][identity_id] < (dec.get("thresholds") or {}).get("split", FACE_UNKNOWN_SPLIT_V2)


def public(dec: dict | None, named: bool = False) -> dict | None:
    """The part of a decision that is stored: never the per-identity scores, and identity/label/candidate only for
    an enrolled match (named=True) or a tracked unnamed identity (identity only, never a label)."""
    if not dec:
        return None
    out = {k: dec.get(k) for k in ("status", "sim", "second_sim", "engine", "reasons", "thresholds") if k in dec}
    if dec.get("status") == "match":
        out["identity"] = dec.get("identity")
        if named:
            out["label"] = dec.get("label")
    return out

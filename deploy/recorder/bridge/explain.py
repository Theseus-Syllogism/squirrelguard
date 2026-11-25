"""Readable scores: one vocabulary for email, ntfy, the UI and logs.

Every person decision is turned into
  band      strong / likely / possible / unlikely   (the same four words everywhere)
  pct       an ESTIMATED probability that the match is right (fixed mapping for now; replace with a model fitted
            on reviewed sightings once there are enough)
  reasons   stable reason codes (APPEARANCE_LIKELY, HEIGHT_MATCH, NIGHT, ...) for rules, Grafana and tests
  sentence  one plain sentence built from the same facts
and recorded in the `decisions` table (and as one JSON log line) with the inputs and model versions.
"""
from __future__ import annotations

import json
import logging
import math
import os

log = logging.getLogger("bridge.explain")
E = os.environ
MATCH_SIM = float(E.get("PERSON_MATCH_SIM", "0.70"))
POSSIBLE_SIM = float(E.get("PERSON_POSSIBLE_SIM", "0.55"))
VERSIONS = {"reid": "osnet_x0_25_msmt17", "pose": "yolov8s-pose", "gait_sig": "gaitbase_grew+yolov8s-seg-320",
            "actions": "stgcnpp_ntu120_hrnet_j", "objects": "yolov8s-worldv2-sg", "face": "inspireface+fasr-qwmean-v1",
            "pulc": "pulc_person_attribute (PPLCNet_x1_0, PA100K)", "explain": "5"}
GAIT_SIG_SAME = float(E.get("GAIT_SIG_SAME", "0.05"))       # provisional: calibrate on real camera events
GAIT_SIG_DIFF = float(E.get("GAIT_SIG_DIFF", "0.15"))

ALERT_REASONS = {"possible_fall": "ACTION_FALL", "night_unknown": "NIGHT", "repeat_unknown": "REPEAT_24H", "linger_unknown": "LINGER",
                 "zone_unknown": "ZONE", "parked_vehicle": "PARKED", "recurring_vehicle": "RECURRING_VEHICLE",
                 "watched_object": "WATCHED_OBJECT"}
REASON_TEXT = {
    "FACE_MATCH": "face recognised (enrolled person) and matched to the identity with that label",
    "FACE_SEEN": "face recognised (enrolled person), but no identity carries that label yet",
    "FACE_TEMPLATE_MATCH": "face matches an enrolled face template (liveness passed)",
    "FACE_NO_MATCH": "face seen, not an enrolled person",
    "FACE_LIVENESS_FAIL": "face liveness check failed (possible photo/screen): the face was not used",
    "FACE_TOO_SMALL": "face too small to compare",
    "FACE_TOO_FEW": "too few usable face frames to decide",
    "FACE_LOW_QUALITY": "face quality too low to decide",
    "FACE_INCONSISTENT": "faces in this visit disagree (more than one person?)",
    "APPEARANCE_STRONG": "clothing and build closely match",
    "APPEARANCE_LIKELY": "clothing and build match",
    "APPEARANCE_POSSIBLE": "clothing and build are somewhat similar",
    "NEW_PERSON": "nobody on file looks similar",
    "FACE_UNKNOWN_MATCH": "face matches earlier faces of this UNNAMED visitor (auto-tracked; not a name)",
    "FACE_UNKNOWN_NEW": "face not seen before (auto-tracked as a new unnamed visitor)",
    "FACE_UNKNOWN_AMBIGUOUS": "face resembles more than one unnamed visitor: kept as a new one",
    "FACE_SPLIT": "body looked like an earlier visitor, but the face clearly differs: kept apart",
    "HEIGHT_MATCH": "height matches",
    "HEIGHT_CONSISTENT": "height is consistent",
    "HEIGHT_MISMATCH": "height differs",
    "GAIT_MATCH": "walking rhythm matches",
    "GAIT_DIFFERS": "walking rhythm differs",
    "GAIT_SIG_MATCH": "gait signature is close", "GAIT_SIG_DIFFERS": "gait signature differs",
    "NIGHT": "at night", "REPEAT_24H": "seen repeatedly today", "LINGER": "stayed a long time",
    "ZONE": "entered a watched zone", "WATCHLIST": "on the watch list", "UNKNOWN": "not a known person",
    "ACTION_FALL": "a possible fall",
    "WATCHED_OBJECT": "carrying an object on the watch list",
    # CARRYING_<LABEL> (e.g. CARRYING_LADDER, CARRYING_PRY_BAR): open-vocabulary object held/worn on >= 2 frames
    "PULC_CONSISTENT": "apparent clothing/bag attributes agree with earlier visits (weak)",
    "PULC_CONTRADICTS": "apparent clothing/bag attributes differ from earlier visits (weak)",
    "PULC_APPARENT_GENDER_DIFFERS": "apparent gender differs from earlier visits (estimate, weak)",
    "PULC_APPARENT_AGE_DIFFERS": "apparent age band differs from earlier visits (estimate, weak)",
}


def band_of(p: float) -> str:
    return "strong" if p >= 0.9 else "likely" if p >= 0.7 else "possible" if p >= 0.4 else "unlikely"


def appearance_prob(cos: float) -> float:
    """Re-ID cosine -> rough probability of 'same person'. Logistic pinned so the decision thresholds land on the
    band edges: PERSON_MATCH_SIM -> 70 % (likely), PERSON_POSSIBLE_SIM -> 40 % (possible). Measured on this system:
    same person 0.90-0.98 (-> strong), different people 0.29-0.36 (-> unlikely)."""
    lo, hi = math.log(0.4 / 0.6), math.log(0.7 / 0.3)
    slope = (hi - lo) / max(1e-6, MATCH_SIM - POSSIBLE_SIM)
    return 1 / (1 + math.exp(-(slope * (cos - MATCH_SIM) + hi)))


def _odds(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return p / (1 - p)


def _pct(p: float) -> int:
    """Estimates never claim certainty: shown as 1-99 %."""
    return min(99, max(1, round(p * 100)))


def person_name(iid, label) -> str:
    return label or f"Unknown #{iid}"


def explain_person(conn, frigate_event: str) -> dict | None:
    """Explanation of who a person sighting was matched to and how sure we are."""
    r = conn.execute("""SELECT s.id, s.identity_id, i.label, i.kind, s.match_sim, s.match_kind, s.possible_id,
                               s.possible_sim, s.height_cm, s.gait, s.explain, s.age_low, s.age_high, s.gender,
                               s.gender_conf, s.behavior, s.attributes, s.ts_start, s.actions, s.gait_vec, s.objects,
                               s.face, s.face_meta
                        FROM sightings s JOIN identities i ON i.id = s.identity_id WHERE s.frigate_event = %s""",
                     (frigate_event,)).fetchone()
    if not r:
        return None
    (sid, iid, label, kind, msim, mkind, pid, psim, hcm, gait, prev, alo, ahi, gender, gconf, beh, attrs, ts, acts, gvec,
     objs, face, fmeta) = r
    fmeta = fmeta or {}
    fdec = fmeta.get("match") or {}
    prev = prev or {}
    closest = prev.get("closest") or {}
    st = conn.execute("""SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY height_cm), count(height_cm),
                                percentile_cont(0.5) WITHIN GROUP (ORDER BY (gait->>'cadence_spm')::real)
                                    FILTER (WHERE (gait->>'usable')::boolean),
                                count(*), min(ts_start)
                         FROM sightings WHERE identity_id = %s AND frigate_event <> %s""",
                      (iid, frigate_event)).fetchone()
    h_med, h_n, cad_med, n_prev, first = st
    reasons, parts, scores = [], [], ({"appearance_cos": round(float(msim), 3)} if mkind not in ("face_template", "face_unknown") else {})
    who = person_name(iid, label)
    if face and face.get("name"):
        # Frigate face recognition (enrolled people only); decides the identity when the name matches its label
        fs = face.get("score")
        reasons.append("FACE_MATCH" if mkind == "face" else "FACE_SEEN")
        parts.append(f"Face recognised as {face['name']}" + (f" ({fs:.2f})" if fs is not None else "")
                     + ("" if mkind == "face" else " (no identity has that label yet)"))
        scores["face"] = fs
    for code in fdec.get("reasons") or []:          # our own face template (enrolled household/known only)
        if code not in reasons:
            reasons.append(code)
    if fdec.get("sim") is not None:
        scores["face_template_cos"] = fdec["sim"]
        scores["face_faces"] = fmeta.get("n_faces")
    if fdec.get("status") == "match":
        parts.append(f"{REASON_TEXT['FACE_TEMPLATE_MATCH']} ({fdec['sim']:.2f}, {fmeta.get('n_faces')} faces)")
    elif fdec.get("status") in ("liveness_fail", "weak"):
        parts.append("face resembles an enrolled person (" + f"{fdec['sim']:.2f}" + ") but was not used: "
                     + ", ".join(REASON_TEXT.get(c, c) for c in fdec.get("reasons") or []))
    elif fdec.get("status") == "no_match":
        parts.append(REASON_TEXT["FACE_NO_MATCH"])
    elif "FACE_TOO_SMALL" in (fdec.get("reasons") or []):
        parts.append(REASON_TEXT["FACE_TOO_SMALL"])
    tr = fmeta.get("track") or {}
    for code in tr.get("reasons") or []:
        if code.startswith("FACE_UNKNOWN") and code not in reasons:
            reasons.append(code)
    if tr.get("sim") is not None:
        scores["face_track_cos"] = tr["sim"]
    if mkind == "face_new":
        reasons.append("FACE_SPLIT")
        parts.append(REASON_TEXT["FACE_SPLIT"] + (f" (body {psim:.2f} to #{pid}, face {tr['sim']:.2f})"
                                                   if pid and psim and tr.get("sim") is not None else ""))
    if mkind == "face_unknown":
        thr = float((tr.get("thresholds") or {}).get("sim", 0.4))
        p = min(0.97, 0.85 + max(0.0, float(msim or 0) - thr) * 0.5)
        band, pct = band_of(p), _pct(p)
        parts.append(f"{REASON_TEXT['FACE_UNKNOWN_MATCH']} ({float(msim or 0):.2f})")
        headline = f"Same unnamed visitor {who} by face: {band} ({pct}%)"
    elif mkind == "face_template":
        p = min(0.99, 0.90 + max(0.0, float(msim or 0) - float((fdec.get("thresholds") or {}).get("sim", 0.4))) * 0.5)
        band, pct = band_of(p), _pct(p)
        headline = f"Recognised {who} by enrolled face: {band} ({pct}%)"
    elif mkind == "face":
        p = min(0.99, max(0.9, float(msim or 0.9)))
        band, pct = band_of(p), _pct(p)
        headline = f"Recognised {who} by face: {band} ({pct}%)"
    elif mkind == "match":
        p = appearance_prob(msim)
        code = "APPEARANCE_STRONG" if msim >= 0.80 else "APPEARANCE_LIKELY"
        reasons.append(code)
        parts.append(f"{REASON_TEXT[code]} ({msim:.2f})")
        odds = _odds(p)
        # height and walk only adjust an existing match (they arrive after the visit ends)
        if hcm and h_med and h_n >= 2:
            d = abs(hcm - h_med)
            scores["height_diff_cm"] = round(d, 1)
            if d <= 5:
                reasons.append("HEIGHT_MATCH"); odds *= 2.0
            elif d <= 10:
                reasons.append("HEIGHT_CONSISTENT"); odds *= 1.2
            elif d > 12:
                reasons.append("HEIGHT_MISMATCH"); odds *= 0.25
            if d <= 10 or d > 12:
                parts.append(f"height {hcm:.0f} cm vs usual {h_med:.0f} cm")
        g = gait or {}
        if g.get("usable") and cad_med:
            d = abs(g["cadence_spm"] - cad_med)
            scores["cadence_diff_spm"] = round(d, 1)
            if d <= 6:
                reasons.append("GAIT_MATCH"); odds *= 1.5
                parts.append(f"walk {g['cadence_spm']:.0f} vs usual {cad_med:.0f} steps/min")
            elif d > 12:
                reasons.append("GAIT_DIFFERS"); odds *= 0.5
                parts.append(f"walk {g['cadence_spm']:.0f} vs usual {cad_med:.0f} steps/min")
        if gvec:   # learned gait signature vs this identity's earlier ones: a weak hint (overlapping distances)
            import numpy as np
            others = conn.execute("""SELECT gait_vec FROM sightings WHERE identity_id = %s AND frigate_event <> %s
                                     AND gait_vec IS NOT NULL ORDER BY ts_start DESC LIMIT 50""",
                                  (iid, frigate_event)).fetchall()
            if others:
                v = np.asarray(gvec, np.float32)
                d = min(1.0 - float(np.dot(v, np.asarray(o, np.float32))) for (o,) in others)
                scores["gait_sig_dist"] = round(d, 3)
                if d <= GAIT_SIG_SAME:
                    reasons.append("GAIT_SIG_MATCH"); odds *= 1.5
                    parts.append(f"gait signature close ({d:.3f})")
                elif d >= GAIT_SIG_DIFF:
                    reasons.append("GAIT_SIG_DIFFERS"); odds *= 0.6
                    parts.append(f"gait signature differs ({d:.3f})")
        odds *= pulc_adjust(conn, iid, frigate_event, reasons, parts, scores)
        p = odds / (1 + odds)
        band, pct = band_of(p), _pct(p)
        headline = f"Matched {who}: {band} ({pct}%)"
    else:
        cid, csim = (pid, psim) if pid else (closest.get("id"), closest.get("sim"))
        if pid:
            reasons.append("APPEARANCE_POSSIBLE")
            parts.append(f"{REASON_TEXT['APPEARANCE_POSSIBLE']} to #{pid} ({psim:.2f}), kept separate until confirmed")
        else:
            reasons.append("NEW_PERSON")
            parts.append(REASON_TEXT["NEW_PERSON"] + (f" (closest #{cid}, {csim:.2f})" if cid else ""))
        p = appearance_prob(csim) if csim else 0.0
        if pid and csim:   # PULC attributes vs the possible identity's visits nudge the shown estimate (weakly)
            o = _odds(p) * pulc_adjust(conn, pid, frigate_event, reasons, parts, scores)
            p = o / (1 + o)
        scores["closest_id"], scores["closest_cos"] = cid, (round(float(csim), 3) if csim else None)
        band, pct = band_of(p), _pct(p)
        headline = f"New visitor {who}" + (f"; closest match #{cid}: {band} ({pct}%)" if cid else "")
    if kind == "watch":
        reasons.append("WATCHLIST")
    elif kind == "unknown":
        reasons.append("UNKNOWN")
    body = []
    if alo and ahi:
        body.append(f"apparent age {alo}-{ahi}")
    if gender and gender != "unclear":
        body.append(f"apparently {gender} ({(gconf or 0) * 100:.0f}%)")
    if hcm:
        body.append(f"about {hcm:.0f} cm")
    fa = fmeta.get("apparent") or {}
    if fa.get("expression"):    # description only: never an alert or identity input
        body.append(f"apparent expression {fa['expression']}")
    did = []
    b = beh or {}
    if b.get("duration_s"):
        d = b["duration_s"]
        did.append(f"stayed {int(d // 60)} min {int(d % 60)} s" if d >= 60 else f"stayed {d:.0f} s")
    if b.get("path_length_m"):
        did.append(f"walked {b['path_length_m']:.1f} m at {b.get('speed_mean_mps') or 0:.1f} m/s"
                   + (f", {len(b['stops'])} stop(s)" if b.get("stops") else ""))
    if b.get("entered_zones"):
        did.append("entered " + ", ".join(b["entered_zones"]))
    for act in acts or []:   # skeleton actions: two consecutive 4 s windows >= 0.5
        reasons.append("ACTION_" + act["tag"].upper())
        did.append(f"{band_of(act['p'])} {act['tag'].replace('_', ' ')} ({_pct(act['p'])}%, {act['label']}, "
                   f"{act['t']:.0f} s in)")
    for o in objs or []:     # open-vocabulary objects: only the carried ones are about this person
        if o.get("carried"):
            reasons.append("CARRYING_" + o["label"].upper().replace(" ", "_"))
            did.append(f"carrying {o['label']} ({o['score'] * 100:.0f}% max, {o['frames']} of {o['sampled']} frames)")
    a = attrs or {}
    if a.get("activity"):
        did.append(a["activity"])
    return {"sighting": sid, "identity": iid, "who": who, "kind": kind, "band": band, "pct": pct,
            "estimated": True, "reasons": reasons, "headline": headline,
            "sentence": headline + ": " + "; ".join(parts) + ".", "evidence": parts, "body": body, "did": did,
            "scores": scores, "seen_before": int(n_prev or 0), "first_seen": first.isoformat() if first else None,
            "closest": prev.get("closest")}


# --- PULC attributes (squirrel_guard/pulc.py) as WEAK evidence ------------------------------------------------------
# Measured (docs/pulc.md): usable on upright full-body views (sleeves, trousers/shorts, coats), poor on close-ups and
# IR, bags over-fire; apparent gender flips under IR. So: few attributes, small factors, total clamped to 0.6-1.4.
# label -> (text, window in hours the earlier visits must fall in, factor when agreeing, factor when contradicting)
PULC_ATTRS = {
    "Backpack": ("backpack", 24 * 30, 1.15, 0.9),        # people often carry the same bag, but not always
    "Hat": ("hat", 12, 1.1, 0.9),
    "LongSleeve": ("long sleeves", 12, 1.1, 0.85),        # clothing: same-day visits only (people change)
    "Shorts": ("shorts", 12, 1.15, 0.85),
    "Skirt&Dress": ("skirt/dress", 12, 1.15, 0.85),
    "LongCoat": ("long coat", 12, 1.1, 0.85),
}
PULC_YES, PULC_NO = 0.7, 0.15            # confident present / absent on the current visit
PULC_CLAMP = (0.6, 1.4)


def pulc_evidence(cur: dict | None, hist: list) -> dict:
    """cur: this visit's PULC summary; hist: [(pulc summary, hours before this visit)] of the identity's earlier
    visits. -> {"factor": odds multiplier, "reasons": [...], "parts": [...], "agree": [...], "differ": [...]}.
    Pure function (no DB) so it is testable; explain_person() calls it through pulc_adjust()."""
    out = {"factor": 1.0, "reasons": [], "parts": [], "agree": [], "differ": []}
    if not cur or not cur.get("crops") or not cur.get("attributes"):
        return out
    hist = [(h, dh) for h, dh in hist if h and h.get("crops") and h.get("attributes")]
    if not hist:
        return out
    f = 1.0
    ca = cur["attributes"]
    for lab, (text, win, up, down) in PULC_ATTRS.items():
        c = ca.get(lab)
        hs = [h["attributes"][lab] for h, dh in hist if dh <= win and lab in h["attributes"]]
        need = 1 if win <= 12 else 2       # cross-day comparisons need at least two earlier visits
        if c is None or len(hs) < need:
            continue
        yes = sum(p >= 0.5 for p in hs) / len(hs)
        if c >= PULC_YES and yes >= 0.6:
            f *= up
            out["agree"].append(text)
        elif (c >= PULC_YES and yes <= 0.1) or (c <= PULC_NO and yes >= 0.9):
            f *= down
            out["differ"].append(("" if c >= PULC_YES else "no ") + text)
    ap = cur.get("apparent") or {}
    hap = [h.get("apparent") or {} for h, _ in hist]
    g = [x["gender_female"] for x in hap if "gender_female" in x]
    if "gender_female" in ap and len(g) >= 2:
        gm = sum(g) / len(g)
        if (ap["gender_female"] >= 0.9 and gm <= 0.1) or (ap["gender_female"] <= 0.1 and gm >= 0.9):
            f *= 0.7
            out["reasons"].append("PULC_APPARENT_GENDER_DIFFERS")
            out["parts"].append(f"apparent gender differs from earlier visits (estimate {ap['gender_female']:.2f} "
                                f"vs {gm:.2f} female)")
    u = [x["age"]["under_18"] for x in hap if "under_18" in (x.get("age") or {})]
    cu = (ap.get("age") or {}).get("under_18")
    if cu is not None and len(u) >= 2:
        um = sum(u) / len(u)
        if (cu >= 0.8 and um <= 0.1) or (cu <= 0.1 and um >= 0.8):
            f *= 0.7
            out["reasons"].append("PULC_APPARENT_AGE_DIFFERS")
            out["parts"].append(f"apparent age band differs from earlier visits (under-18 estimate {cu:.2f} vs {um:.2f})")
    if out["agree"]:
        out["reasons"].insert(0, "PULC_CONSISTENT")
        out["parts"].insert(0, "apparent " + ", ".join(out["agree"]) + " as on earlier visits (weak)")
    if out["differ"]:
        out["reasons"].insert(0, "PULC_CONTRADICTS")
        out["parts"].insert(0, "apparent " + ", ".join(out["differ"]) + " unlike earlier visits (weak)")
    out["factor"] = round(min(PULC_CLAMP[1], max(PULC_CLAMP[0], f)), 3)
    return out


def pulc_adjust(conn, iid, frigate_event: str, reasons: list, parts: list, scores: dict) -> float:
    """Load this sighting's and the identity's earlier PULC summaries, append reasons/parts, return the odds factor.
    1.0 when there is nothing to compare (or the pulc column does not exist yet)."""
    try:
        with conn.transaction():
            r = conn.execute("SELECT pulc, ts_start FROM sightings WHERE frigate_event = %s", (frigate_event,)).fetchone()
            if not r or not r[0]:
                return 1.0
            rows = conn.execute("""SELECT pulc, extract(epoch FROM %s - ts_start) / 3600 FROM sightings
                                   WHERE identity_id = %s AND frigate_event <> %s AND pulc IS NOT NULL
                                     AND ts_start <= %s ORDER BY ts_start DESC LIMIT 30""",
                                (r[1], iid, frigate_event, r[1])).fetchall()
    except Exception as e:
        log.info("pulc evidence unavailable: %s", e)
        return 1.0
    ev = pulc_evidence(r[0], [(p, float(h)) for p, h in rows])
    reasons += ev["reasons"]
    parts += ev["parts"]
    if ev["factor"] != 1.0:
        scores["pulc_factor"] = ev["factor"]
    return ev["factor"]


def record(conn, stage: str, event: str | None, subject: str | None, ex: dict | None = None, *, band=None,
           reasons=None, message=None, scores=None) -> None:
    """One decision row + one JSON log line (same content)."""
    from psycopg.types.json import Jsonb
    ex = ex or {}
    row = {"stage": stage, "event": event, "subject": subject, "band": band or ex.get("band"),
           "pct": ex.get("pct"), "reasons": reasons if reasons is not None else ex.get("reasons", []),
           "scores": scores if scores is not None else ex.get("scores", {}),
           "message": message or ex.get("sentence"), "versions": VERSIONS}
    try:
        conn.execute("""INSERT INTO decisions (stage, frigate_event, subject, band, pct, reasons, scores, message, versions)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                     (stage, event, subject, row["band"], row["pct"], row["reasons"], Jsonb(row["scores"]),
                      row["message"], Jsonb(VERSIONS)))
    except Exception as e:
        log.warning("decision not recorded: %s", e)
    log.info("DECISION %s", json.dumps(row, default=str))

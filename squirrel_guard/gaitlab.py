"""Explainable gait analysis from 2D body keypoints (COCO-17), numpy only.

  analyse(track, fps, cam=None, size=None) -> walk     measurements of one walk, each with its own availability
  profile(walks)                           -> profile  a person's usual walk: median and spread per measurement
  compare(walk, profile)                   -> result   measurement by measurement: this walk vs the usual one
  svg_cycle(walk, profile=None)            -> str      knee angle through the walking cycle, both legs

Measurements (what a gait examiner reads off a video, made repeatable):
  cadence_spm          steps per minute
  step_regularity      how alike consecutive steps are (autocorrelation of the body's up-and-down motion at one step)
  stride_regularity    how alike consecutive strides are (the same, at two steps)
  symmetry             step / stride regularity: 1 = left and right steps alike; a limp lowers it
  vertical_osc         up-and-down travel of the hips, as a share of leg length
  step_time_asym       left-to-right step time against right-to-left, as a share of a step (side/diagonal views)
  knee_rom_l/_r        knee bend range through the cycle, degrees (side views only)
  knee_asym            difference between the two knees' ranges, as a share of their mean
  knee_peak_swing      largest knee bend while the leg swings through, degrees (mean of both legs)
  arm_swing            wrist travel front-to-back as a share of leg length (mean of both arms); arm_asym likewise
  step_rel             step length in leg lengths (side views); step_m and speed_mps when the camera is calibrated
  lean_deg             trunk lean forward (+) or back (-)

Which measurements are valid depends on the view: rhythm (cadence, regularity, symmetry, vertical_osc) works from any
direction; knee, arm and step length need a side or diagonal view. Each walk lists what it could measure.

What this is not: an identification. Two different people can walk alike, and one person walks differently when
carrying, hurrying, injured or on another surface. compare() says "consistent" or "not consistent", never "same".
A copy of this file is in deploy/recorder/persons-ui/ (tests check that the two are identical).
"""
from __future__ import annotations

import math

import numpy as np

VERSION = 2
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
LEGS = (L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK)
CYCLE_POINTS = 51

# How much one person's measurement varies between walks (typical day-to-day spread plus what keypoints from a
# home camera add). Used as the tolerance when a profile has fewer than 3 walks, and as a floor otherwise.
# Sources for the walking part: Hollman et al. 2011 (Gait & Posture 34:111) normative variability; Moe-Nilssen &
# Helbostad 2004 (J Biomech 37:121) for regularity; the keypoint part is our own estimate from synthetic tests.
TOLERANCE = {"step_time_asym": 0.06, "cadence_spm": 5.0, "step_regularity": 0.08, "stride_regularity": 0.08, "symmetry": 0.07,
             "vertical_osc": 0.012, "knee_rom_l": 6.0, "knee_rom_r": 6.0, "knee_asym": 0.08, "knee_peak_swing": 6.0,
             "arm_swing": 0.06, "arm_asym": 0.25, "step_rel": 0.06, "step_m": 0.05, "speed_mps": 0.12, "lean_deg": 3.0}
LABEL = {"step_time_asym": ("Step time difference", ""), "cadence_spm": ("Cadence", "steps/min"), "step_regularity": ("Step regularity", ""),
         "stride_regularity": ("Stride regularity", ""), "symmetry": ("Left/right symmetry", ""),
         "vertical_osc": ("Up-and-down motion", "× leg length"), "knee_rom_l": ("Left knee range", "°"),
         "knee_rom_r": ("Right knee range", "°"), "knee_asym": ("Knee difference", ""), "knee_peak_swing": ("Knee bend in swing", "°"),
         "arm_swing": ("Arm swing", "× leg length"), "arm_asym": ("Arm difference", ""), "step_rel": ("Step length", "× leg length"),
         "step_m": ("Step length", "m"), "speed_mps": ("Walking speed", "m/s"), "lean_deg": ("Trunk lean", "°")}
RHYTHM = ("cadence_spm", "step_regularity", "stride_regularity", "symmetry", "vertical_osc")
SHAPE = ("step_time_asym", "knee_rom_l", "knee_rom_r", "knee_asym", "knee_peak_swing", "arm_swing", "arm_asym", "step_rel", "lean_deg")
METRIC = ("step_m", "speed_mps")


# ---------------------------------------------------------------------------------------------------------- helpers
def _kp(p, i, c=0.4):
    return p.kp[i, :2] if p is not None and p.kp[i, 2] >= c else None


def _longest_run(trk, need, gap=2, c=0.4):
    good = [i for i, p in enumerate(trk) if p is not None and all(p.kp[j, 2] >= c for j in need)]
    if not good:
        return []
    runs, cur = [], [good[0]]
    for i in good[1:]:
        if i - cur[-1] <= gap + 1:
            cur.append(i)
        else:
            runs.append(cur); cur = [i]
    runs.append(cur)
    return max(runs, key=len)


def _series(trk, idx, frames, c=0.0):
    """(len(frames), len(idx), 2) keypoints on a uniform frame grid, gaps linearly filled."""
    out = np.full((frames[-1] - frames[0] + 1, len(idx), 2), np.nan)
    for f in frames:
        p = trk[f]
        for k, j in enumerate(idx):
            if p.kp[j, 2] >= c:
                out[f - frames[0], k] = p.kp[j, :2]
    t = np.arange(len(out))
    for k in range(len(idx)):
        for d in range(2):
            v = out[:, k, d]
            ok = np.isfinite(v)
            if ok.sum() >= 2:
                out[:, k, d] = np.interp(t, t[ok], v[ok])
    return out


def _detrend(x, k):
    k = max(1, int(k))
    if len(x) <= k + 2:
        return x - np.mean(x)
    pad = np.pad(x, (k, k), mode="edge")
    trend = np.convolve(pad, np.ones(k) / k, mode="same")[k:-k]
    return x - trend


def _acf(x):
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    n = np.arange(len(x), 0, -1)                               # unbiased: divide by the overlap at each lag
    ac = ac / n
    return ac / (ac[0] + 1e-12)


def _peak(ac, lo, hi):
    hi = min(hi, len(ac) - 2)
    if hi <= lo:
        return None, 0.0
    lag = lo + int(np.argmax(ac[lo:hi + 1]))
    y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
    den = y0 - 2 * y1 + y2
    return lag + (0.5 * (y0 - y2) / den if den else 0.0), float(y1)


def _angle(a, b, c):
    """Angle at b (degrees) between b->a and b->c, per frame."""
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _fix_swaps(P):
    """Pose estimators swap left and right legs on side views. Keep each leg continuous: swap a frame's legs when
    that brings both knees and ankles closer to the previous frame."""
    P = P.copy()
    for t in range(1, len(P)):
        a = np.linalg.norm(P[t, [2, 4]] - P[t - 1, [2, 4]], axis=-1).sum() + np.linalg.norm(P[t, [3, 5]] - P[t - 1, [3, 5]], axis=-1).sum()
        b = np.linalg.norm(P[t, [3, 5]] - P[t - 1, [2, 4]], axis=-1).sum() + np.linalg.norm(P[t, [2, 4]] - P[t - 1, [3, 5]], axis=-1).sum()
        if b < 0.7 * a:
            P[t, [2, 3]] = P[t, [3, 2]]
            P[t, [4, 5]] = P[t, [5, 4]]
    return P


# ---------------------------------------------------------------------------------------------------------- analyse
def analyse(trk: list, fps: float, cam=None, size=None) -> dict | None:
    """One walk. Returns None when there are no usable legs at all; {"walking": False, "reason"} when too short."""
    run = _longest_run(trk, LEGS)
    if len(run) < 8:
        return None
    frames = list(range(run[0], run[-1] + 1))
    dur = (run[-1] - run[0]) / fps
    if dur < 2.5:
        return {"walking": False, "reason": "under 2.5 s of clear legs", "version": VERSION}
    usable = [i for i in frames if trk[i] is not None]
    P = _fix_swaps(_series(trk, LEGS, usable))                 # hipL hipR kneeL kneeR ankL ankR
    hips = P[:, :2].mean(1)
    leg = np.median(np.linalg.norm(P[:, 0] - P[:, 4], axis=-1) + np.linalg.norm(P[:, 1] - P[:, 5], axis=-1)) / 2
    if not leg > 5:
        return None
    move = hips[-1] - hips[0]
    dirn = move / (np.linalg.norm(move) + 1e-9)
    bh = np.array([trk[i].box[3] - trk[i].box[1] for i in usable])
    sh = [abs(trk[i].kp[L_SH, 0] - trk[i].kp[R_SH, 0]) / (trk[i].box[3] - trk[i].box[1]) for i in usable
          if trk[i].kp[L_SH, 2] >= 0.4 and trk[i].kp[R_SH, 2] >= 0.4]
    shoulder_rel = float(np.median(sh)) if sh else 0.2
    horiz = abs(move[0]) / (np.linalg.norm(move) + 1e-9)
    grow = bh[-1] / max(bh[0], 1)
    if horiz > 0.75 and shoulder_rel < 0.14 and 0.7 < grow < 1.4:
        view = "side"
    elif horiz < 0.5 or not 0.7 < grow < 1.4:
        view = "front/back"
    else:
        view = "diagonal"

    # rhythm: hip up-and-down (two bumps per stride) and ankle separation
    k = max(1, int(round(fps)))
    vert = _detrend(hips[:, 1] / leg, k)
    sep = _detrend(np.abs((P[:, 4] - P[:, 5]) @ (dirn if view != "front/back" else np.array([0.0, 1.0]))) / leg, k)
    lo, hi = int(0.3 * fps), int(math.ceil(0.85 * fps))
    lag_s, q_s = _peak(_acf(sep), lo, hi)
    ac_v = _acf(vert)
    lag_v, q_v = _peak(ac_v, lo, hi)
    cands = [(q, l) for q, l in ((q_s, lag_s), (q_v, lag_v)) if l]
    if not cands:
        return {"walking": False, "reason": "no regular stepping", "version": VERSION}
    quality, step_lag = max(cands)
    step_s = step_lag / fps
    stride_lag = 2 * step_lag
    ad1 = float(np.interp(step_lag, np.arange(len(ac_v)), ac_v))
    ad2 = float(np.interp(stride_lag, np.arange(len(ac_v)), ac_v)) if stride_lag < len(ac_v) - 1 else float("nan")
    cycles = dur / (2 * step_s)
    out = {"walking": True, "version": VERSION, "view": view, "seconds": round(dur, 1), "frames": len(usable),
           "strides": round(cycles, 1), "periodicity": round(quality, 2), "cadence_spm": round(float(60 / step_s), 1),
           "step_regularity": round(ad1, 3), "stride_regularity": round(ad2, 3) if math.isfinite(ad2) else None,
           "vertical_osc": round(float(np.percentile(vert, 95) - np.percentile(vert, 5)), 4),
           "shoulder_rel": round(shoulder_rel, 3)}
    if out["stride_regularity"] and out["stride_regularity"] > 0.2:
        out["symmetry"] = round(min(1.5, max(0.0, ad1 / out["stride_regularity"])), 3)

    if view in ("side", "diagonal"):
        flex_l = 180 - _angle(P[:, 0], P[:, 2], P[:, 4])
        flex_r = 180 - _angle(P[:, 1], P[:, 3], P[:, 5])
        rom_l = float(np.percentile(flex_l, 95) - np.percentile(flex_l, 5))
        rom_r = float(np.percentile(flex_r, 95) - np.percentile(flex_r, 5))
        out.update(knee_rom_l=round(rom_l, 1), knee_rom_r=round(rom_r, 1),
                   knee_asym=round(abs(rom_l - rom_r) / max(1e-6, (rom_l + rom_r) / 2), 3))
        cyc, strike_at = {}, {}
        for name, flex, ank, hip in (("l", flex_l, P[:, 4], P[:, 0]), ("r", flex_r, P[:, 5], P[:, 1])):
            fwd = (ank - hip) @ dirn                              # foot ahead of the hip: largest at heel strike
            strikes = [i for i in range(1, len(fwd) - 1) if fwd[i] >= fwd[i - 1] and fwd[i] > fwd[i + 1]
                       and fwd[i] > np.percentile(fwd, 60)]
            strikes = [s for j, s in enumerate(strikes) if j == 0 or s - strikes[j - 1] > 1.2 * step_lag]
            strike_at[name] = strikes
            curves = []
            for a, b in zip(strikes, strikes[1:]):
                if 1.4 * step_lag <= b - a <= 2.8 * step_lag:
                    curves.append(np.interp(np.linspace(a, b, CYCLE_POINTS), np.arange(len(flex)), flex))
            if curves:
                cyc[name] = [round(float(v), 1) for v in np.mean(curves, 0)]
        # step-time asymmetry: left-to-right step against right-to-left step (a limp shortens one of them)
        sl, sr = strike_at.get("l", []), strike_at.get("r", [])
        lr = [min((b - a for b in sr if b > a), default=None) for a in sl]
        rl = [min((b - a for b in sl if b > a), default=None) for a in sr]
        lr = [d for d in lr if d and d < 1.6 * step_lag]
        rl = [d for d in rl if d and d < 1.6 * step_lag]
        if len(lr) >= 2 and len(rl) >= 2:
            a_, b_ = float(np.mean(lr)), float(np.mean(rl))
            out["step_time_asym"] = round(abs(a_ - b_) / ((a_ + b_) / 2), 3)
        if cyc:
            out["knee_cycle"] = cyc
            peaks = [max(v) for v in cyc.values()]
            out["knee_peak_swing"] = round(float(np.mean(peaks)), 1)
        # step length in leg lengths: hip travel per step, scaled by the leg length in each frame
        legs = (np.linalg.norm(P[:, 0] - P[:, 4], axis=-1) + np.linalg.norm(P[:, 1] - P[:, 5], axis=-1)) / 2
        travel = np.sum(np.abs(np.diff(hips @ dirn)) / np.maximum(legs[1:], 1))
        out["step_rel"] = round(float(travel / max(dur / step_s, 1)), 3)
        arms = []
        for wr, shd in ((L_WR, L_SH), (R_WR, R_SH)):
            v = [((trk[i].kp[wr, :2] - trk[i].kp[shd, :2]) @ dirn) / leg for i in usable
                 if trk[i].kp[wr, 2] >= 0.4 and trk[i].kp[shd, 2] >= 0.4]
            if len(v) >= 8:
                arms.append(float(np.percentile(v, 95) - np.percentile(v, 5)))
        if arms:
            out["arm_swing"] = round(float(np.mean(arms)), 3)
            if len(arms) == 2:
                out["arm_asym"] = round(float(abs(arms[0] - arms[1]) / max(1e-6, np.mean(arms))), 3)
        lean = [math.degrees(math.atan2(((trk[i].kp[[L_SH, R_SH], :2].mean(0) - trk[i].kp[[L_HIP, R_HIP], :2].mean(0)) @ dirn),
                                        trk[i].kp[[L_HIP, R_HIP], 1].mean() - trk[i].kp[[L_SH, R_SH], 1].mean()))
                for i in usable if min(trk[i].kp[[L_SH, R_SH, L_HIP, R_HIP], 2]) >= 0.4]
        if lean:
            out["lean_deg"] = round(float(np.median(lean)), 1)

    if cam is not None and size is not None:
        foot = np.array([[trk[i].kp[[L_ANK, R_ANK], 0].mean(), trk[i].box[3]] for i in usable])
        g = cam.ground(foot, size)
        ok = np.isfinite(g).all(1)
        if ok.sum() >= 8:
            gg = g[ok]
            t = np.array(usable, float)[ok] / fps
            dist = float(np.linalg.norm(gg[-1] - gg[0]))
            if dist > 1.0 and t[-1] > t[0]:
                out["speed_mps"] = round(dist / (t[-1] - t[0]), 2)
                out["step_m"] = round(out["speed_mps"] * step_s, 2)

    measured = [k for k in RHYTHM + SHAPE + METRIC if out.get(k) is not None]
    out["measured"] = measured
    out["quality"] = round(float(np.clip(quality, 0, 1) * min(1.0, cycles / 4) * (1.0 if view == "side" else 0.8)), 2)
    out["usable"] = bool(quality >= 0.35 and cycles >= 2 and dur / step_s >= 4)
    out["side_on"] = view == "side"
    return out


# ---------------------------------------------------------------------------------------------------------- profile
def profile(walks: list[dict]) -> dict:
    """A person's usual walk from several walks (unusable ones are ignored)."""
    ws = [w for w in walks if w and w.get("usable") and w.get("version", 1) >= VERSION]
    feats = {}
    for k in RHYTHM + SHAPE + METRIC:
        v = np.array([w[k] for w in ws if w.get(k) is not None], float)
        if len(v):
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med)) * 1.4826) if len(v) >= 3 else None
            feats[k] = {"median": round(med, 4), "spread": round(mad, 4) if mad is not None else None, "n": int(len(v))}
    cyc = {}
    for leg in ("l", "r"):
        cs = [w["knee_cycle"][leg] for w in ws if w.get("knee_cycle", {}).get(leg)]
        if cs:
            a = np.array(cs, float)
            cyc[leg] = {"mean": [round(float(x), 1) for x in a.mean(0)], "sd": [round(float(x), 1) for x in a.std(0)], "n": len(cs)}
    return {"version": VERSION, "walks": len(ws), "features": feats, "knee_cycle": cyc}


# ---------------------------------------------------------------------------------------------------------- compare
def _fmt(k, v):
    if v is None:
        return "–"
    unit = LABEL[k][1]
    if k in ("cadence_spm", "knee_rom_l", "knee_rom_r", "knee_peak_swing", "lean_deg"):
        return f"{v:.0f}{(' ' + unit) if unit and unit != '°' else unit}"
    if k in ("step_m", "speed_mps"):
        return f"{v:.2f} {unit}"
    return f"{v:.2f}" + (f" {unit}" if unit else "")


def compare(walk: dict, prof: dict) -> dict:
    """This walk against a profile (or against one other walk: pass profile([other]))."""
    if not walk or not walk.get("usable"):
        return {"verdict": "not comparable", "text": "This walk is too short or irregular to compare.", "rows": [], "used": 0}
    rows, z2 = [], []
    for k in RHYTHM + SHAPE + METRIC:
        f = (prof.get("features") or {}).get(k)
        if f is None or walk.get(k) is None:
            continue
        tol = max(TOLERANCE[k], f["spread"] or 0)
        z = (walk[k] - f["median"]) / tol
        v = "within the usual range" if abs(z) < 1.5 else ("somewhat different" if abs(z) < 3 else "different")
        rows.append({"key": k, "label": LABEL[k][0], "this": walk[k], "usual": f["median"], "tolerance": round(tol, 4),
                     "n": f["n"], "z": round(z, 2), "verdict": v, "this_text": _fmt(k, walk[k]), "usual_text": _fmt(k, f["median"])})
        z2.append(min(z * z, 16.0))                               # one wild measurement cannot decide alone
    if len(rows) < 3:
        return {"verdict": "not comparable", "text": "Fewer than three measurements in common (different views).",
                "rows": rows, "used": len(rows)}
    score = float(np.mean(z2))
    n_diff = sum(r["verdict"] == "different" for r in rows)
    if score <= 1.5 and n_diff == 0:
        verdict, text = "consistent", "The walk is consistent with this person's usual walk."
    elif score <= 4 and n_diff <= 1:
        verdict, text = "partly consistent", "Most measurements fit this person's usual walk; some differ."
    else:
        verdict, text = "not consistent", "The walk differs from this person's usual walk in several measurements."
    return {"verdict": verdict, "text": text + " Gait supports or weakens other evidence; it does not identify anyone.",
            "score": round(score, 2), "rows": rows, "used": len(rows), "profile_walks": prof.get("walks", 0)}


# ---------------------------------------------------------------------------------------------------------- chart
def svg_cycle(walk: dict, prof: dict | None = None, width: int = 420, height: int = 200) -> str:
    """Knee bend (degrees) through one walking cycle, left and right; the person's usual curve as a band. Uses CSS
    variables (--accent, --warn, --muted, --line) so it follows the page theme."""
    cyc = (walk or {}).get("knee_cycle") or {}
    if not cyc:
        return ""
    pad_l, pad_b, pad_t, pad_r = 34, 22, 10, 8
    W, H = width - pad_l - pad_r, height - pad_b - pad_t
    ymax = 80.0
    sx = lambda i: pad_l + W * i / (CYCLE_POINTS - 1)
    sy = lambda v: pad_t + H * (1 - min(max(v, -5), ymax) / ymax)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
             f'aria-label="Knee bend through one walking cycle" style="max-width:100%;height:auto;font:11px system-ui">']
    for v in (0, 20, 40, 60, 80):
        parts.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="var(--line,#ccc)" stroke-width="1"/>'
                     f'<text x="{pad_l - 4}" y="{sy(v) + 4:.1f}" text-anchor="end" fill="var(--muted,#666)">{v}°</text>')
    for i, lab in ((0, "heel strike"), (CYCLE_POINTS // 2, "50 %"), (CYCLE_POINTS - 1, "next strike")):
        parts.append(f'<text x="{sx(i):.1f}" y="{height - 6}" text-anchor="{"start" if i == 0 else ("end" if i else "middle")}" fill="var(--muted,#666)">{lab}</text>')
    pc = (prof or {}).get("knee_cycle") or {}
    for leg in ("l", "r"):
        if leg in pc and pc[leg]["n"] >= 2:
            m, s = np.array(pc[leg]["mean"]), np.maximum(np.array(pc[leg]["sd"]), 3.0)
            pts = [f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(m + s)] + [f"{sx(i):.1f},{sy(v):.1f}" for i, v in reversed(list(enumerate(m - s)))]
            parts.append(f'<polygon points="{" ".join(pts)}" fill="var(--muted,#888)" opacity=".18"/>')
    colours = {"l": "var(--accent,#2563eb)", "r": "var(--warn,#c2410c)"}
    for leg, vals in cyc.items():
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colours[leg]}" stroke-width="2"/>')
    lx = pad_l + 8
    for leg, name in (("l", "left knee"), ("r", "right knee")):
        if leg in cyc:
            parts.append(f'<line x1="{lx}" x2="{lx + 16}" y1="{pad_t + 8}" y2="{pad_t + 8}" stroke="{colours[leg]}" stroke-width="2"/>'
                         f'<text x="{lx + 20}" y="{pad_t + 12}" fill="var(--ink-2,#333)">{name}</text>')
            lx += 92
    if pc:
        parts.append(f'<rect x="{lx}" y="{pad_t + 3}" width="16" height="10" fill="var(--muted,#888)" opacity=".25"/>'
                     f'<text x="{lx + 20}" y="{pad_t + 12}" fill="var(--ink-2,#333)">usual range</text>')
    parts.append("</svg>")
    return "".join(parts)

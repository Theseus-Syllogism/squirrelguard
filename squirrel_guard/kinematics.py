"""End-of-event motion analysis for one person: keypoint track -> height estimate + gait features.

Height needs a calibrated camera (geometry.Camera); without one only the image-relative gait features are produced.
Gait is a WEAK signal from one fixed camera: it is only reported when the person walks roughly side-on for a few
steps, and it is stored/compared for review, not used to decide identity on its own.
"""
from __future__ import annotations

import math
import tempfile

import cv2
import numpy as np

from .geometry import Camera
from .pose import L_ANK, L_HIP, L_KNEE, L_SH, L_WR, R_ANK, R_HIP, R_KNEE, R_SH, R_WR, Pose, PoseEstimator


def decode(mp4: bytes, fps: float = 8.0, max_s: float = 40.0):
    """Clip bytes -> (frame generator sampled at ~fps for at most max_s seconds, effective fps).
    A generator, not a list: 40 s of 1080p frames would be ~2 GB."""
    f = tempfile.NamedTemporaryFile(suffix=".mp4")
    f.write(mp4)
    f.flush()
    cap = cv2.VideoCapture(f.name)
    src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not (1 <= src <= 120):
        src = 25.0
    step = max(1, int(round(src / fps)))

    def gen():
        try:
            n, i = 0, 0
            while n < max_s * src / step:
                if not cap.grab():
                    break
                if i % step == 0:
                    ok, fr = cap.retrieve()
                    if ok:
                        n += 1
                        yield fr
                i += 1
        finally:
            cap.release()
            f.close()
    return gen(), src / step


def _iou(a, b) -> float:
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


TORSO = (L_SH, R_SH, L_HIP, R_HIP)
WEAK_BODY = 0.2          # mean shoulder/hip confidence below this = not convincingly a person (coat, shadow, lamp)
WEAK_RUN = 4             # this many weak frames in a row end the track
APPEAR_MAX = 0.55        # Bhattacharyya distance limit for re-acquiring after a loss (0 same, 1 disjoint)


def body_quality(p: Pose) -> float:
    """Mean confidence of the shoulder and hip keypoints: the torso is what makes a detection a person."""
    return float(p.kp[list(TORSO), 2].mean())


def _appearance(fr, p: Pose):
    """Normalised HS histogram of the torso (shoulders..hips, or the box middle): cheap clothing-colour signature."""
    x1, y1, x2, y2 = (int(round(v)) for v in p.box)
    kp = p.kp
    if (kp[list(TORSO), 2] >= 0.3).all():
        ys = kp[list(TORSO), 1]
        y1, y2 = int(ys.min()), int(ys.max())
    else:
        h = y2 - y1
        y1, y2 = y1 + int(0.2 * h), y1 + int(0.6 * h)
    x1, x2 = max(0, x1), min(fr.shape[1], x2)
    y1, y2 = max(0, y1), min(fr.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    hsv = cv2.cvtColor(fr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def _shift(box, v):
    return (box[0] + v[0], box[1] + v[1], box[2] + v[0], box[3] + v[1])


def track(frames, pose: PoseEstimator, start_box=None, on_frame=None) -> tuple[list[Pose | None], tuple[int, int]]:
    """Follow ONE person: start from start_box (or the best-looking person: area x torso confidence), then match
    each frame by IoU with the box PREDICTED from the recent velocity (observation-centric, as in OC-SORT, so a
    short occlusion does not lose a walking person). Guards against drifting onto other objects:
      - torso quality: once the track has shown a clear torso (>= 0.5), WEAK_RUN frames in a row with mean
        shoulder/hip confidence < WEAK_BODY SUSPEND it (a hanging coat or a lamp is "a person" to the detector but
        has no confident torso). A track that starts weak (someone entering close to the lens) is not suspended
        until it has been seen clearly, so arrivals are not cut off;
      - while suspended, only a detection with a clear torso whose colours match the track is taken back;
      - appearance: after a loss, a candidate is only accepted if its torso colours match the track's
        (Bhattacharyya distance <= APPEAR_MAX against a running histogram). Returns one Pose (or None) per frame and the
    frame size (W, H). on_frame(frame, pose_or_None) is called for every frame while it is still in memory."""
    out: list[Pose | None] = []
    last, lost, size = start_box, 0, (0, 0)
    vel = np.zeros(2)                       # box-centre velocity, px per frame (smoothed)
    hist, weak, suspended, confirmed = None, 0, False, False
    for fr in frames:
        size = (fr.shape[1], fr.shape[0])
        p = None
        if suspended:                                             # only a clear, matching person resumes the track
            for c in sorted(pose(fr), key=lambda c: -body_quality(c)):
                if body_quality(c) < 0.5:
                    break
                a = _appearance(fr, c)
                if hist is None or (a is not None and cv2.compareHist(hist, a, cv2.HISTCMP_BHATTACHARYYA) <= APPEAR_MAX):
                    p, suspended, weak, lost, vel = c, False, 0, 0, np.zeros(2)
                    break
        else:
            pred = _shift(last, vel * (lost + 1)) if last is not None else None
            cand = pose.in_region(fr, pred) if pred is not None and lost < 3 else pose(fr)
            if cand:
                if pred is not None:
                    p = max(cand, key=lambda c: _iou(c.box, pred))
                    if _iou(p.box, pred) < 0.1:
                        p = None
                if p is None and (last is None or lost >= 3):
                    p = max(cand, key=lambda c: (c.box[2] - c.box[0]) * (c.box[3] - c.box[1]) * (0.2 + body_quality(c)))
                if p is not None and hist is not None and lost > 0:
                    a = _appearance(fr, p)
                    if a is not None and cv2.compareHist(hist, a, cv2.HISTCMP_BHATTACHARYYA) > APPEAR_MAX:
                        p = None                                  # someone/something else
        if p is not None:
            if body_quality(p) >= 0.5:
                confirmed = True
            weak = weak + 1 if body_quality(p) < WEAK_BODY else 0
            if weak >= WEAK_RUN and confirmed:                    # lost the person's torso: suspend the track
                suspended, p = True, None
                for j in range(1, WEAK_RUN):                      # the weak run itself is not the person either
                    if len(out) >= j:
                        out[-j] = None
        if p is not None:
            if last is not None and lost == 0:
                c_new = np.array([(p.box[0] + p.box[2]) / 2, (p.box[1] + p.box[3]) / 2])
                c_old = np.array([(last[0] + last[2]) / 2, (last[1] + last[3]) / 2])
                vel = 0.6 * vel + 0.4 * (c_new - c_old)
            if body_quality(p) >= 0.5:
                a = _appearance(fr, p)
                if a is not None:
                    hist = a if hist is None else cv2.normalize(0.8 * hist + 0.2 * a, None).flatten()
            last, lost = p.box, 0
        else:
            lost += 1
            if lost > 3:
                vel *= 0.0                                        # stale motion: search where it was last seen
        out.append(p)
        if on_frame is not None:
            on_frame(fr, p)
    return out, size


def _ok(kp, idx, c=0.5) -> bool:
    return all(kp[i, 2] >= c for i in idx)


def _angle(a, b, c) -> float:
    v1, v2 = a - b, c - b
    return math.degrees(math.acos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1)))


def stance_points(p: Pose, size) -> tuple | None:
    """(foot_px, head_px) when the person is standing/walking upright, whole body in frame, else None."""
    W, H = size
    x1, y1, x2, y2 = p.box
    m = 0.01 * max(W, H)
    if x1 < m or y1 < m or x2 > W - m or y2 > H - m:          # cut off by the frame edge
        return None
    kp = p.kp
    if not _ok(kp, (L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK), 0.5):
        return None
    sh, hip = kp[[L_SH, R_SH], :2].mean(0), kp[[L_HIP, R_HIP], :2].mean(0)
    lean = math.degrees(math.atan2(abs(sh[0] - hip[0]), hip[1] - sh[1] + 1e-9))
    if lean > 20:                                                # bending over / sitting / lying
        return None
    legs = [_angle(kp[h, :2], kp[k, :2], kp[a, :2]) for h, k, a in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK))
            if kp[k, 2] >= 0.4]
    if legs and max(legs) < 155:                                # no straight (stance) leg: crouching, mid-jump
        return None
    head_x = kp[:5, 0][kp[:5, 2] >= 0.4].mean() if (kp[:5, 2] >= 0.4).any() else (x1 + x2) / 2
    foot_x = kp[[L_ANK, R_ANK], 0].mean()
    return (foot_x, y2), (head_x, y1)                           # box bottom = sole, box top = top of head/hair


def height(trk: list, cam: Camera | None, size) -> dict | None:
    if cam is None:
        return None
    pts = [sp for p in trk if p is not None and (sp := stance_points(p, size))]
    if len(pts) < 3:
        return None
    z = cam.heights([a for a, _ in pts], [b for _, b in pts], size)
    z = z[np.isfinite(z) & (z > 0.5) & (z < 2.5)]
    if len(z) < 3:
        return None
    q1, med, q3 = np.percentile(z, [25, 50, 75])
    return {"cm": round(float(med) * 100, 1), "iqr_cm": round(float(q3 - q1) * 100, 1), "frames": int(len(z))}


def gait(trk: list, fps: float, cam: Camera | None, size) -> dict | None:
    """Cadence (steps/min), relative step length, arm swing, walking speed, torso lean, from the longest run of
    frames with good leg keypoints. Uses |ankle separation| so left/right keypoint swaps don't matter."""
    good = [i for i, p in enumerate(trk) if p is not None and _ok(p.kp, (L_HIP, R_HIP, L_ANK, R_ANK), 0.4)]
    if len(good) < 8:
        return None
    runs, cur = [], [good[0]]                                   # longest run, allowing 2-frame gaps
    for i in good[1:]:
        if i - cur[-1] <= 3:
            cur.append(i)
        else:
            runs.append(cur); cur = [i]
    runs.append(cur)
    run = max(runs, key=len)
    dur = (run[-1] - run[0]) / fps
    if dur < 2.5:
        return {"walking": False, "reason": "under 2.5 s of clear legs"}
    t = np.array(run, float) / fps
    P = [trk[i] for i in run]
    bh = np.array([p.box[3] - p.box[1] for p in P])
    sep = np.array([abs(p.kp[L_ANK, 0] - p.kp[R_ANK, 0]) for p in P]) / bh
    foot = np.array([[p.kp[[L_ANK, R_ANK], 0].mean(), p.box[3]] for p in P])
    tt = np.arange(t[0], t[-1], 1 / fps)
    s = np.interp(tt, t, sep)
    k = max(1, int(round(fps)))                                  # remove the slow trend (approach/perspective)
    s = s - np.convolve(s, np.ones(k) / k, mode="same")
    s = s[k // 2: len(s) - k // 2] if len(s) > k + 4 else s
    ac = np.correlate(s, s, "full")[len(s) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(0.3 * fps), min(len(ac) - 1, int(math.ceil(0.85 * fps)))   # one step = 0.3-0.85 s
    if hi <= lo + 1:
        return None
    lag = lo + int(np.argmax(ac[lo:hi + 1]))
    if 0 < lag < len(ac) - 1:                                    # parabolic refinement of the peak
        y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
        den = y0 - 2 * y1 + y2
        lagf = lag + (0.5 * (y0 - y2) / den if den else 0)
    else:
        lagf = lag
    quality = float(ac[lag])
    step_s = lagf / fps
    move = foot[-1] - foot[0]
    shoulders = [abs(p.kp[L_SH, 0] - p.kp[R_SH, 0]) / (p.box[3] - p.box[1]) for p in P
                 if p.kp[L_SH, 2] >= 0.4 and p.kp[R_SH, 2] >= 0.4]
    shoulder_rel = float(np.median(shoulders)) if shoulders else 1.0   # ~0.22 facing toward/away, < 0.12 side-on
    side_on_img = shoulder_rel < 0.12 and abs(move[0]) > 1.5 * abs(move[1]) and 0.7 < bh[-1] / bh[0] < 1.4
    res = {"walking": True, "seconds": round(dur, 1), "frames": len(run), "periodicity": round(quality, 2),
           "cadence_spm": round(60 / step_s, 1), "step_rel": round(float(np.percentile(sep, 95)), 3),
           "speed_bh_s": round(float(np.linalg.norm(move) / np.median(bh) / dur), 2), "shoulder_rel": round(shoulder_rel, 3),
           "lean_deg": round(float(np.median([math.degrees(math.atan2(
               p.kp[[L_SH, R_SH], 0].mean() - p.kp[[L_HIP, R_HIP], 0].mean(),
               p.kp[[L_HIP, R_HIP], 1].mean() - p.kp[[L_SH, R_SH], 1].mean())) for p in P])), 1)}
    arms = []
    for wr, sh in ((L_WR, L_SH), (R_WR, R_SH)):
        v = np.array([(p.kp[wr, 0] - p.kp[sh, 0]) / (p.box[3] - p.box[1]) for p in P if p.kp[wr, 2] >= 0.4 and p.kp[sh, 2] >= 0.4])
        if len(v) >= 8:
            arms.append(float(np.percentile(v, 95) - np.percentile(v, 5)))
    if arms:
        res["arm_swing_rel"] = round(float(np.mean(arms)), 3)
    side_on = side_on_img
    if cam is not None:
        g = cam.ground(foot, size)
        okg = np.isfinite(g).all(1)
        if okg.sum() >= 8:
            gg, tg = g[okg], t[okg]
            d = gg[-1] - gg[0]
            dist = float(np.linalg.norm(d))
            res["speed_mps"] = round(dist / (tg[-1] - tg[0]), 2)
            res["step_m"] = round(res["speed_mps"] * step_s, 2)
            view = gg.mean(0) - cam.C[:2]                        # camera -> person, on the ground
            cosang = abs(d @ view) / (dist * np.linalg.norm(view) + 1e-9)
            side_on = dist > 1.0 and cosang < 0.5                 # walking within 30 deg of across the view
    res["side_on"] = bool(side_on)
    res["usable"] = bool(side_on and quality >= 0.35 and res["speed_bh_s"] > 0.2 and dur / step_s >= 4)
    # v2: the explainable measurements (gaitlab.py). The keys above keep their meaning; gaitlab adds rhythm and
    # symmetry from any view and knee/arm shape from side and diagonal views, and decides "usable" by view.
    try:
        from . import gaitlab
        lab = gaitlab.analyse(trk, fps, cam, size)
    except Exception:                                            # never lose the old summary to the new one
        lab = None
    if lab and lab.get("walking"):
        for k, v in lab.items():
            if k not in ("walking", "seconds", "frames", "cadence_spm", "periodicity", "side_on", "usable"):
                res[k] = v
        res["usable_v1"] = res["usable"]
        res["usable"] = bool(lab["usable"])
        res["version"] = gaitlab.VERSION
    return res


def gait_distance(a: dict, b: dict) -> float | None:
    """Rough dissimilarity of two usable gait summaries (0 = identical; > ~2 = clearly different)."""
    if not (a and b and a.get("usable") and b.get("usable")):
        return None
    terms = [((a["cadence_spm"] - b["cadence_spm"]) / 8) ** 2]
    if "step_m" in a and "step_m" in b:
        terms.append(((a["step_m"] - b["step_m"]) / 0.08) ** 2)
    else:
        terms.append(((a["step_rel"] - b["step_rel"]) / 0.05) ** 2)
    if "arm_swing_rel" in a and "arm_swing_rel" in b:
        terms.append(((a["arm_swing_rel"] - b["arm_swing_rel"]) / 0.05) ** 2)
    return round(math.sqrt(sum(terms) / len(terms)), 2)


def _keeping(frames, every: int, store: list, width: int = 960):
    """Pass frames through, keeping every Nth as a downscaled JPEG (for the evidence card / GIF)."""
    for i, fr in enumerate(frames):
        if every and i % every == 0:
            sc = width / fr.shape[1]
            small = cv2.resize(fr, (width, int(fr.shape[0] * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else fr
            ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                store.append((i, jpg.tobytes()))
        yield fr


def analyse(mp4_parts: list[bytes], pose: PoseEstimator, cam: Camera | None, fps: float = 8.0, max_s: float = 40.0,
            keep_every: int = 0, on_frame=None) -> dict:
    """One or more clip pieces of the same event -> {"height": ..., "gait": ...} (either may be None).
    keep_every > 0 also returns "evidence": per piece the kept frames (JPEG), the track, frame size and fps,
    for evidence.card()/gif(). It is not JSON: pop it before publishing."""
    heights, gaits, size, parts = [], [], None, []
    per = max_s / max(1, len(mp4_parts))
    for mp4 in mp4_parts:
        frames, efps = decode(mp4, fps, per)
        kept: list = []
        trk, size = track(_keeping(frames, keep_every, kept), pose, on_frame=on_frame)
        if not trk:
            continue
        parts.append({"frames": kept, "track": trk, "size": size, "fps": efps})
        heights.append((height(trk, cam, size), trk, size))
        g = gait(trk, efps, cam, size)
        if g:
            gaits.append(g)
    hs = [h for h, _, _ in heights if h]
    h = max(hs, key=lambda d: d["frames"]) if hs else None
    if len(hs) > 1:  # pool frames from both pieces
        trk_all, sz = sum((t for _, t, _ in heights), []), heights[0][2]
        h = height(trk_all, cam, sz) or h
    usable = [g for g in gaits if g.get("usable")]
    g = max(usable or gaits, key=lambda d: (d.get("usable", False), d.get("seconds", 0)), default=None)
    out = {"height": h, "gait": g, "calibrated": cam is not None}
    if keep_every:
        out["evidence"] = {"parts": parts}
    return out

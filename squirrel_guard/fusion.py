"""Multi-camera fusion of one visit: the same person seen by 2+ cameras calibrated in ONE world frame
(scripts/multicam_rig.py) -> one IK rig fitted over every view -> one replay (path, stick figure, gait, body) with the
other cameras as "cameras_extra". Shared by the live profiler (LiveFusion: Frigate events -> clips -> fused replay)
and the offline tools (scripts/multicam_demo.py, scripts/fusion_eval.py). docs/fusion.md has the measurements.

Pipeline (fuse_views):
  1. sample every camera's clip on ONE time grid (fps F, t = window start + k / F; nearest decoded frame by its
     timestamp, so 25 / 20 / 15 fps cameras line up), pick the one person per frame (rigpipe.single_person) and run
     RTMW whole-body keypoints on it (extract_view);
  2. time offsets: clips cut by different recorders start a little apart (Frigate cuts on keyframes, camera clocks
     and network delays differ). For every candidate shift of each camera against an already synchronised one, the
     23 body + feet joints seen confidently by both are cast as rays; the shift with the smallest median ray-to-ray
     gap wins (rays of the same joint at the same instant meet), refined to a fraction of a frame (synchronise);
  3. calibration check: a camera whose best median ray gap stays large (it moved since the calibration, or it saw
     somebody else) is dropped, as is a camera with too few common frames (it saw nothing useful);
  4. one rig fit over every remaining view (rig.fit); a view the rig cannot explain (per-view error far above the
     others: moved camera) is dropped and the fit repeated; with fewer than 2 views left the fusion fails and the
     per-camera fits stay;
  5. replay on the PRIMARY camera's time base (the one that saw the person best): path from the rig
     (trajectory.estimate with ground_override), stick figure + gait from the rig, cut-outs/background from the
     primary, "cameras_extra" = the other views, "fusion" = offsets, gaps, members, cost.
"""
from __future__ import annotations

import logging
import tempfile
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

from . import rig, rigpipe

log = logging.getLogger(__name__)
VERSION = "fusion/1"
SYNC_JOINTS = list(range(23))           # COCO-WholeBody body 17 + feet 6
TORSO = [5, 6, 11, 12]


# -- calibration frames ------------------------------------------------------------------------------------------
def frame_id(cam) -> str | None:
    """World frame a calibration lives in: cameras with the same id were calibrated TOGETHER and can be fused.
    An explicit "frame" in the calibration JSON wins (scripts/multicam_rig.py writes one); older joint calibrations
    fall back to method + date. Single-camera calibrations (calibrate.py, map) have no shared frame: None."""
    meta = getattr(cam, "meta", None) or {}
    if meta.get("frame"):
        return str(meta["frame"])
    m = str(meta.get("method") or "")
    if m.startswith("multicam") and meta.get("date"):
        return f"{m}:{meta['date']}"
    return None


# -- geometry ------------------------------------------------------------------------------------------------------
def closest(C1, d1, C2, d2):
    """Midpoints and gaps of the closest approach of ray pairs (C1 + s d1, C2 + t d2); rows of d are unit rays.
    s/t <= 0 (behind a camera) or parallel rays -> gap NaN."""
    d1, d2 = np.atleast_2d(d1), np.atleast_2d(d2)
    w0 = np.asarray(C1, float) - np.asarray(C2, float)
    b = np.einsum("ij,ij->i", d1, d2)
    d = d1 @ w0
    e = d2 @ w0
    den = 1 - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (b * e - d) / den
        t = (e - b * d) / den
    P1, P2 = C1 + s[:, None] * d1, C2 + t[:, None] * d2
    gap = np.linalg.norm(P1 - P2, axis=1)
    bad = ~((s > 0) & (t > 0) & (den > 1e-6))
    gap[bad] = np.nan
    return (P1 + P2) / 2, gap


def view_rays(cam, kp: np.ndarray, size, conf: float = 0.5, joints=SYNC_JOINTS):
    """(n, J, 3) keypoints (px + 0..1 confidence) -> (n, J', 3) unit world rays (NaN where conf < conf or missing)."""
    k = np.asarray(kp, float)[:, joints]
    ok = np.isfinite(k[..., 0]) & (np.nan_to_num(k[..., 2]) >= conf)
    out = np.full(k.shape[:2] + (3,), np.nan)
    if ok.any():
        out[ok] = cam.rays(k[ok][:, :2], size)
    return out


# -- clips on a common time grid ---------------------------------------------------------------------------------
class ClipSampler:
    """Iterate a clip's frames for the time grid t_k = start_s + k / fps (k < n), clip time 0 = its first frame.
    Each target takes the first decoded frame at or after t_k - half a source frame (by container timestamp, so a
    15 fps and a 25 fps camera land on the same instants). A frame serving several targets (fps above the camera's
    rate) is yielded once; `targets` holds the target indices of the frame just yielded, `idx[k]` the unique frame
    number for target k (-1 = clip ended / started later)."""

    def __init__(self, data: bytes | str | Path, fps: float, n: int, start_s: float = 0.0):
        self.data, self.fps, self.n, self.start_s = data, float(fps), int(n), float(start_s)
        self.idx = np.full(self.n, -1, int)
        self.targets: list[int] = []
        self.src_fps = None
        self.decoded = 0
        self.unique = 0

    def __iter__(self):
        tmp = None
        if isinstance(self.data, (bytes, bytearray)):
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4")
            tmp.write(self.data)
            tmp.flush()
            path = tmp.name
        else:
            path = str(self.data)
        cap = cv2.VideoCapture(path)
        try:
            src = cap.get(cv2.CAP_PROP_FPS) or 25.0
            src = src if 1 <= src <= 120 else 25.0
            self.src_fps = src
            half = 0.5 / src
            k, i, t_first, last_t = 0, 0, None, -1e9
            while k < self.n:
                if not cap.grab():
                    break
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if t_first is None:
                    t_first = t
                t = t - t_first
                if not np.isfinite(t) or t <= last_t:          # no usable timestamps: frame counter instead
                    t = i / src
                last_t = t
                i += 1
                self.decoded += 1
                tg = []
                while k < self.n and t >= self.start_s + k / self.fps - half:
                    tg.append(k)
                    k += 1
                if not tg:
                    continue
                ok, fr = cap.retrieve()
                if not ok or fr is None:
                    continue
                for kk in tg:
                    self.idx[kk] = self.unique
                self.unique += 1
                self.targets = tg
                yield fr
        finally:
            cap.release()
            if tmp is not None:
                tmp.close()


def grid_frames(data, fps: float, max_s: float, start_s: float = 0.0):
    """A frame for EVERY grid instant (repeated when the camera is slower than fps) until the clip ends: drop-in for
    kinematics.decode where several cameras must share one time base. -> (generator, fps)."""
    smp = ClipSampler(data, fps, int(round(max_s * fps)), start_s)

    def gen():
        for fr in smp:
            for _ in smp.targets:
                yield fr
    return gen(), float(fps)


def extract_view(data, name: str, cam, pose, wb, fps: float, max_s: float, start_s: float = 0.0,
                 collector=None, replay_step: int = 1) -> dict:
    """One camera's clip -> keypoints of the one person on the time grid (n = max_s * fps targets).
    collector (replay_assets.Collector): fed every replay_step-th target (the replay's frames).
    -> {"name", "cam", "size", "kp" (n, 133, 3) px + 0..1 conf, "box" (n, 4), "picks" [Pose|None] * n, ...}"""
    n = int(round(max_s * fps))
    smp = ClipSampler(data, fps, n, start_s)
    kp_u, box_u, pick_u = [], [], []
    t0 = time.time()
    fed = {"k": 0}                      # next replay frame (grid index) the collector's frame counter stands at

    def on_frame(fr, p):
        if p is None:
            kp_u.append(np.full((133, 3), np.nan, np.float32))
            box_u.append([np.nan] * 4)
        else:
            kp_u.append(wb(fr, p.box))
            box_u.append(list(p.box))
        pick_u.append(p)
        if collector is not None:
            for k in smp.targets:
                if k % replay_step:
                    continue
                while fed["k"] < k:     # replay frames without video (clip starts later): keep frame numbers aligned
                    collector.n += 1
                    fed["k"] += replay_step
                collector.on_frame(fr, p)
                fed["k"] = k + replay_step

    _, size = rigpipe.single_person(smp, pose, on_frame=on_frame)
    kp = np.full((n, 133, 3), np.nan, np.float32)
    box = np.full((n, 4), np.nan, np.float32)
    picks: list = [None] * n
    for k in range(n):
        j = smp.idx[k]
        if 0 <= j < len(kp_u):
            kp[k], box[k], picks[k] = kp_u[j], box_u[j], pick_u[j]
    if getattr(wb, "raw", False):                 # uncalibrated RTMW peaks: map to 0..1
        kp[..., 2] = rig.conf_of(kp[..., 2])
    return {"name": name, "cam": cam, "size": size or tuple(cam.size), "kp": kp, "box": box, "picks": picks,
            "collector": collector, "fps": fps, "start_s": start_s, "src_fps": smp.src_fps,
            "frames_decoded": smp.decoded, "frames_keypointed": smp.unique,
            "frames_person": int(sum(p is not None for p in pick_u)), "seconds": round(time.time() - t0, 1)}


def view_quality(v: dict, conf: float = 0.5) -> float:
    """How well a camera saw the person: summed confident torso/head joints over the visit."""
    k = np.nan_to_num(np.asarray(v["kp"], float)[:, rig.TORSO_HEAD, 2])
    return float((k >= conf).sum())


# -- time offsets ----------------------------------------------------------------------------------------------------
def shift_series(a: np.ndarray, shift: float) -> np.ndarray:
    """out[i] = a[i + shift] (fractional: linear between neighbours where both exist, else the nearer one)."""
    a = np.asarray(a, float)
    n = len(a)
    out = np.full_like(a, np.nan)
    i = np.arange(n)
    x = i + shift
    lo = np.floor(x).astype(int)
    fr = x - lo
    for j in range(n):
        l0, f = lo[j], fr[j]
        if f < 1e-6:
            if 0 <= l0 < n:
                out[j] = a[l0]
            continue
        h0 = l0 + 1
        va = a[l0] if 0 <= l0 < n else None
        vb = a[h0] if 0 <= h0 < n else None
        if va is not None and vb is not None:
            m = (1 - f) * va + f * vb
            if a.ndim > 1:          # a joint missing at either end: take the nearer frame's value as is
                near = va if f < 0.5 else vb
                bad = ~np.isfinite(m)
                m[bad] = near[bad]
            elif not np.isfinite(m):
                m = va if f < 0.5 else vb
            out[j] = m
        elif va is not None and f < 0.5:
            out[j] = va
        elif vb is not None and f >= 0.5:
            out[j] = vb
    return out


def gap_curve(cA, rA, cB, rB, k_lo: int, k_hi: int, min_joints: int = 4, min_frames: int = 5) -> dict:
    """Median joint ray gap for each integer shift k in [k_lo, k_hi] (rB[i + k] vs rA[i]); rA/rB from view_rays.
    -> {k: (median gap m, frames compared)} for shifts with >= min_frames frames of >= min_joints common joints."""
    n_a, n_b = len(rA), len(rB)
    okA = np.isfinite(rA[..., 0])
    okB = np.isfinite(rB[..., 0])
    out = {}
    for k in range(k_lo, k_hi + 1):
        a0, a1 = max(0, -k), min(n_a, n_b - k)
        if a1 - a0 < min_frames:
            continue
        m = okA[a0:a1] & okB[a0 + k:a1 + k]
        rows = m.sum(1) >= min_joints
        if rows.sum() < min_frames:
            continue
        m &= rows[:, None]
        ii, jj = np.nonzero(m)
        _, g = closest(cA.C, rA[a0:a1][ii, jj], cB.C, rB[a0 + k:a1 + k][ii, jj])
        G = np.full(m.shape, np.nan)
        G[ii, jj] = g
        with warnings.catch_warnings():     # per-frame median, then the median over frames (a few frames with
            warnings.simplefilter("ignore", RuntimeWarning)       # many joints must not outvote the rest)
            per = np.nanmedian(G[rows], axis=1)
        per = per[np.isfinite(per)]
        if len(per) >= min_frames:
            out[k] = (float(np.median(per)), int(len(per)))
    return out


def best_shift(curve: dict) -> tuple[float, float, int] | None:
    """Best (shift in frames, refined by a parabola through the neighbours; median gap; frames) or None."""
    if not curve:
        return None
    k = min(curve, key=lambda kk: curve[kk][0])
    if len(curve) > 2 and k in (min(curve), max(curve)):
        return None          # minimum at the edge of the searched offsets: no real match (moved camera, wrong person)
    g, cnt = curve[k]
    sh = float(k)
    if k - 1 in curve and k + 1 in curve:
        f0, f1, f2 = curve[k - 1][0], g, curve[k + 1][0]
        den = f0 - 2 * f1 + f2
        if den > 1e-9:
            sh = k + float(np.clip(0.5 * (f0 - f2) / den, -0.5, 0.5))
    return sh, g, cnt


def project_np(cam, X: np.ndarray, size) -> np.ndarray:
    """World points (..., 3) -> pixels (..., 2) in a frame of `size` (OpenCV model, distortion included); points
    behind the camera -> NaN."""
    X = np.asarray(X, float)
    flat = X.reshape(-1, 3)
    out = np.full((len(flat), 2), np.nan)
    ok = np.isfinite(flat).all(1)
    if ok.any():
        Xc = (flat[ok] - cam.C) @ cam.R
        front = Xc[:, 2] > 0.05
        rvec, _ = cv2.Rodrigues(cam.R.T)
        tvec = -cam.R.T @ cam.C
        uv, _ = cv2.projectPoints(flat[ok].reshape(-1, 1, 3), rvec, tvec, cam.K, cam.D)
        uv = uv.reshape(-1, 2) * [size[0] / cam.size[0], size[1] / cam.size[1]]
        uv[~front] = np.nan
        out[np.nonzero(ok)[0]] = uv
    return out.reshape(X.shape[:-1] + (2,))


def reproj_curve(J: np.ndarray, v: dict, k_lo: int, k_hi: int, conf: float = 0.5, min_joints: int = 4,
                 min_frames: int = 5) -> dict:
    """3D joints J (n, 23, 3) on the reference time base vs camera v's keypoints shifted by k (v.kp[i + k] ~ J[i]):
    {k: (median reprojection error in box heights, frames)}. The 3D points come from OTHER cameras, so a camera
    that moved cannot hide its error behind a time shift as easily as with ray gaps."""
    n = len(J)
    uv = project_np(v["cam"], J, v["size"])                      # (n, 23, 2)
    kp = np.asarray(v["kp"], float)[:, :23]
    bh = np.asarray(v["box"], float)[:, 3] - np.asarray(v["box"], float)[:, 1]
    okv = np.isfinite(kp[..., 0]) & (np.nan_to_num(kp[..., 2]) >= conf)
    okJ = np.isfinite(uv[..., 0])
    out = {}
    nv = len(kp)
    for k in range(k_lo, k_hi + 1):
        a0, a1 = max(0, -k), min(n, nv - k)
        if a1 - a0 < min_frames:
            continue
        m = okJ[a0:a1] & okv[a0 + k:a1 + k]
        rows = (m.sum(1) >= min_joints) & (np.nan_to_num(bh[a0 + k:a1 + k]) > 1)
        if rows.sum() < min_frames:
            continue
        e = np.linalg.norm(uv[a0:a1] - kp[a0 + k:a1 + k, :, :2], axis=-1) / np.maximum(bh[a0 + k:a1 + k], 1)[:, None]
        e[~m] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            per = np.nanmedian(e[rows], axis=1)
        out[k] = (float(np.median(per)), int(rows.sum()))
    return out


def synchronise(views: list[dict], fps: float, max_offset_s: float = 3.0, max_gap_m: float = 0.15,
                min_frames: int = 5, conf: float = 0.5, max_reproj_box: float = 0.06) -> tuple[dict, dict]:
    """Time offsets of every camera. -> (shifts {name: s} meaning view.kp[i + s] ~ reference frame i, with the
    reference = the first ANCHOR camera; report {name: {...}}). Cameras missing from shifts were dropped.

    Anchor = the camera pair whose rays agree best (smallest median joint ray gap at the best shift, <= max_gap_m).
    Every further camera is checked against the anchors' TRIANGULATION: its keypoints, shifted, must reproject onto
    those 3D joints within max_reproj_box box heights. (With only ray gaps, a camera that turned a few degrees while
    the person walked could pair up at a wrong time shift with a small gap; synthetic 8 deg: 3 cm.) With exactly two
    cameras only the ray-gap test is available (docs/fusion.md: limits)."""
    names = [v["name"] for v in views]
    byname = {v["name"]: v for v in views}
    rays = {v["name"]: view_rays(v["cam"], v["kp"], v["size"], conf) for v in views}
    K = int(round(max_offset_s * fps))
    report: dict = {}
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            bs = best_shift(gap_curve(byname[a]["cam"], rays[a], byname[b]["cam"], rays[b], -K, K,
                                      min_frames=min_frames))
            if bs is not None:
                pairs[(a, b)] = bs
    good = sorted((bs[1], a, b) for (a, b), bs in pairs.items() if bs[1] <= max_gap_m)
    if not good:
        both = {}                # frames where a camera and some other one BOTH saw the person (any shift, +-K)
        for nm in names:
            okn = np.isfinite(rays[nm][..., 0]).sum(1) >= 4
            both[nm] = 0
            for o in names:
                if o != nm:
                    oko = np.isfinite(rays[o][..., 0]).sum(1) >= 4
                    both[nm] = max(both[nm], int(okn.sum()) if oko.sum() >= min_frames else 0)
        for nm in names:
            seen = {f"{a}-{b}" if a == nm else f"{b}-{a}": round(bs[1], 3) for (a, b), bs in pairs.items() if nm in (a, b)}
            report[nm] = {"role": "dropped", "pair_ray_gaps_m": seen,
                          "reason": ("no camera pair agrees: median joint ray gaps " +
                                     ", ".join(f"{k} {v * 100:.0f} cm" for k, v in seen.items()) +
                                     f" > {max_gap_m * 100:.0f} cm (a camera moved since the calibration, or they saw "
                                     "different people)") if seen else
                                    (f"seen in {both[nm]} frames, but its joint rays never meet another camera's in "
                                     "front of both: camera moved since the calibration?") if both[nm] >= 3 * min_frames
                                    else "too few frames seen together with another camera (saw nothing useful, or "
                                         "never at the same time)"}
        return {}, report
    g, a, b = good[0]
    sh, _, cnt = pairs[(a, b)]
    shifts = {a: 0.0, b: sh}
    report[a] = {"role": "anchor"}
    report[b] = {"role": "anchor", "synced_to": a, "shift_frames": round(sh, 2), "shift_s": round(sh / fps, 3),
                 "median_ray_gap_m": round(g, 3), "frames_compared": cnt}
    n = len(byname[a]["kp"])
    J, _ = triangulate([align(byname[a], 0.0, n), align(byname[b], sh, n)], conf=conf)
    for nm in sorted((x for x in names if x not in shifts), key=lambda x: -view_quality(byname[x])):
        v = byname[nm]
        bs = best_shift(reproj_curve(J, v, -K + int(round(min(shifts.values()))), K + int(round(max(shifts.values()))),
                                     conf=conf, min_frames=min_frames))
        pg = {f"{nm}-{x}": round(pairs[(x, nm)][1] if (x, nm) in pairs else pairs[(nm, x)][1], 3)
              for x in (a, b) if (x, nm) in pairs or (nm, x) in pairs}
        if bs is None:
            report[nm] = {"role": "dropped", "pair_ray_gaps_m": pg,
                          "reason": "too few frames seen together with the anchor cameras (saw nothing useful, or "
                                    "never at the same time)"}
            continue
        s_nm, err, cnt = bs
        if err > max_reproj_box:
            report[nm] = {"role": "dropped", "pair_ray_gaps_m": pg, "reproj_err_box": round(err, 4),
                          "shift_s": round(s_nm / fps, 3),
                          "reason": f"reprojection of the {a}+{b} triangulation {err:.3f} box heights > "
                                    f"{max_reproj_box:.3f}: camera moved since the calibration, or another person"}
            continue
        shifts[nm] = s_nm
        report[nm] = {"role": "secondary", "synced_to": f"{a}+{b}", "shift_frames": round(s_nm, 2),
                      "shift_s": round(s_nm / fps, 3), "reproj_err_box": round(err, 4), "frames_compared": cnt,
                      "pair_ray_gaps_m": pg}
    return shifts, report


def align(v: dict, shift: float, n: int) -> dict:
    """A view on the primary's time base: kp (n, 133, 3) and box_h (n,) with out[i] = v[i + shift]."""
    kp = np.asarray(v["kp"], float)
    bh = np.asarray(v["box"], float)[:, 3] - np.asarray(v["box"], float)[:, 1]
    if len(kp) < n:
        kp = np.concatenate([kp, np.full((n - len(kp),) + kp.shape[1:], np.nan)])
        bh = np.concatenate([bh, np.full(n - len(bh), np.nan)])
    kp, bh = kp[:n], bh[:n]
    if abs(shift) > 1e-6:
        kp, bh = shift_series(kp, shift), shift_series(bh, shift)
    return {"name": v["name"], "cam": v["cam"], "size": tuple(int(x) for x in v["size"]), "kp": kp, "box_h": bh}


def triangulate(aligned: list[dict], conf: float = 0.5, max_gap: float = 0.25):
    """Aligned views (primary first) -> (n, 23, 3) joint positions (weighted midpoints of every camera pair, NaN where
    fewer than 2 cameras agree) and the list of accepted ray gaps."""
    n = len(aligned[0]["kp"])
    J = np.full((n, 23, 3), np.nan)
    W = np.zeros((n, 23))
    gaps = []
    rays = [view_rays(v["cam"], v["kp"], v["size"], conf) for v in aligned]
    for a in range(len(aligned)):
        for b in range(a + 1, len(aligned)):
            ok = np.isfinite(rays[a][..., 0]) & np.isfinite(rays[b][..., 0])
            ii, jj = np.nonzero(ok)
            if not len(ii):
                continue
            X, g = closest(aligned[a]["cam"].C, rays[a][ii, jj], aligned[b]["cam"].C, rays[b][ii, jj])
            good = np.isfinite(g) & (g <= max_gap) & (X[:, 2] > -0.2) & (X[:, 2] < 2.6)
            for i, j, x, gg in zip(ii[good], jj[good], X[good], g[good]):
                w = 1.0 / (0.02 + gg)
                J[i, j] = (np.nan_to_num(J[i, j]) * W[i, j] + w * x) / (W[i, j] + w)
                W[i, j] += w
                gaps.append(float(gg))
    return J, gaps


# -- fit + checks ----------------------------------------------------------------------------------------------------
def view_error(res: dict, view: dict, conf: float = 0.5) -> dict:
    """Reprojection error of a fitted rig in ONE view (also one the fit never saw: held-out check), on frames where
    the rig is fitted: median over confident joints, in box heights and pixels."""
    import torch
    X = torch.tensor(res["markers"], dtype=torch.float64)
    uv, depth = rig.project(X, view["cam"], view["size"])
    uv, depth = uv.numpy(), depth.numpy()
    kp = np.asarray(view["kp"], float)[:, :23]
    bh = np.nan_to_num(np.asarray(view["box_h"], float), nan=0.0)
    m = np.isfinite(kp[..., 0]) & (np.nan_to_num(kp[..., 2]) >= conf) & res["seen"][:, None] & (bh[:, None] > 1)
    behind = int((m & (depth < 0.05)).sum())        # rig points behind this camera: a grossly wrong fit
    m &= depth >= 0.05
    if not m.any():
        return {"median_err_box": None, "median_err_px": None, "points": 0, "behind_camera": behind}
    e = np.linalg.norm(uv - kp[..., :2], axis=-1)
    return {"median_err_box": round(float(np.median((e / np.maximum(bh, 1)[:, None])[m])), 4),
            "median_err_px": round(float(np.median(e[m])), 1), "points": int(m.sum()), "behind_camera": behind}


def fit_views(aligned: list[dict], fps: float, init_xy=None, stature0: float = 1.70, iters=(300, 900),
              max_err_box: float = 0.06, err_ratio: float = 1.5) -> tuple[dict | None, list[dict], dict]:
    """rig.fit over every aligned view. A view with median error above max_err_box box heights (well fitted views:
    0.01-0.03) that is also >= err_ratio x the best view's is dropped and the fit repeated (3+ views); if any view
    stays above max_err_box the cameras do not agree (moved camera, another person): (None, ...).
    -> (result | None, views used, {name: drop reason})."""
    dropped: dict = {}
    views = list(aligned)
    n = len(views[0]["kp"])
    while True:
        t0 = time.time()
        res = rig.fit(views, n, 1.0 / fps, init_xy=init_xy, stature0=stature0, iters=iters, verbose=False,
                      parts=rigpipe.PARTS or None)
        res["fit_s"] = round(time.time() - t0, 1)
        if not res["seen"].any():
            return None, views, dropped
        errs = [pv.get("median_err_box") for pv in res["per_view"]]
        ok = [e for e in errs if e is not None]
        if not ok or max(ok) <= max_err_box:
            return res, views, dropped
        worst = max(range(len(views)), key=lambda i: errs[i] if errs[i] is not None else -1)
        why = f"rig error {errs[worst]:.3f} box heights (> {max_err_box}): camera moved since the calibration?"
        if len(views) > 2 and errs[worst] >= err_ratio * min(ok):
            dropped[views[worst]["name"]] = why
            views = [v for i, v in enumerate(views) if i != worst]
            continue
        for i, v in enumerate(views):
            if errs[i] is not None and errs[i] > max_err_box:
                dropped[v["name"]] = why if i == worst else f"rig error {errs[i]:.3f} box heights (> {max_err_box})"
        return None, views, dropped


def fuse_views(views: list[dict], fps: float, max_offset_s: float = 4.0, max_gap_m: float = 0.15,
               iters=(300, 900), stature0: float = 1.70, min_quality: float = 20.0,
               primary: str | None = None, max_err_box: float = 0.06) -> dict:
    """extract_view outputs (any order) -> {"ok", "primary", "aligned", "res", "shifts", "report", "triangulated",
    "gaps", "reason"}. Primary = the camera that saw the person best (or `primary`)."""
    out: dict = {"ok": False, "report": {}}
    seen = [v for v in views if view_quality(v) >= min_quality]
    for v in views:
        if v not in seen:
            out["report"][v["name"]] = {"role": "dropped", "reason": "person not seen well enough (torso/head joints "
                                        f"{view_quality(v):.0f} < {min_quality:.0f})"}
    if len(seen) < 2:
        out["reason"] = "fewer than 2 cameras saw the person"
        return out
    seen.sort(key=lambda v: (v["name"] != primary, -view_quality(v)))
    shifts, rep = synchronise(seen, fps, max_offset_s=max_offset_s, max_gap_m=max_gap_m)
    out["report"].update(rep)
    used = [v for v in seen if v["name"] in shifts]       # seen is sorted: the first used = the primary
    if used:
        s_p = shifts[used[0]["name"]]
        shifts = {k: round(v - s_p, 3) for k, v in shifts.items()}  # on the PRIMARY's time base
        out["primary"] = used[0]["name"]
        for k, v in shifts.items():
            r = out["report"][k]
            r["role"] = "primary" if k == out["primary"] else "secondary"
            r["shift_frames"], r["shift_s"] = round(v, 2), round(v / fps, 3)
    out["shifts"] = shifts
    if len(used) < 2:
        out["reason"] = "no second camera could be synchronised within the calibration tolerance"
        return out
    n = len(used[0]["kp"])
    aligned = [align(v, shifts[v["name"]], n) for v in used]
    J, gaps = triangulate(aligned)
    out["triangulated"], out["gaps"] = J, gaps
    tor = J[:, TORSO, :2]
    init = np.full((n, 2), np.nan)
    okt = np.isfinite(tor[..., 0]).sum(1) >= 2
    if okt.any():
        init[okt] = np.nanmean(tor[okt], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        head = np.nanmean(J[:, :5, 2], axis=1)
    s0 = float(np.nanmedian(head) + 0.11) if np.isfinite(head).sum() >= 3 else stature0
    s0 = float(np.clip(s0, 1.4, 2.0))      # seated / crouching visits: keep the prior adult (the rig finds its own)
    res, fitted, dropped = fit_views(aligned, fps, init_xy=init if okt.any() else None, stature0=s0, iters=iters,
                                     max_err_box=max_err_box)
    for name, why in dropped.items():
        out["report"][name] = {**out["report"].get(name, {}), "role": "dropped", "reason": why}
    if res is None:
        out["reason"] = "rig fit failed or fewer than 2 consistent cameras"
        return out
    for v, pv in zip(fitted, res["per_view"]):
        out["report"][v["name"]]["rig_error"] = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in pv.items()}
    out.update({"ok": True, "res": res, "aligned": fitted, "stature0": s0})
    return out


# -- replay ---------------------------------------------------------------------------------------------------------
def replay_step(rig_fps: float, replay_fps: float = 8.0) -> int:
    """Every n-th rig frame becomes a replay frame (replay at rig_fps / n, close to replay_fps)."""
    return max(1, int(round(rig_fps / replay_fps)))


def build_replay(fz: dict, views: list[dict], fps: float, step: int, members: list[dict] | None = None,
                 window: tuple[float, float] | None = None) -> dict | None:
    """Fused result -> replay dict (trajectory.estimate on the primary's picks at fps / step, path from the rig,
    rig block, cameras_extra, fusion block). Assets (cut-outs, background) are built by the caller from the
    primary's collector (views[primary]["collector"]) with replay_assets.build(..., fps=fps / step)."""
    from . import gaze, trajectory
    res = fz["res"]
    byname = {v["name"]: v for v in views}
    P = byname[fz["primary"]]
    cam, size = P["cam"], P["size"]
    rfps = fps / step
    track = P["picks"][::step]
    n_tr = len(track)
    stature = float(res["stature_m"])
    ground = rigpipe.rig_ground(res, n_tr, rfps, fps)
    tr = trajectory.estimate(track, rfps, size, cam, heights_px=True, head_height=stature, ground_override=ground)
    if tr is None:
        return None
    try:
        gz = gaze.estimate(track, tr, cam, size, stature)
        if gz:
            tr["gaze"] = gz
    except Exception:
        log.exception("gaze estimate failed")
    cams = {v["name"]: ({"kp": v["kp"]}, v["cam"], 0) for v in fz["aligned"]}
    blk = rigpipe.attach_rig(tr, cams, fps, n=len(res["markers"]), res=res, views=fz["aligned"])
    used = [v["name"] for v in fz["aligned"]]
    tr["cameras_extra"] = [{"name": nm, "C": byname[nm]["cam"].C.tolist(), "R": byname[nm]["cam"].R.tolist(),
                            "K": byname[nm]["cam"].K.tolist(), "D": byname[nm]["cam"].D.tolist(),
                            "size": list(byname[nm]["cam"].size)} for nm in used if nm != fz["primary"]]
    tr["camera"]["name"] = fz["primary"]
    gaps = fz.get("gaps") or []
    tr["fusion"] = {"version": VERSION, "primary": fz["primary"], "cameras": used, "frame": frame_id(cam),
                    "rig_fps": round(fps, 3), "replay_fps": round(rfps, 3),
                    "cameras_report": fz["report"], "members": members or [],
                    "window": [round(window[0], 2), round(window[1], 2)] if window else None,
                    "joint_ray_gap_median_m": round(float(np.median(gaps)), 3) if gaps else None,
                    "stature_m": round(stature, 3), "frames_fitted": blk["frames_fitted"] if blk else 0}
    tr["multicam"] = {"primary": fz["primary"],
                      "offsets": {k: v for k, v in fz["report"].items() if v.get("role") == "secondary"},
                      "stature_m": round(stature, 3), "stature_measured": True,
                      "joint_ray_gap_median_m": tr["fusion"]["joint_ray_gap_median_m"]}
    tr["calibration"] = {"method": cam.meta.get("method"), "frame": frame_id(cam),
                         **{k: v for k, v in cam.meta.items() if k in ("date", "pitch_deg", "camera_height_m",
                                                                       "hfov_deg")}}
    tr["metrics"]["stature_measured_m"] = round(stature, 3)
    tr["metrics"]["head_height_assumed_m"] = None
    return tr


# -- live: which events to fuse, which window ----------------------------------------------------------------------
def overlaps(a0, a1, b0, b1, pad: float = 0.0) -> bool:
    return a0 < b1 + pad and b0 < a1 + pad


def plan(trigger: dict, events: list[dict], frame_cams: set, pad_s: float = 3.0, max_s: float = 20.0,
         tolerance_s: float = 1.0) -> dict:
    """Which events of the trigger's calibration frame overlap it, and the clip window to fuse.
    trigger/events: Frigate event dicts (id, camera, start_time, end_time, label). -> {"status": "fuse" | "wait" |
    "none", "members": [...], "window": (t0, t1), "core": (t0, t1)}. "wait" = an overlapping event is still in
    progress (its own end triggers the fusion later)."""
    t0, t1 = float(trigger["start_time"]), float(trigger["end_time"])
    mem = [trigger]
    cand = [e for e in events if e["id"] != trigger["id"] and e.get("camera") in frame_cams
            and (e.get("label") or "person") == "person" and not e.get("false_positive")]
    for e in cand:
        if e.get("end_time") is None and e["camera"] != trigger["camera"] and float(e["start_time"]) < t1 + tolerance_s:
            return {"status": "wait", "members": [], "waiting_for": e["id"]}
    cand = [e for e in cand if e.get("end_time") is not None]
    grown = True
    while grown:                 # transitive: an event overlapping any member joins (same camera too: one visit
        grown = False            # often splits into several Frigate events per camera)
        for e in list(cand):
            if any(overlaps(float(m["start_time"]), float(m["end_time"]), float(e["start_time"]), float(e["end_time"]),
                            tolerance_s) for m in mem):
                mem.append(e)
                cand.remove(e)
                grown = True
    cams = {e["camera"] for e in mem}
    if len(cams) < 2:
        return {"status": "none", "members": mem}
    per = {}
    for e in mem:
        a, b = per.get(e["camera"], (np.inf, -np.inf))
        per[e["camera"]] = (min(a, float(e["start_time"])), max(b, float(e["end_time"])))
    u0, u1 = min(v[0] for v in per.values()) - pad_s, max(v[1] for v in per.values()) + pad_s
    # core = where the trigger camera and at least one other camera saw the person together
    others = [v for c, v in per.items() if c != trigger["camera"]]
    c0 = min(max(t0, v[0]) for v in others)
    c1 = max(min(t1, v[1]) for v in others)
    if c1 < c0:
        c0 = c1 = (max(t0, min(v[0] for v in others)) + min(t1, max(v[1] for v in others))) / 2
    if u1 - u0 > max_s:
        mid = (c0 + c1) / 2
        w0 = min(max(u0, mid - max_s / 2), u1 - max_s)
        u0, u1 = w0, w0 + max_s
    return {"status": "fuse", "members": mem, "window": (u0, u1), "core": (c0, c1)}


def coverage(segments: list[dict], w0: float, w1: float, max_hole_s: float = 1.0) -> tuple[float, float] | None:
    """Frigate recording segments [{start_time, end_time}] -> the contiguous recorded range inside [w0, w1] that
    overlaps it most (a hole in the recordings would silently shift every later frame of a concatenated clip)."""
    segs = sorted((float(s["start_time"]), float(s["end_time"])) for s in segments
                  if float(s["end_time"]) > w0 and float(s["start_time"]) < w1)
    if not segs:
        return None
    runs, cur = [], list(segs[0])
    for a, b in segs[1:]:
        if a <= cur[1] + max_hole_s:
            cur[1] = max(cur[1], b)
        else:
            runs.append(tuple(cur))
            cur = [a, b]
    runs.append(tuple(cur))
    runs = [(max(a, w0), min(b, w1)) for a, b in runs]
    best = max(runs, key=lambda r: r[1] - r[0])
    return best if best[1] - best[0] > 0.5 else None


class LiveFusion:
    """Profiler side: after an event's own analysis, fuse it with overlapping events of other cameras in the same
    calibration frame. Frigate access is injected (get_json(path) -> obj, get_bytes(path) -> bytes | None) so the
    planning is testable without a server. The heavy work (extract + fit) runs in run()."""

    def __init__(self, get_json, get_bytes, calib_of, cameras: list[str], fps: float = 12.0, max_s: float = 20.0,
                 max_cameras: int = 3, pad_s: float = 3.0, max_offset_s: float = 4.0, max_gap_m: float = 0.15,
                 iters=(300, 900), replay_fps: float = 8.0, lookback_s: float = 900.0):
        self.get_json, self.get_bytes, self.calib_of = get_json, get_bytes, calib_of
        self.cameras = list(cameras)
        self.fps, self.max_s, self.max_cameras, self.pad_s = fps, max_s, max_cameras, pad_s
        self.max_offset_s, self.max_gap_m, self.iters, self.replay_fps = max_offset_s, max_gap_m, iters, replay_fps
        self.lookback_s = lookback_s
        self.done: dict = {}               # event id -> fusion id (bounded)

    def frame_cameras(self, cam: str) -> set:
        c = self.calib_of(cam)
        fid = frame_id(c) if c is not None else None
        if fid is None:
            return set()
        out = set()
        for name in self.cameras:
            cc = self.calib_of(name)
            if cc is not None and frame_id(cc) == fid:
                out.add(name)
        return out

    def candidates(self, a: dict) -> dict:
        """Plan for a finished event a (Frigate event dict)."""
        if a.get("end_time") is None or a["id"] in self.done:
            return {"status": "none", "members": []}
        cams = self.frame_cameras(a["camera"])
        if len(cams) < 2:
            return {"status": "none", "members": [], "why": "camera not in a shared calibration frame"}
        t0, t1 = float(a["start_time"]), float(a["end_time"])
        evs = self.get_json(f"/api/events?labels=person&cameras={','.join(sorted(cams))}&after={t0 - self.lookback_s:.1f}"
                            f"&before={t1 + self.pad_s:.1f}&limit=100") or []
        p = plan(a, evs, cams, pad_s=self.pad_s, max_s=self.max_s)
        if p["status"] == "fuse":
            if any(e["id"] in self.done for e in p["members"]):
                return {"status": "none", "members": p["members"], "why": "already fused"}
            by = {}
            for e in p["members"]:                     # at most max_cameras cameras: the trigger + longest others
                by.setdefault(e["camera"], []).append(e)
            keep = [a["camera"]] + sorted((c for c in by if c != a["camera"]),
                                          key=lambda c: -sum(float(e["end_time"]) - float(e["start_time"])
                                                             for e in by[c]))[:self.max_cameras - 1]
            p["members"] = [e for e in p["members"] if e["camera"] in keep]
        return p

    def mark(self, members: list[dict], fid: str) -> None:
        for e in members:
            self.done[e["id"]] = fid
        while len(self.done) > 2000:
            self.done.pop(next(iter(self.done)))

    def clips(self, p: dict) -> dict:
        """-> {camera: (clip bytes, start offset in the window s)} for the planned window."""
        w0, w1 = p["window"]
        out = {}
        for cam in sorted({e["camera"] for e in p["members"]}):
            segs = self.get_json(f"/api/{cam}/recordings?after={w0:.1f}&before={w1:.1f}") or []
            cov = coverage(segs, w0, w1)
            if cov is None:
                log.info("fusion: no recordings for %s in %.1f-%.1f", cam, w0, w1)
                continue
            data = self.get_bytes(f"/api/{cam}/start/{cov[0]:.2f}/end/{cov[1]:.2f}/clip.mp4")
            if data:
                out[cam] = (data, cov[0] - w0)
        return out

    def run(self, p: dict, pose, wb, make_collector=None) -> tuple[dict | None, dict]:
        """Planned fusion -> (fz result incl. views, info). make_collector(): a fresh replay_assets.Collector."""
        t_start = time.time()
        w0, w1 = p["window"]
        clips = self.clips(p)
        info = {"window": [w0, w1], "cameras_fetched": sorted(clips)}
        if len(clips) < 2:
            info["reason"] = "fewer than 2 cameras have recordings for the window"
            return None, info
        step = replay_step(self.fps, self.replay_fps)
        views = []
        for cam, (data, off) in clips.items():
            calib = self.calib_of(cam)
            v = extract_view(data, cam, calib, pose, wb, self.fps, w1 - w0, start_s=-off,
                             collector=make_collector() if make_collector else None, replay_step=step)
            views.append(v)
            log.info("fusion: %s %d grid frames, %d keypointed, person in %d (%.0fs)", cam, len(v["kp"]),
                     v["frames_keypointed"], v["frames_person"], v["seconds"])
        trig = p["members"][0]["camera"]
        fz = fuse_views(views, self.fps, max_offset_s=self.max_offset_s, max_gap_m=self.max_gap_m, iters=self.iters)
        fz["views"] = views
        fz["step"] = step
        info.update({"seconds": round(time.time() - t_start, 1), "trigger_camera": trig,
                     "keypoint_s": round(sum(v["seconds"] for v in views), 1)})
        return fz, info

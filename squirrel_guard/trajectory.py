"""Gaussian ground trajectory of one tracked person: pixels -> metres with honest uncertainty.

What it does
    Takes the per-frame keypoint track of one person (kinematics.track) and a calibrated camera (geometry.Camera)
    and produces a metric path on the ground plane, as a Gaussian at every analysed frame:
      measurement   foot pixel -> ground XY through Camera.ground; its 2x2 covariance is pixel noise pushed through
                    a numerical Jacobian of the ground projection (far points: large, stretched along the ray)
      motion        constant-velocity Kalman filter on [x, y, vx, vy] with white-noise acceleration, Mahalanobis
                    gating of outliers, then a Rauch-Tung-Striebel smoother; missing frames are predictions whose
                    covariance grows until the next observation pulls it back
      prediction    a short continuation (2.5 s) from a Gaussian process on time, per axis: mean function = the last
                    smoothed state extrapolated at constant velocity, kernel = Matern-3/2 (wobble) + a linear kernel
                    anchored at the last time (velocity uncertainty), fitted to the last seconds of the smoothed path
      footprints    alternating left/right steps placed along the smoothed path at the gait cadence, 0.1 m to either
                    side of the path; they are ESTIMATED, not detected ("estimated": true), the first side is a guess
      metrics       path length, speeds, stops, entry/exit heading, time per ground zone
      skeleton3d    a rough stick figure: every COCO joint on the vertical plane through the ground point, facing
                    the camera (display only)

Assumptions and limits
    Flat ground at Z = 0 and a calibrated camera (K, D, R, C). Any calibration error (pitch, height) becomes a
    systematic position error that the covariances do NOT contain: they only express pixel noise and motion model
    uncertainty. The foot point is the ankle midpoint at the bottom of the detection box; when the feet are cut off
    by the frame edge or hidden, the frame counts as unobserved. The 2.5 s prediction is a statistical continuation:
    people turn, and a turn is not predictable from the path alone, so the cone is wide by design. Footprint spacing
    follows speed x step period; actual foot placement is not measured. Pure numpy + cv2 (through Camera): runs in
    the runtime container without scipy.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import Camera
from .pose import L_ANK, R_ANK

VERSION = "1"

# -- tuning -------------------------------------------------------------------------------------------------------
PIX_SIGMA = 3.0          # px: keypoint/box noise at full pose confidence (1920x1080 frame)
PIX_SIGMA_V_SCALE = 1.5  # the box bottom (foot row) is noisier than the ankle column
ACCEL_SIGMA = 1.2        # m/s^2: white-noise acceleration of a walking person (turns, starts, stops)
GATE_CHI2 = 13.8         # chi^2(2) at 0.999: measurements beyond this Mahalanobis distance are outliers
INIT_V_SIGMA = 1.2       # m/s: velocity uncertainty before the second observation
FLOOR_SIGMA = 0.08       # m: floor on measurement std: the lowest foot swings ~0.3 m fore/aft of the body centre
SYSTEMATIC_SIGMA = 0.06  # m: added to every reported position covariance (not filtered away): the measured foot
                         # point is the leading sole, offset from the body centre in the walking direction
MAX_RANGE_M = 40.0       # ground points further than this are unusable (horizon)
STOP_SPEED = 0.2         # m/s
STOP_MIN_S = 2.0
DEFAULT_CADENCE = 110.0  # steps/min when gait gave none (flagged as assumed)
LATERAL_M = 0.10         # half the step width
PRED_HORIZON_S = 2.5
PRED_DT = 0.1
GP_WINDOW_S = 4.0
GP_LENGTH_S = 1.0        # Matern-3/2 lengthscale
GP_SIGMA_V = 0.35        # m/s per axis: how much the velocity may differ from the last smoothed one
GP_SIGMA_F_MIN = 0.08    # m: minimum wobble amplitude
GP_SIGMA_A = 0.7         # m/s^2: future-only acceleration noise (integrated Wiener process, variance sa^2 tau^3 / 3)
REACQUIRE_AFTER = 3      # accept a measurement after this many consecutive gated rejections (lost track)


# -- measurement -------------------------------------------------------------------------------------------------
def foot_point(p, size) -> tuple[tuple[float, float], float] | None:
    """(foot pixel, confidence) for one Pose: ankle-midpoint column, detection-box bottom row.
    None when the feet are at/over the bottom edge (cut off) or the person is too small."""
    W, H = size
    x1, y1, x2, y2 = p.box
    if y2 >= H - 0.01 * H or (y2 - y1) < 0.04 * H:
        return None
    kp = p.kp
    ca, cb = float(kp[L_ANK, 2]), float(kp[R_ANK, 2])
    if ca >= 0.3 and cb >= 0.3:
        fx, conf = float(kp[[L_ANK, R_ANK], 0].mean()), min(ca, cb)
    elif max(ca, cb) >= 0.3:
        fx, conf = float(kp[L_ANK if ca > cb else R_ANK, 0]), 0.6 * max(ca, cb)
    else:
        fx, conf = (x1 + x2) / 2, 0.25
    if not (0 <= fx < W):
        return None
    return (fx, float(y2)), float(min(1.0, max(0.1, conf)))


def head_point(p, size) -> tuple[tuple[float, float], float] | None:
    """(top-of-head pixel, confidence) for one Pose: face-keypoint column, detection-box top row. None when the head
    is cut off by the top edge or the person is too small. Used with estimate(head_height=...) when the feet are
    out of frame (a camera below head level, e.g. on a desk looking up)."""
    W, H = size
    x1, y1, x2, y2 = p.box
    if y1 <= 0.01 * H or (y2 - y1) < 0.04 * H:
        return None
    face = p.kp[:5]
    ok = face[:, 2] >= 0.4
    hx = float(face[ok, 0].mean()) if ok.any() else (x1 + x2) / 2
    conf = float(face[ok, 2].mean()) if ok.any() else 0.25
    return (hx, float(y1)), float(min(1.0, max(0.1, conf)))


def measure(cam: Camera, px, size, conf: float = 1.0, h_px: float = 1.0, z: float = 0.0):
    """One foot pixel (or head-top pixel with z = stature) -> (ground xy, 2x2 covariance) or None. The covariance is
    J diag(s_u^2, s_v^2) J^T with J the central-difference Jacobian of Camera.plane, s = PIX_SIGMA scaled by
    (1 + 2 (1 - conf))."""
    u, v = float(px[0]), float(px[1])
    pts = np.array([[u, v], [u + h_px, v], [u - h_px, v], [u, v + h_px], [u, v - h_px]])
    g = cam.plane(pts, size, z)
    if not np.isfinite(g).all():
        return None
    J = np.c_[(g[1] - g[2]) / (2 * h_px), (g[3] - g[4]) / (2 * h_px)]
    s = PIX_SIGMA * (1.0 + 2.0 * (1.0 - conf))
    cov = J @ np.diag([s * s, (PIX_SIGMA_V_SCALE * s) ** 2]) @ J.T + np.eye(2) * FLOOR_SIGMA ** 2
    if np.linalg.norm(g[0] - cam.C[:2]) > MAX_RANGE_M or np.trace(cov) > 100.0:
        return None
    return g[0], cov


# -- Kalman filter + RTS smoother --------------------------------------------------------------------------------
def _models(dt: float):
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    q = ACCEL_SIGMA ** 2
    Q1 = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1
    H = np.zeros((2, 4))
    H[0, 0] = H[1, 1] = 1.0
    return F, Q, H


def smooth(meas: list, dt: float, speed_prior: float | None = None):
    """meas: per tick (xy, cov) or None, starting and ending with an observation.
    Returns smoothed means (N, 4), covariances (N, 4, 4) and the per-tick 'used' flags (gated outliers -> False).
    speed_prior (m/s, from gait): only shrinks the initial velocity spread (a pseudo-measurement of |v| was tried
    and dropped: it drags a person who has STOPPED during an occlusion). Measurements beyond the gate are rejected
    unless REACQUIRE_AFTER in a row were, in which case the filter re-acquires the track."""
    n = len(meas)
    F, Q, H = _models(dt)
    xf, Pf = np.zeros((n, 4)), np.zeros((n, 4, 4))     # filtered
    xp, Pp = np.zeros((n, 4)), np.zeros((n, 4, 4))     # predicted (prior)
    used = np.zeros(n, bool)
    z0, R0 = meas[0]
    x = np.r_[z0, 0.0, 0.0]
    P = np.zeros((4, 4))
    P[:2, :2] = R0
    P[2, 2] = P[3, 3] = (min(INIT_V_SIGMA, max(0.4, 0.5 * speed_prior)) if speed_prior else INIT_V_SIGMA) ** 2
    rejected = 0
    for k in range(n):
        if k > 0:
            x, P = F @ x, F @ P @ F.T + Q
        xp[k], Pp[k] = x, P
        m = meas[k]
        if m is not None:
            z, R = m
            y = z - H @ x
            S = H @ P @ H.T + R
            d2 = float(y @ np.linalg.solve(S, y))
            if d2 <= GATE_CHI2 or k < 2 or rejected >= REACQUIRE_AFTER:
                if rejected >= REACQUIRE_AFTER:               # lost: restart around this measurement
                    x = np.r_[z, x[2], x[3]]
                    P = np.zeros((4, 4)); P[:2, :2] = R; P[2, 2] = P[3, 3] = INIT_V_SIGMA ** 2
                    y, S = z - H @ x, H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ y
                P = (np.eye(4) - K @ H) @ P @ (np.eye(4) - K @ H).T + K @ R @ K.T
                used[k] = True
                rejected = 0
            else:
                rejected += 1
        xf[k], Pf[k] = x, P
    xs, Ps = xf.copy(), Pf.copy()
    for k in range(n - 2, -1, -1):          # Rauch-Tung-Striebel
        G = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + G @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + G @ (Ps[k + 1] - Pp[k + 1]) @ G.T
        Ps[k] = 0.5 * (Ps[k] + Ps[k].T)
    return xs, Ps, used


# -- Gaussian-process continuation -------------------------------------------------------------------------------
def _matern32(ta, tb, ell):
    r = np.abs(ta[:, None] - tb[None, :]) / ell
    s3 = math.sqrt(3.0)
    return (1 + s3 * r) * np.exp(-s3 * r)


def gp_predict(t, xs, Ps, horizon=PRED_HORIZON_S, step=PRED_DT):
    """Continue the smoothed path past its last sample. Per axis: y(t) = m(t) + f(t), m = constant-velocity
    extrapolation of the last smoothed state, f ~ GP(0, sf^2 Matern32(|dt|/GP_LENGTH_S) + GP_SIGMA_V^2 (t-T)(t'-T)),
    observed through the last GP_WINDOW_S of smoothed positions with their smoothed variances as noise.
    The linear kernel lets the data correct the velocity; the Matern term adds the short-range wobble (sf = rms
    residual of the window, >= GP_SIGMA_F_MIN). Both are learnt from the past, so on their own they would predict
    the future far too confidently: an integrated-Wiener kernel k(tau, tau') = sa^2 (m^3/3 + |tau - tau'| m^2/2),
    m = min(tau, tau'), tau = lead time (zero for the past), adds the acceleration the data cannot know about.
    Returns (t_pred, mean (M, 2), cov (M, 2, 2)); x and y are modelled independently (diagonal covariance)."""
    T = t[-1]
    sel = t >= T - GP_WINDOW_S
    tt, X, V = t[sel], xs[sel], Ps[sel]
    tp = T + np.arange(step, horizon + 1e-9, step)
    mean, cov = np.zeros((len(tp), 2)), np.zeros((len(tp), 2, 2))
    for a in range(2):
        m_tr = xs[-1, a] + xs[-1, 2 + a] * (tt - T)
        m_te = xs[-1, a] + xs[-1, 2 + a] * (tp - T)
        r = X[:, a] - m_tr
        sf = max(GP_SIGMA_F_MIN, float(np.sqrt(np.mean(r * r))))
        noise = V[:, a, a] + 1e-4
        Kxx = sf ** 2 * _matern32(tt, tt, GP_LENGTH_S) + GP_SIGMA_V ** 2 * np.outer(tt - T, tt - T) + np.diag(noise)
        Kpx = sf ** 2 * _matern32(tp, tt, GP_LENGTH_S) + GP_SIGMA_V ** 2 * np.outer(tp - T, tt - T)
        Kpp = sf ** 2 * _matern32(tp, tp, GP_LENGTH_S) + GP_SIGMA_V ** 2 * np.outer(tp - T, tp - T)
        L = np.linalg.cholesky(Kxx)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, r))
        Wm = np.linalg.solve(L, Kpx.T)
        mean[:, a] = m_te + Kpx @ alpha
        tau = tp - T
        cov[:, a, a] = np.diag(Kpp) - np.sum(Wm * Wm, 0) + Ps[-1, a, a] + GP_SIGMA_A ** 2 * tau ** 3 / 3
    return tp, mean, cov


# -- footprints and metrics --------------------------------------------------------------------------------------
def footprints(t, xs, Ps, cadence_spm: float):
    """Alternating steps along the path at the cadence, LATERAL_M off the path; the step clock pauses while the
    person is slower than STOP_SPEED. Uncertainty = path std at that time plus 5 cm of placement noise."""
    period = 60.0 / cadence_spm
    out, acc, side = [], period, 0
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        sp = math.hypot(xs[k, 2], xs[k, 3])
        if sp < STOP_SPEED:
            continue
        acc += dt
        if acc < period:
            continue
        acc -= period
        hx, hy = xs[k, 2] / sp, xs[k, 3] / sp
        lx, ly = -hy, hx                           # left of the heading
        sgn = 1.0 if side == 0 else -1.0
        out.append({"t": round(float(t[k]), 3), "x": round(float(xs[k, 0] + sgn * LATERAL_M * lx), 3),
                    "y": round(float(xs[k, 1] + sgn * LATERAL_M * ly), 3), "side": "L" if side == 0 else "R",
                    "sx": round(float(math.sqrt(Ps[k, 0, 0] + 0.05 ** 2)), 3),
                    "sy": round(float(math.sqrt(Ps[k, 1, 1] + 0.05 ** 2)), 3), "estimated": True})
        side ^= 1
    return out


def _point_in_poly(x, y, poly) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def metrics(t, xs, zones=None) -> dict:
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    speed = np.hypot(xs[:, 2], xs[:, 3])
    seg = np.linalg.norm(np.diff(xs[:, :2], axis=0), axis=1)
    stops, k = [], 0
    while k < len(t):
        if speed[k] < STOP_SPEED:
            j = k
            while j + 1 < len(t) and speed[j + 1] < STOP_SPEED:
                j += 1
            if t[j] - t[k] + dt >= STOP_MIN_S:
                stops.append({"t0": round(float(t[k]), 2), "t1": round(float(t[j]), 2),
                              "x": round(float(xs[k:j + 1, 0].mean()), 2), "y": round(float(xs[k:j + 1, 1].mean()), 2)})
            k = j + 1
        else:
            k += 1

    def heading(sl):
        v = xs[sl, 2:4].mean(0)
        return round(math.degrees(math.atan2(v[1], v[0])), 1) if np.linalg.norm(v) > 0.05 else None

    w = max(1, int(round(0.5 / dt))) if dt else 1
    m = {"duration_s": round(float(t[-1] - t[0]), 2), "path_length_m": round(float(seg.sum()), 2),
         "speed_mean_mps": round(float(speed.mean()), 2), "speed_max_mps": round(float(speed.max()), 2),
         "stops": stops, "heading_entry_deg": heading(slice(0, w)), "heading_exit_deg": heading(slice(len(t) - w, len(t))),
         "heading_note": "degrees, 0 = world +X, 90 = world +Y (counter-clockwise seen from above)"}
    if zones:
        zt = {}
        for z in zones:
            inside = sum(1 for p in xs[:, :2] if _point_in_poly(p[0], p[1], z["polygon"]))
            zt[z.get("name", "zone")] = round(inside * dt, 2)
        m["zone_seconds"] = zt
    return m


# -- public API --------------------------------------------------------------------------------------------------
def estimate(track: list, fps: float, size, cam: Camera, gait: dict | None = None, heights_px: bool = False,
             head_height: float | None = None,
             zones: list | None = None, ground_override: list | None = None) -> dict | None:
    """Per-frame Pose|None track at fps in frames of `size` (W, H) -> JSON-serialisable trajectory, or None when
    fewer than 3 frames give a usable ground point. heights_px: also report the head height per frame (cam.heights
    of box-bottom/box-top; NaN -> null). zones: [{"name", "polygon": [[x, y], ...]}] in ground metres.
    ground_override: per frame (xy, 2x2 cov) or None, from another source (e.g. multi-camera triangulation); where
    it is not None it replaces this camera's foot/head measurement for that frame."""
    n = len(track)
    if n < 3 or fps <= 0:
        return None
    dt = 1.0 / fps
    meas: list = [None] * n
    heads: list = [None] * n
    via_head = [False] * n
    for i, p in enumerate(track):
        if ground_override is not None and i < len(ground_override) and ground_override[i] is not None:
            meas[i] = ground_override[i]
            continue
        if p is None:
            continue
        fp = foot_point(p, size)
        if fp is None:
            # feet out of frame: place the person by the top of the head on the plane Z = head_height
            hp = head_point(p, size) if head_height else None
            m = measure(cam, hp[0], size, 0.7 * hp[1], z=head_height) if hp else None
            if m is not None:
                meas[i], via_head[i] = m, True
            continue
        m = measure(cam, fp[0], size, fp[1])
        if m is None:
            continue
        meas[i] = m
        if heights_px:
            z = cam.heights([fp[0]], [((p.box[0] + p.box[2]) / 2, p.box[1])], size)[0]
            heads[i] = round(float(z), 3) if np.isfinite(z) and 0.3 < z < 2.6 else None
    idx = [i for i, m in enumerate(meas) if m is not None]
    if len(idx) < 3:
        return None
    i0, i1 = idx[0], idx[-1]
    speed_prior = None
    if gait and gait.get("usable") and gait.get("speed_mps"):
        speed_prior = float(gait["speed_mps"])
    xs, Ps, used = smooth(meas[i0:i1 + 1], dt, speed_prior)
    t = np.arange(i0, i1 + 1) * dt
    Ps[:, 0, 0] += SYSTEMATIC_SIGMA ** 2        # reported uncertainty includes the foot-vs-centre offset
    Ps[:, 1, 1] += SYSTEMATIC_SIGMA ** 2
    speed = np.hypot(xs[:, 2], xs[:, 3])
    tp, pm, pc = gp_predict(t, xs, Ps)
    cad, assumed = DEFAULT_CADENCE, True
    if gait and gait.get("walking") and gait.get("cadence_spm"):
        cad, assumed = float(gait["cadence_spm"]), False
    fp_list = footprints(t, xs, Ps, cad)
    met = metrics(t, xs, zones)
    met["cadence_spm"] = round(cad, 1)
    met["cadence_assumed"] = assumed
    met["frames_observed"] = int(used.sum())
    met["frames_head_anchored"] = int(sum(via_head[i] for i in range(i0, i1 + 1)))
    if head_height:
        met["head_height_assumed_m"] = head_height
    met["frames_total"] = int(len(t))
    met["frames_rejected"] = int(sum(1 for i in range(i0, i1 + 1) if meas[i] is not None) - used.sum())
    out = {"version": VERSION, "fps": fps, "frame0": int(i0),
           "t": [round(float(v), 3) for v in t],
           "mean": np.round(xs[:, :2], 3).tolist(),
           "cov": np.round(Ps[:, :2, :2], 5).tolist(),
           "velocity": np.round(xs[:, 2:], 3).tolist(),
           "observed": used.tolist(),
           "speed": np.round(speed, 3).tolist(),
           "speed_std": np.round(np.sqrt(np.maximum(Ps[:, 2, 2] + Ps[:, 3, 3], 0)), 3).tolist(),
           "footprints": fp_list,
           "prediction": {"t": [round(float(v), 3) for v in tp], "mean": np.round(pm, 3).tolist(),
                          "cov": np.round(pc, 5).tolist(),
                          "note": "GP continuation; x/y independent; a turn is not predictable from the path"},
           "metrics": met,
           "camera": {"C": cam.C.tolist(), "R": cam.R.tolist(), "K": cam.K.tolist(), "D": cam.D.tolist(),
                      "size": list(cam.size)}}
    if heights_px:
        out["head_z"] = heads[i0:i1 + 1]
    return out


def skeleton3d(track: list, cam: Camera, size, head_height: float | None = None) -> list:
    """Per frame: 17 joints as [x, y, z] metres or null, or null for frames without a usable foot point.
    APPROXIMATE: each joint is where its pixel ray meets the vertical plane through the person's ground point
    whose normal points at the camera (the person is treated as a flat cut-out facing the lens). Joints with
    confidence < 0.3 are null; z is clamped to >= 0."""
    out = []
    for p in track:
        fp = foot_point(p, size) if p is not None else None
        hp = head_point(p, size) if (p is not None and fp is None and head_height) else None
        if fp is None and hp is None:
            out.append(None)
            continue
        g = cam.ground([fp[0]], size)[0] if fp else cam.plane([hp[0]], size, head_height)[0]
        if not np.isfinite(g).all():
            out.append(None)
            continue
        G = np.array([g[0], g[1], 0.0])
        nrm = np.array([cam.C[0] - g[0], cam.C[1] - g[1], 0.0])
        nn = np.linalg.norm(nrm)
        if nn < 1e-6:
            out.append(None)
            continue
        nrm /= nn
        d = cam.rays(p.kp[:, :2], size)
        joints = []
        for j in range(17):
            den = float(nrm @ d[j])
            if p.kp[j, 2] < 0.3 or abs(den) < 1e-6:
                joints.append(None)
                continue
            s = float(nrm @ (G - cam.C)) / den
            if s <= 0:
                joints.append(None)
                continue
            P = cam.C + s * d[j]
            joints.append([round(float(P[0]), 3), round(float(P[1]), 3), round(max(0.0, float(P[2])), 3)])
        out.append(joints)
    return out

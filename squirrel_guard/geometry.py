"""Single-camera metrology: pixel -> ground position (metres) and person height, from a calibrated camera.

World frame: Z up, ground plane Z = 0, camera centre at C = (0, 0, camera_height) unless points were measured.
A calibration file (JSON) holds:
  image_size [W, H]           resolution the calibration was made at (other resolutions are scaled)
  K, D                        intrinsics + distortion (OpenCV model) from a checkerboard, or a fitted pinhole
  R                           3x3, world_from_camera rotation (OpenCV camera axes: x right, y down, z forward)
  C                           camera centre in world metres
Height = where the vertical line standing on the foot's ground point comes closest to the head pixel's ray.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

# camera looking horizontally along world +Y: cam x -> world X, cam y (down) -> world -Z, cam z -> world +Y
R0 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)


def rot_from_angles(pitch_deg: float, roll_deg: float) -> np.ndarray:
    """world_from_camera for a camera tilted DOWN by pitch and rolled by roll (yaw is irrelevant for height)."""
    a, r = math.radians(pitch_deg), math.radians(roll_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(a), math.sin(a)], [0, -math.sin(a), math.cos(a)]])  # forward -> down
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return R0 @ rx @ rz


def nominal_K(w: int, h: int, hfov_deg: float = 100.0) -> np.ndarray:
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)


class Camera:
    def __init__(self, K, D, R, C, image_size, meta: dict | None = None):
        self.K, self.D = np.asarray(K, float), np.asarray(D if D is not None else [0, 0, 0, 0, 0], float).reshape(-1)
        self.R, self.C = np.asarray(R, float), np.asarray(C, float)
        self.size = tuple(int(v) for v in image_size)
        self.meta = meta or {}

    @classmethod
    def load(cls, path: str | Path) -> "Camera | None":
        p = Path(path)
        if not p.is_file():
            return None
        d = json.loads(p.read_text())
        if not all(k in d for k in ("K", "R", "C", "image_size")):
            return None
        return cls(d["K"], d.get("D"), d["R"], d["C"], d["image_size"], d)

    def save(self, path: str | Path, **extra) -> None:
        d = dict(self.meta)
        d.update(extra)
        d.update({"image_size": list(self.size), "K": self.K.tolist(), "D": self.D.tolist(),
                  "R": self.R.tolist(), "C": self.C.tolist()})
        Path(path).write_text(json.dumps(d, indent=1))

    # -- rays ------------------------------------------------------------------------------------------------
    def rays(self, pts_px, frame_size) -> np.ndarray:
        """(N, 2) pixels in a frame of frame_size (W, H) -> (N, 3) unit ray directions in world coordinates."""
        pts = np.asarray(pts_px, float).reshape(-1, 2).copy()
        pts[:, 0] *= self.size[0] / frame_size[0]
        pts[:, 1] *= self.size[1] / frame_size[1]
        n = cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)
        d = np.c_[n, np.ones(len(n))] @ self.R.T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def ground(self, pts_px, frame_size) -> np.ndarray:
        """Pixels -> (N, 2) ground positions in metres; NaN for rays at/above the horizon."""
        return self.plane(pts_px, frame_size, 0.0)

    def plane(self, pts_px, frame_size, z: float) -> np.ndarray:
        """Pixels -> (N, 2) xy where each ray meets the horizontal plane Z = z (metres); NaN when it never does.
        z = 0 is the ground; z = a person's height places them by the top of the head (camera below head level)."""
        d = self.rays(pts_px, frame_size)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z - self.C[2]) / d[:, 2]
        g = self.C[None, :2] + s[:, None] * d[:, :2]
        g[~(s > 0)] = np.nan
        return g

    def heights(self, foot_px, head_px, frame_size) -> np.ndarray:
        """Foot (on the ground) + head-top pixels -> heights in metres (NaN when the foot is above the horizon)."""
        g = self.ground(foot_px, frame_size)
        r = self.rays(head_px, frame_size)
        out = np.full(len(g), np.nan)
        ez = np.array([0., 0., 1.])
        for i, (gi, ri) in enumerate(zip(g, r)):
            if np.isnan(gi).any():
                continue
            G = np.array([gi[0], gi[1], 0.])
            # closest points of the head ray C + s*r and the vertical G + z*ez: [r, -ez] [s, z]^T ~ G - C
            A = np.c_[ri, -ez]
            (s, z), *_ = np.linalg.lstsq(A, G - self.C, rcond=None)
            if s > 0:
                out[i] = z
        return out


# -- fitting ----------------------------------------------------------------------------------------------------
def nelder_mead(f, x0, step, iters=400, tol=1e-9):
    n = len(x0)
    pts = [np.asarray(x0, float)] + [np.asarray(x0, float) + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        cen = np.mean(pts[:-1], 0)
        xr = cen + (cen - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = cen + 2 * (cen - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def fit_walk(obs: list[tuple], frame_size, height_m: float, K=None, D=None, k_size=None,
             fit_focal: bool | None = None) -> tuple[Camera, dict]:
    """Calibrate from a person of KNOWN height walking around the scene.

    obs: [(foot_px, head_px), ...] from many positions (near/far, left/right) in frames of frame_size. Unknowns:
    camera pitch and roll (and focal length when no checkerboard intrinsics are given); camera height follows in
    closed form because every height scales linearly with it. k_size = the resolution K was calibrated at.
    """
    W, H = frame_size
    foot = np.array([o[0] for o in obs], float)
    head = np.array([o[1] for o in obs], float)
    have_k = K is not None
    K0 = np.asarray(K, float) if have_k else nominal_K(W, H)
    D0 = np.asarray(D, float) if D is not None else np.zeros(5)
    size = tuple(k_size) if (have_k and k_size) else (W, H)
    fit_focal = (not have_k) if fit_focal is None else fit_focal

    def build(p, cam_h=1.0):
        Kp, Dp = K0.copy(), D0.copy()
        if fit_focal:  # no checkerboard: also fit focal length and the main radial distortion term
            Kp[0, 0] = Kp[1, 1] = K0[0, 0] * math.exp(p[2])
            Dp[0] = p[3]
        return Camera(Kp, Dp, rot_from_angles(p[0], p[1]), [0, 0, cam_h], size)

    def cost(p):
        if not (0 < p[0] < 89) or abs(p[1]) > 20 or (fit_focal and abs(p[3]) > 0.6):
            return 1e6
        z = build(p).heights(foot, head, (W, H))
        ok = np.isfinite(z) & (z > 0)
        if ok.sum() < max(5, 0.6 * len(z)):
            return 1e6
        z = z[ok]
        r = z / np.median(z) - 1.0                     # relative spread: scale (camera height) drops out
        return float(np.mean(np.minimum(r * r, 0.04)))  # robust: one bad keypoint can't dominate

    best = None
    for p0 in (10, 20, 30, 45, 60):
        x0 = [p0, 0.0] + ([0.0, -0.2] if fit_focal else [])
        x, v = nelder_mead(cost, x0, [5, 2] + ([0.2, 0.1] if fit_focal else []), iters=1500 if fit_focal else 400)
        if best is None or v < best[1]:
            best = (x, v)
    x, _ = best
    z1 = build(x).heights(foot, head, (W, H))
    ok = np.isfinite(z1) & (z1 > 0)
    cam_h = height_m / float(np.median(z1[ok]))
    cam = build(x, cam_h)
    z = cam.heights(foot, head, (W, H))[ok]
    stats = {"method": "walk", "pitch_deg": round(float(x[0]), 2), "roll_deg": round(float(x[1]), 2),
             "camera_height_m": round(cam_h, 3), "focal_px": round(float(cam.K[0, 0]), 1), "focal_fitted": bool(fit_focal), "k1": round(float(cam.D[0]), 4),
             "intrinsics": "checkerboard" if have_k else "nominal", "observations": int(ok.sum()),
             "reference_height_m": height_m,
             "residual_cm_median_abs": round(float(np.median(np.abs(z - height_m))) * 100, 1),
             "residual_cm_p90_abs": round(float(np.percentile(np.abs(z - height_m), 90)) * 100, 1)}
    return cam, stats


def fit_points(points: list[dict], frame_size, K, D=None, image_size=None, refine_lens: bool = False) -> tuple[Camera, dict]:
    """Calibrate from tape-measured points: [{"px": [u, v], "world": [x, y, z]}, ...] (>= 4; ground points z = 0,
    tops of verticals z = their height). Needs intrinsics (checkerboard, or nominal_K as a rough fallback)."""
    obj = np.array([p["world"] for p in points], float)
    img = np.array([p["px"] for p in points], float)
    W, H = frame_size
    size = tuple(image_size or (W, H))
    img[:, 0] *= size[0] / W
    img[:, 1] *= size[1] / H
    K, D = np.asarray(K, float), np.asarray(D if D is not None else np.zeros(5), float)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE if len(obj) >= 6 else cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise ValueError("solvePnP failed")
    Rcw, _ = cv2.Rodrigues(rvec)
    R, C = Rcw.T, (-Rcw.T @ tvec).reshape(3)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
    st = {"method": "points", "points": len(obj), "reproj_px_mean": round(float(err.mean()), 2),
          "reproj_px_max": round(float(err.max()), 2), "camera_height_m": round(float(C[2]), 3)}
    if refine_lens and len(obj) >= 8:
        # Nominal intrinsics (no checkerboard): also fit focal length and radial distortion k1, k2 from the same
        # points, principal point fixed at the centre. Kept only if clearly better and the focal is plausible.
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
                 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
        try:
            rms, K2, D2, rv, tv = cv2.calibrateCamera([obj.astype(np.float32)], [img.astype(np.float32)], size,
                                                      K.copy(), np.r_[D.reshape(-1)[:5], np.zeros(max(0, 5 - D.size))].astype(np.float64),
                                                      flags=flags)
            hfov = 2 * np.degrees(np.arctan(size[0] / 2 / K2[0, 0]))
            p2, _ = cv2.projectPoints(obj, rv[0], tv[0], K2, D2)
            err2 = np.linalg.norm(p2.reshape(-1, 2) - img, axis=1)
            st["lens_refit"] = {"hfov_deg": round(float(hfov), 1), "k1": round(float(D2.reshape(-1)[0]), 3),
                                "reproj_px_mean": round(float(err2.mean()), 2), "kept": False}
            if 50 <= hfov <= 140 and err2.mean() < 0.7 * err.mean():
                Rcw2, _ = cv2.Rodrigues(rv[0])
                K, D, R, C, err = K2, D2.reshape(-1), Rcw2.T, (-Rcw2.T @ tv[0]).reshape(3), err2
                st.update({"reproj_px_mean": round(float(err.mean()), 2), "reproj_px_max": round(float(err.max()), 2),
                           "camera_height_m": round(float(C[2]), 3)})
                st["lens_refit"]["kept"] = True
        except cv2.error as e:
            st["lens_refit"] = {"error": str(e)[:120]}
    cam = Camera(K, D, R, C, size)
    return cam, st

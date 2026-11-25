"""Lucky-imaging face composite for HUMAN VIEWING (tier A of docs/faceseq.md): no learned model, nothing invented.

  1. pick the best frames of a face sequence (faceseq.py): AdaFace norm x sharpness x frontal pose, and only frames
     whose pose is close to the reference frame's (stacking different poses blurs);
  2. align each to the reference: similarity transform from the 5 landmarks, refined to sub-pixel accuracy with
     OpenCV ECC (enhanced correlation coefficient) on an upsampled grey crop;
  3. reject outlier pixels per frame (blink, motion blur, compression blocks): |x - median| > k * robust sigma;
  4. drizzle (STScI `drizzle`, BSD-3) every frame's real pixels onto a finer output grid (shift-and-add with a
     shrunken pixel footprint), weighted by frame quality;
  5. per-pixel uncertainty = weighted spread of the aligned frames (where they disagree, the composite is a guess).
Every output pixel is a weighted average of camera pixels. It is still ENHANCED imagery: label it, keep the
original beside it, and never feed it to recognition (recognition uses faceseq_agg on camera pixels).
"""
from __future__ import annotations

import cv2
import numpy as np
from drizzle.resample import Drizzle

LABEL = "ENHANCED (lucky-imaging stack) - not original pixels"
# Failed-stack test (provisional, 4 self-test points 2026-11-10: good stacks face-uncertainty 8.7-9.5 and
# consistency 20-23 dB; the failed 15 px stack 16.9 and 17.8 dB). Suspect -> show the single reference frame.
SUSPECT_UNC = 13.0
SUSPECT_DB = 19.0


def _sim(src_pts, dst_pts):
    M, _ = cv2.estimateAffinePartial2D(np.asarray(src_pts, np.float32), np.asarray(dst_pts, np.float32),
                                       method=cv2.LMEDS)
    return M


def _to3(M):
    return np.vstack([M, [0, 0, 1]])


def select(obs, norms, k: int = 12, yaw_tol: float = 0.18):
    sharp = np.array([o.sharpness for o in obs], float)
    yaw = np.array([o.yaw_proxy for o in obs], float)
    q = np.clip(np.asarray(norms, float) - 14, 0, None) ** 2 * (sharp / (np.median(sharp) + 1e-6)) ** 0.5 \
        * np.clip(1 - np.abs(yaw) / 0.6, 0.05, 1)
    ref = int(np.argmax(q))
    cand = [i for i in np.argsort(-q) if abs(yaw[i] - yaw[ref]) <= yaw_tol and q[i] > 0][:k]
    return ref, cand, q


def composite(frames: list[np.ndarray], obs, norms, scale: int = 2, k: int = 12, clip_k: float = 2.5,
              pixfrac: float = 0.7) -> dict:
    ref, idx, q = select(obs, norms, k)
    r = obs[ref]
    fr_ref = frames[r.frame]
    c = r.pts5.mean(0)
    side = int(max(24, 1.7 * r.face_px))
    X0, Y0 = c[0] - side / 2, c[1] - side / 2                     # canvas origin in ref-frame pixels
    out = side * scale
    to_canvas = np.array([[scale, 0, -X0 * scale], [0, scale, -Y0 * scale], [0, 0, 1]], np.float64)

    # align: frame_i -> ref frame (landmarks), then ECC refinement on the canvas
    ref_canvas = cv2.warpAffine(fr_ref, to_canvas[:2], (out, out), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REFLECT)
    ref_g = cv2.cvtColor(ref_canvas, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mats, warped, used = [], [], []
    for i in idx:
        o = obs[i]
        M = _to3(_sim(o.pts5, r.pts5)) if i != ref else np.eye(3)
        A = to_canvas @ M                                          # frame_i pixels -> canvas pixels
        w = cv2.warpAffine(frames[o.frame], A[:2], (out, out), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        if i != ref:
            g = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY).astype(np.float32)
            E = np.eye(2, 3, dtype=np.float32)
            try:
                _, E = cv2.findTransformECC(ref_g, g, E, cv2.MOTION_AFFINE,
                                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 5)
                A = np.linalg.inv(_to3(E.astype(np.float64))) @ A    # E maps ref canvas -> this canvas
                w = cv2.warpAffine(frames[o.frame], A[:2], (out, out), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REFLECT)
            except cv2.error:
                continue                                           # ECC did not converge: leave the frame out
        mats.append(A)
        warped.append(w.astype(np.float32))
        used.append(i)
    W = np.stack(warped)                                           # (n, out, out, 3)
    wq = np.array([q[i] for i in used], np.float64)
    wq = wq / wq.max()

    med = np.median(W, axis=0)
    mad = np.median(np.abs(W - med), axis=0) * 1.4826 + 2.0
    inlier = (np.abs(W - med) <= clip_k * mad).all(axis=-1)        # (n, out, out)
    wts = inlier * wq[:, None, None]
    mean = (W * wts[..., None]).sum(0) / np.maximum(wts.sum(0), 1e-6)[..., None]
    spread = np.sqrt((wts[..., None] * (W - mean) ** 2).sum(0) / np.maximum(wts.sum(0), 1e-6)[..., None]).mean(-1)

    # drizzle the real pixels: input = each frame's native region, pixmap = its pixels' canvas coordinates
    drz = [Drizzle(out_shape=(out, out), kernel="square", fillval=np.nan) for _ in range(3)]
    for n, (i, A) in enumerate(zip(used, mats)):
        o = obs[i]
        Ainv = np.linalg.inv(A)
        corners = (Ainv @ np.array([[0, out, 0, out], [0, 0, out, out], [1, 1, 1, 1]], float))[:2]
        x0, y0 = np.floor(corners.min(1)).astype(int) - 2
        x1, y1 = np.ceil(corners.max(1)).astype(int) + 2
        H, Wd = frames[o.frame].shape[:2]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(Wd, x1), min(H, y1)
        patch = frames[o.frame][y0:y1, x0:x1].astype(np.float32)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        P = A @ np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
        pixmap = np.dstack([P[0].reshape(xx.shape), P[1].reshape(xx.shape)]).astype(np.float64)
        # per-input-pixel weight: frame quality x inlier test against the median (sampled at the pixel's position)
        u = np.clip(np.round(P[0]).astype(int), 0, out - 1).reshape(xx.shape)
        v = np.clip(np.round(P[1]).astype(int), 0, out - 1).reshape(xx.shape)
        ok = (np.abs(patch - med[v, u]) <= clip_k * mad[v, u]).all(-1)
        wmap = (ok * wq[n]).astype(np.float32)
        for ch in range(3):
            drz[ch].add_image(np.ascontiguousarray(patch[..., ch]), exptime=1.0, pixmap=pixmap,
                              weight_map=wmap, pixfrac=pixfrac, in_units="cps")
    img = np.dstack([d.out_img for d in drz])
    hole = ~np.isfinite(img).all(-1) | (np.dstack([d.out_wht for d in drz]).min(-1) <= 0)
    img[hole] = mean[hole]                                          # uncovered output pixels: interpolated stack
    ref_bic = cv2.resize(fr_ref[int(Y0):int(Y0) + side, int(X0):int(X0) + side], (out, out),
                         interpolation=cv2.INTER_CUBIC)
    # self-consistency on the face: the composite shrunk back to camera resolution must still look like the real
    # reference frame there. A failed alignment (tiny faces: unreliable landmarks) shows up as a large difference.
    face = np.zeros((out, out), np.uint8)
    fc = (r.pts5.mean(0) - np.array([X0, Y0])) * scale
    cv2.ellipse(face, (int(fc[0]), int(fc[1])), (int(0.42 * r.face_px * scale), int(0.55 * r.face_px * scale)),
                0, 0, 360, 1, -1)
    back = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (side, side), interpolation=cv2.INTER_AREA)
    real = fr_ref[int(Y0):int(Y0) + side, int(X0):int(X0) + side]
    fm = cv2.resize(face, (side, side), interpolation=cv2.INTER_NEAREST).astype(bool)
    if real.shape[:2] == back.shape[:2] and fm.any():
        d = back.astype(np.float64)[fm] - real.astype(np.float64)[fm]
        consistency = float(10 * np.log10(255 ** 2 / max(np.mean(d ** 2), 1e-9)))
    else:
        consistency = 0.0
    face_unc = float(spread[face.astype(bool)].mean()) if face.any() else 99.0
    return {"composite": np.clip(img, 0, 255).astype(np.uint8), "stack_mean": np.clip(mean, 0, 255).astype(np.uint8),
            "reference_bicubic": ref_bic, "uncertainty": spread, "frames_used": [obs[i].frame for i in used],
            "reference_frame": r.frame, "weights": [round(float(x), 3) for x in wq], "canvas": (X0, Y0, side, scale),
            "hole_frac": round(float(hole.mean()), 3), "consistency_db": round(consistency, 2),
            "face_uncertainty": round(face_unc, 2),
            "suspect": bool(face_unc > SUSPECT_UNC or consistency < SUSPECT_DB or len(used) < 4)}


def stamp(img: np.ndarray) -> np.ndarray:
    img = img.copy()
    h = max(14, img.shape[0] // 18)
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    cv2.putText(img, LABEL, (3, h - 4), cv2.FONT_HERSHEY_SIMPLEX, h / 40, (0, 220, 255), 1, cv2.LINE_AA)
    return img

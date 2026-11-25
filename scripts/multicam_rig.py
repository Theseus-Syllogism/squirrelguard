#!/usr/bin/env python
"""Joint calibration of several fixed cameras from ONE simultaneous frame each (VGGT), in one metric world frame.

  ~/src/splat/.venv/bin/python scripts/multicam_rig.py WORK --raw FRAMES_DIR --cams frontyard=1920x1080 \
      v4=2560x1440 testcam=1280x720 --lens frontyard=-0.3 v4=-0.3 testcam=-0.2 [--ceiling 2.30] [--desk 0.75]
      [--out-calib DIR] [--ply rig.ply] [--splat-work DIR]

FRAMES_DIR/<cam>.jpg: one frame per camera taken at the same moment (same aspect ratio for all). Lens: each frame
is first undistorted with a one-parameter division model (r_u = r_d / (1 + lam r_d^2), r in units of the image
half-diagonal; lam < 0 = barrel; pick lam by the plumb-line rule: ceiling joists and door frames straight), resized
to 1280x720 into WORK/images and given to splat/vggt_infer.py (-> WORK/vggt.npz). The saved calibrations carry the
lens as OpenCV k1, k2, k3 fitted to the division model over the whole raw image, so geometry.Camera rays are right
out to the frame edges (a plain pinhole was off by degrees there and broke triangulation). VGGT gives every camera's pose and pinhole intrinsics up to one unknown
scale. This script then:
  * gravity: fits the dominant horizontal plane ABOVE the cameras (ceiling joists, RANSAC on the fused depth
    points, normal within 35 deg of the cameras' mean up); its normal is world +Z;
  * scale: a second horizontal plane BELOW the cameras (here the desk they stand on) and the assumed
    floor-to-ceiling (--ceiling) and floor-to-that-plane (--desk) heights. This is the weak part: replace it with
    a measured distance (--scale-ref A,B,metres is not implemented; re-run with corrected heights) or with a
    walker of known height through replay_demo/fit_walk;
  * frame: Z up, floor Z = 0 directly below the first camera, +Y = the first camera's horizontal forward;
  * writes <out-calib>/<cam>.json (squirrel_guard.geometry.Camera: K at native size, D = 0, R world_from_camera,
    C metres) + rig.json (all cameras, planes, assumptions) + an optional coloured point cloud (PLY, metres).
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard.geometry import Camera  # noqa: E402


def undistort_division(img, lam: float):
    """Raw frame -> same-size undistorted frame (centre crop of the wider undistorted view)."""
    import cv2
    h, w = img.shape[:2]
    R = np.hypot(w / 2, h / 2)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    xu, yu = (x - w / 2) / R, (y - h / 2) / R
    ru = np.hypot(xu, yu)
    if abs(lam) < 1e-12:
        sc = np.ones_like(ru)
    else:   # invert r_u = r_d / (1 + lam r_d^2) for r_d
        disc = np.maximum(1 - 4 * lam * ru * ru, 0)
        rd = (1 - np.sqrt(disc)) / (2 * lam * np.maximum(ru, 1e-12))
        sc = np.where(ru > 1e-12, rd / np.maximum(ru, 1e-12), 1.0)
    return cv2.remap(img, (xu * sc * R + w / 2).astype(np.float32), (yu * sc * R + h / 2).astype(np.float32),
                     cv2.INTER_LINEAR)


def opencv_from_division(lam: float, f: float, W: int, H: int):
    """OpenCV [k1, k2, p1, p2, k3] (with focal f px, principal point at the centre) matching the division model
    over every raw pixel radius of a W x H image."""
    if abs(lam) < 1e-12:
        return [0.0] * 5
    R = np.hypot(W / 2, H / 2)
    rd = np.linspace(1e-3, 1.0, 400) * R
    ru = rd / (1 + lam * (rd / R) ** 2)
    rn = ru / f
    A = np.stack([rn ** 2, rn ** 4, rn ** 6], 1)
    k = np.linalg.lstsq(A, rd / ru - 1, rcond=None)[0]
    return [float(k[0]), float(k[1]), 0.0, 0.0, float(k[2])]


def unproject(z, i, keep_pct=70.0):
    d, c, K, E = z["depth"][i].astype(np.float32), z["conf"][i], z["intrinsic"][i], z["extrinsic"][i]
    y, x = np.nonzero(c >= np.percentile(c, 100 - keep_pct))
    dd = d[y, x]
    cam = np.stack([(x - K[0, 2]) / K[0, 0] * dd, (y - K[1, 2]) / K[1, 1] * dd, dd], 1)
    return (cam - E[:, 3]) @ E[:, :3], z["rgb"][i][y, x]


def ransac_plane(P, up, rng, tol=0.01, it=1500, max_tilt=35.0):
    S = P[rng.integers(0, len(P), (it, 3))]
    N = np.cross(S[:, 1] - S[:, 0], S[:, 2] - S[:, 0])
    L = np.linalg.norm(N, axis=1)
    ok = L > 1e-9
    N, S = N[ok] / L[ok, None], S[ok]
    N *= np.sign(N @ up)[:, None]
    keep = N @ up > np.cos(np.radians(max_tilt))
    best, bk = None, 0
    for nrm, s in zip(N[keep], S[keep]):
        k = int((np.abs((P - s[0]) @ nrm) < tol).sum())
        if k > bk:
            bk, best = k, (nrm, nrm @ s[0])
    if best is None:
        raise SystemExit("no horizontal plane found")
    nrm, o = best
    Q = P[np.abs(P @ nrm - o) < tol]
    m = Q.mean(0)
    nn = np.linalg.svd(Q - m)[2][2]
    nn *= np.sign(nn @ up)
    return nn, float(nn @ m), len(Q), len(P)


def depth_scale(z, ref: Path, cams: dict, order: list):
    """Metres per VGGT unit of this solve from the INSTALLED calibration's own VGGT run (ref/vggt.npz + ref/rig.json
    world_from_vggt.scale): per camera, the median ratio of the two metric depth maps over pixels both runs are
    confident about and agree on to 25 % (people, moved objects and the rest drop out). (scale, per-camera list)."""
    zr = np.load(ref / "vggt.npz")
    sr = json.loads((ref / "rig.json").read_text())["world_from_vggt"]["scale"]
    nr = [str(n) for n in zr["names"]]
    per = []
    for n, i in zip(cams, order):
        if f"{n}.jpg" not in nr or zr["depth"].shape[1:] != z["depth"].shape[1:]:
            continue
        j = nr.index(f"{n}.jpg")
        dn, dr = z["depth"][i].reshape(-1).astype(float), zr["depth"][j].reshape(-1).astype(float) * sr
        cn, cr = z["conf"][i].reshape(-1), zr["conf"][j].reshape(-1)
        ok = (cn >= np.percentile(cn, 50)) & (cr >= np.percentile(cr, 50)) & (dn > 1e-6) & (dr > 1e-6)
        r = dr[ok] / dn[ok]
        if len(r) < 500:
            continue
        m = np.median(r)
        r = r[np.abs(r / m - 1) < 0.25]
        per.append(float(np.median(r)))
    return (float(np.median(per)) if per else None), [round(v, 4) for v in per]


def align_to_installed(cdir: Path, cams: dict, order: list, E, centres, s_planes=None,
                       s_depth=None):
    """world_from_vggt (R, scale, origin) that maps this solve onto INSTALLED calibrations (geometry.Camera json: R =
    world_from_camera, C metres): rotation = chordal mean of R_inst @ cam_from_vggt over the cameras; scale from
    s_depth (depth_scale) when given, else least squares on the camera centres; then the translation. Returns (R, s, origin, report) with per-camera residuals
    (centre metres, rotation degrees): large residuals mean a camera moved since the calibration."""
    inst = {n: json.loads((cdir / f"{n}.json").read_text()) for n in cams}
    frames = {m.get("frame") for m in inst.values()}
    if len(frames) != 1 or None in frames:
        raise SystemExit(f"--align-to: calibrations are not one joint frame: {frames}")
    Ri = {n: np.asarray(inst[n]["R"], float) for n in cams}
    Ci = {n: np.asarray(inst[n]["C"], float) for n in cams}
    M = sum(Ri[n] @ E[i, :, :3] for n, i in zip(cams, order))
    U, _, Vt = np.linalg.svd(M)
    Rw = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    A = np.array([Rw @ centres[i] for i in order])
    B = np.array([Ci[n] for n in cams])
    am, bm = A.mean(0), B.mean(0)
    s_c = float(((A - am) * (B - bm)).sum() / max(((A - am) ** 2).sum(), 1e-12))
    base = max(float(np.linalg.norm(p - q)) for p in B for q in B)
    # scale: the installed run's depth maps when available (scene_worker passes --align-ref), else the camera
    # centres. Measured 2026-11-08: depth vs centres within 2-3 % on 3 cameras (1.5 m baseline) and on 2 (0.33 m);
    # the ceiling/desk planes were off by up to 20 % on clip frames, so they are only reported (scale_planes)
    s, src = s_c, "centres"
    if s_depth:                                  # the installed run's depth maps: the most direct reference
        s, src = float(s_depth), "depth"
    t = bm - s * am
    origin = -(Rw.T @ t) / s                       # world = (p - origin) @ Rw.T * s
    per = {}
    for n, i in zip(cams, order):
        Cn = s * (Rw @ centres[i]) + t
        Rn = Rw @ E[i, :, :3].T
        ang = np.degrees(np.arccos(np.clip((np.trace(Rn.T @ Ri[n]) - 1) / 2, -1, 1)))
        per[n] = {"centre_m": round(float(np.linalg.norm(Cn - Ci[n])), 3), "rot_deg": round(float(ang), 2)}
    return Rw, s, origin, {"frame": frames.pop(), "calib_dir": str(cdir), "scale": round(s, 4),
                           "scale_from": src, "scale_centres": round(s_c, 4),
                           "scale_depth": round(float(s_depth), 4) if s_depth else None,
                           "scale_planes": round(float(s_planes), 4) if s_planes else None,
                           "baseline_m": round(base, 3), "cameras": per,
                           "max_centre_m": max(v["centre_m"] for v in per.values()),
                           "max_rot_deg": max(v["rot_deg"] for v in per.values())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work")
    ap.add_argument("--cams", nargs="+", required=True, help="NAME=WxH (native resolution), file WORK/images/NAME.jpg")
    ap.add_argument("--raw", help="directory of simultaneous raw frames <cam>.jpg: undistort, then run VGGT")
    ap.add_argument("--lens", nargs="*", default=[], help="NAME=LAM division-model distortion per camera (default 0)")
    ap.add_argument("--net-size", default="1280x720", help="size the undistorted frames are resized to for VGGT")
    ap.add_argument("--vggt-python", default=sys.executable)
    ap.add_argument("--vggt-repo", default=str(Path.home() / "src/vggt"))
    ap.add_argument("--ceiling", type=float, default=2.30, help="floor to ceiling plane (m), ASSUMED")
    ap.add_argument("--desk", type=float, default=0.75, help="floor to the plane under the cameras (m), ASSUMED")
    ap.add_argument("--out-calib", help="write <cam>.json calibrations here (default WORK/calib)")
    ap.add_argument("--ply", help="also write the fused point cloud (metres) here")
    ap.add_argument("--align-to", help="directory of INSTALLED calibrations <cam>.json: instead of the plane-based "
                                       "gravity/scale/frame, put this solve into their world frame (similarity from "
                                       "camera rotations + centres) and report the residual (scene_worker.py)")
    ap.add_argument("--align-ref", help="with --align-to: the installed calibration's own work dir (vggt.npz + "
                                        "calib/rig.json or rig.json) to take the metric scale from its depth maps")
    ap.add_argument("--splat-work", help="also write a make_splat.py `train` work dir (undistorted/{sparse,images}) "
                                         "in the metric world frame: one PINHOLE camera per view, rig points as init")
    a = ap.parse_args()
    work = Path(a.work)
    cams = {}
    for spec in a.cams:
        n, wh = spec.split("=")
        cams[n] = tuple(int(v) for v in wh.lower().split("x"))
    lens = {n: 0.0 for n in cams}
    lens.update({k: float(v) for k, v in (t.split("=") for t in a.lens)})
    if a.raw:
        import cv2
        nw, nh = (int(v) for v in a.net_size.split("x"))
        (work / "images").mkdir(parents=True, exist_ok=True)
        for n in cams:
            img = cv2.imread(str(Path(a.raw) / f"{n}.jpg"))
            if img is None:
                sys.exit(f"missing {a.raw}/{n}.jpg")
            und = cv2.resize(undistort_division(img, lens[n]), (nw, nh), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(work / "images" / f"{n}.jpg"), und, [cv2.IMWRITE_JPEG_QUALITY, 95])
        subprocess.run([a.vggt_python, str(ROOT / "splat/vggt_infer.py"), str(work / "images"), str(work / "vggt.npz"),
                        "--repo", a.vggt_repo], check=True)
    z = np.load(work / "vggt.npz")
    names = [str(n) for n in z["names"]]
    order = [names.index(f"{n}.jpg") for n in cams]
    E, Kn = z["extrinsic"], z["intrinsic"]
    h, w = z["depth"].shape[1:]
    rng = np.random.default_rng(0)

    pts, cols = zip(*(unproject(z, i) for i in range(len(names))))
    X, RGB = np.vstack(pts), np.vstack(cols)
    sub = rng.choice(len(X), min(len(X), 80000), replace=False)
    centres = np.array([-E[i, :, :3].T @ E[i, :, 3] for i in range(len(names))])
    up0 = -np.mean([E[i, 1, :3] for i in order], 0)            # cameras' "up" (-y) in the VGGT frame
    up0 /= np.linalg.norm(up0)
    align = None
    if a.align_to:
        s_planes = None
        try:                                    # independent scale: the calibration's own ceiling/desk definition
            Rw0 = align_to_installed(Path(a.align_to), cams, order, E, centres)[0]
            up = Rw0[2]
            hx, hc = X[sub] @ up, centres @ up
            cn_, co_, _, _ = ransac_plane(X[sub][hx > hc.max() + 0.05], up, rng)
            bl = X[sub][hx < hc.min() - 0.05]
            dn_, do_, _, _ = ransac_plane(bl, cn_, rng)
            dp = bl[np.abs(bl @ dn_ - do_) < 0.01]
            s_planes = (a.ceiling - a.desk) / (co_ - float(np.median(dp @ cn_)))
        except (SystemExit, ValueError, IndexError):
            pass
        s_depth, per_depth = (depth_scale(z, Path(a.align_ref), cams, order) if a.align_ref else (None, []))
        Rw, s, origin, align = align_to_installed(Path(a.align_to), cams, order, E, centres, s_planes,
                                                  s_depth=s_depth)
        align["scale_depth_per_camera"] = per_depth
        zw = Rw[2]
        ck = cn = dk = dn = 0
        desk_n = ceil_n = zw
    else:
        hx, hc = X[sub] @ up0, centres @ up0
        ceil_n, ceil_o, ck, cn = ransac_plane(X[sub][hx > hc.max() + 0.05], up0, rng)
        below = X[sub][hx < hc.min() - 0.05]
        desk_n, desk_o, dk, dn = ransac_plane(below, ceil_n, rng)
        # desk offset measured along the ceiling normal (the better-determined direction) from its inliers
        dpts = below[np.abs(below @ desk_n - desk_o) < 0.01]
        desk_along = float(np.median(dpts @ ceil_n))
        gap = ceil_o - desk_along
        s = (a.ceiling - a.desk) / gap                               # metres per VGGT unit
        zw = ceil_n
        i0 = order[0]
        fwd = E[i0, 2, :3] - (E[i0, 2, :3] @ zw) * zw
        yw = fwd / np.linalg.norm(fwd)
        xw = np.cross(yw, zw)
        Rw = np.stack([xw, yw, zw])                                   # world_from_vggt rotation (rows = world axes)
        floor_vggt = ceil_o - a.ceiling / s                           # floor offset along zw in VGGT units
        origin = centres[i0] - (centres[i0] @ zw - floor_vggt) * zw   # floor point under the first camera

    def to_world(P):
        return (np.asarray(P) - origin) @ Rw.T * s

    out = Path(a.out_calib or work / "calib")
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.date.today().isoformat()
    # one id per joint solve: cameras carrying the same "frame" can be fused (squirrel_guard.fusion.frame_id)
    frame = f"multicam-vggt:{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M')}:{'+'.join(cams)}"
    if align:
        frame = align["frame"]                       # same world as the installed calibration (not a new frame)
    rig = {"method": "multicam-vggt", "date": stamp, "frame": frame, "world": "Z up, floor Z=0 below " + list(cams)[0] +
           ", +Y = its horizontal forward", "assumptions": {"ceiling_m": a.ceiling, "desk_m": a.desk},
           "planes": {"ceiling_inliers": f"{ck}/{cn}", "desk_inliers": f"{dk}/{dn}",
                      "desk_vs_ceiling_tilt_deg": round(float(np.degrees(np.arccos(min(1.0, abs(float(desk_n @ ceil_n)))))), 1)},
           "metres_per_unit": round(float(s), 4), "cameras": {},
           # world = (p_vggt - origin) @ R.T * scale, for tools that reuse WORK/vggt.npz (rig_objects.py)
           "world_from_vggt": {"origin": origin.tolist(), "R": Rw.tolist(), "scale": float(s)}}
    if align:
        rig["method"] = "multicam-vggt aligned"
        rig["world"] = f"aligned to the installed calibration {align['frame']}"
        rig["alignment"] = align
        rig.pop("assumptions")
    for n, i in zip(cams, order):
        W, H = cams[n]
        K = Kn[i].astype(float).copy()
        K[0] *= W / w
        K[1] *= H / h
        Rcw = E[i, :, :3]                                         # cam_from_vggt
        R = Rw @ Rcw.T                                            # world_from_camera
        C = to_world(centres[i])
        fwd_w = R[:, 2]
        pitch = float(np.degrees(np.arcsin(-fwd_w[2])))           # positive = looking down
        yaw = float(np.degrees(np.arctan2(fwd_w[0], fwd_w[1])))
        f = float(np.sqrt(K[0, 0] * K[1, 1]))
        K[0, 0] = K[1, 1] = f                                     # square pixels
        K[0, 2], K[1, 2] = W / 2, H / 2                           # the division model is centred
        D = opencv_from_division(lens[n], f, W, H)
        hfov = float(np.degrees(2 * np.arctan(W / 2 / f)))        # of the undistorted centre crop
        Camera(K, D, R, C, (W, H)).save(
            out / f"{n}.json", method="multicam-vggt", date=stamp, frame=frame, pitch_deg=round(pitch, 1),
            camera_height_m=round(float(C[2]), 3), hfov_deg=round(hfov, 1), lens_division_lambda=lens[n],
            note=f"interim joint calibration with {', '.join(cams)} (scripts/multicam_rig.py); scale from ASSUMED "
                 f"ceiling {a.ceiling} m and desk {a.desk} m; lens from the plumb-line division model")
        rig["cameras"][n] = {"C": C.round(3).tolist(), "height_m": round(float(C[2]), 3), "pitch_deg": round(pitch, 1),
                             "yaw_deg": round(yaw, 1), "hfov_deg": round(hfov, 1),
                             "lens_lambda": lens[n], "D": [round(v, 4) for v in D]}
    (out / "rig.json").write_text(json.dumps(rig, indent=1))
    print(json.dumps(rig, indent=1))
    if a.ply:
        P = to_world(X)
        keep = (P[:, 2] > -0.3) & (P[:, 2] < a.ceiling + 0.5)
        P, C3 = P[keep].astype(np.float32), RGB[keep].astype(np.uint8)
        with open(a.ply, "wb") as f:
            f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\n"
                    "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
            rec = np.zeros(len(P), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
            rec["p"], rec["c"] = P, C3
            f.write(rec.tobytes())
        print(f"{len(P)} points -> {a.ply}")
    if a.splat_work:
        import shutil
        import pycolmap
        sw = Path(a.splat_work)
        und = sw / "undistorted"
        shutil.rmtree(und, ignore_errors=True)
        (und / "sparse").mkdir(parents=True)
        (und / "images").mkdir()
        import cv2
        camlines, imglines = [], []
        for k, (n, i) in enumerate(zip(cams, order), 1):
            src = work / "images" / f"{n}.jpg"
            shutil.copy(src, und / "images" / f"{n}.jpg")
            Hs, Ws = cv2.imread(str(src)).shape[:2]
            K = Kn[i].astype(float)
            camlines.append(f"{k} PINHOLE {Ws} {Hs} {K[0, 0] * Ws / w!r} {K[1, 1] * Hs / h!r} "
                            f"{K[0, 2] * Ws / w!r} {K[1, 2] * Hs / h!r}\n")
            Rcw = E[i, :, :3] @ Rw.T                                  # cam_from_world (metric world)
            t = -Rcw @ to_world(centres[i])
            qx, qy, qz, qw = pycolmap.Rotation3d(Rcw).quat
            imglines.append(f"{k} {qw!r} {qx!r} {qy!r} {qz!r} {t[0]!r} {t[1]!r} {t[2]!r} {k} {n}.jpg\n\n")
        (und / "sparse" / "cameras.txt").write_text("".join(camlines))
        (und / "sparse" / "images.txt").write_text("".join(imglines))
        (und / "sparse" / "points3D.txt").write_text("")
        rec = pycolmap.Reconstruction(und / "sparse")
        P = to_world(X)
        sel = rng.choice(len(P), min(len(P), 150_000), replace=False)
        for p, c in zip(P[sel], RGB[sel]):
            rec.add_point3D(p, pycolmap.Track(), c)
        for f in ("cameras.txt", "images.txt", "points3D.txt"):
            (und / "sparse" / f).unlink()
        rec.write(und / "sparse")
        print(f"splat work dir: {und} ({len(sel)} init points)")


if __name__ == "__main__":
    main()

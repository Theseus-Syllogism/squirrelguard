"""Camera calibration for height estimates.

  board       write a printable checkerboard (landscape letter paper, 9x6 inner corners, 25 mm squares)
  intrinsics  lens calibration from the checkerboard (can be done BEFORE mounting: the lens doesn't change)
  walk        mounted camera: someone of known height walks around the yard -> camera pose (main method)
  points      mounted camera: tape-measured points marked on a snapshot -> camera pose (alternative);
              or map points from deploy/recorder/replay/calib.html (aerial photo pairs, east/north metres)
  check       heights the current calibration gives for a Frigate event (validate with another known person)
  snapshot    save the camera's current frame (to read pixel coordinates for `points`)

Run inside the profiler container on gpuserver (it has the models, the Frigate API and /calib):
  docker compose exec profiler python -m squirrel_guard.calibrate walk --camera frontyard --height-cm 180 --event <id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from .geometry import Camera, fit_points, fit_walk, nominal_K
from . import kinematics as motion
from .pose import PoseEstimator

CALIB = Path(os.environ.get("SG_CALIB_DIR", "/calib"))
API = os.environ.get("SG_FRIGATE_API", "http://frigate:5000")
POSE = os.environ.get("SG_POSE_MODEL", "/models/yolov8s-pose.onnx")


def _intrinsics(camera: str, path: str | None):
    p = Path(path) if path else CALIB / f"{camera}.intrinsics.json"
    if p.is_file():
        d = json.loads(p.read_text())
        return np.array(d["K"]), np.array(d["D"]), tuple(d["image_size"]), p
    return None, None, None, None


def _event_clip(eid: str) -> bytes:
    r = requests.get(f"{API}/api/events/{eid}/clip.mp4", timeout=120)
    r.raise_for_status()
    return r.content


def cmd_board(a):
    cols, rows = (int(v) for v in a.board.split("x"))
    dpi, sq = 300, a.square_mm / 25.4 * 300
    w, h = int(11 * dpi), int(8.5 * dpi)  # landscape letter
    img = np.full((h, w), 255, np.uint8)
    bw, bh = (cols + 1) * sq, (rows + 1) * sq
    if bw > w - 150 or bh > h - 150:
        sys.exit("board does not fit on letter paper; use smaller squares")
    x0, y0 = (w - bw) / 2, (h - bh) / 2
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                cv2.rectangle(img, (int(x0 + c * sq), int(y0 + r * sq)), (int(x0 + (c + 1) * sq) - 1, int(y0 + (r + 1) * sq) - 1), 0, -1)
    cv2.putText(img, f"{cols}x{rows} inner corners, {a.square_mm:g} mm squares - print at 100% (no fit-to-page), check with a ruler",
                (60, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    cv2.imwrite(a.out, img)
    print(f"wrote {a.out}: print at 100 %, measure one square (should be {a.square_mm:g} mm), tape it to something flat")


def cmd_intrinsics(a):
    cols, rows = (int(v) for v in a.board.split("x"))
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * (a.square_mm / 1000)
    src = a.source
    live = not Path(src).is_dir()
    if not live:
        frames = (cv2.imread(str(p)) for p in sorted(Path(src).glob("*.jpg")) + sorted(Path(src).glob("*.png")))
    else:
        cap = cv2.VideoCapture(src)

        def gen():
            t_end = time.time() + a.seconds
            while time.time() < t_end:
                ok, fr = cap.read()
                if not ok:
                    if not src.startswith("rtsp"):
                        return
                    continue
                yield fr
        frames = gen()
    objs, imgs, cells, size = [], [], set(), None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    last, nread = 0.0, 0
    for fr in frames:
        if fr is None:
            continue
        nread += 1
        if live and nread % 125 == 0:
            print(f"... {nread} frames read, {len(objs)} views so far", flush=True)
        if live and time.time() - last < 0.4:
            continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        small = cv2.resize(gray, None, fx=0.5, fy=0.5)
        ok, c = cv2.findChessboardCorners(small, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
        if not ok:
            continue
        last = time.time()
        c = cv2.cornerSubPix(gray, c * 2, (11, 11), (-1, -1), crit)
        m = c.reshape(-1, 2)
        span = np.ptp(m, 0).max() / max(size)
        cell = (int(m[:, 0].mean() / size[0] * 4), int(m[:, 1].mean() / size[1] * 3), int(span * 4),
                int(np.degrees(np.arctan2(*(m[cols - 1] - m[0])[::-1])) // 20))
        if cell in cells:                                   # keep only views that add a new position/size/angle
            continue
        cells.add(cell)
        objs.append(objp); imgs.append(c)
        print(f"view {len(objs)}: board at {cell[:2]} size {span:.2f}", flush=True)
        if len(objs) >= a.views:
            break
    if len(objs) < 10:
        sys.exit(f"read {nread} frames but only {len(objs)} usable views; need >= 10 (move the board around: corners, edges, near, far, tilted)")
    rms, K, D, *_ = cv2.calibrateCamera(objs, imgs, size, None, None)
    out = Path(a.out) if a.out else CALIB / f"{a.camera}.intrinsics.json"
    out.write_text(json.dumps({"image_size": list(size), "K": K.tolist(), "D": D.reshape(-1).tolist(),
                               "rms_px": rms, "views": len(objs), "made": time.strftime("%F %T")}, indent=1))
    hfov = 2 * np.degrees(np.arctan(size[0] / 2 / K[0, 0]))
    print(f"wrote {out}: rms {rms:.2f} px (good < 1), fx {K[0, 0]:.0f}, k1 {D.reshape(-1)[0]:.3f}, pinhole hfov {hfov:.0f} deg")


def _stance_obs(clips: list[bytes], pose, fps: float):
    obs, size = [], None
    for mp4 in clips:
        frames, _ = motion.decode(mp4, fps, 300)
        trk, size = motion.track(frames, pose)
        obs += [sp for p in trk if p is not None and (sp := motion.stance_points(p, size))]
    return obs, size


def cmd_walk(a):
    pose = PoseEstimator(a.pose)
    clips = [_event_clip(e) for e in a.event] + [Path(f).read_bytes() for f in a.clip]
    if not clips:
        sys.exit("give --event (Frigate event id) and/or --clip")
    obs, size = _stance_obs(clips, pose, a.fps)
    print(f"{len(obs)} upright full-body frames at {size}")
    if len(obs) < 30:
        sys.exit("too few: walk slowly through the WHOLE yard (near, far, left, right) for a minute, whole body in view")
    ys = np.array([f[1] for f, _ in obs]) / size[1]
    if np.ptp(ys) < 0.25:
        print("WARNING: feet covered only a narrow band of the image; walk both close to and far from the camera")
    K, D, ksize, kpath = _intrinsics(a.camera, a.intrinsics)
    cam, st = fit_walk(obs, size, a.height_cm / 100, K=K, D=D, k_size=ksize)
    st.update({"camera": a.camera, "events": a.event, "made": time.strftime("%F %T"),
               "intrinsics_file": str(kpath) if kpath else None})
    out = Path(a.out) if a.out else CALIB / f"{a.camera}.json"
    cam.meta = st
    cam.save(out)
    print(json.dumps(st, indent=1))
    print(f"wrote {out}. Residual = how far the known person's per-frame heights scatter around {a.height_cm:g} cm.")
    print("Validate with someone else of known height: calibrate check --event <their event>")


def cmd_points(a):
    pts = json.loads(Path(a.json).read_text())
    W, H = pts["frame_size"]
    K, D, ksize, _ = _intrinsics(a.camera, a.intrinsics)
    nominal = K is None
    if nominal:
        print("no checkerboard intrinsics: starting from a nominal 100 deg lens; with >= 8 points the focal length "
              "and distortion are fitted too")
        K, D, ksize = nominal_K(W, H), np.zeros(5), (W, H)
    cam, st = fit_points(pts["points"], (W, H), K, D, ksize, refine_lens=nominal)
    st.update({"camera": a.camera, "made": time.strftime("%F %T")})
    if pts.get("geo"):                         # map calibration (replay/calib.html): keep the georeference + aerial
        st["geo"] = pts["geo"]
    hint = (pts.get("camera_hint") or {}).get("position_m")
    if hint:                                   # hand-entered mount position: report how far the fit landed from it
        st["camera_hint_m"] = hint
        st["camera_vs_hint_m"] = round(float(np.linalg.norm(np.asarray(cam.C, float) - np.asarray(hint, float))), 2)
    cam.meta = st
    out = Path(a.out) if a.out else CALIB / f"{a.camera}.json"
    cam.save(out)
    print(json.dumps(st, indent=1), f"\nwrote {out}")


def cmd_check(a):
    cam = Camera.load(a.calib or CALIB / f"{a.camera}.json")
    if cam is None:
        sys.exit("no calibration yet")
    pose = PoseEstimator(a.pose)
    res = motion.analyse([_event_clip(e) for e in a.event], pose, cam, fps=a.fps, max_s=120)
    print(json.dumps(res, indent=1))


def cmd_snapshot(a):
    r = requests.get(f"{API}/api/{a.camera}/latest.jpg", params={"height": a.height} if a.height else None, timeout=15)
    r.raise_for_status()
    Path(a.out).write_bytes(r.content)
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    print(f"wrote {a.out} ({img.shape[1]}x{img.shape[0]}): read pixel coordinates in an image editor; use this size as frame_size")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("board"); b.add_argument("--board", default="9x6"); b.add_argument("--square-mm", type=float, default=25)
    b.add_argument("--out", default="checkerboard.png")
    i = sub.add_parser("intrinsics"); i.add_argument("--source", required=True, help="rtsp:// URL, video file or image folder")
    i.add_argument("--camera", default="frontyard"); i.add_argument("--board", default="9x6")
    i.add_argument("--square-mm", type=float, default=25); i.add_argument("--views", type=int, default=30)
    i.add_argument("--seconds", type=float, default=180); i.add_argument("--out")
    for name in ("walk", "check"):
        w = sub.add_parser(name); w.add_argument("--camera", default="frontyard")
        w.add_argument("--event", action="append", default=[]); w.add_argument("--fps", type=float, default=6)
        w.add_argument("--pose", default=POSE)
        if name == "walk":
            w.add_argument("--height-cm", type=float, required=True); w.add_argument("--clip", action="append", default=[])
            w.add_argument("--intrinsics"); w.add_argument("--out")
        else:
            w.add_argument("--calib")
    p = sub.add_parser("points"); p.add_argument("--json", required=True); p.add_argument("--camera", default="frontyard")
    p.add_argument("--intrinsics"); p.add_argument("--out")
    s = sub.add_parser("snapshot"); s.add_argument("--camera", default="frontyard"); s.add_argument("--out", default="snapshot.jpg")
    s.add_argument("--height", type=int)
    a = ap.parse_args()
    {"board": cmd_board, "intrinsics": cmd_intrinsics, "walk": cmd_walk, "points": cmd_points,
     "check": cmd_check, "snapshot": cmd_snapshot}[a.cmd](a)


if __name__ == "__main__":
    main()

"""Visual evidence for one person visit, rendered on gpuserver where the frames and keypoint track already are.

  card  (JPEG, 1440 px wide): key frame with box + skeleton + timestamp | path through the scene over a dimmed
        frame, with zones | the visit's facts as text | a strip of 6 moments across the visit
  gif   (animated, 480 px wide): flip-book of the visit cropped around where the person went, ~3 frames/s

Input comes from kinematics.analyse(..., keep_every=N): per clip piece, the kept frames as JPEG plus the track.
Text is ASCII only (OpenCV Hershey fonts).
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
         (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (24, 26, 30)
FG = (235, 238, 242)
MUTED = (160, 166, 178)
ACCENT = (40, 170, 255)      # BGR orange-ish box colour
PATH_A, PATH_B = (255, 160, 60), (60, 60, 255)   # start (blue) -> end (red), BGR


def _dec(jpg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


def _text(img, s, org, scale=0.6, color=FG, thick=1):
    if img[max(0, int(org[1]) - 12):int(org[1]) + 4, int(org[0]):int(org[0]) + 40].mean() > 60:  # outline only on bright areas
        cv2.putText(img, s, (int(org[0]), int(org[1])), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, (int(org[0]), int(org[1])), FONT, scale, color, thick, cv2.LINE_AA)


def _fit(img, w, h):
    """Letterbox img into w x h."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), BG, np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def _draw_pose(img, p, s):
    x1, y1, x2, y2 = (int(v * s) for v in p.box)
    cv2.rectangle(img, (x1, y1), (x2, y2), ACCENT, 2, cv2.LINE_AA)
    kp = p.kp
    for a, b in EDGES:
        if kp[a, 2] >= 0.4 and kp[b, 2] >= 0.4:
            cv2.line(img, (int(kp[a, 0] * s), int(kp[a, 1] * s)), (int(kp[b, 0] * s), int(kp[b, 1] * s)),
                     (90, 230, 120), 2, cv2.LINE_AA)
    for x, y, c in kp:
        if c >= 0.4:
            cv2.circle(img, (int(x * s), int(y * s)), 3, (255, 255, 255), -1, cv2.LINE_AA)


def _zones(zones: dict, size, detect_size) -> list[tuple[str, np.ndarray]]:
    """Frigate zone coordinates ("x,y,x,y,..." normalised 0-1, or pixels at the detect resolution) -> polygons in
    pixels of a frame of `size`."""
    W, H = size
    out = []
    for name, z in (zones or {}).items():
        c = z.get("coordinates") if isinstance(z, dict) else z
        if isinstance(c, list):
            c = ",".join(str(v) for v in c)
        try:
            v = np.array([float(t) for t in str(c).split(",")], float).reshape(-1, 2)
        except ValueError:
            continue
        v *= (W, H) if v.max() <= 1.0 else (W / detect_size[0], H / detect_size[1])
        out.append((name, v))
    return out


def _primary(ev: dict):
    parts = [p for p in ev.get("parts", []) if p["frames"] and any(t is not None for t in p["track"])]
    return max(parts, key=lambda p: sum(t is not None for t in p["track"])) if parts else None


def card(ev: dict, facts: list[str], title: str, zones: dict | None = None, detect_size=(1280, 720)) -> bytes | None:
    part = _primary(ev)
    if part is None:
        return None
    trk, size, fps = part["track"], part["size"], part["fps"]
    kept = [(i, _dec(j)) for i, j in part["frames"]]
    kept = [(i, f) for i, f in kept if f is not None]
    s = kept[0][1].shape[1] / size[0]                   # kept-frame scale vs analysed frame
    with_p = [(i, f) for i, f in kept if trk[i] is not None]
    if not with_p:
        return None

    def area(i):
        b = trk[i].box
        return (b[2] - b[0]) * (b[3] - b[1]) * float(np.mean(trk[i].kp[:, 2]))

    ki, kf = max(with_p, key=lambda t: area(t[0]))
    key = kf.copy()
    _draw_pose(key, trk[ki], s)
    _text(key, f"+{ki / fps:.0f}s", (12, key.shape[0] - 14), 0.7)
    key = _fit(key, 960, 540)

    # path panel: foot points over a dimmed frame, zones shaded
    bgf = kept[len(kept) // 2][1]
    path = (bgf * 0.45).astype(np.uint8)
    if zones:
        ov = path.copy()
        polys = [(name, (poly * s).astype(np.int32)) for name, poly in _zones(zones, size, detect_size)]
        for _, pts in polys:
            cv2.fillPoly(ov, [pts], (90, 70, 40))
        path = cv2.addWeighted(ov, 0.35, path, 0.65, 0)
        for name, pts in polys:
            cv2.polylines(path, [pts], True, (200, 160, 90), 1, cv2.LINE_AA)
            _text(path, name, pts.min(0) + (4, 16), 0.45, (220, 190, 120))
    foot = [((p.box[0] + p.box[2]) / 2 * s, p.box[3] * s) for p in trk if p is not None]
    n = len(foot)
    for k in range(1, n):
        a = k / max(1, n - 1)
        col = tuple(int(PATH_A[c] * (1 - a) + PATH_B[c] * a) for c in range(3))
        cv2.line(path, tuple(int(v) for v in foot[k - 1]), tuple(int(v) for v in foot[k]), col, 3, cv2.LINE_AA)
    if n:
        cv2.circle(path, tuple(int(v) for v in foot[0]), 7, (90, 220, 90), -1, cv2.LINE_AA)
        cv2.circle(path, tuple(int(v) for v in foot[-1]), 7, (60, 60, 255), -1, cv2.LINE_AA)
        _text(path, "start", (foot[0][0] + 9, foot[0][1] - 6), 0.5, (150, 240, 150))
        _text(path, "end", (foot[-1][0] + 9, foot[-1][1] - 6), 0.5, (150, 150, 255))
    path = _fit(path, 480, 270)
    _text(path, "path through the scene", (10, 22), 0.5, MUTED)

    # facts panel
    info = np.full((270, 480, 3), BG, np.uint8)
    y = 30
    for line in facts[:9]:
        _text(info, line[:46], (14, y), 0.55, MUTED if line.startswith("(") else FG)
        y += 27

    # moments strip: 6 crops spread over the visit
    pick = [with_p[int(round(k))] for k in np.linspace(0, len(with_p) - 1, min(6, len(with_p)))]
    tiles = []
    for i, f in pick:
        x1, y1, x2, y2 = (v * s for v in trk[i].box)
        w, h = x2 - x1, y2 - y1
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(w, h) * 1.3
        X1, Y1 = int(max(0, cx - side * 0.4)), int(max(0, cy - side / 2))
        X2, Y2 = int(min(f.shape[1], cx + side * 0.4)), int(min(f.shape[0], cy + side / 2))
        t = _fit(f[Y1:Y2, X1:X2], 240, 250)
        _text(t, f"+{i / fps:.0f}s", (8, 242), 0.55)
        tiles.append(t)
    while len(tiles) < 6:
        tiles.append(np.full((250, 240, 3), BG, np.uint8))
    strip = np.hstack(tiles)

    head = np.full((44, 1440, 3), BG, np.uint8)
    _text(head, title[:110], (14, 30), 0.75, FG, 2)
    out = np.vstack([head, np.hstack([key, np.vstack([path, info])]), strip])
    ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return jpg.tobytes() if ok else None


def gif(ev: dict, max_frames: int = 16, width: int = 480, fps: float = 3.0) -> bytes | None:
    """Flip-book of the visit, cropped to the region the person moved through."""
    part = _primary(ev)
    if part is None:
        return None
    trk, size = part["track"], part["size"]
    kept = [(i, j) for i, j in part["frames"] if trk[i] is not None]
    if len(kept) < 2:
        return None
    first = _dec(kept[0][1])
    s = first.shape[1] / size[0]
    boxes = np.array([trk[i].box for i, _ in kept]) * s
    x1, y1 = boxes[:, 0].min(), boxes[:, 1].min()
    x2, y2 = boxes[:, 2].max(), boxes[:, 3].max()
    pw, ph = (x2 - x1) * 0.15 + 20, (y2 - y1) * 0.15 + 20
    H, W = first.shape[:2]
    X1, Y1, X2, Y2 = int(max(0, x1 - pw)), int(max(0, y1 - ph)), int(min(W, x2 + pw)), int(min(H, y2 + ph))
    pick = [kept[int(round(k))] for k in np.linspace(0, len(kept) - 1, min(max_frames, len(kept)))]
    with tempfile.TemporaryDirectory() as td:
        for n, (i, j) in enumerate(pick):
            f = _dec(j)
            b = trk[i].box
            cv2.rectangle(f, (int(b[0] * s), int(b[1] * s)), (int(b[2] * s), int(b[3] * s)), ACCENT, 2, cv2.LINE_AA)
            c = f[Y1:Y2, X1:X2]
            h = int(c.shape[0] * width / c.shape[1]) // 2 * 2
            c = cv2.resize(c, (width, max(2, h)), interpolation=cv2.INTER_AREA)
            _text(c, f"+{i / part['fps']:.0f}s", (8, c.shape[0] - 10), 0.5)
            cv2.imwrite(str(Path(td) / f"{n:03d}.png"), c)
        out = Path(td) / "v.gif"
        r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-framerate", str(fps), "-i",
                            str(Path(td) / "%03d.png"), "-vf",
                            "split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer",
                            "-loop", "0", str(out)], capture_output=True)
        return out.read_bytes() if r.returncode == 0 and out.exists() else None

"""Face sequences for small/distant faces: find the face in EVERY frame of a person's track, so recognition can
combine many weak views (faceseq_aggregate) and a viewer can stack them (lucky.py). docs/faceseq.md

Front end (per frame):
  1. person box -> YOLOv8s-pose (COCO) on a zoomed window -> head position from nose/eyes/ears keypoints
     (works when the face is only ~8 px: no face detector needed to know where the head is);
  2. head window upscaled so the face is ~100 px -> YuNet (OpenCV Zoo, MIT) face + 5 landmarks;
  3. landmarks mapped back to camera pixels; 112x112 ArcFace-aligned crop warped from the ORIGINAL frame
     (bilinear interpolation only -- never restoration or super-resolution: recognition sees camera pixels).
Each observation carries its true face size in camera pixels, so the recognition rules can refuse faces that are
too small ("too far") instead of guessing.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np

from .faceid_v2 import align112
from .pose import PoseEstimator

YUNET = Path.home() / ".cache/squirrel-guard/models/face_detection_yunet_2023mar.onnx"
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4


@dataclasses.dataclass
class FaceSeqObs:
    frame: int                      # frame index in the clip
    t: float                        # seconds from clip start
    pts5: np.ndarray                # (5, 2) landmarks in camera pixels (YuNet order: r-eye, l-eye, nose, r-mouth, l-mouth
                                    # as seen in the image = ArcFace order: image-left eye first)
    face_px: float                  # face size in camera pixels (YuNet box, or 2.5 x eye distance)
    det_score: float
    crop112: np.ndarray             # ArcFace-aligned 112x112 BGR from camera pixels
    native: np.ndarray              # the face region at camera resolution (for stacking / viewing)
    native_origin: tuple            # (x0, y0) of `native` in the frame
    sharpness: float                # variance of Laplacian on the native face region
    yaw_proxy: float                # nose offset from the eye midpoint / eye distance (0 = frontal)
    person_box: tuple


class FaceSeq:
    def __init__(self, pose_model: str, yunet: str | Path = YUNET, target_face_px: int = 100, threads: int = 4,
                 backend: str = "ort"):
        self.pose = PoseEstimator(pose_model, conf=0.35, threads=threads, backend=backend)
        self.det = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), score_threshold=0.6, nms_threshold=0.3,
                                             top_k=20)
        self.target = target_face_px

    # -- head from body keypoints -------------------------------------------------------------------------------
    def head(self, frame: np.ndarray, person_box) -> tuple | None:
        ps = self.pose.in_region(frame, person_box, grow=1.4)
        if not ps:
            return None
        cx0, cy0 = (person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2
        p = min(ps, key=lambda q: abs((q.box[0] + q.box[2]) / 2 - cx0) + abs((q.box[1] + q.box[3]) / 2 - cy0))
        kp = p.kp
        good = [i for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if kp[i, 2] > 0.3]
        ph = p.box[3] - p.box[1]
        if len(good) >= 2:
            hx, hy = kp[good, 0].mean(), kp[good, 1].mean()
        else:
            hx, hy = (p.box[0] + p.box[2]) / 2, p.box[1] + 0.07 * ph
        size = max(12.0, 0.16 * ph)                     # head ~ 1/7 of stature
        return hx, hy, size, p.box

    # -- face in an upscaled head window ------------------------------------------------------------------------
    def face(self, frame: np.ndarray, hx: float, hy: float, size: float):
        H, W = frame.shape[:2]
        half = 1.6 * size
        x0, y0 = int(max(0, hx - half)), int(max(0, hy - half))
        x1, y1 = int(min(W, hx + half)), int(min(H, hy + half))
        win = frame[y0:y1, x0:x1]
        if win.size == 0:
            return None
        z = max(1.0, min(8.0, self.target / max(size * 0.8, 1)))
        big = cv2.resize(win, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
        self.det.setInputSize((big.shape[1], big.shape[0]))
        _, faces = self.det.detect(big)
        if faces is None or not len(faces):
            return None
        f = max(faces, key=lambda r: r[-1])
        pts = f[4:14].reshape(5, 2) / z + np.array([x0, y0], np.float32)
        return pts.astype(np.float32), float(f[-1]), float(max(f[2], f[3]) / z)

    def observe(self, frame: np.ndarray, idx: int, t: float, person_box) -> tuple[FaceSeqObs | None, tuple | None]:
        h = self.head(frame, person_box)
        if h is None:
            return None, None
        hx, hy, size, pbox = h
        r = self.face(frame, hx, hy, size)
        if r is None:
            return None, pbox
        pts, sc, box_px = r
        eye_d = float(np.linalg.norm(pts[0] - pts[1]))
        face_px = max(box_px, 2.5 * eye_d)              # detector box: eye spacing collapses on profile faces
        crop = align112(frame, pts)
        if crop is None:
            return None, pbox
        c = pts.mean(0)
        half = max(6, int(face_px * 0.75))
        X0, Y0 = int(max(0, c[0] - half)), int(max(0, c[1] - half))
        native = frame[Y0:int(c[1] + half), X0:int(c[0] + half)].copy()
        g = cv2.cvtColor(native, cv2.COLOR_BGR2GRAY) if native.size else None
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var()) if g is not None and g.size else 0.0
        mid = (pts[0] + pts[1]) / 2
        yaw = float((pts[2, 0] - mid[0]) / max(eye_d, 1e-3))
        return FaceSeqObs(idx, t, pts, face_px, sc, crop, native, (X0, Y0), sharp, yaw, tuple(pbox)), pbox

    # -- a whole clip -------------------------------------------------------------------------------------------
    def find(self, frame: np.ndarray, near=None, tile: int = 320) -> tuple | None:
        """Zoomed-tile person search (small/distant people); the one nearest `near` (x, y), else the tallest."""
        H, W = frame.shape[:2]
        found = []
        for ty in range(0, max(1, H - tile + 1), tile * 3 // 4):
            for tx in range(0, max(1, W - tile + 1), tile * 3 // 4):
                for p in self.pose(frame[ty:ty + tile, tx:tx + tile]):
                    if p.score >= 0.5:
                        found.append((p.box[0] + tx, p.box[1] + ty, p.box[2] + tx, p.box[3] + ty))
        if not found:
            return None
        if near is None:
            return max(found, key=lambda b: b[3] - b[1])
        return min(found, key=lambda b: abs((b[0] + b[2]) / 2 - near[0]) + abs((b[1] + b[3]) / 2 - near[1]))

    def track(self, frames: list[np.ndarray], fps: float, start_box, step: int = 1,
              reacquire_s: float = 0.5) -> list[FaceSeqObs]:
        """Follow one person from `start_box` (x1, y1, x2, y2 in the first frame) and collect face observations.
        When the person is lost for `reacquire_s` (turned away, occluded), search again with zoomed tiles near
        their last position."""
        out, box = [], tuple(float(v) for v in start_box)
        lost = 0
        for i in range(0, len(frames), step):
            o, pbox = self.observe(frames[i], i, i / fps, box)
            if pbox is not None:
                box, lost = tuple(float(v) for v in pbox), 0
            else:
                lost += step
                if lost >= reacquire_s * fps:
                    b = self.find(frames[i], near=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
                    if b is not None:
                        box, lost = tuple(float(v) for v in b), 0
            if o is not None:
                out.append(o)
        return out

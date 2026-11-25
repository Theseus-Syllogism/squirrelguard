"""Release copies: every face and number plate blurred, for disclosure (public-records requests, court bundles).

Detectors (both MIT-licensed code, downloaded at image build with a fixed SHA-256, see the Dockerfile):
  faces   OpenCV YuNet (face_detection_yunet_2023mar.onnx); frames narrower than 1280 px are enlarged 2x first so
          small, distant faces are found
  plates  open-image-models YOLOv9-t 640 licence-plate detector (end-to-end, NMS inside the model)

Redaction is irreversible (pixelate to 8x8-pixel blocks, then blur) and generous: boxes grow by 30 % and stay on for
two frames either side of a detection so a missed frame does not flash a face. Automatic detection misses things
(faces turned away, very small, behind glass): every release copy says so, and a person must check it before release.
The release copy has no sound and no continuous-recording segments.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import os

MODELS = Path(os.environ.get("REDACT_MODELS") or Path(__file__).with_name("models"))
FACE_MODEL = "face_detection_yunet_2023mar.onnx"
PLATE_MODEL = "yolo-v9-t-640-license-plates-end2end.onnx"
GROW = 0.30
HOLD = 2                                                   # frames a box stays on before and after a detection


class Redactor:
    def __init__(self, models: Path | None = None, face_score: float = 0.55, plate_score: float = 0.25):
        models = models or MODELS
        self.face = cv2.FaceDetectorYN.create(str(models / FACE_MODEL), "", (320, 320), face_score, 0.3, 5000)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.plate = ort.InferenceSession(str(models / PLATE_MODEL), so, providers=["CPUExecutionProvider"])
        self.plate_score = plate_score

    def faces(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = img.shape[:2]
        k = 2.0 if w < 1280 else 1.0
        big = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_CUBIC) if k != 1 else img
        self.face.setInputSize((big.shape[1], big.shape[0]))
        _, found = self.face.detect(big)
        out = []
        for f in found if found is not None else []:
            x, y, bw, bh = (f[:4] / k).tolist()
            out.append((int(x), int(y), int(x + bw), int(y + bh)))
        return out

    def plates(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = img.shape[:2]
        r = min(640 / h, 640 / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw, dh = (640 - nw) / 2, (640 - nh) / 2
        canvas = np.full((640, 640, 3), 114, np.uint8)
        canvas[int(round(dh - 0.1)):int(round(dh - 0.1)) + nh, int(round(dw - 0.1)):int(round(dw - 0.1)) + nw] = cv2.resize(img, (nw, nh))
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = []
        for row in self.plate.run(None, {"images": x})[0]:
            x1, y1, x2, y2, score = row[1], row[2], row[3], row[4], row[6]
            if score >= self.plate_score:
                out.append((int((x1 - dw) / r), int((y1 - dh) / r), int((x2 - dw) / r), int((y2 - dh) / r)))
        return out

    def detect(self, img: np.ndarray) -> tuple[list, list]:
        return self.faces(img), self.plates(img)


def _grow(b, w, h, g=GROW):
    x1, y1, x2, y2 = b
    dx, dy = (x2 - x1) * g / 2, (y2 - y1) * g / 2
    return max(0, int(x1 - dx)), max(0, int(y1 - dy)), min(w, int(x2 + dx)), min(h, int(y2 + dy))


def blur(img: np.ndarray, boxes) -> np.ndarray:
    h, w = img.shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = _grow(b, w, h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = img[y1:y2, x1:x2]
        small = cv2.resize(roi, (max(1, (x2 - x1) // 8), max(1, (y2 - y1) // 8)), interpolation=cv2.INTER_AREA)
        img[y1:y2, x1:x2] = cv2.GaussianBlur(cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST), (0, 0), 3)
    return img


def redact_image(r: Redactor, src: Path, dst: Path) -> dict:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"{src.name} is not a readable image")
    f, p = r.detect(img)
    cv2.imwrite(str(dst), blur(img, f + p), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"faces": len(f), "plates": len(p), "frames": 1}


def redact_video(r: Redactor, src: Path, dst: Path, max_frames: int = 20000) -> dict:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"{src.name} is not a readable video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames, dets = [], []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
        dets.append(r.detect(fr))
    cap.release()
    if not frames:
        raise ValueError(f"{src.name} has no frames")
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    nf = sum(len(d[0]) for d in dets)
    npl = sum(len(d[1]) for d in dets)
    for i, fr in enumerate(frames):
        boxes = []
        for j in range(max(0, i - HOLD), min(len(dets), i + HOLD + 1)):
            boxes += dets[j][0] + dets[j][1]
        out.write(blur(fr, boxes))
    out.release()
    return {"faces": nf, "plates": npl, "frames": len(frames)}


RELEASE_NOTE = ("Release copy: every face and number plate that the automatic detectors found is blurred, irreversibly. "
                "Detectors miss faces that are small, turned away or behind glass, and plates at steep angles. A person "
                "must check every frame before this copy is released. It has no sound and no continuous-recording segments. "
                "It is linked to the unredacted package by that package's manifest hash.")


def make_release(package: Path, dest: Path, meta: dict, signer, work: Path, redactor: Redactor | None = None) -> dict:
    """Read the unredacted package, write a redacted, signed release package. Returns its record."""
    import hashlib
    import zipfile

    import evidence_export as evx
    r = redactor or Redactor()
    files, counts = [], {"faces": 0, "plates": 0, "files": 0}
    with zipfile.ZipFile(package) as z:
        src_manifest = json.loads(z.read("manifest.json"))
        src_hash = hashlib.sha256(z.read("manifest.json")).hexdigest()
        for f in src_manifest["files"]:
            p = f["path"]
            kind = "video" if p.endswith("/original/clip.mp4") else ("image" if p.endswith((".jpg", ".jpeg")) else None)
            if not kind or "/segments/" in p:
                continue
            src = work / ("src-" + p.replace("/", "_"))
            src.write_bytes(z.read(p))
            out_name = p.replace("original/", "").replace("derived/", "")
            out_name = out_name[:-4] + (".redacted.mp4" if kind == "video" else ".redacted.jpg")
            dst = work / out_name.replace("/", "_")
            try:
                res = (redact_video if kind == "video" else redact_image)(r, src, dst)
            except ValueError as e:                                # unreadable: left out, and said so
                counts.setdefault("left_out", []).append(f"{p}: {e}")
                continue
            counts["faces"] += res["faces"]
            counts["plates"] += res["plates"]
            counts["files"] += 1
            files.append({"path": out_name, "src": dst, "role": "redacted",
                          "description": f"Redacted from {p} (sha256 {f['sha256'][:16]}…): {res['faces']} face and "
                                         f"{res['plates']} plate detections over {res['frames']} frame(s)",
                          "source": f"{p} sha256 {f['sha256']}"})
    counts.setdefault("left_out", [])
    summary = dict(counts, source_export=src_manifest["export_id"], source_manifest_sha256=src_hash)
    manifest = {"format": "squirrel-guard-release/1", **meta, "source": {"export_id": src_manifest["export_id"],
                "case_ref": src_manifest["case_ref"], "manifest_sha256": src_hash}, "note": RELEASE_NOTE,
                "left_out": counts["left_out"],
                "signature": {"algorithm": "Ed25519", "fingerprint": signer.fingerprint}, "files": []}
    for f in files:
        digest, size = evx.sha256_file(f["src"])
        manifest["files"].append({k: f[k] for k in ("path", "role", "description", "source")} | {"sha256": digest, "bytes": size})
    mbytes = json.dumps(manifest, indent=1, ensure_ascii=False).encode()
    sums = "".join(f"{f['sha256']}  {f['path']}\n" for f in manifest["files"]) + f"{hashlib.sha256(mbytes).hexdigest()}  manifest.json\n"
    tmp = dest.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(f["src"], f["path"])
        z.writestr("manifest.json", mbytes)
        z.writestr("manifest.sig", signer.sign(mbytes))
        z.writestr("signing-key.pem", signer.public_pem)
        z.writestr("SHA256SUMS", sums)
        z.writestr("verify.py", evx.VERIFY_PY)
        z.writestr("README.txt", f"Release copy of evidence package {src_manifest['export_id']} (case {src_manifest['case_ref']}).\n\n"
                                 f"{RELEASE_NOTE}\n\nSource manifest sha256: {src_hash}\nCheck: python3 verify.py\n")
    tmp.replace(dest)
    dest.chmod(0o440)
    psha, pbytes = evx.sha256_file(dest)
    return {"package_sha256": psha, "package_bytes": pbytes, "manifest_sha256": hashlib.sha256(mbytes).hexdigest(), "summary": summary}

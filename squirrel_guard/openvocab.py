"""Open-vocabulary objects on a person event: "carrying a ladder", "backpack", "knife near the person".

Model: YOLO-World v2 (yolov8s-worldv2) exported by scripts/export_openvocab.py with the vocabulary baked in, run
with onnxruntime on CPU. The text side (CLIP) is computed at export time only, so the runtime is a plain YOLO
detector whose classes are text prompts. Several prompts can report one label ("crowbar" and "pry bar" -> "pry bar")
and some prompts are distractors ("coat", "lamp", "person"): every box takes its single best-scoring prompt, and a
box won by a distractor is dropped. That is the main false-positive control on top of the score threshold.

Three steps:
  OpenVocabDetector.detect(frame)       -> [{"label", "score", "box", "prompt"}]          one frame
  relation(obj_box, person_pose_or_box) -> "carried" | "near" | "scene"                   one object vs the person
  ObjectAggregator.add(...) / summary() -> per label: max score, frames seen, carried fraction  over a clip
summary() keeps only labels seen on >= min_frames sampled frames (single-frame hits are mostly noise), and never
calls a static object "carried" (the person walked past it while it stayed put: a chair or railing touched in passing).

detect_around(frame, person_box) runs on a square window around the tracked person (like PoseEstimator.in_region):
a person 300 px tall in a 1920x1080 frame is ~100 px in a 640 letterbox of the whole frame, too small to see a
crowbar; the window gives the hands ~3x the pixels at the same cost. Objects far from the person are not looked for.

Change the vocabulary: edit scripts/openvocab_vocab.txt (or pass --vocab) and re-export:
    .venv/bin/python scripts/export_openvocab.py --out models/yolov8s-worldv2-sg.onnx
then copy the ONNX file to gpuserver's model cache (deploy/gpuserver/compose.yaml mounts it). The runtime cannot add words
(the text embeddings are inside the graph); [profiler] openvocab_classes only narrows what is reported.

Licences: Ultralytics code/weights AGPL-3.0, YOLO-World GPL-3.0 (Tencent AILab), CLIP (export time only) MIT.
See docs/openvocab.md for measurements.
"""
from __future__ import annotations

import ast
import json

import cv2
import numpy as np
import onnxruntime as ort

L_WR, R_WR = 9, 10   # COCO wrist keypoints (pose.py)


class OpenVocabDetector:
    def __init__(self, model_path: str, classes=None, min_score: float = 0.25, iou: float = 0.5, cross_iou: float = 0.7,
                 threads: int = 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.inp = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        meta = self.sess.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(meta["names"]) if "names" in meta else {}
        self.prompts = [names.get(i, str(i)) for i in range(len(names))]
        # reported label per prompt (None = distractor); a plain Ultralytics export reports every prompt as itself
        self.labels: list[str | None] = json.loads(meta["sg_labels"]) if "sg_labels" in meta else list(self.prompts)
        wanted = {c.strip().lower() for c in (classes or []) if c.strip()}
        if wanted:
            unknown = wanted - {lab.lower() for lab in self.labels if lab}
            if unknown:
                raise ValueError(f"openvocab classes not in the model's vocabulary: {sorted(unknown)} "
                                 f"(re-export with scripts/export_openvocab.py)")
        self.report = np.array([lab is not None and (not wanted or lab.lower() in wanted) for lab in self.labels])
        self.min_score, self.iou, self.cross_iou = float(min_score), float(iou), float(cross_iou)

    @property
    def vocabulary(self) -> list[str]:
        return sorted({lab for lab, r in zip(self.labels, self.report) if r})

    def _infer(self, img: np.ndarray) -> np.ndarray:
        """-> raw (anchors, 4 + n_prompts) for one image, boxes still in letterbox pixels; plus the transform."""
        h, w = img.shape[:2]
        s = self.size / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        dy, dx = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.inp: x})[0][0].T
        return out, s, dx, dy

    def detect(self, frame: np.ndarray, offset=(0, 0)) -> list[dict]:
        """One BGR frame -> [{"label", "score", "box": (x1, y1, x2, y2) frame pixels, "prompt"}], best first."""
        out, s, dx, dy = self._infer(frame)
        nc = len(self.labels)
        scores = out[:, 4:4 + nc]
        best = scores.argmax(1)                      # every box competes across ALL prompts, distractors included
        conf = scores[np.arange(len(best)), best]
        keep = (conf >= self.min_score) & self.report[best]
        if not keep.any():
            return []
        xywh, best, conf = out[keep, :4], best[keep], conf[keep]
        x1 = (xywh[:, 0] - xywh[:, 2] / 2 - dx) / s
        y1 = (xywh[:, 1] - xywh[:, 3] / 2 - dy) / s
        boxes = np.stack([x1, y1, xywh[:, 2] / s, xywh[:, 3] / s], 1)
        # NMS per reported LABEL (two prompts of one label on the same box are one object)
        lab_ids = {lab: i for i, lab in enumerate(sorted({self.labels[b] for b in best}))}
        groups = [lab_ids[self.labels[b]] for b in best]
        idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), conf.tolist(), groups, self.min_score, self.iou)
        # then across labels for near-identical boxes: one object is "backpack" OR "bag", not both
        idx = np.array(idx, int).reshape(-1)
        if len(idx) > 1:
            keep2 = cv2.dnn.NMSBoxes(boxes[idx].tolist(), conf[idx].tolist(), self.min_score, self.cross_iou)
            idx = idx[np.array(keep2, int).reshape(-1)]
        H, W = frame.shape[:2]
        ox, oy = offset
        res = []
        for i in idx:
            x, y, bw, bh = (float(v) for v in boxes[i])   # plain floats: the summary goes into JSON
            res.append({"label": self.labels[best[i]], "score": round(float(conf[i]), 3), "prompt": self.prompts[best[i]],
                        "box": (round(max(0., x) + ox, 1), round(max(0., y) + oy, 1),
                                round(min(W - 1., x + bw) + ox, 1), round(min(H - 1., y + bh) + oy, 1))})
        return sorted(res, key=lambda d: -d["score"])

    def detect_around(self, frame: np.ndarray, person_box, grow: float = 2.2, min_side: int = 320) -> list[dict]:
        """Run on a square window around the person box (grown), boxes returned in full-frame pixels.
        min_side keeps a tiny distant person from being blown up into a blur."""
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = person_box
        side = max(max(x2 - x1, y2 - y1) * grow, min_side)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        X1, Y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        X2, Y2 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
        if X2 - X1 < 32 or Y2 - Y1 < 32:
            return []
        return self.detect(frame[Y1:Y2, X1:X2], offset=(X1, Y1))


# ---------------------------------------------------------------- association with the tracked person
def _inter(a, b) -> float:
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def _gap(a, b) -> float:
    """Distance between two boxes (0 when they touch or overlap)."""
    dx = max(0., max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0., max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def relation(obj_box, person, wrist_conf: float = 0.3, near: float = 0.5) -> str:
    """How an object relates to the tracked person.
    person: a pose.Pose (box + COCO keypoints) or a plain (x1, y1, x2, y2) box, or None.
      "carried"  a confident wrist is in/at the object box (grown by 10 % of the person's height), or the object
                 lies mostly inside the person box (worn backpack, box held against the body)
      "near"     within `near` x person height of the person box
      "scene"    anywhere else, or no person on this frame
    The overlap test alone would call a ladder leaning on the wall behind someone "carried"; the wrist test is the
    stronger one, so for a pose the overlap rule needs the object to be small relative to the person too."""
    if person is None:
        return "scene"
    pbox = getattr(person, "box", person)
    kp = getattr(person, "kp", None)
    ph = max(1., pbox[3] - pbox[1])
    ob = obj_box
    oarea = max(1., (ob[2] - ob[0]) * (ob[3] - ob[1]))
    if kp is not None:
        m = 0.10 * ph
        for w in (L_WR, R_WR):
            x, y, c = kp[w]
            if c >= wrist_conf and ob[0] - m <= x <= ob[2] + m and ob[1] - m <= y <= ob[3] + m:
                return "carried"
    ioa = _inter(ob, pbox) / oarea
    parea = max(1., (pbox[2] - pbox[0]) * ph)
    if ioa >= 0.6 and (kp is None or oarea <= 0.6 * parea):
        return "carried"
    return "near" if _gap(ob, pbox) <= near * ph else "scene"


class ObjectAggregator:
    """Per-clip summary of the per-frame detections. Memory: one small dict per label, never frames."""

    def __init__(self, min_frames: int = 2, carried_fraction: float = 0.3, static_ratio: float = 0.25):
        self.min_frames, self.carried_fraction = int(min_frames), float(carried_fraction)
        self.static_ratio = float(static_ratio)
        self.frames = 0
        self.t: dict[str, dict] = {}

    def add(self, dets: list[dict], person=None, t: float | None = None, size=None) -> None:
        """One SAMPLED frame: its detections (any number, incl. none) and the tracked person (Pose/box/None).
        A label counts once per frame (its best box); `size` (W, H) stores the best box normalised to 0-1."""
        self.frames += 1
        best: dict[str, dict] = {}
        for d in dets:
            if d["label"] not in best or d["score"] > best[d["label"]]["score"]:
                best[d["label"]] = d
        for lab, d in best.items():
            rel = relation(d["box"], person)
            e = self.t.setdefault(lab, {"label": lab, "score": 0.0, "frames": 0, "carried": 0, "near": 0,
                                        "first_t": t, "box": None, "prompts": set(), "track": []})
            e["frames"] += 1
            e["carried"] += rel == "carried"
            e["near"] += rel == "near"
            e["prompts"].add(d.get("prompt") or lab)
            if person is not None:   # object centre, person centre, person height (for the static test)
                pb, ob = getattr(person, "box", person), d["box"]
                e["track"].append(((ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2, (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2,
                                   pb[3] - pb[1]))
            if d["score"] > e["score"]:
                e["score"], e["best_t"], e["best_rel"] = d["score"], t, rel
                if size:
                    W, H = size
                    e["box"] = [round(d["box"][0] / W, 3), round(d["box"][1] / H, 3),
                                round(d["box"][2] / W, 3), round(d["box"][3] / H, 3)]

    def is_static(self, track: list) -> bool:
        """A fixture, not a carried thing: the person moved (> half a body height) while the object stayed put
        (moved < static_ratio of that). Catches a chair/railing that touches a passing hand on a few frames."""
        if len(track) < 2:
            return False
        a = np.asarray(track, np.float32)
        obj = np.ptp(a[:, 0:2], 0).max()
        per = np.ptp(a[:, 2:4], 0).max()
        return bool(per > 0.5 * np.median(a[:, 4]) and obj < self.static_ratio * per)

    def summary(self) -> list[dict]:
        """-> [{"label", "score" (max), "frames", "sampled", "carried_frames", "carried_fraction", "carried",
        "relation" (carried | near | scene: the majority view), "first_t", "best_t", "box", "static", "prompts"}],
        best first; only labels seen on >= min_frames sampled frames. A static object (is_static) is never carried."""
        out = []
        for e in self.t.values():
            if e["frames"] < self.min_frames:
                continue
            cf = e["carried"] / e["frames"]
            static = self.is_static(e["track"])
            carried = e["carried"] >= self.min_frames and cf >= self.carried_fraction and not static
            rel = "carried" if carried else "near" if (e["carried"] + e["near"]) * 2 >= e["frames"] else "scene"
            out.append({"label": e["label"], "score": round(e["score"], 3), "frames": e["frames"], "sampled": self.frames,
                        "carried_frames": e["carried"], "carried_fraction": round(cf, 2), "carried": carried,
                        "relation": rel, "first_t": e["first_t"], "best_t": e.get("best_t"), "box": e["box"],
                        "static": static, "prompts": sorted(e["prompts"])})
        return sorted(out, key=lambda o: (not o["carried"], -o["score"]))


def describe(objects: list[dict]) -> str:
    """One readable phrase: 'carrying a ladder (0.41); backpack nearby (0.33)'."""
    parts = []
    for o in objects:
        verb = "carrying" if o["relation"] == "carried" else ""
        where = {"near": " nearby", "scene": " in view"}.get(o["relation"], "")
        parts.append(f"{verb + ' ' if verb else ''}{o['label']}{where} ({o['score']:.2f})")
    return "; ".join(parts)

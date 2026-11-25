"""PaddleClas PULC attribute classifiers (ONNX, CPU): pedestrian attributes and vehicle colour/type.

Models come from scripts/export_pulc.py (PaddleClas release/2.6 inference models, Apache-2.0, converted with
paddle2onnx). The label order, input size and normalisation are read from the ONNX metadata the export writes, so a
model cannot be paired with the wrong label list.

  PersonAttributes  PPLCNet_x1_0 trained on PA100K: 26 sigmoid outputs (hat, glasses, sleeves, upper/lower clothing
                    style, boots, bags, holding something, age band, female, front/side/back).
  VehicleAttributes PPLCNet_x1_0 trained on VeRi-776: 10 colours + 9 body types (sigmoid each).

Preprocessing is PaddleClas' own (deploy/configs/PULC/*/inference_*.yaml): plain resize (no letterbox) to 192x256
(person) / 256x192 (vehicle) with cv2 bilinear, BGR->RGB, /255, ImageNet mean/std, CHW.

Policy: everything here is an APPARENT, per-image estimate. Age band and gender are PA100K's annotators' guesses,
learned by a model; they are never identity and are labelled "apparent" wherever they are shown. PA100K has no
ethnicity/skin attribute and nothing here infers one.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

ORIENT = {"Front": "front", "Side": "side", "Back": "back"}
AGE = {"AgeLess18": "under_18", "Age18-60": "18_60", "AgeOver60": "over_60"}
# PA100K/PaddleClas label -> readable tag. Sleeves are a pair (PaddleClas: long if LongSleeve > ShortSleeve).
PERSON_TAGS = {"Hat": "hat", "Glasses": "glasses", "UpperStride": "striped_top", "UpperLogo": "logo_top",
               "UpperPlaid": "plaid_top", "UpperSplice": "colour_block_top", "LowerStripe": "striped_bottoms",
               "LowerPattern": "patterned_bottoms", "LongCoat": "long_coat", "Trousers": "trousers", "Shorts": "shorts",
               "Skirt&Dress": "skirt_or_dress", "Boots": "boots", "HandBag": "handbag", "ShoulderBag": "shoulder_bag",
               "Backpack": "backpack", "HoldObjectsInFront": "holding_something"}
VEHICLE_COLOURS = ["yellow", "orange", "green", "gray", "red", "blue", "white", "golden", "brown", "black"]
VEHICLE_TYPES = ["sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus", "truck", "estate"]


class _Model:
    kind = ""

    def __init__(self, path: str, threads: int = 2, batch: int = 8):
        import onnxruntime as ort
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        meta = self.sess.get_modelmeta().custom_metadata_map
        if self.kind and meta.get("sg_kind") != self.kind:
            raise ValueError(f"{path}: sg_kind {meta.get('sg_kind')!r}, expected {self.kind!r} (run scripts/export_pulc.py)")
        self.labels: list[str] = json.loads(meta["sg_labels"])
        self.wh = tuple(json.loads(meta["sg_input_wh"]))
        self.mean = np.array(json.loads(meta.get("sg_mean", "null")) or MEAN, np.float32)
        self.std = np.array(json.loads(meta.get("sg_std", "null")) or STD, np.float32)
        self.thresholds: dict = json.loads(meta.get("sg_thresholds", '{"default": 0.5}'))
        self.thresholds.setdefault("default", 0.5)
        self.batch = batch

    def prep(self, crop_bgr: np.ndarray) -> np.ndarray:
        x = cv2.resize(crop_bgr, self.wh, interpolation=cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32) / 255.0
        return ((x - self.mean) / self.std).transpose(2, 0, 1)

    def raw(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """(N, n_labels) sigmoid probabilities."""
        out = []
        for i in range(0, len(crops_bgr), self.batch):
            x = np.stack([self.prep(c) for c in crops_bgr[i:i + self.batch]]).astype(np.float32)
            out.append(self.sess.run(None, {self.inp: x})[0])
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)), np.float32)

    def probs(self, crops_bgr: list[np.ndarray]) -> list[dict]:
        """Per crop: {label: probability}."""
        return [{lab: float(p) for lab, p in zip(self.labels, row)} for row in self.raw(crops_bgr)]


class PersonAttributes(_Model):
    kind = "pulc_person_attribute"

    def visit(self, crops_bgr: list[np.ndarray], weights=None, max_crops: int = 6) -> dict | None:
        """Best `max_crops` crops (by weight, e.g. detector score x height) -> aggregated visit summary."""
        if not crops_bgr:
            return None
        order = np.argsort(-np.asarray(weights, float))[:max_crops] if weights is not None else range(min(len(crops_bgr), max_crops))
        return aggregate_person(self.probs([crops_bgr[i] for i in order]), self.thresholds)


class VehicleAttributes(_Model):
    kind = "pulc_vehicle_attribute"

    def visit(self, crops_bgr: list[np.ndarray], weights=None, max_crops: int = 6) -> dict | None:
        if not crops_bgr:
            return None
        order = np.argsort(-np.asarray(weights, float))[:max_crops] if weights is not None else range(min(len(crops_bgr), max_crops))
        return aggregate_vehicle(self.probs([crops_bgr[i] for i in order]), self.thresholds)


def _thr(thresholds: dict | None, label: str) -> float:
    t = thresholds or {}
    return float(t.get(label, t.get("default", 0.5)))


def person_tags(probs: dict, thresholds: dict | None = None) -> list[str]:
    """Readable tags for one probability dict (a crop or a visit mean), PaddleClas' decision rules."""
    tags = []
    if "LongSleeve" in probs and "ShortSleeve" in probs and max(probs["LongSleeve"], probs["ShortSleeve"]) >= 0.3:
        tags.append("long_sleeves" if probs["LongSleeve"] > probs["ShortSleeve"] else "short_sleeves")
    for lab, tag in PERSON_TAGS.items():
        if lab in probs and probs[lab] > _thr(thresholds, lab):
            tags.append(tag)
    return tags


def aggregate_person(per_crop: list[dict], thresholds: dict | None = None) -> dict | None:
    """Visit summary from per-crop probabilities.

    attributes   mean probability per PULC label over the crops (orientation/age/gender excluded: see below)
    orientation  front/side/back distribution: mean of each crop's normalised Front/Side/Back scores
    views        how many crops were mostly front/side/back (argmax)
    apparent     {"gender_female": mean p, "age": {"under_18", "18_60", "over_60"} normalised} - estimates only
    tags         readable tags (PaddleClas thresholds on the mean probabilities)
    spread       max per-label standard deviation (high = the crops disagree: occlusion, turning, bad crops)
    """
    if not per_crop:
        return None
    labels = list(per_crop[0])
    P = np.array([[c[lab] for lab in labels] for c in per_crop], np.float64)
    mean = dict(zip(labels, P.mean(0)))
    out: dict = {"model": "pulc_person_attribute", "crops": len(per_crop)}
    o = [lab for lab in ORIENT if lab in mean]
    if o:
        idx = [labels.index(lab) for lab in o]
        O = P[:, idx] / np.maximum(P[:, idx].sum(1, keepdims=True), 1e-6)
        out["orientation"] = {ORIENT[lab]: round(float(v), 3) for lab, v in zip(o, O.mean(0))}
        am = O.argmax(1)
        out["views"] = {ORIENT[lab]: int((am == k).sum()) for k, lab in enumerate(o)}
    app: dict = {}
    if "Female" in mean:
        app["gender_female"] = round(float(mean["Female"]), 3)
    a = [lab for lab in AGE if lab in mean]
    if a:
        s = sum(mean[lab] for lab in a) or 1e-6
        app["age"] = {AGE[lab]: round(float(mean[lab] / s), 3) for lab in a}
    if app:
        out["apparent"] = app
    skip = set(ORIENT) | set(AGE) | {"Female"}
    out["attributes"] = {lab: round(float(v), 3) for lab, v in mean.items() if lab not in skip}
    out["tags"] = person_tags(out["attributes"], thresholds)
    out["spread"] = round(float(P.std(0).max()), 3) if len(P) > 1 else 0.0
    return out


def aggregate_vehicle(per_crop: list[dict], thresholds: dict | None = None) -> dict | None:
    """Visit summary: mean colour/type probabilities, the top colour and type if above threshold (else None)."""
    if not per_crop:
        return None
    labels = list(per_crop[0])
    mean = {lab: float(np.mean([c[lab] for c in per_crop])) for lab in labels}
    cols = {c: mean[c] for c in VEHICLE_COLOURS if c in mean}
    typs = {t: mean[t] for t in VEHICLE_TYPES if t in mean}
    top_c = max(cols, key=cols.get) if cols else None
    top_t = max(typs, key=typs.get) if typs else None
    return {"model": "pulc_vehicle_attribute", "crops": len(per_crop),
            "color": top_c if top_c and cols[top_c] >= _thr(thresholds, top_c) else None,
            "type": top_t if top_t and typs[top_t] >= _thr(thresholds, top_t) else None,
            "colors": {k: round(v, 3) for k, v in sorted(cols.items(), key=lambda kv: -kv[1])[:3]},
            "types": {k: round(v, 3) for k, v in sorted(typs.items(), key=lambda kv: -kv[1])[:3]}}


def describe_person(v: dict | None) -> str:
    """One short line for logs/cards: 'apparent: backpack, long sleeves, trousers; mostly back view'."""
    if not v:
        return ""
    parts = [t.replace("_", " ") for t in v.get("tags", [])]
    o = v.get("orientation") or {}
    s = "apparent: " + (", ".join(parts) or "no clear attributes")
    if o:
        k = max(o, key=o.get)
        s += f"; mostly {k} view ({o[k] * 100:.0f}%)"
    return s

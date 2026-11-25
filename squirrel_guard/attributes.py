"""Structured person attributes from the local vision LLM (Qwen3.6 on gpuserver), as JSON.

Age and gender are APPARENT estimates from one image, with the model's own confidence; they are guesses, never
identity. No race/ethnicity is asked for. Nothing is looked up outside this system.
"""
from __future__ import annotations

import base64
import json
import logging
import re

import cv2
import numpy as np
import requests

log = logging.getLogger(__name__)

SCHEMA = {
    "type": "object",
    "properties": {
        "upper_clothing": {"type": "string"}, "upper_color": {"type": "string"},
        "lower_clothing": {"type": "string"}, "lower_color": {"type": "string"},
        "footwear": {"type": "string"}, "headwear": {"type": "string"},
        "accessories": {"type": "array", "items": {"type": "string"}},
        "carried_items": {"type": "array", "items": {"type": "string"}},
        "hair": {"type": "string"}, "build": {"type": "string", "enum": ["slim", "average", "heavy", "unclear"]},
        "apparent_age_low": {"type": "integer"}, "apparent_age_high": {"type": "integer"},
        "apparent_gender": {"type": "string", "enum": ["male", "female", "unclear"]},
        "gender_confidence": {"type": "number"},
        "activity": {"type": "string"}, "facing": {"type": "string", "enum": ["toward", "away", "side", "unclear"]},
        "visible_fraction": {"type": "string", "enum": ["full_body", "upper_body", "partial", "unclear"]},
        "description": {"type": "string"},
    },
    "required": ["upper_clothing", "upper_color", "lower_clothing", "lower_color", "apparent_age_low",
                 "apparent_age_high", "apparent_gender", "gender_confidence", "build", "activity", "description"],
}

PROMPT = (
    "You are describing a person in a home security camera crop for a private visitor log. Describe only what is "
    "visible. Use 'unclear' or 'none' when something cannot be seen. Give the APPARENT age as a range (e.g. 25-35) "
    "and APPARENT gender with a confidence between 0 and 1; these are estimates, not facts. Do not guess names, "
    "identity, race or ethnicity. 'description' is one short factual sentence. Reply with JSON only."
)


def describe(url: str, crop_bgr: np.ndarray, timeout: float = 45, max_side: int = 512, model: str = "") -> dict | None:
    h, w = crop_bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        crop_bgr = cv2.resize(crop_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    body = {
        "model": model or "local", "temperature": 0, "max_tokens": 400,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()}},
            {"type": "text", "text": PROMPT}]}],
        "response_format": {"type": "json_object", "schema": SCHEMA},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = requests.post(url.rstrip("/") + "/v1/chat/completions", json=body, headers=_headers(), timeout=timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"] or ""
    except Exception as e:
        log.warning("attribute LLM unavailable: %s", e)
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        a = json.loads(m.group(0) if m else text)
    except ValueError:
        log.info("attribute LLM returned non-JSON: %r", text[:120])
        return None
    # clamp the guesses into sane shapes
    lo, hi = a.get("apparent_age_low"), a.get("apparent_age_high")
    if isinstance(lo, int) and isinstance(hi, int) and lo > hi:
        a["apparent_age_low"], a["apparent_age_high"] = hi, lo
    try:
        a["gender_confidence"] = max(0.0, min(1.0, float(a.get("gender_confidence", 0))))
    except (TypeError, ValueError):
        a["gender_confidence"] = 0.0
    return a


VEHICLE_SCHEMA = {
    "type": "object",
    "properties": {
        "body_type": {"type": "string", "enum": ["sedan", "hatchback", "suv", "pickup", "van", "minivan", "coupe",
                                                  "wagon", "convertible", "truck", "bus", "motorcycle", "other", "unclear"]},
        "color": {"type": "string"},
        "apparent_make": {"type": "string"}, "apparent_model": {"type": "string"},
        "make_model_confidence": {"type": "number"},
        "apparent_era": {"type": "string"},
        "distinguishing_features": {"type": "array", "items": {"type": "string"}},
        "plate_visible_text": {"type": "string"},
        "activity": {"type": "string", "enum": ["parked", "arriving", "leaving", "passing", "unclear"]},
        "description": {"type": "string"},
    },
    "required": ["body_type", "color", "apparent_make", "make_model_confidence", "distinguishing_features",
                 "activity", "description"],
}

VEHICLE_PROMPT = (
    "You are describing a vehicle in a home security camera crop for a private log. Describe only what is "
    "visible. Give the APPARENT make/model with a confidence 0-1 ('unclear' if you cannot tell). List distinguishing "
    "features that would help recognise this exact vehicle again (damage, stickers, roof rack, aftermarket wheels, "
    "tinted windows, missing hubcap). 'plate_visible_text' is the plate text only if clearly legible, else ''. "
    "Do not guess the owner or identity. Reply with JSON only."
)


def describe_vehicle(url: str, crop_bgr: np.ndarray, timeout: float = 45, max_side: int = 640, model: str = "") -> dict | None:
    return _ask(url, crop_bgr, VEHICLE_PROMPT, VEHICLE_SCHEMA, timeout, max_side, model)


def _headers() -> dict:
    # profiler model key; falls back to the verifier's key (the profiler falls back to the verifier's address too)
    from .verifier import auth_headers
    return auth_headers("SG_LLM_API_KEY", "SG_VERIFIER_API_KEY")


def _ask(url, crop_bgr, prompt, schema, timeout, max_side, model=""):
    h, w = crop_bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        crop_bgr = cv2.resize(crop_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    body = {"model": model or "local", "temperature": 0, "max_tokens": 400,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()}},
                {"type": "text", "text": prompt}]}],
            "response_format": {"type": "json_object", "schema": schema},
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        r = requests.post(url.rstrip("/") + "/v1/chat/completions", json=body, headers=_headers(), timeout=timeout)
        r.raise_for_status()
        text = re.sub(r"<think>.*?</think>", "", r.json()["choices"][0]["message"]["content"] or "", flags=re.S).strip()
        m = re.search(r"\{.*\}", text, flags=re.S)
        return json.loads(m.group(0) if m else text)
    except Exception as e:
        log.warning("vehicle attribute LLM failed: %s", e)
        return None

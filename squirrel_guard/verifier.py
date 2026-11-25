"""Second opinion from a local vision LLM (llama.cpp server, OpenAI-compatible API) on a detection crop."""
from __future__ import annotations

import base64
import logging
import os
import re

import cv2
import numpy as np
import requests

log = logging.getLogger(__name__)

LABELS = ["squirrel", "cat", "dog", "bird", "rabbit", "raccoon", "person", "other", "nothing"]

PROMPT = (
    "This is a crop from an outdoor security camera. What animal or object is the main subject? "
    f"Answer with exactly one word from this list: {', '.join(LABELS)}."
)


def auth_headers(*env_names: str) -> dict:
    """Authorization header for an OpenAI-compatible server started with an API key (llama.cpp --api-key):
    the first of env_names that is set. Keys live in the private .env, never in config.toml."""
    for name in env_names:
        key = os.environ.get(name, "").strip()
        if key:
            return {"Authorization": f"Bearer {key}"}
    return {}


class Verifier:
    def __init__(self, url: str, model: str = "local", timeout: float = 60.0, pad: float = 0.35,
                 max_side: int = 384, headers: dict | None = None):
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.model = model or "local"     # llama.cpp ignores it; model routers and other servers need the real name
        self.headers = auth_headers("SG_VERIFIER_API_KEY") if headers is None else headers
        self.timeout = timeout
        self.pad = pad
        self.max_side = max_side  # image tokens (and CPU vision-encoder time) scale with pixel count

    def crop(self, frame: np.ndarray, box) -> np.ndarray:
        x1, y1, x2, y2 = box
        H, W = frame.shape[:2]
        pw, ph = int((x2 - x1) * self.pad), int((y2 - y1) * self.pad)
        return frame[max(0, y1 - ph):min(H, y2 + ph), max(0, x1 - pw):min(W, x2 + pw)]

    def __call__(self, image: np.ndarray) -> str | None:
        """Return one of LABELS, or None if the server is unreachable or the answer is unusable."""
        h, w = image.shape[:2]
        if max(h, w) > self.max_side:
            s = self.max_side / max(h, w)
            image = cv2.resize(image, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return None
        data_url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": PROMPT},
            ]}],
            # Qwen3.x: skip the thinking phase; we only want the label
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            r = requests.post(self.url, json=body, headers=self.headers, timeout=self.timeout)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:  # server down, model loading, bad JSON
            log.warning("verifier unavailable: %s", e)
            return None
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip().lower()
        for label in LABELS:
            if re.search(rf"\b{label}s?\b", text):
                return label
        log.info("verifier gave unusable answer: %r", text[:80])
        return None

"""Read the straightened page with the local vision LLM (gpuserver Qwen 35B, OpenAI-compatible API) into structured
notes, tasks and reminders. Nothing leaves the LAN."""
from __future__ import annotations

import base64
import datetime as dt
import json
import re
import time
import urllib.request

import cv2
import numpy as np

PROMPT = """This is a straightened scan of a large easel pad with handwritten marker notes in ruled boxes and columns.
Faint grey marks are bleed-through from the sheet underneath: ignore them. The writer's vocabulary often includes:
{vocab}.
Today is {today} ({weekday}); resolve relative dates ("tmrw", "Fri", "next week") against today.

Transcribe exactly, keeping each box/column separate, reading top to bottom. Join wrapped lines of one sentence.
REMINDERS are ONLY: lines in a box whose heading is exactly "Reminders" or "Reminder" (a mixed heading such as
"Notes/reminders" does NOT count); lines starting with "R:" or "R -"; lines next to a drawn bell or a boxed letter R.
If none of these appear, "reminders" is []. Give a due date/time only if one is written.
TASKS are lines with a checkbox or bullet that read like to-dos; done = ticked or crossed out.
Return JSON only, no prose:
{{"date_written": str|null,
  "sections": [{{"heading": str, "position": "left|middle|right top|middle|bottom", "lines": [str]}}],
  "reminders": [{{"text": str, "due": "YYYY-MM-DD" | "YYYY-MM-DDTHH:MM" | null, "marker": "heading|R:|bell"}}],
  "tasks": [{{"text": str, "done": bool}}],
  "drawings": [str],
  "unreadable": [str]}}
Never guess words you cannot read; put them in "unreadable" and write [?] in the line."""


def _png_b64(gray_or_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", gray_or_bgr)
    return base64.b64encode(buf.tobytes()).decode()


def read_page(page: np.ndarray, llm_url: str, vocab: list[str], today: dt.date | None = None,
              timeout: float = 1200, upscale: float = 1.0, think: bool = False) -> tuple[dict, dict]:
    today = today or dt.date.today()
    img = cv2.resize(page, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC) if upscale != 1 else page
    prompt = PROMPT.format(vocab=", ".join(vocab) or "(none)", today=today.isoformat(), weekday=today.strftime("%A"))
    # the server caps an image at ~1024 tokens, so the native ~12 px/cm page already uses the full budget;
    # thinking off: it was ~60 % of the output tokens and most of the 4-minute read time
    body = {"model": "local", "temperature": 0, "max_tokens": 4000,
            "chat_template_kwargs": {"enable_thinking": think},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _png_b64(img)}}]}]}
    t0 = time.time()
    req = urllib.request.Request(llm_url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    msg = resp["choices"][0]["message"]
    text = msg.get("content") or ""
    meta = {"seconds": round(time.time() - t0, 1), "usage": resp.get("usage"), "model": resp.get("model")}
    return parse_json(text), meta


def parse_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S) or re.search(r"(\{.*\})", text, re.S)
    if not m:
        raise ValueError("no JSON in the model output: " + text[:200])
    d = json.loads(m.group(1))
    for k in ("sections", "reminders", "tasks", "drawings", "unreadable"):
        d.setdefault(k, [])
    d["reminders"] = [r for r in d["reminders"] if keep_reminder(r, d["sections"])]
    for r in d["reminders"]:
        r["due"] = normalize_due(r.get("due"))
    return d


def keep_reminder(r: dict, sections: list[dict]) -> bool:
    """Code-side check of the model's reminder list: a marker it claims must exist on the page."""
    text = (r.get("text") or "").strip()
    if not text:
        return False
    if re.match(r"^\s*R\s*[:\-]", text) or r.get("marker") == "bell":
        return True
    for s in sections:
        if re.fullmatch(r"\s*reminders?\s*:?\s*", (s.get("heading") or ""), re.I):
            if any(text.lower()[:20] in ln.lower() or ln.lower()[:20] in text.lower() for ln in s.get("lines", [])):
                return True
    return False


def normalize_due(due) -> str | None:
    if not due:
        return None
    s = str(due).strip()
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            v = dt.datetime.strptime(s[:16], fmt)
            return v.strftime("%Y-%m-%dT%H:%M") if "H" in fmt else v.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

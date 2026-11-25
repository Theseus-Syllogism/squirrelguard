"""Scan outputs: page image, searchable PDF (page + typed transcript), Markdown, JSON, reminders .ics, ntfy push."""
from __future__ import annotations

import datetime as dt
import json
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np


def markdown(d: dict, when: dt.datetime) -> str:
    out = [f"# Easel notes {when:%Y-%m-%d %H:%M}", ""]
    if d.get("date_written"):
        out += [f"*Written:* {d['date_written']}", ""]
    for s in d.get("sections", []):
        out.append(f"## {s.get('heading') or '(untitled)'} *({s.get('position', '')})*")
        out += [f"- {ln}" for ln in s.get("lines", [])] + [""]
    if d.get("reminders"):
        out.append("## Reminders")
        out += [f"- [ ] {r['text']}" + (f" — **{r['due']}**" if r.get("due") else "") for r in d["reminders"]]
        out.append("")
    if d.get("tasks"):
        out.append("## Tasks")
        out += [f"- [{'x' if t.get('done') else ' '}] {t['text']}" for t in d["tasks"]] + [""]
    if d.get("drawings"):
        out += ["## Drawings"] + [f"- {x}" for x in d["drawings"]] + [""]
    if d.get("unreadable"):
        out += ["## Could not read"] + [f"- {x}" for x in d["unreadable"]] + [""]
    return "\n".join(out)


def pdf(path: Path, page_gray: np.ndarray, md: str) -> None:
    """Page 1: the scanned page. Page 2+: the transcript as real text (searchable, copyable)."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    ok, png = cv2.imencode(".png", page_gray)
    import io
    img = ImageReader(io.BytesIO(png.tobytes()))
    W, H = letter
    c = canvas.Canvas(str(path), pagesize=letter)
    h, w = page_gray.shape[:2]
    s = min((W - inch) / w, (H - inch) / h)
    c.drawImage(img, (W - w * s) / 2, (H - h * s) / 2, w * s, h * s)
    c.showPage()
    y = H - 0.75 * inch
    for line in md.splitlines():
        size = 14 if line.startswith("# ") else 12 if line.startswith("## ") else 10
        font = "Helvetica-Bold" if line.startswith("#") else "Helvetica"
        text = line.lstrip("# ").replace("**", "").replace("*", "")
        c.setFont(font, size)
        for chunk in [text[i:i + 95] for i in range(0, max(1, len(text)), 95)]:
            if y < 0.75 * inch:
                c.showPage(); y = H - 0.75 * inch; c.setFont(font, size)
            c.drawString(0.75 * inch, y, chunk)
            y -= size * 1.4
    c.save()


def ics(reminders: list[dict], when: dt.datetime, source: str) -> str | None:
    """VTODO per reminder (DUE when written, else none) with a 15-minute alarm for timed ones."""
    if not reminders:
        return None
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//camstack//easelscan//EN"]
    stamp = when.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for r in reminders:
        lines += ["BEGIN:VTODO", f"UID:{uuid.uuid5(uuid.NAMESPACE_URL, source + r['text'])}@easelscan",
                  f"DTSTAMP:{stamp}", f"SUMMARY:{r['text'][:200]}", f"DESCRIPTION:From the easel scan {source}"]
        due = r.get("due")
        if due and "T" in due:
            lines += [f"DUE;TZID=America/Denver:{due.replace('-', '').replace(':', '')}00",
                      "BEGIN:VALARM", "ACTION:DISPLAY", f"DESCRIPTION:{r['text'][:200]}", "TRIGGER:-PT15M", "END:VALARM"]
        elif due:
            lines.append(f"DUE;VALUE=DATE:{due.replace('-', '')}")
        lines.append("END:VTODO")
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def write_scan(out_dir: Path, when: dt.datetime, page_gray: np.ndarray, d: dict, meta: dict) -> Path:
    sd = out_dir / when.strftime("%Y/%m/%Y-%m-%d_%H%M")
    sd.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sd / "page.png"), page_gray)
    md = markdown(d, when)
    (sd / "notes.md").write_text(md)
    (sd / "scan.json").write_text(json.dumps({"scanned": when.isoformat(), "read": meta, **d}, indent=1))
    pdf(sd / "notes.pdf", page_gray, md)
    cal = ics(d.get("reminders", []), when, sd.name)
    if cal:
        (sd / "reminders.ics").write_text(cal)
    latest = out_dir / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(sd.relative_to(out_dir))
    return sd


def ntfy(url: str, token: str | None, title: str, body: str, tags: str = "memo") -> None:
    if not url:
        return
    h = {"Title": title, "Tags": tags}
    if token:
        h["Authorization"] = f"Bearer {token}"
    urllib.request.urlopen(urllib.request.Request(url, body.encode(), h), timeout=15).read()

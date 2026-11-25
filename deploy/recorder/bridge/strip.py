"""Match strip: this sighting's crop next to the 3 most similar earlier sightings it was compared with, each with
its similarity band, so a person can judge "is this really the same person?" at a glance."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TILE_W, TILE_H, CAP = 180, 300, 46
BG, FG, MUTED = (30, 26, 24), (242, 238, 235), (178, 166, 160)
BAND_COL = {"strong": (60, 200, 110), "likely": (120, 200, 90), "possible": (240, 180, 40), "unlikely": (230, 90, 80)}


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _tile(path: Path | None, title: str, sub: str, band: str | None) -> Image.Image:
    t = Image.new("RGB", (TILE_W, TILE_H + CAP), BG)
    if path and path.exists():
        im = Image.open(path).convert("RGB")
        im.thumbnail((TILE_W, TILE_H))
        t.paste(im, ((TILE_W - im.width) // 2, (TILE_H - im.height) // 2))
    d = ImageDraw.Draw(t)
    if band:
        d.rectangle([0, 0, TILE_W - 1, TILE_H - 1], outline=BAND_COL.get(band, MUTED), width=4)
    d.text((8, TILE_H + 4), title, fill=FG, font=_font(15))
    d.text((8, TILE_H + 24), sub, fill=MUTED, font=_font(13))
    return t


def render(conn, snap_dir: Path, frigate_event: str, compare_id: int | None, band_fn, prob_fn, tz) -> bytes | None:
    """compare_id = the identity it was matched to (or the closest one for a new visitor)."""
    me = conn.execute("SELECT embedding, crop_file, ts_start FROM sightings WHERE frigate_event = %s",
                      (frigate_event,)).fetchone()
    if not me:
        return None
    v = np.asarray(me[0], np.float32)
    tiles = [_tile(snap_dir / "persons" / me[1] if me[1] else None, "this visit",
                   me[2].astimezone(tz).strftime("%b %d %H:%M"), None)]
    if compare_id is not None:
        rows = conn.execute("""SELECT embedding, crop_file, ts_start FROM sightings
                               WHERE identity_id = %s AND frigate_event <> %s AND crop_file IS NOT NULL
                               ORDER BY ts_start DESC LIMIT 300""", (compare_id, frigate_event)).fetchall()
        scored = sorted(((float(np.dot(v, np.asarray(e, np.float32))), c, ts) for e, c, ts in rows), reverse=True)[:3]
        for sim, c, ts in scored:
            p = prob_fn(sim)
            tiles.append(_tile(snap_dir / "persons" / c, f"#{compare_id}  {band_fn(p)} {p * 100:.0f}%",
                               f"{ts.astimezone(tz):%b %d %H:%M}  sim {sim:.2f}", band_fn(p)))
    if len(tiles) == 1:
        tiles.append(_tile(None, "no earlier", "sightings to compare", None))
    out = Image.new("RGB", (len(tiles) * (TILE_W + 6) - 6, TILE_H + CAP), BG)
    for k, t in enumerate(tiles):
        out.paste(t, (k * (TILE_W + 6), 0))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=85)
    return buf.getvalue()

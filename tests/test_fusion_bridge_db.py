"""Bridge side of multi-camera fusion against a THROWAWAY PostgreSQL: a squirrelguard/<cam>/fusion message stores
<primary>_fused_traj.json (+ allowed assets only) and links it from EVERY member sighting; each keeps its own replay.
Skipped unless SG_TEST_PG_URL is set (same setup as tests/test_face_bridge_db.py):

  docker run -d --rm --name sgfuse-pg-test -e POSTGRES_PASSWORD=test -p 127.0.0.1:55433:5432 postgres:16-alpine
  SG_TEST_PG_URL=postgresql://postgres:test@127.0.0.1:55433/postgres <venv with psycopg, paho-mqtt, requests, flask,
      numpy>/bin/python tests/test_fusion_bridge_db.py
The database is wiped (DROP SCHEMA public CASCADE) at the start: never point this at a real database.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
PG = os.environ.get("SG_TEST_PG_URL")


def main() -> None:
    if not PG:
        print("skipped (SG_TEST_PG_URL unset)")
        return
    from test_face_bridge_db import setup, sighting, unit
    app, snap, psycopg = setup()
    rng = np.random.default_rng(3)
    e1, e2 = "1794044518.517957-gjcnr4", "1794044517.350524-rh7zw8"
    sighting(app, e1, unit(rng))
    sighting(app, e2, unit(rng))
    fid = f"{e1}_fused"
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    tr = {"t": [0, 0.125], "fusion": {"cameras": ["v4", "frontyard"], "stature_m": 1.7},
          "replay_files": {f"{fid}_replay_atlas.webp": b64(b"RIFFwebp"), f"{fid}_body.bin": b64(b"\0\1"),
                           "../evil.json": b64(b"x"), f"{e2}_replay_bg.jpg": b64(b"not ours")}}
    msg = {"camera": "v4", "fusion_id": fid, "frigate_event": e1,
           "members": [{"camera": "v4", "frigate_event": e1}, {"camera": "frontyard", "frigate_event": e2}],
           "trajectory": tr}
    app.handle_fusion(json.dumps(msg).encode())
    ev = snap / "evidence"
    assert (ev / f"{fid}_traj.json").is_file()
    assert (ev / f"{fid}_replay_atlas.webp").read_bytes() == b"RIFFwebp"
    assert (ev / f"{fid}_body.bin").is_file()
    assert not (ev / f"{e2}_replay_bg.jpg").exists() and not (snap / "evil.json").exists()
    stored = json.loads((ev / f"{fid}_traj.json").read_text())
    assert "replay_files" not in stored and stored["fusion"]["cameras"] == ["v4", "frontyard"]
    with psycopg.connect(PG, autocommit=True) as c:
        rows = dict(c.execute("SELECT frigate_event, fused_trajectory_file FROM sightings").fetchall())
    assert rows == {e1: f"{fid}_traj.json", e2: f"{fid}_traj.json"}, rows
    # bad ids / single member: ignored
    app.handle_fusion(json.dumps({**msg, "fusion_id": "../x_fused"}).encode())
    app.handle_fusion(json.dumps({**msg, "members": msg["members"][:1]}).encode())
    print("ok test_fusion_bridge_db")


if __name__ == "__main__":
    main()

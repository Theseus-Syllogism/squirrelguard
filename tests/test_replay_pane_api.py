"""Replay investigation pane API (persons-ui /api/visit, /api/identity) against a THROWAWAY PostgreSQL.

Uses SG_TEST_PG_URL when set; otherwise starts its own postgres:16-alpine container (docker) and removes it at the
end; skipped when neither is available.

  <venv with flask, psycopg[binary], numpy>/bin/python tests/test_replay_pane_api.py      (or pytest)
The database is wiped (DROP SCHEMA public CASCADE) at the start: never point SG_TEST_PG_URL at a real database.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "deploy/recorder/persons-ui"
EV = "1762393012.481234-v1sit0"
H = {"Remote-User": "tester", "Remote-Groups": "admins"}   # SSO user + group (roles: access.py)


def unit(rng, n=512):
    v = rng.normal(size=n)
    return v / np.linalg.norm(v)


def toward(u, cos, rng):
    r = unit(rng, len(u))
    r -= (r @ u) * u
    r /= np.linalg.norm(r)
    return cos * u + np.sqrt(1 - cos ** 2) * r


def _docker_pg():
    """-> (url, container id) of a fresh postgres:16-alpine on a free loopback port, or (None, None)."""
    if not shutil.which("docker"):
        return None, None
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    try:
        cid = subprocess.run(["docker", "run", "-d", "--rm", "--name", f"sg-pane-pg-test-{os.getpid()}", "-e",
                              "POSTGRES_PASSWORD=test", "-p", f"127.0.0.1:{port}:5432", "postgres:16-alpine"],
                             check=True, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception:
        return None, None
    url = f"postgresql://postgres:test@127.0.0.1:{port}/postgres"
    import psycopg
    for _ in range(60):
        try:
            with psycopg.connect(url, connect_timeout=2) as c:
                c.execute("SELECT 1")
            return url, cid
        except Exception:
            time.sleep(0.5)
    subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
    return None, None


def load_app(pg):
    os.environ.update({"PG_URL": pg, "TRUST_PROXY_USER": "1", "PERSONS_UI_USER": "u", "PERSONS_UI_PASS": "p",
                       "FRIGATE_URL": "https://frigate.example.test"})
    sys.path.insert(0, str(UI))
    spec = importlib.util.spec_from_file_location("persons_app", UI / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.app


def fixtures(pg):
    import psycopg
    from psycopg.types.json import Jsonb
    rng = np.random.default_rng(7)
    q = unit(rng)
    face = unit(rng)
    gait = unit(rng, 4096)
    now = dt.datetime(2026, 11, 5, 18, 56, tzinfo=dt.timezone.utc)
    with psycopg.connect(pg, autocommit=True) as c:
        c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        c.execute((ROOT / "deploy/recorder/schema.sql").read_text())
        c.execute((ROOT / "deploy/recorder/persons_schema.sql").read_text())
        ida = c.execute("INSERT INTO identities (label, kind, centroid, first_seen, last_seen) VALUES "
                        "('Alex', 'household', %s, %s, %s) RETURNING id", ([float(x) for x in q], now, now)).fetchone()[0]
        idb = c.execute("INSERT INTO identities (kind, centroid, first_seen, last_seen) VALUES "
                        "('unknown', %s, %s, %s) RETURNING id", ([float(x) for x in q], now, now)).fetchone()[0]

        def sight(ev, iid, ago_h, emb, **kw):
            cols = {"frigate_event": ev, "camera": "frontyard", "ts_start": now - dt.timedelta(hours=ago_h),
                    "ts_end": now - dt.timedelta(hours=ago_h) + dt.timedelta(seconds=40), "embedding": [float(x) for x in emb],
                    "identity_id": iid, "crop_file": f"{ev}.jpg"}
            cols.update(kw)
            ks = list(cols)
            c.execute(f"INSERT INTO sightings ({', '.join(ks)}) VALUES ({', '.join(['%s'] * len(ks))})",
                      [cols[k] for k in ks])

        f = lambda v: [float(x) for x in v]  # noqa: E731
        sight(EV, ida, 0, q, match_kind="match", match_sim=0.83, possible_id=idb, possible_sim=0.6,
              face_vec=f(face), gait_vec=f(gait), trajectory_file=f"{EV}_traj.json",
              face_meta=Jsonb({"n_faces": 6, "quality": 0.4, "liveness": {"ok": True, "score": 0.9},
                               "crop_files": [f"{EV}_face0.jpg"], "apparent": {"expression": "neutral"},
                               "match": {"status": "no_match", "sim": 0.12, "reasons": ["FACE_NO_MATCH"]}}),
              explain=Jsonb({"headline": "Matched Alex: likely (80%)", "band": "likely", "pct": 80,
                             "reasons": ["APPEARANCE_LIKELY", "CARRYING_LADDER"], "evidence": ["clothing and build match (0.83)"]}),
              objects=Jsonb([{"label": "ladder", "score": 0.6, "frames": 4, "sampled": 8, "carried": True}]),
              routine=Jsonb({"bits": 3.1, "threshold": 9, "history": 20, "ready": True, "unusual": False,
                             "parts": [{"code": "HOUR", "bits": 2.0, "text": "hour of day"}]}),
              pulc=Jsonb({"crops": 3, "tags": ["long_sleeves"], "orientation": {"front": 0.7, "back": 0.3}}),
              attributes=Jsonb({"upper_color": "red", "upper_clothing": "jacket"}), height_cm=178)
        sight("a1", ida, 24, toward(q, 0.95, rng), match_kind="match", gait_vec=f(toward(gait, 0.98, rng)))
        sight("a2", ida, 48, toward(q, 0.60, rng), match_kind="match")
        sight("b1", idb, 30, toward(q, 0.85, rng), match_kind="new", face_vec=f(toward(face, 0.7, rng)))
        sight("b2", idb, 5, toward(q, 0.20, rng), match_kind="new")
        sight("b3_old", idb, 24 * 120, toward(q, 0.99, rng), match_kind="new")      # outside the 90-day window
        sight("later", idb, -2, toward(q, 0.99, rng), match_kind="new")             # after the visit: not "prior"
        for vev, mins, plate in (("veh_near", 4, "ABC123"), ("veh_far", 35, "ZZZ999")):
            vid = c.execute("INSERT INTO vehicles (plate, color, body_type, centroid, first_seen, last_seen) VALUES "
                            "(%s, 'blue', 'sedan', %s, %s, %s) RETURNING id", (plate, f(q[:8]), now, now)).fetchone()[0]
            c.execute("INSERT INTO vehicle_sightings (frigate_event, camera, ts_start, embedding, vehicle_id) VALUES "
                      "(%s, 'frontyard', %s, %s, %s)", (vev, now + dt.timedelta(minutes=mins), f(q[:8]), vid))
        c.execute("INSERT INTO alerts (kind, subject, dedupe_key, message, sighting) VALUES "
                  "('night_unknown', %s, 'k1', 'test alert', %s)", (f"person:{ida}", EV))
        c.execute("INSERT INTO decisions (stage, frigate_event, band, pct, reasons, message) VALUES "
                  "('identity_match', %s, 'likely', 80, '{APPEARANCE_LIKELY}', 'Matched Alex')", (EV,))
    return ida, idb


def run(pg):
    ida, idb = fixtures(pg)
    app = load_app(pg)
    cl = app.test_client()
    # auth: nothing without the SSO header, not even a 404 that would reveal whether an event exists
    for path in (f"/api/visit/{EV}", f"/api/identity/{ida}", "/api/visit/nope"):
        r = cl.get(path)
        assert r.status_code == 401, (path, r.status_code)
        assert b"Alex" not in r.data
    assert cl.get("/api/visit/nope", headers=H).status_code == 404
    assert cl.get("/api/visit/..%2Fetc", headers=H).status_code == 404
    r = cl.get(f"/api/visit/{EV}", headers=H)
    assert r.status_code == 200, r.data
    assert r.headers["Cache-Control"].startswith("private")
    v = r.get_json()
    assert v["identity"]["name"] == "Alex" and v["explain"]["band"] == "likely" and v["explain"]["pct"] == 80
    assert any(x["text"] == "carrying ladder" for x in v["explain"]["reasons"])
    assert v["face"]["expression"] == "neutral" and v["face"]["crops"] == [f"/face/{EV}_face0.jpg"]
    assert "enrolled_crop" not in v["face"]                       # no gated match -> no enrolled face shown
    assert v["links"]["clip"] == f"https://frigate.example.test/api/events/{EV}/clip.mp4"
    assert v["links"]["replay_data"] == f"/evidence/{EV}_traj.json"
    assert v["routine"]["bits"] == 3.1 and v["objects"][0]["label"] == "ladder"
    assert len(v["alerts"]) == 1 and len(v["decisions"]) == 1
    rel = v["related"]
    types = [x["type"] for x in rel]
    assert types[0] == "identity" and "possible_identity" in types
    sights = [x for x in rel if x["type"] == "sighting"]
    evs = [x["event"] for x in sights]
    assert "b3_old" not in evs and "later" not in evs and EV not in evs, evs
    assert set(evs[:2]) == {"a1", "b1"}, evs                     # best: same identity + gait / strong face
    assert evs[-1] == "b2", evs                                   # weakest re-ID last
    ps = [x["strength"]["p"] for x in sights]
    assert ps == sorted(ps, reverse=True)
    a1 = next(x for x in sights if x["event"] == "a1")
    b1 = next(x for x in sights if x["event"] == "b1")
    assert {e["kind"] for e in a1["strength"]["edges"]} >= {"reid", "gait", "identity"}
    assert {e["kind"] for e in b1["strength"]["edges"]} >= {"reid", "face"} and b1["face_cos"] > 0.6
    assert a1["strength"]["band"] == "strong"
    a2 = next(x for x in sights if x["event"] == "a2")
    assert a2["strength"]["p"] > next(x for x in sights if x["event"] == "b2")["strength"]["p"]
    vehicles = [x for x in rel if x["type"] == "vehicle"]
    assert [x["plate"] for x in vehicles] == ["ABC123"], vehicles   # +-10 min only
    # identity timeline
    r = cl.get(f"/api/identity/{ida}", headers=H)
    assert r.status_code == 200
    j = r.get_json()
    assert j["identity"]["name"] == "Alex" and [x["event"] for x in j["visits"]] == [EV, "a1", "a2"]
    assert cl.get("/api/identity/999999", headers=H).status_code == 404
    print("replay pane API: ok", [(x["event"], x["strength"]["pct"]) for x in sights])


def test_pane_api():
    pg, cid = os.environ.get("SG_TEST_PG_URL"), None
    if not pg:
        pg, cid = _docker_pg()
    if not pg:
        try:
            import pytest
            pytest.skip("no SG_TEST_PG_URL and docker unavailable")
        except ImportError:
            print("SKIP: no SG_TEST_PG_URL and docker unavailable")
            return
    try:
        run(pg)
    finally:
        if cid:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True)


if __name__ == "__main__":
    test_pane_api()

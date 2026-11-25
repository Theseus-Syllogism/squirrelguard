"""End-to-end face template flow against a THROWAWAY PostgreSQL: bridge (handle_sighting / handle_behavior /
retention sweep), explain codes, persons-UI enrolment. Skipped unless SG_TEST_PG_URL is set.

  docker run -d --rm --name sgface-pg-test -e POSTGRES_PASSWORD=test -p 127.0.0.1:55432:5432 postgres:16-alpine
  SG_TEST_PG_URL=postgresql://postgres:test@127.0.0.1:55432/postgres <venv with psycopg, paho-mqtt, requests, flask,
      numpy>/bin/python tests/test_face_bridge_db.py
The database is wiped (DROP SCHEMA public CASCADE) at the start: never point this at a real database.
"""
from __future__ import annotations

import base64
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PG = os.environ.get("SG_TEST_PG_URL")


def unit(rng):
    v = rng.normal(size=512)
    return v / np.linalg.norm(v)


def toward(u, cos, rng):
    r = unit(rng)
    r -= (r @ u) * u
    r /= np.linalg.norm(r)
    return cos * u + np.sqrt(1 - cos ** 2) * r


JPG = base64.b64encode(b"\xff\xd8fakejpeg\xff\xd9").decode()


def faceid(vec, live=True, n=12):
    return {"vec": [float(x) for x in vec], "n_faces": n, "n_rejected": 0, "faces_seen": n, "quality": 0.4,
            "quality_max": 0.6, "weight_sum": 2.0 + n / 10, "consistency": 0.9, "size_px": 70, "max_size_px": 90,
            "yaw_abs_med": 10, "liveness": {"ok": live, "score": 0.9 if live else 0.1, "pass_frac": 1.0 if live else 0.0,
                                            "frames": n},
            "apparent": {"expression": "happy", "expression_share": 0.8, "expression_note": "apparent expression"},
            "best_crop_b64": JPG, "crops_b64": [JPG, JPG], "model": "Megatron", "aggregator": "fasr-qwmean-v1",
            "faces": []}


def setup():
    snap = Path(tempfile.mkdtemp(prefix="sgface-"))
    os.environ.update({"PG_URL": PG, "SNAPSHOT_DIR": str(snap), "NTFY_URL": "http://127.0.0.1:9/x",
                       "NTFY_PUB_TOKEN": "t", "PERSONS_UI_PASS": "x", "CROP_DIR": str(snap / "persons"),
                       "TRUST_PROXY_USER": "1", "EVIDENCE_MAIL": "off"})
    import psycopg
    with psycopg.connect(PG, autocommit=True) as c:
        c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        c.execute((ROOT / "deploy/recorder/schema.sql").read_text())
        c.execute((ROOT / "deploy/recorder/persons_schema.sql").read_text())
    sys.path.insert(0, str(ROOT / "deploy/recorder/bridge"))
    app = importlib.import_module("app")
    app.visitors_ntfy = lambda *a, **k: None
    app.intel.on_person_sighting = lambda *a, **k: None   # alerts are not under test here
    app.intel.on_behavior = lambda *a, **k: None
    app.match_strip.render = lambda *a, **k: None
    return app, snap, psycopg


def sighting(app, ev, emb, fid=None):
    msg = {"camera": "basement", "frigate_event": ev, "start": time.time() - 30, "end": None, "score": 0.8, "crops": 3,
           "embedding": [float(x) for x in emb], "attributes": None, "crop_jpeg_b64": None, "face": None}
    if fid is not None:
        msg["faceid"] = fid
    app.handle_sighting(json.dumps(msg).encode())


def row(psycopg, ev):
    with psycopg.connect(PG, autocommit=True) as c:
        return c.execute("""SELECT s.identity_id, s.match_kind, s.face_meta, s.explain, i.label, s.face_vec IS NOT NULL
                            FROM sightings s JOIN identities i ON i.id = s.identity_id WHERE frigate_event = %s""",
                         (ev,)).fetchone()


def test_flow():
    if not PG:
        print("skip: SG_TEST_PG_URL not set")
        return
    rng = np.random.default_rng(5)
    app, snap, psycopg = setup()
    A = unit(rng)
    with psycopg.connect(PG, autocommit=True) as c:
        nid = c.execute("INSERT INTO identities (label, kind, centroid, first_seen, last_seen) VALUES "
                        "('Robin', 'household', %s, now(), now()) RETURNING id", ([float(x) for x in unit(rng)],)).fetchone()[0]
        c.execute("INSERT INTO face_templates (identity_id, vec, model, n_faces) VALUES (%s, %s, 'Megatron', 10)",
                  (nid, [float(x) for x in A]))

    # 1. live face matching the enrolled template decides the identity (re-ID embedding is unrelated)
    sighting(app, "ev1", unit(rng), faceid(toward(A, 0.55, rng)))
    iid, kind, fm, ex, label, has = row(psycopg, "ev1")
    assert iid == nid and kind == "face_template" and has, (iid, kind)
    assert fm["match"]["status"] == "match" and "vec" not in fm and "crops_b64" not in fm
    assert "FACE_TEMPLATE_MATCH" in ex["reasons"] and ex["headline"].startswith("Recognised Robin by enrolled face")
    assert "apparent expression happy" in ex["body"]
    assert (snap / "faces/ev1_face0.jpg").exists() and (snap / "faces/ev1_face1.jpg").exists()

    # 2. same face but liveness fails (a photo): NOT named, explained
    sighting(app, "ev2", unit(rng), faceid(toward(A, 0.7, rng), live=False))
    iid, kind, fm, ex, label, _ = row(psycopg, "ev2")
    assert iid != nid and kind == "new" and fm["match"]["status"] == "liveness_fail"
    assert "FACE_LIVENESS_FAIL" in ex["reasons"]

    # 3. stranger: face seen, no match, new identity
    sighting(app, "ev3", unit(rng), faceid(unit(rng)))
    iid3, kind, fm, ex, label, _ = row(psycopg, "ev3")
    assert iid3 != nid and fm["match"]["status"] == "no_match" and "FACE_NO_MATCH" in ex["reasons"]

    # 4. no face at the 10 s sighting; the end-of-visit template matches -> moved, empty identity deleted
    sighting(app, "ev4", unit(rng), {"vec": None, "n_faces": 0, "reason": "too_small", "max_size_px": 12,
                                     "faces_seen": 1, "aggregator": "fasr-qwmean-v1"})
    old, kind, fm, ex, _, has = row(psycopg, "ev4")
    assert old != nid and "FACE_TOO_SMALL" in ex["reasons"] and not has
    app.handle_behavior(json.dumps({"kind": "person", "camera": "basement", "frigate_event": "ev4", "end": time.time(),
                                    "behavior": {"duration_s": 20}, "faceid": faceid(toward(A, 0.6, rng))}).encode())
    iid, kind, fm, ex, label, _ = row(psycopg, "ev4")
    assert iid == nid and kind == "face_template" and "FACE_TEMPLATE_MATCH" in ex["reasons"]
    with psycopg.connect(PG, autocommit=True) as c:
        assert c.execute("SELECT count(*) FROM identities WHERE id = %s", (old,)).fetchone()[0] == 0

    # 5. persons UI: enrolment only for household/known, consent required; kind change drops templates
    import importlib.util
    spec = importlib.util.spec_from_file_location("persons_ui_app", ROOT / "deploy/recorder/persons-ui/app.py")
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    cl = ui.app.test_client()
    H = {"Remote-User": "tester"}
    with psycopg.connect(PG, autocommit=True) as c:
        sid3 = c.execute("SELECT id FROM sightings WHERE frigate_event = 'ev3'").fetchone()[0]
        sid2 = c.execute("SELECT id FROM sightings WHERE frigate_event = 'ev2'").fetchone()[0]
    page = cl.get(f"/sighting/{sid3}", headers=H).get_data(as_text=True)
    assert "Use this visit&#39;s face for" in page or "Use this visit's face for" in page
    assert "Apparent expression: happy" in page and "never used for alerts" in page
    assert "Not usable for enrolment" in cl.get(f"/sighting/{sid2}", headers=H).get_data(as_text=True)  # liveness
    assert cl.post(f"/sighting/{sid3}/enroll", data={"identity": iid3, "consent": "yes"}, headers=H).status_code == 400
    assert cl.post(f"/sighting/{sid3}/enroll", data={"identity": nid}, headers=H).status_code == 400   # no consent
    assert cl.post(f"/sighting/{sid2}/enroll", data={"identity": nid, "consent": "yes"}, headers=H).status_code == 400
    r = cl.post(f"/sighting/{sid3}/enroll", data={"identity": nid, "consent": "yes"}, headers=H)
    assert r.status_code == 302, r.get_data(as_text=True)
    with psycopg.connect(PG, autocommit=True) as c:
        tids = [t for (t,) in c.execute("SELECT id FROM face_templates WHERE identity_id = %s ORDER BY id", (nid,))]
        assert len(tids) == 2
        assert c.execute("SELECT crop_jpeg IS NOT NULL FROM face_templates WHERE id = %s", (tids[1],)).fetchone()[0]
    page = cl.get(f"/identity/{nid}", headers=H).get_data(as_text=True)
    assert "Enrolled faces" in page and f"/face_template/{tids[1]}.jpg" in page
    assert cl.get(f"/face_template/{tids[1]}.jpg", headers=H).status_code == 200
    assert cl.post(f"/face_template/{tids[1]}/delete", headers=H).status_code == 302
    cl.post(f"/identity/{nid}/edit", data={"label": "Robin", "kind": "unknown", "notes": ""}, headers=H)
    with psycopg.connect(PG, autocommit=True) as c:
        assert c.execute("SELECT count(*) FROM face_templates").fetchone()[0] == 0

    # 6. retention: crops of deleted sightings are swept
    with psycopg.connect(PG, autocommit=True) as c:
        c.execute("DELETE FROM sightings WHERE frigate_event = 'ev3'")
        old_t = time.time() - 7200
        for f in (snap / "faces").glob("*.jpg"):
            os.utime(f, (old_t, old_t))
        app.face_crop_sweep(c)
    assert not list((snap / "faces").glob("ev3_face*.jpg")) and (snap / "faces/ev1_face0.jpg").exists()
    print("ok test_flow")


if __name__ == "__main__":
    test_flow()

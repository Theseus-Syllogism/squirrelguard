"""Vehicle occupant faces + unnamed-face auto-tracking against a THROWAWAY PostgreSQL: bridge (handle_vehicle,
handle_sighting, handle_behavior, retention), persons UI (vehicle and identity pages). Skipped unless SG_TEST_PG_URL
is set; same setup as tests/test_face_bridge_db.py (the database is wiped: never point this at a real database).

Checks the core rules: names only for ENROLLED identities; unusable occupant faces leave no vector, crop or identity;
non-matching faces become UNNAMED identities only with FACE_AUTO_TRACK on; thresholds/margins/gates; 90-day retention.
"""
from __future__ import annotations

import base64
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PG = os.environ.get("SG_TEST_PG_URL")
ADA = "adaface_ir101_webface12m"
JPG = base64.b64encode(b"\xff\xd8fakejpeg\xff\xd9").decode()


def unit(rng):
    v = rng.normal(size=512)
    return v / np.linalg.norm(v)


def toward(u, cos, rng):
    r = unit(rng)
    r -= (r @ u) * u
    r /= np.linalg.norm(r)
    return cos * u + np.sqrt(1 - cos ** 2) * r


def tpl(isf, ada=None, live=True, n=8, crop=True):
    t = {"vec": [float(x) for x in isf], "n_faces": n, "quality": 0.35, "size_px": 48, "consistency": 0.9,
         "weight_sum": 1.5, "liveness": {"ok": live, "score": 0.9 if live else 0.1}, "model": "Megatron",
         "rel_x": 0.3, "apparent": {}, "best_crop_b64": JPG if crop else None, "aggregator": "fasr-qwmean-v1"}
    if ada is not None:
        t["v2"] = {"model": ADA, "vec": [float(x) for x in ada], "n_faces": n}
    return t


def setup():
    snap = Path(tempfile.mkdtemp(prefix="sgocc-"))
    os.environ.update({"PG_URL": PG, "SNAPSHOT_DIR": str(snap), "NTFY_URL": "http://127.0.0.1:9/x",
                       "NTFY_PUB_TOKEN": "t", "PERSONS_UI_PASS": "x", "CROP_DIR": str(snap / "persons"),
                       "TRUST_PROXY_USER": "1", "EVIDENCE_MAIL": "off", "FACE_AUTO_TRACK": "1"})
    import psycopg
    with psycopg.connect(PG, autocommit=True) as c:
        c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        c.execute((ROOT / "deploy/recorder/schema.sql").read_text())
        c.execute((ROOT / "deploy/recorder/persons_schema.sql").read_text())
        c.execute((ROOT / "deploy/recorder/persons_schema.sql").read_text())     # additive: applies twice cleanly
    sys.path.insert(0, str(ROOT / "deploy/recorder/bridge"))
    for m in ("app", "facematch", "intel", "explain"):
        sys.modules.pop(m, None)
    app = importlib.import_module("app")
    app.visitors_ntfy = lambda *a, **k: None
    app.intel.on_person_sighting = lambda *a, **k: None
    app.intel.on_vehicle_sighting = lambda *a, **k: None
    app.intel.on_behavior = lambda *a, **k: None
    app.match_strip.render = lambda *a, **k: None
    return app, snap, psycopg


def vehicle(app, ev, emb, occ, n=None, plate=None):
    msg = {"camera": "v4", "frigate_event": ev, "label": "car", "start": time.time() - 30, "end": time.time(),
           "score": 0.8, "crops": 3, "plate": plate, "embedding": [float(x) for x in emb], "attributes": None,
           "crop_jpeg_b64": None, "occupants_detected": len(occ) if n is None else n, "occupant_faces": occ,
           "occupant_stats": {"faces_seen": 9, "frames_with_vehicle": 8, "max_size_px": 60, "cpu_s": 5.0}}
    app.handle_vehicle(json.dumps(msg).encode())


def person(app, ev, emb, fid):
    app.handle_sighting(json.dumps({"camera": "frontyard", "frigate_event": ev, "start": time.time() - 20, "end": None,
                                    "score": 0.8, "crops": 3, "embedding": [float(x) for x in emb], "attributes": None,
                                    "crop_jpeg_b64": None, "face": None, "faceid": fid}).encode())


def vs(psycopg, ev):
    with psycopg.connect(PG, autocommit=True) as c:
        return c.execute("""SELECT occupants, occupant_face, driver_identity_id, driver_kind, driver_sim, id
                            FROM vehicle_sightings WHERE frigate_event = %s""", (ev,)).fetchone()


def occ_rows(psycopg, ev):
    with psycopg.connect(PG, autocommit=True) as c:
        return c.execute("""SELECT o.identity_id, o.match_kind, o.vec IS NOT NULL, o.vec_v2 IS NOT NULL, o.crop_file,
                                   i.label, i.kind, i.origin FROM occupant_faces o JOIN identities i ON i.id = o.identity_id
                            WHERE o.frigate_event = %s ORDER BY o.slot""", (ev,)).fetchall()


def n_identities(psycopg):
    with psycopg.connect(PG, autocommit=True) as c:
        return c.execute("SELECT count(*) FROM identities").fetchone()[0]


def test_occupant_flow():
    if not PG:
        print("skip: SG_TEST_PG_URL not set")
        return
    rng = np.random.default_rng(23)
    app, snap, psycopg = setup()
    fm = app.facematch
    A_isf, A_ada = unit(rng), unit(rng)                 # the enrolled owner (both models)
    with psycopg.connect(PG, autocommit=True) as c:
        nid = c.execute("INSERT INTO identities (label, kind, centroid, first_seen, last_seen) VALUES "
                        "('Robin', 'household', %s, now(), now()) RETURNING id", ([float(x) for x in unit(rng)],)).fetchone()[0]
        c.execute("INSERT INTO face_templates (identity_id, vec, model, n_faces) VALUES (%s, %s, 'Megatron', 10)",
                  (nid, [float(x) for x in A_isf]))
        c.execute("INSERT INTO face_templates (identity_id, vec, model, n_faces) VALUES (%s, %s, %s, 10)",
                  (nid, [float(x) for x in A_ada], ADA))
    base = n_identities(psycopg)

    # 1. enrolled driver: named, linked, one crop, NO vector stored (enrolled people match via face_templates)
    vehicle(app, "v1", unit(rng), [tpl(toward(A_isf, 0.5, rng), toward(A_ada, 0.6, rng))])
    occ, summ, did, dk, dsim, _ = vs(psycopg, "v1")
    assert occ == 1 and did == nid and dk == "face_template" and abs(dsim - 0.6) < 0.01, (did, dk, dsim)
    assert summ["occupants"][0]["match"]["label"] == "Robin" and summ["occupants"][0]["match"]["engine"] == ADA
    rows = occ_rows(psycopg, "v1")
    assert len(rows) == 1 and rows[0][:4] == (nid, "face_template", False, False) and rows[0][4] == "v1_occ0_face0.jpg"
    assert (snap / "faces/v1_occ0_face0.jpg").exists() and n_identities(psycopg) == base

    # 2. unusable occupant faces (liveness fail behind glass; too few faces): counted only. No identity, no row, no
    #    vector, no crop; a weak look-alike of the enrolled owner leaks neither id nor label
    vehicle(app, "v2", unit(rng), [tpl(toward(A_isf, 0.6, rng), toward(A_ada, 0.7, rng), live=False),
                                   tpl(unit(rng), unit(rng), n=1)])
    occ, summ, did, dk, _, _ = vs(psycopg, "v2")
    assert occ == 2 and did is None and dk is None and occ_rows(psycopg, "v2") == []
    assert not list((snap / "faces").glob("v2_*")) and n_identities(psycopg) == base
    blob = json.dumps(summ)
    assert '"identity"' not in blob and "Robin" not in blob and "candidate" not in blob and "vec" not in blob
    assert summ["occupants"][0]["match"]["status"] == "liveness_fail"

    # 3. a stranger driving (usable face): a NEW UNNAMED identity (no label, kind unknown, origin vehicle_occupant)
    S_isf, S_ada = unit(rng), unit(rng)
    vehicle(app, "v3", unit(rng), [tpl(toward(S_isf, 0.8, rng), toward(S_ada, 0.8, rng))])
    _, summ, sid_ident, dk, _, _ = vs(psycopg, "v3")
    rows = occ_rows(psycopg, "v3")
    assert dk == "face_new" and rows[0][0] == sid_ident and rows[0][1:4] == ("face_new", True, True)
    assert rows[0][5:] == (None, "unknown", "vehicle_occupant") and n_identities(psycopg) == base + 1
    assert summ["occupants"][0]["match"]["status"] == "no_match"

    # 4. the same stranger again (AdaFace visit-vs-visit >= 0.40): same unnamed identity; a different stranger whose
    #    face is just below the threshold (0.35) gets a new one
    vehicle(app, "v4", unit(rng), [tpl(toward(S_isf, 0.7, rng), toward(S_ada, 0.75, rng))])
    _, summ, d4, dk, s4, _ = vs(psycopg, "v4")
    assert d4 == sid_ident and dk == "face_unknown" and s4 >= fm.FACE_UNKNOWN_SIM_V2
    assert summ["occupants"][0]["track"]["status"] == "match" and "label" not in summ["occupants"][0]["track"]
    vehicle(app, "v5", unit(rng), [tpl(unit(rng), toward(S_ada, 0.30, rng))])
    _, _, d5, dk, _, _ = vs(psycopg, "v5")
    assert d5 not in (None, sid_ident) and dk == "face_new"

    # 5. the stranger later walks up the drive: the person sighting joins the unnamed identity by face
    #    (body re-ID has nothing on an occupant-only identity: its centroid is all zeros)
    person(app, "p1", unit(rng), {**tpl(toward(S_isf, 0.6, rng), toward(S_ada, 0.7, rng)), "crops_b64": [JPG]})
    with psycopg.connect(PG, autocommit=True) as c:
        iid, mk, fmeta, ex = c.execute("SELECT identity_id, match_kind, face_meta, explain FROM sightings "
                                       "WHERE frigate_event = 'p1'").fetchone()
    assert iid == sid_ident and mk == "face_unknown" and fmeta["track"]["status"] == "match"
    assert "FACE_UNKNOWN_MATCH" in ex["reasons"] and ex["headline"].startswith(f"Same unnamed visitor Unknown #{sid_ident}")

    # 6. person path: body re-ID points at an unnamed identity whose faces clearly differ -> kept apart (face_new);
    #    an unusable face never splits
    body = unit(rng)
    person(app, "p2", body, tpl(unit(rng), toward(S_ada, 0.05, rng)))          # founds X (new face)
    with psycopg.connect(PG, autocommit=True) as c:
        x = c.execute("SELECT identity_id FROM sightings WHERE frigate_event = 'p2'").fetchone()[0]
    T_ada = unit(rng)
    person(app, "p3", toward(body, 0.9, rng), tpl(unit(rng), T_ada))         # same clothes, a different face
    person(app, "p4", toward(body, 0.9, rng), tpl(unit(rng), unit(rng), live=False))   # same clothes, unusable face
    with psycopg.connect(PG, autocommit=True) as c:
        r3 = c.execute("SELECT identity_id, match_kind, possible_id FROM sightings WHERE frigate_event = 'p3'").fetchone()
        r4 = c.execute("SELECT identity_id, match_kind FROM sightings WHERE frigate_event = 'p4'").fetchone()
        ex3 = c.execute("SELECT explain FROM sightings WHERE frigate_event = 'p3'").fetchone()[0]
    assert r3[0] != x and r3[1] == "face_new" and r3[2] == x and "FACE_SPLIT" in ex3["reasons"], r3
    assert r4 == (x, "match"), r4

    # 7. a user-labelled (not enrolled) identity leaves the auto-track gallery: tracking never names anyone
    with psycopg.connect(PG, autocommit=True) as c:
        c.execute("UPDATE identities SET label = 'Delivery guy' WHERE id = %s", (sid_ident,))
        gal = app.unknown_gallery(c)
    assert sid_ident not in {g["identity_id"] for g in gal}
    vehicle(app, "v6", unit(rng), [tpl(toward(S_isf, 0.7, rng), toward(S_ada, 0.8, rng))])
    _, summ, d6, dk, _, _ = vs(psycopg, "v6")
    assert d6 != sid_ident and dk == "face_new" and "Delivery guy" not in json.dumps(summ)
    with psycopg.connect(PG, autocommit=True) as c:
        c.execute("UPDATE identities SET label = NULL WHERE id = %s", (sid_ident,))

    # 8. persons UI: driver lines and the identity's vehicles
    spec = importlib.util.spec_from_file_location("persons_ui_occ", ROOT / "deploy/recorder/persons-ui/app.py")
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    cl = ui.app.test_client()
    H = {"Remote-User": "tester"}
    with psycopg.connect(PG, autocommit=True) as c:
        vid1 = c.execute("SELECT vehicle_id FROM vehicle_sightings WHERE frigate_event = 'v1'").fetchone()[0]
        vid2 = c.execute("SELECT vehicle_id FROM vehicle_sightings WHERE frigate_event = 'v2'").fetchone()[0]
        vid4 = c.execute("SELECT vehicle_id FROM vehicle_sightings WHERE frigate_event = 'v4'").fetchone()[0]
    page = cl.get(f"/vehicle/{vid1}", headers=H).get_data(as_text=True)
    assert "Driver: Robin (enrolled, face match 0.60)" in page and "v1_occ0_face0.jpg" in page
    page = cl.get(f"/vehicle/{vid2}", headers=H).get_data(as_text=True)
    assert "Occupant(s) detected (2), not recognised" in page
    page = cl.get(f"/vehicle/{vid4}", headers=H).get_data(as_text=True)
    assert f"Driver: Unknown #{sid_ident} (unnamed, same face as before" in page and f"/identity/{sid_ident}" in page
    page = cl.get(f"/identity/{sid_ident}", headers=H).get_data(as_text=True)
    assert "In vehicles (face seen inside)" in page and "Unnamed: grouped automatically" in page
    page = cl.get(f"/identity/{nid}", headers=H).get_data(as_text=True)
    assert "In vehicles (face seen inside)" in page and "enrolled face" in page
    j = cl.get(f"/api/identity/{sid_ident}", headers=H).get_json()
    assert j["in_vehicles"] and j["identity"]

    # 9. retention: occupant faces of non-enrolled people older than 90 days go (the vehicle may stay), the emptied
    #    occupant-only identity goes, the driver link is cleared, crops are swept; the enrolled owner's row stays
    with psycopg.connect(PG, autocommit=True) as c:
        c.execute("UPDATE occupant_faces SET ts = now() - interval '100 days' WHERE frigate_event IN ('v1', 'v5')")
        c.execute("UPDATE identities SET last_seen = now() - interval '2 hours'")
        d5_crop = c.execute("SELECT crop_file FROM occupant_faces WHERE frigate_event = 'v5'").fetchone()[0]
        app.occupant_retention(c)
        old_t = time.time() - 7200
        for f in (snap / "faces").glob("*.jpg"):
            os.utime(f, (old_t, old_t))
        app.face_crop_sweep(c)
        assert c.execute("SELECT count(*) FROM occupant_faces WHERE frigate_event = 'v5'").fetchone()[0] == 0
        assert c.execute("SELECT count(*) FROM identities WHERE id = %s", (d5,)).fetchone()[0] == 0
        assert c.execute("SELECT driver_identity_id FROM vehicle_sightings WHERE frigate_event = 'v5'").fetchone()[0] is None
        assert c.execute("SELECT count(*) FROM vehicle_sightings WHERE frigate_event = 'v5'").fetchone()[0] == 1
        assert c.execute("SELECT count(*) FROM occupant_faces WHERE frigate_event = 'v1'").fetchone()[0] == 1
        assert c.execute("SELECT count(*) FROM identities WHERE id = %s", (sid_ident,)).fetchone()[0] == 1
    assert not (snap / "faces" / d5_crop).exists() and (snap / "faces/v1_occ0_face0.jpg").exists()

    # 10. deleting an identity in the UI removes its occupant faces (cascade) and the driver links
    assert cl.post(f"/identity/{sid_ident}/delete", headers=H).status_code == 302
    with psycopg.connect(PG, autocommit=True) as c:
        assert c.execute("SELECT count(*) FROM occupant_faces WHERE identity_id = %s", (sid_ident,)).fetchone()[0] == 0
        assert c.execute("SELECT count(*) FROM vehicle_sightings WHERE driver_identity_id = %s",
                         (sid_ident,)).fetchone()[0] == 0
    print("ok test_occupant_flow")


def test_auto_track_off():
    """FACE_AUTO_TRACK=0: only enrolled matches are linked; strangers leave no identity, row, vector or crop."""
    if not PG:
        print("skip: SG_TEST_PG_URL not set")
        return
    rng = np.random.default_rng(29)
    app, snap, psycopg = setup()
    app.facematch.FACE_AUTO_TRACK = False
    try:
        before = n_identities(psycopg)
        vehicle(app, "w1", unit(rng), [tpl(unit(rng), unit(rng))])
        occ, summ, did, dk, _, _ = vs(psycopg, "w1")
        assert occ == 1 and did is None and occ_rows(psycopg, "w1") == [] and n_identities(psycopg) == before
        assert not list((snap / "faces").glob("w1_*")) and "track" not in summ["occupants"][0]
        person(app, "q1", unit(rng), tpl(unit(rng), unit(rng)))
        with psycopg.connect(PG, autocommit=True) as c:
            mk, fmeta = c.execute("SELECT match_kind, face_meta FROM sightings WHERE frigate_event = 'q1'").fetchone()
        assert mk == "new" and "track" not in fmeta
    finally:
        app.facematch.FACE_AUTO_TRACK = True
    print("ok test_auto_track_off")


if __name__ == "__main__":
    test_occupant_flow()
    test_auto_track_off()

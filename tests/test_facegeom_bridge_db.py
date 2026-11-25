"""Face geometry through the bridge against a THROWAWAY PostgreSQL: face_geom stored with files, enrolled templates
take their visit's geometry, a later matching visit gets the explanatory comparison, orphans are swept.
Skipped unless SG_TEST_PG_URL is set (see tests/test_face_bridge_db.py for the docker one-liner); the database is
wiped. MediaPipe is mocked (tests/test_facegeom.MockMesher); needs numpy, opencv, psycopg, paho-mqtt, requests.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))
from test_face_bridge_db import PG, faceid, row, setup, sighting, toward, unit  # noqa: E402
from test_facegeom import MockMesher, _obs  # noqa: E402

from squirrel_guard import facegeom as fg  # noqa: E402


def geom_payload(seed: int) -> dict:
    c = fg.GeomCollector(MockMesher())
    frame = np.broadcast_to(np.linspace(40, 220, 1920, dtype=np.uint8)[None, :, None], (1080, 1920, 3)).copy()
    rng = np.random.default_rng(seed)
    for i in range(6):
        c.offer(frame, _obs(size=80 + rng.integers(0, 20), yaw=4.0, t=i), 1.0 + i)
    return c.result()


def test_facegeom_flow():
    if not PG:
        print("skip: SG_TEST_PG_URL not set")
        return
    rng = np.random.default_rng(11)
    app, snap, psycopg = setup()
    A = unit(rng)
    with psycopg.connect(PG, autocommit=True) as c:
        nid = c.execute("INSERT INTO identities (label, kind, centroid, first_seen, last_seen) VALUES "
                        "('Robin', 'household', %s, now(), now()) RETURNING id", ([float(x) for x in unit(rng)],)).fetchone()[0]
        c.execute("INSERT INTO face_templates (identity_id, vec, model, n_faces) VALUES (%s, %s, 'Megatron', 10)",
                  (nid, [float(x) for x in A]))

    # 1. a matching visit with geometry: stored, files written, no enrolled geometry yet -> no comparison
    f1 = faceid(toward(A, 0.6, rng))
    f1["facegeom"] = geom_payload(1)
    sighting(app, "gv1", unit(rng), f1)
    iid, kind, fm, ex, _, _ = row(psycopg, "gv1")
    assert iid == nid and kind == "face_template" and "facegeom" not in fm
    with psycopg.connect(PG, autocommit=True) as c:
        g = c.execute("SELECT face_geom FROM sightings WHERE frigate_event = 'gv1'").fetchone()[0]
    assert g["n_faces"] >= 3 and g["explanatory"] and "compare" not in g and "mesh" not in g
    for k in ("overlay", "depth", "mesh"):
        assert (snap / "evidence" / g["files"][k]).is_file(), k
    json.loads((snap / "evidence" / g["files"]["mesh"]).read_text())

    # 2. the user enrols that visit's face (persons UI inserts the template with source_event) -> geometry synced
    with psycopg.connect(PG, autocommit=True) as c:
        c.execute("INSERT INTO face_templates (identity_id, vec, model, n_faces, source_event) "
                  "VALUES (%s, %s, 'Megatron', 10, 'gv1')", (nid, [float(x) for x in A]))
        app.sync_enrolled_geometry(c)
        assert c.execute("SELECT count(*) FROM face_templates WHERE geom IS NOT NULL").fetchone()[0] == 1

    # 3. next matching visit: explanatory comparison with the enrolled geometry
    f2 = faceid(toward(A, 0.6, rng))
    f2["facegeom"] = geom_payload(2)
    sighting(app, "gv2", unit(rng), f2)
    with psycopg.connect(PG, autocommit=True) as c:
        g2 = c.execute("SELECT face_geom FROM sightings WHERE frigate_event = 'gv2'").fetchone()[0]
    cmp = g2["compare"]
    assert cmp["explanatory"] and cmp["identity"] == nid and cmp["enrolled_visits"] == 1
    assert cmp["features"]["nose_width"]["verdict"] == "consistent", cmp["features"]["nose_width"]

    # 4. the sighting goes (retention / persons UI): its geometry files are swept; the template keeps its copy
    with psycopg.connect(PG, autocommit=True) as c:
        c.execute("DELETE FROM sightings WHERE frigate_event = 'gv1'")
        old_t = time.time() - 7200
        for f in (snap / "evidence").glob("*_facegeom_*"):
            os.utime(f, (old_t, old_t))
        app.facegeom_sweep(c)
        assert c.execute("SELECT count(*) FROM face_templates WHERE geom IS NOT NULL").fetchone()[0] == 1
    assert not list((snap / "evidence").glob("gv1_facegeom_*")) and list((snap / "evidence").glob("gv2_facegeom_*"))
    print("ok test_facegeom_flow")


if __name__ == "__main__":
    test_facegeom_flow()

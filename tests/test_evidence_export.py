"""Evidence packages, signatures, outside timestamps, capture-time seals, roles, audit log, human review and release
copies. Module tests run anywhere with `cryptography`; the route tests need Flask + psycopg and a THROWAWAY PostgreSQL
(SG_TEST_PG_URL, or a postgres:16-alpine container started here) and are skipped otherwise. The database is wiped:
never point SG_TEST_PG_URL at a real one. A local RFC 3161 service is made with openssl, so no network is used."""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import http.server
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "deploy/recorder/persons-ui"
REC = ROOT / "deploy/recorder/recorder"
sys.path.insert(0, str(UI))
sys.path.insert(0, str(Path(__file__).parent))
import evidence_export as evx  # noqa: E402
import tsa  # noqa: E402

EV = "1762393012.481234-v1sit0"
EV2 = "1762396612.000000-v2sit0"
SUP = {"Remote-User": "sgt.okafor", "Remote-Groups": "admins"}
INV = {"Remote-User": "det.rivera", "Remote-Groups": "sg-investigator"}
INV2 = {"Remote-User": "det.chen", "Remote-Groups": "sg-investigator"}
VIEW = {"Remote-User": "clerk.ames", "Remote-Groups": "sg-viewer"}
NOBODY = {"Remote-User": "visitor", "Remote-Groups": "family"}
CLIP = b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 400


# ------------------------------------------------------------------ a local timestamp service (openssl)
@pytest.fixture(scope="module")
def tsa_service(tmp_path_factory):
    if not shutil.which("openssl"):
        pytest.skip("openssl not installed")
    d = tmp_path_factory.mktemp("tsa")
    (d / "ts.cnf").write_text("""[ tsa ]
default_tsa = tsa_config1
[ tsa_config1 ]
dir = .
serial = ./serial
signer_cert = ./tsa.crt
signer_key = ./tsa.key
signer_digest = sha256
default_policy = 1.2.3.4.1
digests = sha256
accuracy = secs:1
ordering = no
tsa_name = no
ess_cert_id_chain = no
ess_cert_id_alg = sha256
[ v3_tsa ]
extendedKeyUsage = critical,timeStamping
basicConstraints = CA:false
keyUsage = critical,digitalSignature
[ v3_ca ]
basicConstraints = critical,CA:true
keyUsage = critical,keyCertSign
""")
    (d / "serial").write_text("01\n")
    run = lambda *a: subprocess.run(a, cwd=d, check=True, capture_output=True)
    run("openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes", "-keyout", "ca.key",
        "-out", "ca.crt", "-subj", "/CN=Test TSA Root", "-days", "2", "-config", "ts.cnf", "-extensions", "v3_ca")
    run("openssl", "req", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes", "-keyout", "tsa.key", "-out", "tsa.csr",
        "-subj", "/CN=Test TSA")
    run("openssl", "x509", "-req", "-in", "tsa.csr", "-CA", "ca.crt", "-CAkey", "ca.key", "-CAcreateserial", "-out", "tsa.crt",
        "-days", "2", "-extfile", "ts.cnf", "-extensions", "v3_tsa")
    lock = threading.Lock()

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            q = self.rfile.read(int(self.headers["Content-Length"]))
            with lock:
                (d / "q.tsq").write_bytes(q)
                run("openssl", "ts", "-reply", "-queryfile", "q.tsq", "-config", "ts.cnf", "-out", "r.tsr")
                body = (d / "r.tsr").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/timestamp-reply")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{srv.server_address[1]}/", "ca": str(d / "ca.crt")}
    srv.shutdown()


def test_tsa_request_parses_with_openssl(tmp_path):
    (tmp_path / "q.tsq").write_bytes(tsa.request(hashlib.sha256(b"x").digest(), 12345))
    out = subprocess.run(["openssl", "ts", "-query", "-in", "q.tsq", "-text"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert "sha256" in out and "Nonce: 0x3039" in out and "Certificate required: yes" in out


def test_tsa_round_trip_and_openssl_verify(tsa_service):
    data = b"manifest bytes"
    st = tsa.stamp(hashlib.sha256(data).hexdigest(), tsa_service["url"])
    assert abs(st.offset_s) < 5 and st.hash == hashlib.sha256(data).hexdigest()
    assert tsa.verify_openssl(data, st.tsr, tsa_service["ca"])[0] is True
    assert tsa.verify_openssl(data + b"!", st.tsr, tsa_service["ca"])[0] is False
    with pytest.raises(tsa.TSAError):
        tsa.parse_response(b"\x30\x03\x02\x01\x02")                       # a refusal, no token


def test_recorder_copy_of_tsa_is_identical():
    assert (UI / "tsa.py").read_bytes() == (REC / "tsa.py").read_bytes()


# ------------------------------------------------------------------ packages
def _signer(tmp_path):
    return evx.Signer(tmp_path / "keys")


def _items(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(CLIP)
    (tmp_path / "crop.jpg").write_bytes(b"\xff\xd8crop")
    (tmp_path / "seg.mkv").write_bytes(b"\x1a\x45\xdf\xa3segment" * 1000)
    seg_sha = hashlib.sha256((tmp_path / "seg.mkv").read_bytes()).hexdigest()
    prev = "1" * 64
    led = []
    for seq, (f, h) in enumerate([("cam/2026-11-05_11-50-00.mkv", seg_sha), ("cam/2026-11-05_12-00-00.mkv", "b" * 64)], 41):
        fields = {"seq": seq, "camera": "cam", "file": f, "bytes": 1, "sha256": h, "start_local": f[4:-4], "mtime_utc": "t", "sealed_utc": "s"}
        hh = evx.chain_hash(prev, fields)
        led.append(dict(fields, prev=prev, hash=hh))
        prev = hh
    return [{"n": 1, "subject": {"sighting": 1, "event": EV, "camera": "cam<script>", "start": "s", "end": "e", "identity": "Unknown #2"},
             "visit": {"explain": {"headline": "Probably <i>the same</i> person", "band": "likely", "pct": 81, "evidence": ["re-ID 0.71"],
                                   "reasons": [{"code": "REID_MATCH", "text": "appearance matches"}]}},
             "review": {"latest": None, "history": []}, "missing": [],
             "files": [{"path": "original/clip.mp4", "src": tmp_path / "clip.mp4", "role": "original", "description": "clip"},
                       {"path": "original/segments/2026-11-05_11-50-00.mkv", "src": tmp_path / "seg.mkv", "role": "original", "description": "seg"},
                       {"path": "derived/crop.jpg", "src": tmp_path / "crop.jpg", "role": "derived", "description": "crop"}],
             "segments": [{"file": "cam/2026-11-05_11-50-00.mkv", "status": "sealed", "status_text": "matches its capture-time seal",
                           "included": True, "sha256": seg_sha, "sealed_utc": "s", "anchor": "a"}],
             "ledger": led, "anchors": []}]


META = {"export_id": "EX-20261119-00001", "case_ref": "2026-<b>1</b>", "purpose": "Incident report", "note": "", "exported_by": "tester",
        "exported_role": "investigator", "created_utc": "2026-11-19T00:00:00.000000Z", "created_local": "2026-11-18T17:00:00-07:00",
        "system": {"version": "test"}, "custody_prev": evx.GENESIS, "clock": {"export_host_text": "x"}, "timestamp_note": "test"}


def _pkg(tmp_path, timestamper=None):
    dest = tmp_path / "p.zip"
    s = _signer(tmp_path)
    return dest, evx.build_package(dest, _items(tmp_path), dict(META), s, timestamper), s


def test_package_contents_signature_and_report(tmp_path):
    dest, res, s = _pkg(tmp_path)
    with zipfile.ZipFile(dest) as z:
        names = set(z.namelist())
        assert {"items/01/original/clip.mp4", "items/01/record/visit.json", "items/01/record/review.json", "items/01/record/ledger.jsonl",
                "report.html", "manifest.json", "manifest.sig", "signing-key.pem", "SHA256SUMS", "verify.py", "README.txt"} <= names
        assert z.read("items/01/original/clip.mp4") == CLIP
        mb = z.read("manifest.json")
        assert evx.check_signature(z.read("signing-key.pem"), mb, z.read("manifest.sig"))
        assert not evx.check_signature(z.read("signing-key.pem"), mb + b" ", z.read("manifest.sig"))
        report = z.read("report.html").decode()
        assert "<script>" not in report and "&lt;b&gt;1&lt;/b&gt;" in report
        assert "NOT CONFIRMED BY A PERSON" in report and "investigative lead" in report
        assert s.fingerprint in z.read("README.txt").decode()
    assert evx.verify_package(dest, res["package_sha256"], res["manifest_sha256"], s.public_pem) == []
    assert oct(dest.stat().st_mode & 0o777) == "0o440"
    other = evx.Signer(tmp_path / "other-keys")
    assert evx.verify_package(dest, res["package_sha256"], res["manifest_sha256"], other.public_pem) == \
        ["the signature does not match this installation's key"]


def test_openssl_checks_the_signature_and_the_timestamp(tmp_path, tsa_service):
    dest, res, _ = _pkg(tmp_path, lambda h: tsa.stamp(h, tsa_service["url"]))
    assert res["timestamp"] and res["timestamp_error"] is None
    out = tmp_path / "x"
    zipfile.ZipFile(dest).extractall(out)
    r = subprocess.run(["openssl", "pkeyutl", "-verify", "-pubin", "-inkey", "signing-key.pem", "-rawin", "-in", "manifest.json",
                        "-sigfile", "manifest.sig"], cwd=out, capture_output=True, text=True)
    assert r.returncode == 0 and "Signature Verified Successfully" in r.stdout
    r = subprocess.run(["openssl", "ts", "-verify", "-data", "manifest.json", "-in", "manifest.tsr", "-CAfile", tsa_service["ca"]],
                       cwd=out, capture_output=True, text=True)
    assert "Verification: OK" in r.stdout
    assert evx.verify_package(dest, res["package_sha256"], res["manifest_sha256"], None,
                              lambda d, t: tsa.verify_openssl(d, t, tsa_service["ca"])) == []


def test_timestamp_failure_still_makes_the_package(tmp_path):
    def down(h):
        raise tsa.TSAError("the timestamp service did not answer (URLError)")
    dest, res, _ = _pkg(tmp_path, down)
    assert res["timestamp"] is None and "did not answer" in res["timestamp_error"] and "manifest.tsr" not in zipfile.ZipFile(dest).namelist()


def test_verify_py_checks_files_seals_and_signature(tmp_path):
    dest, _, _ = _pkg(tmp_path)
    out = tmp_path / "unzipped"
    zipfile.ZipFile(dest).extractall(out)
    assert subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=out).returncode == 0
    r = subprocess.run([sys.executable, "verify.py"], cwd=out, capture_output=True, text=True)
    assert r.returncode == 0 and "signature: valid" in r.stdout and "matches its seal" in r.stdout, r.stdout
    seg = out / "items/01/original/segments/2026-11-05_11-50-00.mkv"
    seg.write_bytes(seg.read_bytes()[:-1] + b"X")
    r = subprocess.run([sys.executable, "verify.py"], cwd=out, capture_output=True, text=True)
    assert r.returncode == 1 and "CHANGED  items/01/original/segments" in r.stdout and "DIFFERS FROM ITS CAPTURE-TIME SEAL" in r.stdout
    (out / "planted.txt").write_text("x")
    assert "EXTRA    planted.txt" in subprocess.run([sys.executable, "verify.py"], cwd=out, capture_output=True, text=True).stdout


def test_chains_detect_edit_removal_and_reorder():
    rows, prev = [], evx.GENESIS
    for i, action in enumerate(["created", "downloaded", "verified", "downloaded"], 1):
        ts = f"2026-11-19T00:00:0{i}.000000Z"
        h = evx.entry_hash(prev, 7, ts, "a", action, "d", "127.0.0.1")
        rows.append({"id": i, "export_id": 7, "ts": ts, "actor": "a", "action": action, "detail": "d", "remote": "127.0.0.1",
                     "prev_hash": prev, "entry_hash": h})
        prev = h
    assert evx.verify_chain(rows) == {"ok": True, "entries": 4, "broken_at": None, "head": prev}
    edited = [dict(r) for r in rows]; edited[1]["actor"] = "someone-else"
    assert evx.verify_chain(edited)["broken_at"] == 2
    assert evx.verify_chain(rows[:1] + rows[2:])["broken_at"] == 3
    assert evx.verify_chain([rows[0], rows[2], rows[1], rows[3]])["ok"] is False


# ------------------------------------------------------------------ capture-time sealer
def _sealer(tmp_path, monkeypatch, tsa_url=""):
    rec, led = tmp_path / "rec", tmp_path / "ledger"
    (rec / "frontyard").mkdir(parents=True)
    monkeypatch.setenv("RECORDINGS", str(rec)); monkeypatch.setenv("LEDGER", str(led)); monkeypatch.setenv("TSA_URL", tsa_url)
    sys.path.insert(0, str(REC))
    spec = importlib.util.spec_from_file_location("seal_under_test", REC / "seal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, rec, led


def test_sealer_seals_closed_segments_and_anchors(tmp_path, monkeypatch, tsa_service):
    seal, rec, led = _sealer(tmp_path, monkeypatch, tsa_service["url"])
    a, b = rec / "frontyard" / "2026-11-05_11-50-00.mkv", rec / "frontyard" / "2026-11-05_12-00-00.mkv"
    a.write_bytes(b"first"); b.write_bytes(b"still recording")
    led.mkdir()
    cams = {}
    seal.once(cams)
    lines = seal.read_ledger(led / "frontyard.jsonl")
    assert [l["file"] for l in lines] == ["frontyard/2026-11-05_11-50-00.mkv"]          # the open segment waits
    assert lines[0]["sha256"] == hashlib.sha256(b"first").hexdigest() and seal.verify_lines(lines) is None
    anchors = seal.read_ledger(led / "anchors" / "frontyard.jsonl")
    assert anchors[0]["head"] == lines[-1]["hash"]
    assert tsa.verify_openssl(bytes.fromhex(anchors[0]["head"]), base64.b64decode(anchors[0]["tsr_b64"]), tsa_service["ca"])[0] is False
    # the token covers the hash itself, so openssl checks it with -digest rather than -data:
    d = tmp_path / "a"; d.mkdir(); (d / "t.tsr").write_bytes(base64.b64decode(anchors[0]["tsr_b64"]))
    r = subprocess.run(["openssl", "ts", "-verify", "-digest", anchors[0]["head"], "-in", "t.tsr", "-CAfile", tsa_service["ca"]],
                       cwd=d, capture_output=True, text=True)
    assert "Verification: OK" in r.stdout
    (rec / "frontyard" / "2026-11-05_12-10-00.mkv").write_bytes(b"third")
    seal.once(cams)
    assert len(seal.read_ledger(led / "frontyard.jsonl")) == 2
    # a tampered ledger stops sealing for that camera instead of being rewritten
    text = (led / "frontyard.jsonl").read_text().replace('"bytes": 5', '"bytes": 6', 1)
    (led / "frontyard.jsonl").write_text(text)
    assert seal.Camera("frontyard").broken


# ------------------------------------------------------------------ routes, against a throwaway database
@pytest.fixture(scope="module")
def ui(tmp_path_factory, tsa_service):
    pytest.importorskip("flask"); pytest.importorskip("psycopg")
    import test_replay_pane_api as pane
    pg, cid = os.environ.get("SG_TEST_PG_URL"), None
    if not pg:
        pg, cid = pane._docker_pg()
    if not pg:
        pytest.skip("no throwaway PostgreSQL (set SG_TEST_PG_URL or install docker)")
    import psycopg
    base = tmp_path_factory.mktemp("ev")
    crops, store, keys, rec, led = base / "snapshots" / "persons", base / "store", base / "keys", base / "rec", base / "ledger"
    for d in (crops, store, rec / "frontyard", led):
        d.mkdir(parents=True)
    (crops / f"{EV}.jpg").write_bytes(b"\xff\xd8crop")
    os.environ["TZ"] = "America/Denver"
    import time as _time
    _time.tzset()
    t0 = dt.datetime(2026, 11, 5, 18, 56, tzinfo=dt.timezone.utc)
    seg_name = t0.astimezone().strftime("%Y-%m-%d_%H-%M") [:-1] + "0-00.mkv"          # the 10-minute segment holding t0
    (rec / "frontyard" / seg_name).write_bytes(b"segment bytes " * 500)
    (rec / "frontyard" / "2099-01-01_00-00-00.mkv").write_bytes(b"later")               # closes the first one
    with psycopg.connect(pg, autocommit=True) as c:
        c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        c.execute((ROOT / "deploy/recorder/schema.sql").read_text())
        c.execute((ROOT / "deploy/recorder/persons_schema.sql").read_text())
        c.execute((ROOT / "deploy/recorder/persons_schema.sql").read_text())               # idempotent: setup.sh reruns it
        iid = c.execute("INSERT INTO identities (kind, centroid, first_seen, last_seen) VALUES ('unknown', %s, %s, %s) RETURNING id",
                        ([0.0] * 512, t0, t0)).fetchone()[0]
        from psycopg.types.json import Jsonb
        sys.path.insert(0, str(ROOT))
        from squirrel_guard import gaitlab
        from test_gaitlab import walker
        walks = [gaitlab.analyse(walker(110, fps=10, seed=k), 10) for k in range(2)]
        sids = [c.execute("""INSERT INTO sightings (frigate_event, camera, ts_start, ts_end, embedding, identity_id, crop_file, gait)
                             VALUES (%s, 'frontyard', %s, %s, %s, %s, %s, %s) RETURNING id""",
                          (ev, t0 + dt.timedelta(minutes=k), t0 + dt.timedelta(minutes=k, seconds=40), [0.0] * 512, iid, f"{EV}.jpg",
                           Jsonb(walks[k]))).fetchone()[0]
                for k, ev in enumerate((EV, EV2))]
    monkey_env = {"PG_URL": pg, "TRUST_PROXY_USER": "1", "PERSONS_UI_USER": "u", "PERSONS_UI_PASS": "p", "CROP_DIR": str(crops),
                  "EVIDENCE_STORE": str(store), "EVIDENCE_KEYS": str(keys), "RECORDINGS": str(rec), "LEDGER": str(led),
                  "TSA_URL": tsa_service["url"], "FRIGATE_URL": "https://frigate.example.test", "SG_VERSION": "test"}
    os.environ.update(monkey_env)
    seal_env = {"RECORDINGS": str(rec), "LEDGER": str(led), "TSA_URL": tsa_service["url"]}
    sys.path.insert(0, str(REC))
    spec = importlib.util.spec_from_file_location("seal_for_ui", REC / "seal.py")
    seal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seal)
    seal.once({})
    spec = importlib.util.spec_from_file_location("persons_app_ev", UI / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    clip = CLIP
    try:                                                                            # a real clip with faces, for the release copy
        import cv2
        img = cv2.imread(str(ROOT / "data/testimg/cars/8_Blue_sports_cars_parked_on_a_city_street.jpg"))
        if img is not None:
            img = cv2.resize(img, (424, 640))
            w = cv2.VideoWriter(str(base / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5, (424, 640))
            for _ in range(5):
                w.write(img)
            w.release()
            clip = (base / "clip.mp4").read_bytes()
    except ImportError:
        pass

    def fake_frigate(path, dest, actor):
        if path.endswith("snapshot.jpg"):
            raise OSError("gone")                                                   # retention removed it
        return (*evx.save_stream(io.BytesIO(clip), dest), 2.0)
    mod._frigate_file = fake_frigate
    real_verify = tsa.verify_openssl                                               # same module object as mod.tsalib
    mod.tsalib.verify_openssl = lambda d, t, ca=None: real_verify(d, t, tsa_service["ca"])
    yield {"clip": clip, "client": mod.app.test_client(), "mod": mod, "sids": sids, "iid": iid, "store": store, "pg": pg, "rec": rec, "seg": seg_name,
           "env": seal_env}
    if cid:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)


def _wait_job(c, loc, h):
    import time
    for _ in range(200):
        r = c.get(loc, headers=h)
        if r.status_code == 302:
            return r
        assert b"was not created" not in r.data, r.data.decode()[-600:]
        time.sleep(0.05)
    raise AssertionError("export did not finish")


def test_roles(ui):
    c, sid = ui["client"], ui["sids"][0]
    assert c.get("/", headers=NOBODY).status_code == 403                               # signed in, no role
    assert c.get(f"/sighting/{sid}", headers=VIEW).status_code == 200
    page = c.get(f"/sighting/{sid}", headers=VIEW).data.decode()
    assert "NOT CONFIRMED BY A PERSON" in page and "Record review" not in page and "Export evidence package" not in page
    assert c.post(f"/sighting/{sid}/review", headers=VIEW, data={"outcome": "confirmed"}).status_code == 403
    assert c.get("/exports", headers=VIEW).status_code == 403
    assert c.post(f"/identity/{ui['iid']}", headers=VIEW, data={"label": "x"}).status_code == 403
    assert c.get("/audit", headers=INV).status_code == 403
    nav = c.get("/", headers=INV).data.decode()
    assert "Evidence" in nav and ">Audit<" not in nav
    assert ">Audit<" in c.get("/", headers=SUP).data.decode()


def test_review_is_recorded_and_shown(ui):
    c, sid = ui["client"], ui["sids"][0]
    assert c.post(f"/sighting/{sid}/review", headers=INV, data={"outcome": "rejected", "note": ""}).status_code == 400
    assert c.post(f"/sighting/{sid}/review", headers=INV, data={"outcome": "unsure", "note": "face hidden"}).status_code == 302
    assert c.post(f"/sighting/{sid}/review", headers=INV2, data={"outcome": "confirmed", "note": "same coat, same gait"}).status_code == 302
    page = c.get(f"/sighting/{sid}", headers=VIEW).data.decode()
    assert "Confirmed by det.chen" in page and "Review history (2)" in page
    assert "Walk" in page and "consistent with this person" in page and "<svg" in page       # gait vs the identity's other walk
    ident = c.get(f"/identity/{ui['iid']}", headers=VIEW).data.decode()
    assert "Walk profile" in ident and "Cadence" in ident


def test_multi_item_export_with_segments_timestamp_and_release(ui):
    import psycopg
    c, (s1, s2) = ui["client"], ui["sids"]
    form = c.get(f"/exports/new?identity={ui['iid']}", headers=INV).data.decode()
    assert form.count('name="sid"') == 2 and "not reviewed" in form
    r = c.post(f"/exports/new?identity={ui['iid']}", headers=INV, data={"sid": [str(s1), str(s2)], "case_ref": "2026-004512",
                                                                        "purpose": "Incident report", "segments": "on"})
    assert r.status_code == 400                                                        # authorisation not confirmed
    r = c.post(f"/exports/new?identity={ui['iid']}", headers=INV, data={"sid": [str(s1), str(s2)], "case_ref": "2026-004512",
                                                                        "purpose": "Incident report", "segments": "on", "ack": "on"})
    assert r.status_code == 302 and "/exports/job/" in r.headers["Location"]
    assert c.get(r.headers["Location"], headers=INV2).status_code == 404               # someone else's job
    done = _wait_job(c, r.headers["Location"], INV)
    loc = done.headers["Location"]
    xid = int(loc.split("/export/")[1].split("?")[0])
    page = c.get(loc, headers=INV).data.decode()
    assert "2026-004512" in page and "matches its capture-time seal" in page and "Outside" in page
    assert "1 of 2 items NOT confirmed" in page                                         # item 1 confirmed above, item 2 not
    assert c.get(loc, headers=INV2).status_code == 404                                 # investigators see their own only
    assert c.get(loc, headers=SUP).status_code == 200
    zips = sorted(ui["store"].glob("EX-*.zip"))
    with zipfile.ZipFile(zips[-1]) as z:
        names = z.namelist()
        m = json.loads(z.read("manifest.json"))
        msha = hashlib.sha256(z.read("manifest.json")).hexdigest()
        assert z.read("items/01/original/clip.mp4") == ui["clip"] and "manifest.tsr" in names
        assert f"items/01/original/segments/{ui['seg']}" in names and "items/01/record/ledger.jsonl" in names
        assert m["items"][0]["review"]["reviewer"] == "det.chen" and m["items"][1]["review"] is None
        assert any("snapshot.jpg" in t for t in m["items"][0]["missing"])
        assert m["clock"]["recorder_offset_s"] == 2 and "the outside time service" in m["clock"]["export_host_text"]
        assert m["exported_role"] == "investigator"
        v1 = json.loads(z.read("items/01/record/visit.json"))
        assert v1["gait_compare"]["verdict"] == "consistent" and "Knee" in z.read("report.html").decode()
    d = c.get(f"/export/{xid}/download", headers=INV)
    assert d.status_code == 200 and d.data == zips[-1].read_bytes()
    assert c.post(f"/export/{xid}/verify", headers=INV).headers["Location"].endswith("done=verified")
    assert c.get(f"/export/{xid}/download", headers=INV2).status_code == 404

    # release copy (needs the detector models; SG_REDACT_MODELS points at them outside the container)
    models = Path(os.environ.get("SG_REDACT_MODELS", "/app/models"))
    if (models / "face_detection_yunet_2023mar.onnx").exists():
        import redact
        redact.MODELS = models
        assert c.post(f"/export/{xid}/release", headers=INV).status_code == 302
        import time
        for _ in range(300):
            if ui["mod"].RELEASES.get(xid) != "running":
                break
            time.sleep(0.1)
        assert ui["mod"].RELEASES[xid] == "done", ui["mod"].RELEASES[xid]
        rz = sorted(ui["store"].glob("EX-*-R*.zip"))[-1]
        with zipfile.ZipFile(rz) as z:
            rm = json.loads(z.read("manifest.json"))
            assert rm["source"] == {"export_id": m["export_id"], "case_ref": m["case_ref"], "manifest_sha256": msha}
            assert any(f["path"].endswith("clip.redacted.mp4") for f in rm["files"])
            # the fixture's crop.jpg is not a real picture: it is left out and listed, not silently dropped
            assert rm["left_out"] and all("crop.jpg" in t for t in rm["left_out"])
            if ui["clip"] != CLIP:                                                     # real clip with six bystanders
                red = next(f for f in rm["files"] if f["path"] == "items/01/clip.redacted.mp4")
                n_faces = int(red["description"].split(": ")[1].split(" face")[0])
                assert n_faces >= 5 and red["source"].startswith("items/01/original/clip.mp4 sha256 ")
            assert not any("segments" in n for n in z.namelist())

    log = c.get("/exports/custody.json", headers=SUP).get_json()
    acts = [e["action"] for e in log["entries"] if e["export_id"] == xid]
    assert acts[:3] == ["created", "downloaded", "verified"] and log["check"]["ok"]
    assert c.get("/exports/custody.json", headers=INV).status_code == 403

    # audit log: views, searches, refusals and changes are there, and the chain holds
    c.get("/search?q=red+coat", headers=VIEW)
    a = c.get("/audit.json", headers=SUP).get_json()
    acts = [(e["actor"], e["action"]) for e in a["entries"]]
    assert ("clerk.ames", "view:search") in acts and ("clerk.ames", "denied") in acts and ("det.chen", "review:confirmed") in acts
    assert any(e["detail"] == "q=red coat" for e in a["entries"]) and a["check"]["ok"]

    # the database refuses edits; a forced edit is caught by the chain
    with psycopg.connect(ui["pg"], autocommit=True) as db:
        for sql in ("UPDATE audit_log SET actor = 'x'", "DELETE FROM custody_log", "TRUNCATE match_reviews",
                    "UPDATE evidence_exports SET case_ref = 'x'"):
            with pytest.raises(psycopg.Error, match="append-only"):
                db.execute(sql)
        db.execute("ALTER TABLE audit_log DISABLE TRIGGER audit_log_append_only")
        db.execute("UPDATE audit_log SET actor = 'someone-else' WHERE actor = 'clerk.ames'")
    assert b"audit log has been altered" in c.get("/audit", headers=SUP).data


def test_changed_segment_is_reported(ui):
    c = ui["client"]
    seg = ui["rec"] / "frontyard" / ui["seg"]
    seg.write_bytes(seg.read_bytes() + b"edited")
    r = c.post(f"/exports/new?sids={ui['sids'][0]}", headers=SUP, data={"sid": str(ui["sids"][0]), "case_ref": "c2", "purpose": "Other",
                                                                        "ack": "on"})
    done = _wait_job(c, r.headers["Location"], SUP)
    assert "DOES NOT MATCH its capture-time seal" in c.get(done.headers["Location"], headers=SUP).data.decode()


@pytest.mark.skipif(not os.environ.get("SG_SCREENSHOT_DIR"), reason="only when SG_SCREENSHOT_DIR is set")
def test_save_pages_for_review(ui):
    """Writes the new pages as self-contained HTML (styles inlined) for a visual check."""
    out = Path(os.environ["SG_SCREENSHOT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    css = (UI / "static/ui.css").read_text()
    c = ui["client"]
    xid = c.get("/exports", headers=SUP).data.decode().split('href="/export/')[1].split('"')[0]
    for name, path, h in (("sighting", f"/sighting/{ui['sids'][0]}", INV), ("identity", f"/identity/{ui['iid']}", INV),
                          ("export-form", f"/exports/new?identity={ui['iid']}", INV), ("export", f"/export/{xid}", SUP),
                          ("exports", "/exports", SUP), ("audit", "/audit", SUP)):
        html_ = c.get(path, headers=h).data.decode()
        html_ = html_.split('<link rel="stylesheet"')[0] + f"<style>{css}</style>" + html_.split(".css?v=")[1].split(">", 1)[1]
        (out / f"{name}.html").write_text(html_)

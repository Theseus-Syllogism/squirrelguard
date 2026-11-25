"""Console core, command line and web GUI (squirrel_guard/console). No network, no real install: a throwaway checkout."""
import json
import stat
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from squirrel_guard.console import cli, web
from squirrel_guard.console.core import ConfigError, Site, parse_fields, scrub

REPO = Path(__file__).resolve().parents[1]
SECRET = "rtsp://user:hunter2@127.0.0.1:9/ch1"


@pytest.fixture
def site(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "config.example.toml").write_text((REPO / "config.example.toml").read_text())
    return Site(root=root, home=tmp_path / "home")


def test_parse_fields_help_and_commented_settings():
    f = {x.id: x for x in parse_fields((REPO / "config.example.toml").read_text())}
    assert f["camera.fps"].value == 3 and "frames per second" in f["camera.fps"].help
    assert "motion.roi" not in f                                  # commented-out setting is not a setting
    assert "roi" not in f["detector.model"].help                  # ...and not somebody else's help text
    assert f["detector.targets"].value == ["Squirrel"] and f["detector.targets"].kind == "list"


def test_set_value_keeps_comments_and_backs_up(site):
    site.create_config()
    assert site.set_value("camera.fps", "2.5") == 2.5
    text = site.config_path.read_text()
    assert "fps = 2.5            # frames per second actually processed" in text
    assert site.config()["camera"]["fps"] == 2.5
    assert "fps = 3 " in site.config_path.with_name("config.toml.bak").read_text()
    assert site.set_value("actuator.dry_run", "no") is False
    assert site.set_value("detector.targets", "Squirrel, Raccoon") == ["Squirrel", "Raccoon"]
    assert site.set_value("actuator.command", 'paplay "my sound.wav"  # loud') == 'paplay "my sound.wav"  # loud'
    assert site.config()["actuator"]["command"].endswith("# loud")   # a # inside a string survives the round trip
    assert "# log only" in site.config_path.read_text()


@pytest.mark.parametrize("fid,value,msg", [
    ("camera.fps", "99", "between"), ("camera.width", "abc", "whole number"), ("detector.conf", "0", "between"),
    ("verifier.url", "ftp://x", "http"), ("actuator.dry_run", "maybe", "yes or no"), ("trigger.need", "9", "trigger.window"),
    ("nope.nothing", "1", "no setting"),
])
def test_set_value_refuses_bad_input_and_leaves_file_alone(site, fid, value, msg):
    site.create_config()
    before = site.config_path.read_text()
    with pytest.raises(ConfigError, match=msg):
        site.set_value(fid, value)
    assert site.config_path.read_text() == before


def test_missing_key_is_added_to_its_section(site):
    site.config_path.write_text("[camera]\nwidth = 640\n\n[detector]\nconf = 0.25\n")
    site.set_value("camera.fps", "4")
    assert site.config_path.read_text() == "[camera]\nwidth = 640\nfps = 4\n\n[detector]\nconf = 0.25\n"
    site.set_value("output.dir", "data/x")                       # section not in the file yet
    assert site.config()["output"]["dir"] == "data/x"


def test_secrets_are_private_and_never_returned(site):
    site.env_set("SG_RTSP_URL", SECRET)
    assert stat.S_IMODE(site.env_path.stat().st_mode) == 0o600
    assert "hunter2" not in json.dumps(site.env_status())
    assert next(e for e in site.env_status() if e["name"] == "SG_RTSP_URL")["set"] is True
    assert "hunter2" not in json.dumps([c.__dict__ for c in site.checks("detector")]) + json.dumps(site.status())
    assert "127.0.0.1" not in json.dumps([c.__dict__ for c in site.checks("detector") if c.id == "stream"])
    with pytest.raises(ConfigError):
        site.env_set("SG_RTSP_URL", "http://not-rtsp")
    with pytest.raises(ConfigError):
        site.env_set("bad name", "x")
    with pytest.raises(ConfigError):
        site.env_set("SG_MQTT_PASS", "a\nINJECTED=1")
    site.env_set("SG_RTSP_URL", None)
    assert site.env_path.read_text() == ""


def test_checks_on_an_empty_checkout_fail_with_fixes(site):
    checks = {c.id: c for c in site.checks("detector")}
    assert checks["venv"].status == "fail" and checks["config"].status == "fail" and checks["stream"].status == "fail"
    assert all(c.fix for c in checks.values() if c.status == "fail")


def test_steps_dry_run_changes_nothing(site):
    out = []
    for step in site.steps("detector"):
        assert site.run_step("detector", step.id, out.append, dry_run=True) == 0
    assert not site.config_path.exists() and not site.env_path.exists() and not site._unit_path().exists()
    assert any("config.example.toml" in line for line in out) and all(line.startswith("would ") for line in out)
    with pytest.raises(ConfigError):
        site.run_step("detector", "rm -rf", out.append)


def test_config_step_and_service_unit(site, tmp_path):
    assert site.run_step("detector", "config", lambda _: None) == 0 and site.config_path.exists()
    out = []
    assert site.run_step("detector", "service", out.append) == 2          # no environment yet: refuses, explains
    assert "environment" in out[0]
    unit = site._unit_text()
    assert f"WorkingDirectory={site.root}" in unit and "-m squirrel_guard.main -c" in unit and "hunter2" not in unit


def test_recorder_stack_copy_keeps_secrets(site, tmp_path):
    src = site.root / "deploy" / "recorder"
    (src / "mosquitto").mkdir(parents=True)
    (src / "compose.yaml").write_text("services: {}\n")
    (src / "mosquitto" / "passwd").write_text("")
    dst = tmp_path / "stack"
    site.set_setting("recorder_dir", str(dst))
    dst.mkdir()
    (dst / "mosquitto").mkdir()
    (dst / ".env").write_text("KEEP=1\n")
    (dst / "mosquitto" / "passwd").write_text("user:hash\n")
    assert site.run_step("recorder", "stackdir", lambda _: None) == 0
    assert (dst / "compose.yaml").exists() and (dst / ".env").read_text() == "KEEP=1\n"
    assert (dst / "mosquitto" / "passwd").read_text() == "user:hash\n"
    assert site.run_step("recorder", "database", lambda _: None, answer="postgresql://u:pw@db/x?sslmode=require") == 0
    assert stat.S_IMODE((dst / "pg.url").stat().st_mode) == 0o600
    with pytest.raises(ConfigError):
        site.set_setting("recorder_dir", "relative/path")


def test_events_tail_and_snapshot_name_guard(site):
    site.create_config()
    d = site.root / "data" / "events"
    d.mkdir(parents=True)
    (d / "events.jsonl").write_text("".join(json.dumps({"time": f"t{i}", "label": "Squirrel", "score": 0.9, "action": "dry_run",
                                                        "confirmed": True, "snapshot": f"{i}_event.jpg"}) + "\n" for i in range(5)) + "garbage\n")
    (d / "4_event.jpg").write_bytes(b"\xff\xd8")
    assert [e["time"] for e in site.events(3)] == ["t4", "t3", "t2"]
    assert site.snapshot_path("4_event.jpg") and site.snapshot_path("../../.env") is None and site.snapshot_path("x.png") is None


def test_cli(site, monkeypatch, capsys):
    monkeypatch.setenv("SG_ROOT", str(site.root))
    assert cli.main(["setup", "--role", "detector", "--dry-run"]) == 0
    assert "Create the Python environment" in capsys.readouterr().out and not site.config_path.exists()
    assert cli.main(["config", "set", "camera.fps", "5"]) == 0 and Site(root=site.root).config()["camera"]["fps"] == 5
    assert cli.main(["config", "set", "camera.fps", "500"]) == 2
    assert cli.main(["check"]) == 1                                # failures -> exit code 1, for scripts
    assert cli.main(["status", "--json"]) == 0


def test_web_access_control_and_api(site):
    site.env_set("SG_RTSP_URL", SECRET)
    srv, url = web.make_server(site, port=18760, token="tok123")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = url.split("/?")[0]

    def call(path, body=None, cookie=True, token="tok123", host=None, origin=None):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None)
        if cookie:
            req.add_header("Cookie", "sg_console=tok123")
        if body is not None:
            req.add_header("Content-Type", "application/json")
            req.add_header("X-SG-Token", token)
        if host:
            req.add_header("Host", host)
        if origin:
            req.add_header("Origin", origin)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()
    try:
        assert call("/api/status", cookie=False)[0] == 403                         # no key
        assert call("/api/status", host="evil.example:18760")[0] == 403            # DNS rebinding
        assert call("/api/config", {"id": "camera.fps", "value": "5"}, token="")[0] == 403          # cross-site post
        assert call("/api/config", {"id": "camera.fps", "value": "5"}, origin="http://evil.example")[0] == 403
        assert call("/snapshot/..%2f..%2f.env")[0] == 404
        code, body = call("/api/config", {"id": "camera.fps", "value": "5"})
        assert code == 200 and site.config()["camera"]["fps"] == 5
        code, body = call("/api/config", {"id": "camera.fps", "value": "500"})
        assert code == 400 and "between" in json.loads(body)["error"]
        assert call("/api/run", {"steps": ["nope"]})[0] == 400
        for path in ("/", "/api/status", "/api/env", "/api/checks", "/api/steps", "/api/config", "/api/service", "/api/events"):
            code, body = call(path)
            assert code == 200 and "hunter2" not in body, path
        code, body = call("/api/run", {"steps": ["venv", "config"], "dry_run": True})
        job = json.loads(body)["job"]
        for _ in range(100):
            j = json.loads(call(f"/api/job?id={job}")[1])
            if j["done"]:
                break
            threading.Event().wait(0.05)
        assert j["rc"] == 0 and any("would" in line for line in j["lines"])
    finally:
        srv.shutdown()
        srv.server_close()


def test_logs_are_scrubbed():
    line = "\x1b[2K2026 Error opening input file rtsp://user:hunter2@198.51.100.5:554/ch1."
    assert scrub(line) == "2026 Error opening input file rtsp://***@198.51.100.5:554/ch1."
    assert scrub("see http://example.org/a:b@c") == "see http://example.org/a:b@c"      # not a credential


def test_docker_runtime_steps(site):
    site.set_setting("runtime", "docker")
    ids = [s.id for s in site.steps("detector")]
    assert ids == ["docker", "config", "stream", "build", "model", "start"]
    out = []
    for sid in ids[1:]:
        assert site.run_step("detector", sid, out.append, dry_run=True) == 0
    assert any("docker compose up -d detector" in l for l in out) and not site.config_path.exists()
    assert site.run_step("detector", "start", out.append) == 2          # refuses to start without config.toml
    with pytest.raises(ConfigError):
        site.set_setting("runtime", "podman")
    with pytest.raises(ConfigError):
        site.service("enable")

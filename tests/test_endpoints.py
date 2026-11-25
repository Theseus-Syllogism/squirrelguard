"""Endpoints page (squirrel_guard/console/endpoints.py): storing addresses, keys and model names, and the tests
against a fake llama.cpp / OpenAI-compatible server. No outside network."""
import json
import stat
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from squirrel_guard.console import cli, endpoints as ep, web
from squirrel_guard.console.core import ConfigError, Site

REPO = Path(__file__).resolve().parents[1]


class Fake:
    """A llama.cpp-like server. Settings are changed between requests by the tests."""
    def __init__(self):
        self.key, self.vision, self.loading, self.models, self.props = None, True, False, ["qwen-vl"], True
        self.seen = []
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def reply(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                fake.seen.append((self.path, self.headers.get("Authorization")))
                if self.path == "/health":
                    return self.reply(503, {"error": "loading"}) if fake.loading else self.reply(200, {"status": "ok"})
                if fake.key and self.headers.get("Authorization") != f"Bearer {fake.key}":
                    return self.reply(401, {"error": "invalid api key"})
                if self.path == "/v1/models":
                    return self.reply(200, {"object": "list", "data": [{"id": m} for m in fake.models]})
                if self.path == "/props" and fake.props:
                    return self.reply(200, {"modalities": {"vision": fake.vision, "audio": False}})
                if self.path == "/api/version":
                    self.send_response(200)
                    self.end_headers()
                    return self.wfile.write(b"0.16.1")
                self.reply(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                fake.seen.append((self.path, self.headers.get("Authorization"), body.get("model")))
                self.reply(200, {"choices": [{"message": {"content": "squirrel"}}]})

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


@pytest.fixture
def fake():
    f = Fake()
    yield f
    f.srv.shutdown()


@pytest.fixture
def site(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "config.example.toml").write_text((REPO / "config.example.toml").read_text())
    s = Site(root=root, home=tmp_path / "home")
    s.create_config()
    return s


def test_example_has_no_private_address():
    import ipaddress
    import re
    text = (REPO / "config.example.toml").read_text()
    for ip in re.findall(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d])", text):
        a = ipaddress.ip_address(ip)
        assert a.is_loopback or not a.is_private or a in ipaddress.ip_network("192.0.2.0/24"), ip


def test_listing_and_defaults(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    (root / "config.example.toml").write_text((REPO / "config.example.toml").read_text())
    s = Site(root=root, home=tmp_path / "h")
    rows = {r["id"]: r for r in ep.listing(s, "detector")}
    assert rows["verifier"]["address"] == "http://127.0.0.1:8080" and rows["verifier"]["source"] == "default"
    assert rows["mqtt"]["source"] == "none" and "stay on this machine" in rows["mqtt"]["fallback"]
    assert rows["verifier"]["api_key"] == "SG_VERIFIER_API_KEY" and rows["verifier"]["api_key_set"] is False


def test_llm_ok_lists_models_and_images(site, fake):
    ep.set_address(site, "detector", "verifier", fake.url + "/")
    assert site.config()["verifier"]["url"] == fake.url                 # trailing slash dropped
    r = ep.test(site, "verifier")
    assert r["status"] == "ok" and r["models"] == ["qwen-vl"] and r["vision"] is True
    assert "qwen-vl" in r["detail"]


def test_llm_refuses_text_only_model(site, fake):
    fake.vision = False
    ep.set_address(site, "detector", "verifier", fake.url)
    r = ep.test(site, "verifier")
    assert r["status"] == "fail" and "does not accept images" in r["detail"] and "mmproj" in r["fix"]


def test_llm_vision_unknown_on_other_servers(site, fake):
    fake.props = False                                                  # not llama.cpp: no /props
    ep.set_address(site, "detector", "verifier", fake.url)
    r = ep.test(site, "verifier")
    assert r["status"] == "ok" and r["vision"] is None and "not reported" in r["detail"]


def test_llm_loading_and_down(site, fake):
    ep.set_address(site, "detector", "verifier", fake.url)
    fake.loading = True
    assert ep.test(site, "verifier")["status"] == "warn"
    fake.srv.shutdown()
    fake.srv.server_close()
    r = ep.test(site, "verifier")
    assert r["status"] == "fail" and r["detail"] == "no answer"


def test_api_key_flow_never_shows_the_key(site, fake, capsys, monkeypatch):
    fake.key = "s3cret-key-xyz"
    ep.set_address(site, "detector", "verifier", fake.url)
    assert ep.test(site, "verifier")["detail"] == "the server wants an API key"
    ep.set_api_key(site, "detector", "verifier", "wrong")
    assert ep.test(site, "verifier")["detail"] == "the API key was refused"
    ep.set_api_key(site, "detector", "verifier", fake.key)
    assert ep.test(site, "verifier")["status"] == "ok"
    assert stat.S_IMODE(site.env_path.stat().st_mode) == 0o600
    assert "s3cret" not in site.config_path.read_text()
    assert "s3cret" not in json.dumps(ep.listing(site))
    monkeypatch.setenv("SG_ROOT", str(site.root))
    assert cli.main(["endpoints", "list"]) == 0
    assert cli.main(["endpoints", "test", "verifier"]) == 0
    assert "s3cret" not in capsys.readouterr().out
    ep.set_api_key(site, "detector", "verifier", None)
    assert ep.listing(site)[0]["api_key_set"] is False


def test_model_name_checked_against_server(site, fake):
    fake.models = ["qwen-vl", "gemma"]
    ep.set_address(site, "detector", "verifier", fake.url)
    ep.set_model(site, "detector", "verifier", "nosuch")
    assert site.config()["verifier"]["model"] == "nosuch"
    r = ep.test(site, "verifier")
    assert r["status"] == "warn" and "nosuch" in r["detail"]
    ep.set_model(site, "detector", "verifier", "gemma")
    assert ep.test(site, "verifier")["status"] == "ok"
    with pytest.raises(ConfigError):
        ep.set_model(site, "detector", "verifier", 'a"b')


def test_bad_addresses_refused(site):
    for bad, words in (("ftp://x", "http://"), ("http://h:1/v1", "without /v1"), ("http://u:p@h:1", "API key"),
                       ("http://a b", "without spaces")):
        with pytest.raises(ConfigError, match=words):
            ep.set_address(site, "detector", "verifier", bad)
    with pytest.raises(ConfigError):
        ep.find("detector", "frigate")


def test_mqtt_host_and_port(site):
    ep.set_address(site, "detector", "mqtt", "127.0.0.1:1")
    env = site.env_path.read_text()
    assert "SG_MQTT_HOST=127.0.0.1" in env and "SG_MQTT_PORT=1" in env
    assert ep.listing(site)[1]["address"] == "127.0.0.1:1"
    assert ep.test(site, "mqtt")["status"] == "fail"
    with pytest.raises(ConfigError):
        ep.set_address(site, "detector", "mqtt", "user@host")


def test_gpu_role_profiler_falls_back_and_container_names(site, tmp_path, fake):
    d = site.root / "deploy" / "gpuserver" / "squirrel-guard"
    d.mkdir(parents=True)
    (d / "config.toml").write_text('[frigate]\napi = "http://frigate:5000"  # internal\n\n'
                                   f'[verifier]\nurl = "{fake.url}"\n\n[profiler]\nframes = 6\n')
    rows = {r["id"]: r for r in ep.listing(site, "gpu")}
    assert rows["profiler_llm"]["source"] == "fallback" and rows["profiler_llm"]["address"] == fake.url
    assert ep.test(site, "profiler_llm", "gpu")["status"] == "ok"
    fr = ep.test(site, "frigate", "gpu")
    assert fr["status"] == "info" and "container name" in fr["detail"]
    ep.set_address(site, "gpu", "profiler_llm", fake.url)
    assert "llm_url" in (d / "config.toml").read_text() and "# internal" in (d / "config.toml").read_text()
    ep.set_address(site, "gpu", "frigate", fake.url)
    assert ep.test(site, "frigate", "gpu")["status"] == "ok"
    ep.set_api_key(site, "gpu", "profiler_llm", "k")
    assert "SG_LLM_API_KEY=k" in (site.root / "deploy" / "gpuserver" / "sg.env").read_text()


def test_recorder_tsa_empty_means_off(site, tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    site.set_setting("recorder_dir", str(rec))
    row = ep.listing(site, "recorder")[0]
    assert row["source"] == "default" and "digicert" in row["address"]
    ep.set_address(site, "recorder", "tsa", "")
    assert (rec / ".env").read_text() == "TSA_URL=\n"
    assert ep.listing(site, "recorder")[0]["source"] == "none"
    assert ep.test(site, "tsa", "recorder")["status"] == "info"
    ep.set_address(site, "recorder", "tsa", "http://tsa.example")
    assert (rec / ".env").read_text() == "TSA_URL=http://tsa.example\n"


def test_compose_reads_tsa_from_env():
    text = (REPO / "deploy" / "recorder" / "compose.yaml").read_text()
    assert text.count("TSA_URL: ${TSA_URL-http://timestamp.digicert.com}") == 2


def test_programs_send_key_and_model(fake, monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from squirrel_guard import attributes
    from squirrel_guard.verifier import Verifier
    monkeypatch.setenv("SG_VERIFIER_API_KEY", "vk")
    img = np.zeros((40, 40, 3), np.uint8)
    assert Verifier(fake.url, model="qwen-vl")(img) == "squirrel"
    assert fake.seen[-1] == ("/v1/chat/completions", "Bearer vk", "qwen-vl")
    attributes.describe(fake.url, img, model="")                      # falls back to the verifier's key
    assert fake.seen[-1][1:] == ("Bearer vk", "local")
    monkeypatch.setenv("SG_LLM_API_KEY", "lk")
    attributes.describe_vehicle(fake.url, img, model="m2")
    assert fake.seen[-1][1:] == ("Bearer lk", "m2")


def test_web_endpoint_api(site, fake):
    srv, url = web.make_server(site, port=18790, token="tok")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = url.split("/?")[0]

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None)
        req.add_header("Cookie", "sg_console=tok")
        if body is not None:
            req.add_header("Content-Type", "application/json")
            req.add_header("X-SG-Token", "tok")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    try:
        out = call("/api/endpoint", {"id": "verifier", "what": "address", "value": fake.url})
        assert out["test"]["status"] == "ok"
        assert call("/api/endpoint", {"id": "verifier", "what": "key", "value": "zz"})["ok"]
        listing = call("/api/endpoints")
        assert listing["endpoints"][0]["api_key_set"] is True and "zz" not in json.dumps(listing)
        assert call("/api/endpoint", {"id": "verifier", "what": "test"})["test"]["status"] == "ok"
    finally:
        srv.shutdown()
        srv.server_close()

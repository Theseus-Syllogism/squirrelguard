"""Outside addresses the installation talks to: what they are, where each is stored, and a test for each.

One list per role. Each endpoint has an address (kept in a config file, or in a private env file when the address
is not a secret but belongs to the containers' environment), an optional API key (always in a private env file,
never shown) and, for model servers, an optional model name.

Tests only read: GET requests, a TCP connect, or (timestamp service) one timestamp of a fixed test hash.
Nothing here returns a secret: API keys are reported as set / not set, and addresses are shown with any
user:password part masked.
"""
from __future__ import annotations

import importlib.util
import json
import socket
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import ConfigError, Site, _atomic_write, _read_env, scrub, set_toml

UA = "squirrel-guard-console"


@dataclass(frozen=True)
class Endpoint:
    id: str
    title: str
    kind: str                 # llm | frigate | http | mqtt | tsa
    help: str
    store: str                # config | env
    key: str                  # section.key in the config file, or the env variable name
    api_key: str = ""         # env variable for an API key ("" = this endpoint takes none)
    model_key: str = ""       # section.key for the model name ("" = no model name)
    fallback: str = ""        # what happens when the address is empty
    default: str = ""
    needs_vision: bool = False


DEFS: dict[str, list[Endpoint]] = {
    "detector": [
        Endpoint("verifier", "Second-opinion model (llama.cpp or other OpenAI-compatible server)", "llm",
                 "Looks at each detection crop and says what it is. Must accept images (llama.cpp: --mmproj).",
                 "config", "verifier.url", "SG_VERIFIER_API_KEY", "verifier.model",
                 "no second opinion: see verifier.trust_detector_if_down", "http://127.0.0.1:8080", True),
        Endpoint("mqtt", "Message broker (MQTT)", "mqtt", "Where detections are published for the recorder.",
                 "env", "SG_MQTT_HOST", fallback="events stay on this machine"),
    ],
    "gpu": [
        Endpoint("verifier", "Second-opinion model (llama.cpp or other OpenAI-compatible server)", "llm",
                 "Checks Frigate's squirrel events before the deterrent plays. Must accept images.",
                 "config", "verifier.url", "SG_VERIFIER_API_KEY", "verifier.model",
                 "no second opinion", "http://host.docker.internal:8080", True),
        Endpoint("profiler_llm", "Description model for people and vehicles", "llm",
                 "Writes the apparent attributes of each person and vehicle visit. Must accept images.",
                 "config", "profiler.llm_url", "SG_LLM_API_KEY", "profiler.llm_model",
                 "uses the second-opinion model", "", True),
        Endpoint("frigate", "Frigate API", "frigate", "Events, snapshots and clips. Use the internal port 5000, not 8971.",
                 "config", "frigate.api", default="http://frigate:5000"),
        Endpoint("review", "Review link in notifications", "http", "The address phones open from a notification.",
                 "config", "notify.review_url"),
        Endpoint("mqtt", "Message broker (MQTT)", "mqtt", "Frigate's events arrive here; detections are published here.",
                 "env", "SG_MQTT_HOST", fallback="required: the listener and the profiler cannot start without it"),
    ],
    "recorder": [
        Endpoint("tsa", "Timestamp service (RFC 3161)", "tsa",
                 "Signs the time of every evidence package and of the daily recording seals. Only a hash is sent.",
                 "env", "TSA_URL", fallback="no outside timestamps (packages say so)",
                 default="http://timestamp.digicert.com"),
    ],
}


def _files(site: Site, role: str) -> tuple[Path, Path]:
    """(config file, env file) that the role's programs read."""
    if role == "gpu":
        d = site._stack_dir("gpu")
        return d / "squirrel-guard" / "config.toml", d / "sg.env"
    if role == "recorder":
        d = site._stack_dir("recorder")
        return d / "compose.yaml", d / ".env"
    return site.config_path, site.env_path


def _toml(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path.name} has a syntax error: {e}") from None


def _get(cfg: dict, dotted: str) -> Any:
    sec, key = dotted.split(".", 1)
    return (cfg.get(sec) or {}).get(key)


def find(role: str, eid: str) -> Endpoint:
    for e in DEFS.get(role, []):
        if e.id == eid:
            return e
    raise ConfigError(f"no endpoint {eid!r} for this role; choose one of: {', '.join(e.id for e in DEFS.get(role, [])) or 'none'}")


def _address(site: Site, role: str, e: Endpoint) -> tuple[str, str]:
    """(address actually used, where it comes from: set | default | fallback | none)."""
    cfg_path, env_path = _files(site, role)
    if e.store == "env":
        env = _read_env(env_path)
        if e.key in env:
            v = env[e.key]
            if e.kind == "mqtt" and v:
                v = f"{v}:{env.get('SG_MQTT_PORT') or 1883}"
            return v, "set" if v else "none"
        return (e.default, "default") if e.default else ("", "none")
    cfg = _toml(cfg_path)
    if not cfg and role == "detector" and site.example_path.exists():
        cfg, from_example = _toml(site.example_path), True        # the install step copies the example
    else:
        from_example = False
    v = _get(cfg, e.key)
    if isinstance(v, str) and v:
        return v, "default" if from_example else "set"
    if e.id == "profiler_llm":
        ver = _get(cfg, "verifier.url")
        return (ver, "fallback") if ver else ("", "none")
    return ("", "none")


def listing(site: Site, role: str | None = None) -> list[dict]:
    role = role or site.role
    cfg_path, env_path = _files(site, role)
    env = _read_env(env_path)
    cfg = _toml(cfg_path) if cfg_path.suffix == ".toml" else {}
    out = []
    for e in DEFS.get(role, []):
        addr, source = _address(site, role, e)
        out.append({
            "id": e.id, "title": e.title, "kind": e.kind, "help": e.help,
            "address": scrub(addr), "source": source, "fallback": e.fallback, "default": e.default,
            "stored_in": f"{(env_path if e.store == 'env' else cfg_path).name} ({e.key})",
            "api_key": e.api_key, "api_key_set": bool(e.api_key and env.get(e.api_key)),
            "model_key": e.model_key, "model": str(_get(cfg, e.model_key) or "") if e.model_key else "",
        })
    return out


# ------------------------------------------------------------------------------------------------ changing
def _writes(fn):
    """File permission problems become a message, not a traceback (the recorder's folder is usually root's)."""
    import functools

    @functools.wraps(fn)
    def wrap(*a, **k):
        try:
            return fn(*a, **k)
        except PermissionError as err:
            raise ConfigError(f"cannot write {err.filename}: run the console as a user that may change it (e.g. with sudo)") from None
    return wrap


@_writes
def set_address(site: Site, role: str, eid: str, value: str) -> str:
    """Store a new address ("" clears it). Returns the stored value as shown."""
    e = find(role, eid)
    value = value.strip()
    if any(c in value for c in "\n\r\0 \t"):
        raise ConfigError("enter one address, without spaces")
    cfg_path, env_path = _files(site, role)
    if e.kind == "mqtt":
        host, port = value, ""
        if value.count(":") == 1:
            host, port = value.split(":")
        if value and (not host or "/" in host or "@" in host):
            raise ConfigError("enter the broker's host name or address, optionally with :port (e.g. 192.0.2.10:1883)")
        site.env_set("SG_MQTT_HOST", host or None, path=env_path)
        if port:
            site.env_set("SG_MQTT_PORT", port, path=env_path)
        return value
    if value:
        u = urllib.parse.urlsplit(value)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ConfigError("enter an address that starts with http:// or https://")
        if u.username or u.password:
            raise ConfigError("do not put a user name or password in the address; use the API key instead")
        if e.kind in ("llm", "frigate") and u.path.rstrip("/") not in ("",):
            if u.path.rstrip("/").endswith("/v1"):
                raise ConfigError("give the server address without /v1 (e.g. http://127.0.0.1:8080)")
            raise ConfigError("give the server address only, without a path")
        value = value.rstrip("/") if e.kind in ("llm", "frigate") else value
    if e.store == "env":
        if e.kind == "tsa" and value == "":
            _env_line(env_path, e.key, "")           # present but empty = switched off (absent = the default)
        else:
            site.env_set(e.key, value or None, path=env_path)
        return value
    if not cfg_path.exists():
        raise ConfigError(f"{cfg_path} does not exist yet: run the install step that creates it first")
    set_toml(cfg_path, e.key, value)
    return value


def _env_line(path: Path, name: str, value: str) -> None:
    """Write NAME= with an empty value (env_set refuses empty values on purpose)."""
    lines = path.read_text().splitlines() if path.exists() else []
    lines = [l for l in lines if not l.strip().startswith(name + "=")] + [f"{name}={value}"]
    _atomic_write(path, "\n".join(lines) + "\n", 0o600)


@_writes
def set_api_key(site: Site, role: str, eid: str, value: str | None) -> None:
    e = find(role, eid)
    if not e.api_key:
        raise ConfigError(f"{e.title} takes no API key")
    site.env_set(e.api_key, value, path=_files(site, role)[1])


@_writes
def set_model(site: Site, role: str, eid: str, value: str) -> str:
    e = find(role, eid)
    if not e.model_key:
        raise ConfigError(f"{e.title} has no model name")
    value = value.strip()
    if any(c in value for c in "\n\r\0\"'\\") or len(value) > 200:
        raise ConfigError("enter a model name on one line, without quotes")
    cfg_path = _files(site, role)[0]
    if not cfg_path.exists():
        raise ConfigError(f"{cfg_path} does not exist yet: run the install step that creates it first")
    set_toml(cfg_path, e.model_key, value)
    return value


# ------------------------------------------------------------------------------------------------ testing
def _http(url: str, headers: dict | None = None, timeout: float = 4.0) -> tuple[int, bytes, float]:
    """(status, body, seconds). status 0 = no answer. Follows no redirects."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    t0 = time.monotonic()
    try:
        with opener.open(urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})}), timeout=timeout) as r:
            return r.status, r.read(1 << 20), time.monotonic() - t0
    except urllib.error.HTTPError as err:
        return err.code, err.read(1 << 16) if err.fp else b"", time.monotonic() - t0
    except Exception:
        return 0, b"", time.monotonic() - t0


def _local_target(url: str) -> tuple[str, str]:
    """Container addresses as seen from this machine. Returns (url to test, note)."""
    u = urllib.parse.urlsplit(url)
    host = u.hostname or ""
    if host == "host.docker.internal":
        return url.replace("host.docker.internal", "127.0.0.1", 1), "tested as 127.0.0.1 (host.docker.internal is this machine)"
    if host and "." not in host and host != "localhost":
        try:
            socket.getaddrinfo(host, None)
        except OSError:
            return "", f"“{host}” is a container name inside the stack's network; it cannot be reached from here"
    return url, ""


def _result(status: str, detail: str, **extra) -> dict:
    return {"status": status, "detail": detail, **extra}


def _test_llm(e: Endpoint, url: str, key: str, model: str) -> dict:
    auth = {"Authorization": f"Bearer {key}"} if key else {}
    code, body, secs = _http(url + "/health", auth)
    if code == 0:
        return _result("fail", "no answer", fix="Start the model server, or correct the address.")
    if code == 503:
        return _result("warn", "the server is up but still loading its model", ms=round(secs * 1000))
    code_m, body_m, _ = _http(url + "/v1/models", auth)
    if code_m in (401, 403):
        return _result("fail", "the API key was refused" if key else "the server wants an API key",
                       fix="Set the API key for this endpoint." if not key else "Store the server's current key.")
    if code_m != 200:
        return _result("fail", f"answers, but /v1/models gave HTTP {code_m}: not an OpenAI-compatible server?",
                       fix="Give the server's base address (llama.cpp: http://host:8080).")
    try:
        data = json.loads(body_m)
        models = [str(m.get("id")) for m in data.get("data", []) if isinstance(m, dict)][:30]
    except (ValueError, AttributeError):
        models = []
    vision = None
    code_p, body_p, _ = _http(url + "/props", auth)
    if code_p == 200:
        try:
            props = json.loads(body_p)
            mod = props.get("modalities")
            if isinstance(mod, dict) and "vision" in mod:
                vision = bool(mod["vision"])
        except (ValueError, AttributeError):
            pass
    out = {"ms": round(secs * 1000), "models": models, "vision": vision, "llama_cpp": code_p == 200}
    names = ", ".join(models) or "none listed"
    if e.needs_vision and vision is False:
        return _result("fail", f"answers ({names}), but the loaded model does not accept images",
                       fix="Start llama.cpp with the model's --mmproj file, or point this at a vision model.", **out)
    if model and models and model not in models and len(models) > 1:
        return _result("warn", f"answers, but has no model “{model}” (it has: {names})",
                       fix="Pick one of the listed models.", **out)
    note = "" if vision else (" (image support not reported by this server)" if e.needs_vision else "")
    return _result("ok", f"answers in {out['ms']} ms; models: {names}{note}", **out)


def _test_frigate(url: str) -> dict:
    code, body, secs = _http(url + "/api/version")
    if code == 200:
        return _result("ok", f"Frigate {body.decode(errors='replace').strip()[:40]} answers in {round(secs * 1000)} ms")
    if code in (401, 403):
        return _result("fail", "this is Frigate's signed-in port", fix="Use the internal port 5000 (http://frigate:5000).")
    if code == 0:
        return _result("fail", "no answer", fix="Check that Frigate is running and the address.")
    return _result("fail", f"HTTP {code} from /api/version: is this Frigate?")


def _test_http(url: str) -> dict:
    code, _, secs = _http(url, timeout=6)
    if code == 0:
        return _result("fail", "no answer from this machine", fix="Phones may still reach it; check from a phone.")
    if 300 <= code < 400:
        return _result("ok", f"answers (HTTP {code}, a sign-in redirect is normal) in {round(secs * 1000)} ms")
    return _result("ok" if code < 400 or code in (401, 403) else "warn", f"answers with HTTP {code} in {round(secs * 1000)} ms")


def _test_mqtt(addr: str) -> dict:
    host, _, port = addr.rpartition(":")
    try:
        t0 = time.monotonic()
        with socket.create_connection((host, int(port)), timeout=3):
            pass
        return _result("ok", f"the broker port answers in {round((time.monotonic() - t0) * 1000)} ms")
    except (OSError, ValueError):
        return _result("fail", "the broker port did not answer within 3 s", fix="Check the host, the port and the firewall.")


def _test_tsa(site: Site, url: str) -> dict:
    src = site.root / "deploy" / "recorder" / "persons-ui" / "tsa.py"
    spec = importlib.util.spec_from_file_location("sg_tsa", src)
    if not spec or not src.exists():
        return _result("warn", "the timestamp client (deploy/recorder/persons-ui/tsa.py) is not in this checkout")
    tsa = sys.modules.get("sg_tsa")
    if tsa is None:
        tsa = importlib.util.module_from_spec(spec)
        sys.modules["sg_tsa"] = tsa                     # dataclasses look the module up while it loads
        spec.loader.exec_module(tsa)
    try:
        s = tsa.stamp(tsa.sha256_hex(b"squirrel-guard endpoint test"), url, timeout=15)
    except Exception as err:          # TSAError carries a readable message
        return _result("fail", str(err) or type(err).__name__, fix="Try another RFC 3161 service, or clear it to switch timestamps off.")
    return _result("ok", f"timestamp issued in {s.round_trip_s:.1f} s; its clock differs from this machine's by {s.offset_s:+.1f} s")


def test(site: Site, eid: str, role: str | None = None) -> dict:
    role = role or site.role
    e = find(role, eid)
    addr, source = _address(site, role, e)
    base = {"id": e.id, "title": e.title, "address": scrub(addr), "source": source}
    if not addr:
        return {**base, **_result("info", "not set: " + (e.fallback or "nothing uses it"))}
    if e.kind == "mqtt":
        return {**base, **_test_mqtt(addr)}
    target, note = _local_target(addr)
    if not target:
        return {**base, **_result("info", note)}
    env = _read_env(_files(site, role)[1])
    cfg_path = _files(site, role)[0]
    cfg = _toml(cfg_path) if cfg_path.suffix == ".toml" else {}
    if e.kind == "llm":
        # same order as the programs: the description model falls back to the verifier's key, address and model
        key = env.get(e.api_key, "") or (env.get("SG_VERIFIER_API_KEY", "") if e.id == "profiler_llm" else "")
        mkey = "verifier.model" if source == "fallback" else e.model_key
        model = str(_get(cfg, mkey) or "") if mkey else ""
        r = _test_llm(e, target.rstrip("/"), key, model)
    elif e.kind == "frigate":
        r = _test_frigate(target.rstrip("/"))
    elif e.kind == "tsa":
        r = _test_tsa(site, target)
    else:
        r = _test_http(target)
    if note:
        r["detail"] += f" ({note})"
    return {**base, **r}


def test_all(site: Site, role: str | None = None) -> list[dict]:
    role = role or site.role
    return [test(site, e.id, role) for e in DEFS.get(role, [])]

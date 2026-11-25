"""squirrel-guard console core: everything the `sg` command, the terminal UI and the web GUI share.

Standard library only, so the installer runs before any dependency is installed (Python 3.11+ for tomllib).

Rules this module keeps:
- Secrets (the camera stream address, passwords, the database address) are written, never read back: no function
  here returns a secret value, and checks report "reachable" without naming the host.
- config.toml is edited line by line, so comments and layout survive; every write is validated by parsing the
  whole file again, and the previous version is kept as config.toml.bak.
- Install steps are a fixed list. Nothing outside it can be run through the console.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
UNIT = "squirrel-guard.service"
IMAGE = "squirrel-guard-detector"
RUNTIMES = {"venv": "Python environment + systemd service", "docker": "Docker container"}

ROLES = {
    "detector": ("Detector workstation", "Watches one camera stream, confirms detections and triggers the deterrent."),
    "recorder": ("Recorder host", "Runs the container stack: recording, event database, alerts, dashboards, review UI."),
    "gpu": ("GPU server", "Runs the object detector for all cameras (Frigate) and the listener that acts on its events."),
}

# ---------------------------------------------------------------------------------------------------------------
# TOML: comment-preserving, line-based editing
# ---------------------------------------------------------------------------------------------------------------
_SEC = re.compile(r"^\s*\[([A-Za-z0-9_.\-]+)\]\s*(#.*)?$")
_ARR = re.compile(r"^\s*\[\[")
_KV = re.compile(r"^(\s*)([A-Za-z0-9_\-]+)(\s*=\s*)(.*)$")

# section.key -> rule. kind "num" lets a whole number become a decimal (and back).
RULES: dict[str, dict[str, Any]] = {
    "camera.width": {"min": 160, "max": 3840}, "camera.height": {"min": 90, "max": 2160},
    "camera.fps": {"min": 0.2, "max": 30, "kind": "num"},
    "motion.min_area_frac": {"min": 0, "max": 1, "kind": "num"}, "motion.threshold": {"min": 1, "max": 255},
    "detector.conf": {"min": 0.01, "max": 1, "kind": "num"}, "detector.threads": {"min": 1, "max": 64},
    "trigger.window": {"min": 1, "max": 60}, "trigger.need": {"min": 1, "max": 60},
    "verifier.timeout": {"min": 1, "max": 300, "kind": "num"}, "actuator.cooldown_s": {"min": 0, "max": 86400, "kind": "num"},
}
URL_KEYS = {"url", "api", "review_url"}


class ConfigError(ValueError):
    """A value or file the console refuses, with a message fit to show the operator."""


@dataclass
class Field:
    section: str
    key: str
    value: Any
    raw: str = ""
    help: str = ""
    line: int = -1            # index in the file, -1 = not in the file (default from the example)
    editable: bool = True
    default: Any = None

    @property
    def id(self) -> str:
        return f"{self.section}.{self.key}"

    @property
    def kind(self) -> str:
        v = self.value if self.value is not None else self.default
        return {bool: "bool", int: "int", float: "float", str: "text", list: "list"}.get(type(v), "other")

    def public(self) -> dict:
        d = asdict(self)
        d.update(id=self.id, kind=self.kind, in_file=self.line >= 0, rule=RULES.get(self.id, {}))
        return d


def _split_comment(rest: str) -> tuple[str, str]:
    """'"a # b"  # note' -> ('"a # b"', '  # note'). Tracks quotes so a # inside a string is kept."""
    q = None
    i = 0
    while i < len(rest):
        c = rest[i]
        if q:
            if c == "\\" and q == '"':
                i += 1
            elif c == q:
                q = None
        elif c in "\"'":
            q = c
        elif c == "#":
            val = rest[:i].rstrip()
            return val, rest[len(val):]
        i += 1
    return rest.rstrip(), ""


def _parse_scalar(text: str) -> tuple[bool, Any]:
    try:
        return True, tomllib.loads("x = " + text)["x"]
    except tomllib.TOMLDecodeError:
        return False, None


def parse_fields(text: str) -> list[Field]:
    """Every `key = value` line of a TOML text, with its section and its comments as help."""
    out: list[Field] = []
    section, skip, pending = "", False, []
    for n, line in enumerate(text.splitlines()):
        s = line.strip()
        if _ARR.match(line):
            skip, pending = True, []          # arrays of tables are left to a text editor
            continue
        m = _SEC.match(line)
        if m:
            section, skip, pending = m.group(1), False, []
            continue
        if s.startswith("#"):
            body = s.lstrip("#").strip()
            if not _KV.match(body) or not _parse_scalar(_split_comment(_KV.match(body).group(4))[0])[0]:
                pending.append(body)          # prose comment (a commented-out setting is not help text)
            continue
        if not s:
            pending = []
            continue
        m = _KV.match(line)
        if not m or skip:
            pending = []
            continue
        val, comment = _split_comment(m.group(4))
        ok, value = _parse_scalar(val)
        helps = pending + ([comment.strip().lstrip("#").strip()] if comment.strip() else [])
        out.append(Field(section, m.group(2), value if ok else None, val, " ".join(h for h in helps if h), n,
                         editable=ok and not isinstance(value, dict)))
        pending = []
    return out


def to_toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(to_toml(v) for v in value) + "]"
    raise ConfigError(f"cannot write a value of type {type(value).__name__}")


def coerce(fld: Field, text: str) -> Any:
    """Turn what a person typed into the field's type. Raises ConfigError with a plain message."""
    text = text.strip()
    kind, rule = fld.kind, RULES.get(fld.id, {})
    if kind == "bool":
        low = text.lower()
        if low in ("true", "yes", "on", "1", "y"):
            return True
        if low in ("false", "no", "off", "0", "n"):
            return False
        raise ConfigError("enter yes or no")
    if kind in ("int", "float"):
        try:
            num = float(text) if (kind == "float" or rule.get("kind") == "num") else int(text)
        except ValueError:
            raise ConfigError("enter a whole number" if kind == "int" and rule.get("kind") != "num" else "enter a number") from None
        if num != num or num in (float("inf"), float("-inf")):
            raise ConfigError("enter a finite number")
        if isinstance(num, float) and num.is_integer() and kind == "int":
            num = int(num)
        if "min" in rule and num < rule["min"] or "max" in rule and num > rule["max"]:
            raise ConfigError(f"must be between {rule.get('min', '-inf')} and {rule.get('max', 'inf')}")
        if fld.key == "volume" and not 0 <= num <= 100:
            raise ConfigError("must be between 0 and 100")
        return num
    if kind == "list":
        ok, val = _parse_scalar(text) if text.startswith("[") else (True, [p.strip() for p in text.split(",") if p.strip()])
        if not ok or not isinstance(val, list):
            raise ConfigError("enter a list: items separated by commas, or a TOML array")
        old = fld.value if fld.value is not None else fld.default
        if old and all(isinstance(o, (int, float)) and not isinstance(o, bool) for o in old) and not text.startswith("["):
            try:
                val = [float(v) if "." in v else int(v) for v in val]
            except ValueError:
                raise ConfigError("enter numbers separated by commas") from None
        return val
    if kind == "text":
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
            text = text[1:-1]
        if "\n" in text or "\r" in text:
            raise ConfigError("one line only")
        if fld.key in URL_KEYS and text and not re.match(r"^https?://[^\s/]+", text):
            raise ConfigError("enter an address that starts with http:// or https://")
        return text
    raise ConfigError("this setting has to be changed in a text editor")


def set_toml(path: Path, fid: str, value: Any) -> None:
    """Store section.key = value in a TOML file line by line (comments kept; a missing key or section is added).
    The whole file is parsed again before writing; the previous version is kept as <name>.bak."""
    section, key = fid.split(".", 1)
    text = path.read_text() if path.exists() else ""
    lines = text.split("\n")
    have = next((f for f in parse_fields(text) if f.id == fid), None)
    if have is not None:
        m = _KV.match(lines[have.line])
        _, comment = _split_comment(m.group(4))
        lines[have.line] = f"{m.group(1)}{m.group(2)}{m.group(3)}{to_toml(value)}{comment}"
    else:
        new = f"{key} = {to_toml(value)}"
        start = next((i for i, l in enumerate(lines) if (m := _SEC.match(l)) and m.group(1) == section), None)
        if start is None:
            while lines and not lines[-1].strip():
                lines.pop()
            lines += ["", f"[{section}]", new, ""]
        else:
            end = next((i for i in range(start + 1, len(lines)) if _SEC.match(lines[i]) or _ARR.match(lines[i])), len(lines))
            while end > start + 1 and not lines[end - 1].strip():
                end -= 1
            lines.insert(end, new)
    out = "\n".join(lines)
    try:
        cfg = tomllib.loads(out)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"that value would break the file ({e}); nothing was saved") from None
    _cross_check(cfg)
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    _atomic_write(path, out)


def _read_env(path: Path) -> dict[str, str]:
    """NAME=value lines of an env file. Internal only: values never leave the console modules."""
    out: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("'\"")
    except OSError:
        pass
    return out


def _cross_check(cfg: dict) -> None:
    trig = cfg.get("trigger", {})
    if isinstance(trig.get("need"), int) and isinstance(trig.get("window"), int) and trig["need"] > trig["window"]:
        raise ConfigError(f"trigger.need ({trig['need']}) cannot be larger than trigger.window ({trig['window']})")


def _atomic_write(path: Path, text: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode is None:
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".")
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------------------------------------------
# Environment file (secrets): names and "set / not set" only
# ---------------------------------------------------------------------------------------------------------------
ENV_VARS = [
    ("SG_RTSP_URL", "Camera stream address, including its user name and password (rtsp://user:password@camera:554/ch1)", True),
    ("SG_MQTT_HOST", "Message broker host on the recorder (leave unset to run without publishing events)", False),
    ("SG_MQTT_PORT", "Message broker port (default 1883)", False),
    ("SG_MQTT_USER", "Message broker user name", False),
    ("SG_MQTT_PASS", "Message broker password", True),
    ("SG_FRIGATE_API", "Address of the Frigate API, for the listener on the GPU server", False),
    ("SG_VERIFIER_API_KEY", "API key of the second-opinion model server, if it was started with one (llama.cpp --api-key)", True),
    ("SG_LLM_API_KEY", "API key of the description model server (default: the second-opinion model's key)", True),
]
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")


@dataclass
class Check:
    id: str
    title: str
    status: str               # ok | warn | fail | info
    detail: str = ""
    fix: str = ""


@dataclass
class Step:
    id: str
    title: str
    why: str
    done: bool | None         # None = cannot tell
    commands: list[list[str]] = field(default_factory=list)   # shown before running; empty = done by the console itself
    needs: str = ""           # name of a secret the step asks for
    needs_help: str = ""
    manual: str = ""          # set when the console cannot do the step; what the operator does instead
    shown: list[str] = field(default_factory=list)            # the commands as text, paths relative to the checkout


def run(argv: list[str], cwd: Path | None = None, timeout: float = 20, env: dict | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(argv, cwd=cwd, timeout=timeout, capture_output=True, text=True, env=env)
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, f"{argv[0]}: not installed"
    except subprocess.TimeoutExpired:
        return 124, f"{argv[0]}: no answer after {timeout:g} s"
    except OSError as e:
        return 126, str(e)


_CRED = re.compile(r"(\b[a-z][a-z0-9+.\-]*://)[^\s/@:]*:[^\s/@]*@", re.I)
_CTRL = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|[\x00-\x08\x0b-\x1f\x7f]")


def scrub(line: str) -> str:
    """A log line fit to display: passwords inside addresses masked (ffmpeg prints the stream address), no control codes."""
    return _CRED.sub(r"\1***@", _CTRL.sub("", line))


def _compose_env() -> dict:
    """The environment for commands: containers write ./data as the person running the console."""
    return {**os.environ, "SG_UID": str(os.getuid()), "SG_GID": str(os.getgid())}


class Site:
    """One checkout of squirrel-guard and the machine it runs on."""

    def __init__(self, root: Path | str | None = None, config: Path | str | None = None, home: Path | str | None = None):
        self.root = Path(root or ROOT).resolve()
        self.config_path = Path(config).resolve() if config else self.root / "config.toml"
        self.example_path = self.root / "config.example.toml"
        self.env_path = self.config_path.parent / ".env"
        self.home = Path(home) if home else Path.home()
        self.settings_path = self.root / "data" / "console.json"
        self.venv_py = self.root / ".venv" / "bin" / "python"

    # ------------------------------------------------------------------ console settings
    def settings(self) -> dict:
        try:
            s = json.loads(self.settings_path.read_text())
        except (OSError, ValueError):
            s = {}
        s.setdefault("role", "detector")
        s.setdefault("recorder_dir", "/srv/camstack")
        s.setdefault("runtime", "venv")
        return s

    def set_setting(self, key: str, value: str) -> None:
        if key == "role" and value not in ROLES:
            raise ConfigError(f"unknown role {value!r}: choose one of {', '.join(ROLES)}")
        if key == "recorder_dir":
            if not value.startswith("/") or any(c in value for c in "\n\r\0") or ".." in Path(value).parts:
                raise ConfigError("enter a full path such as /srv/camstack")
            value = str(Path(value))
        if key == "runtime" and value not in RUNTIMES:
            raise ConfigError("choose venv (Python environment + systemd service) or docker (container)")
        if key not in ("role", "recorder_dir", "runtime"):
            raise ConfigError(f"unknown setting {key!r}")
        s = self.settings()
        s[key] = value
        _atomic_write(self.settings_path, json.dumps(s, indent=1) + "\n")

    @property
    def role(self) -> str:
        return self.settings()["role"]

    @property
    def docker(self) -> bool:
        """The detector runs as a container (compose.yaml in the checkout) instead of a Python environment."""
        return self.settings()["runtime"] == "docker"

    # ------------------------------------------------------------------ configuration
    def config_text(self) -> str:
        return self.config_path.read_text() if self.config_path.exists() else ""

    def config(self) -> dict:
        """Parsed config.toml ({} when missing). Raises ConfigError when it does not parse."""
        try:
            return tomllib.loads(self.config_text())
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{self.config_path.name} has a syntax error: {e}") from None

    def fields(self) -> list[Field]:
        """Settings in config.toml, plus settings the example knows that the file does not have yet."""
        have = parse_fields(self.config_text())
        ex = parse_fields(self.example_path.read_text()) if self.example_path.exists() else []
        exmap = {f.id: f for f in ex}
        for f in have:
            e = exmap.get(f.id)
            if e:
                f.default = e.value
                f.help = f.help or e.help
        seen = {f.id for f in have}
        missing = [Field(e.section, e.key, None, "", e.help, -1, e.editable, e.value) for e in ex if e.id not in seen]
        order: list[str] = []
        for f in have + missing:
            if f.section not in order:
                order.append(f.section)
        return sorted(have + missing, key=lambda f: (order.index(f.section), f.line if f.line >= 0 else 10 ** 6))

    def field(self, fid: str) -> Field:
        for f in self.fields():
            if f.id == fid:
                return f
        raise ConfigError(f"no setting called {fid!r} (see `sg config list`)")

    def create_config(self) -> bool:
        if self.config_path.exists():
            return False
        if not self.example_path.exists():
            raise ConfigError("config.example.toml is missing from this checkout")
        _atomic_write(self.config_path, self.example_path.read_text().replace(
            "# Copy to config.toml. ", "# ", 1), 0o644)
        return True

    def set_value(self, fid: str, typed: str) -> Any:
        """Validate and store one setting. Returns the stored value."""
        self.create_config()
        fld = self.field(fid)
        if not fld.editable:
            raise ConfigError(f"{fid} spans several lines or is a table: change it in a text editor")
        value = coerce(fld, typed)
        set_toml(self.config_path, fid, value)
        return value

    # ------------------------------------------------------------------ secrets
    def _env(self) -> dict[str, str]:
        """Internal only: values never leave the console modules."""
        return _read_env(self.env_path)

    def env_status(self) -> list[dict]:
        env = self._env()
        known = [{"name": n, "help": h, "secret": s, "set": bool(env.get(n))} for n, h, s in ENV_VARS]
        extra = [{"name": n, "help": "", "secret": True, "set": bool(v)} for n, v in env.items()
                 if n not in {k[0] for k in ENV_VARS}]
        return known + extra

    def env_set(self, name: str, value: str | None, path: Path | None = None) -> None:
        """Store (or with value=None remove) one secret. The file is kept private (mode 600)."""
        path = path or self.env_path
        if not _ENV_NAME.match(name):
            raise ConfigError("names are capital letters, digits and underscores, e.g. SG_RTSP_URL")
        if value is not None:
            value = value.strip()
            if not value or any(c in value for c in "\n\r\0"):
                raise ConfigError("enter a value on one line")
            if name == "SG_RTSP_URL" and not re.match(r"^rtsps?://[^\s]+$", value):
                raise ConfigError("the stream address starts with rtsp:// and has no spaces")
            if name.endswith("_PORT") and not (value.isdigit() and 0 < int(value) < 65536):
                raise ConfigError("enter a port number between 1 and 65535")
        lines = path.read_text().splitlines() if path.exists() else []
        kept = [l for l in lines if not re.match(rf"^\s*{re.escape(name)}\s*=", l)]
        if value is not None:
            kept.append(f"{name}={value}")
        _atomic_write(path, "\n".join(kept) + ("\n" if kept else ""), 0o600)

    # ------------------------------------------------------------------ helpers
    def _py(self) -> str:
        return str(self.venv_py) if self.venv_py.exists() else sys.executable

    def _missing_modules(self, mods: list[str]) -> list[str] | None:
        if not self.venv_py.exists():
            return None
        code = "import importlib.util as u,sys;print(' '.join(m for m in sys.argv[1:] if u.find_spec(m) is None))"
        rc, out = run([str(self.venv_py), "-c", code, *mods], timeout=30)
        return out.split() if rc == 0 else None

    def _model_path(self) -> Path | None:
        try:
            m = self.config().get("detector", {}).get("model")
        except ConfigError:
            return None
        return (self.config_path.parent / m) if isinstance(m, str) and m else None

    def _output_dir(self) -> Path:
        try:
            d = self.config().get("output", {}).get("dir", "data/events")
        except ConfigError:
            d = "data/events"
        return self.config_path.parent / str(d)

    def _unit_path(self) -> Path:
        return self.home / ".config" / "systemd" / "user" / UNIT

    def _stack_dir(self, role: str) -> Path:
        if role == "detector":
            return self.root
        return Path(self.settings()["recorder_dir"]) if role == "recorder" else self.root / "deploy" / "gpuserver"

    def _docker(self) -> tuple[bool, bool]:
        if not shutil.which("docker"):
            return False, False
        return True, run(["docker", "compose", "version"], timeout=10)[0] == 0

    def stack_status(self, role: str) -> list[dict] | None:
        """Containers of the role's stack: [{name, state}], or None when docker cannot be asked."""
        d = self._stack_dir(role)
        if not (d / "compose.yaml").exists():
            return None
        rc, out = run(["docker", "compose", "ps", "-a", "--format", "json"], cwd=d, timeout=20, env=_compose_env())
        if rc != 0:
            return None
        rows = []
        for chunk in ([out] if out.startswith("[") else out.splitlines()):
            try:
                item = json.loads(chunk)
            except ValueError:
                continue
            for c in item if isinstance(item, list) else [item]:
                rows.append({"name": c.get("Service") or c.get("Name", "?"), "state": c.get("State", "?"),
                             "status": c.get("Status", "")})
        return sorted(rows, key=lambda r: r["name"])

    # ------------------------------------------------------------------ checks
    def checks(self, role: str | None = None) -> list[Check]:
        role = role or self.role
        out: list[Check] = []
        add = lambda *a, **k: out.append(Check(*a, **k))
        v = sys.version_info
        add("python", "Python 3.11 or newer", "ok" if v >= (3, 11) else "fail", f"running {v.major}.{v.minor}.{v.micro}",
            "" if v >= (3, 11) else "Install Python 3.11 or newer.")
        if role == "detector":
            self._detector_checks(add)
        else:
            self._stack_checks(role, add)
        return out

    def _detector_checks(self, add: Callable) -> None:
        if self.docker:
            return self._detector_docker_checks(add)
        add("venv", "Private Python environment", "ok" if self.venv_py.exists() else "fail",
            str(self.venv_py.parent.parent) if self.venv_py.exists() else "not created",
            "" if self.venv_py.exists() else "Run the install step “Create the Python environment”.")
        miss = self._missing_modules(["onnxruntime", "cv2", "numpy", "requests", "paho"])
        if miss is None:
            add("packages", "Python packages", "fail" if not self.venv_py.exists() else "warn", "could not be checked",
                "Create the Python environment first.")
        else:
            add("packages", "Python packages", "fail" if miss else "ok", ("missing: " + ", ".join(miss)) if miss else "all present",
                "Run the install step “Install the packages”." if miss else "")
        ff = shutil.which("ffmpeg")
        add("ffmpeg", "ffmpeg (reads the camera stream)", "ok" if ff else "fail", ff or "not found on PATH",
            "" if ff else "Install ffmpeg with your package manager (Debian/Ubuntu: sudo apt install ffmpeg).")
        cfg = self._common_checks(add)
        svc = self.service_status()
        add("service", "Background service", "ok" if svc["active"] else ("info" if svc["installed"] else "info"),
            "running" if svc["active"] else ("installed, stopped" if svc["installed"] else "not installed"),
            "" if svc["active"] else "Install and start it from the Install menu, or run it by hand.")

    def _common_checks(self, add: Callable) -> dict:
        """Checks that hold for both ways of running the detector. Returns the parsed configuration."""
        cfg: dict = {}
        if not self.config_path.exists():
            add("config", "Configuration file", "fail", f"{self.config_path.name} does not exist",
                "Run the install step “Create the configuration”.")
        else:
            try:
                cfg = self.config()
                _cross_check(cfg)
                need = [s for s in ("camera", "detector", "verifier", "actuator", "trigger") if s not in cfg]
                add("config", "Configuration file", "fail" if need else "ok",
                    ("missing sections: " + ", ".join(need)) if need else str(self.config_path),
                    "Copy the missing sections from config.example.toml." if need else "")
            except ConfigError as e:
                add("config", "Configuration file", "fail", str(e), "Fix the file in a text editor, or restore config.toml.bak.")
        mp = self._model_path()
        if mp is not None:
            add("model", "Detector model", "ok" if mp.exists() else "fail",
                f"{mp.name} ({mp.stat().st_size / 1e6:.0f} MB)" if mp.exists() else f"{mp} is missing",
                "" if mp.exists() else "Run the install step “Export the detector model”.")
        env = self._env()
        url = env.get("SG_RTSP_URL", "")
        if not url:
            add("stream", "Camera stream address", "fail", "not set", "Set SG_RTSP_URL under Secrets (or `sg env set SG_RTSP_URL`).")
        else:
            mode = stat.S_IMODE(self.env_path.stat().st_mode)
            if mode & 0o077:
                add("envperm", "Secrets file is private", "warn", f".env is readable by other users (mode {mode:o})",
                    f"chmod 600 {self.env_path}")
            u = urllib.parse.urlsplit(url)
            try:
                port = u.port or 554
                with socket.create_connection((u.hostname or "", port), timeout=2.5):
                    pass
                add("stream", "Camera stream reachable", "ok", "the stream host answers on its port")
            except (OSError, ValueError):
                add("stream", "Camera stream reachable", "fail", "the stream host did not answer within 2.5 s",
                    "Check that the camera or stream relay is on, that this machine is on its network, and the address under Secrets.")
        ver = cfg.get("verifier", {})
        if ver.get("enabled", True) and ver.get("url"):
            from . import endpoints                      # same test as the Endpoints page (address, key, model, images)
            r = endpoints.test(self, "verifier", "detector")
            if r["status"] == "fail" and r["detail"] == "no answer":
                down = "acts on the detector alone" if ver.get("trust_detector_if_down") else "does nothing"
                add("verifier", "Second-opinion model", "warn", f"no answer from {r['address']}; while it is down the detector {down}",
                    "Start the vision model server, change verifier.url, or set verifier.enabled to no.")
            else:
                add("verifier", "Second-opinion model", r["status"], f"{r['address']}: {r['detail']}", r.get("fix", ""))
        elif cfg:
            add("verifier", "Second-opinion model", "info", "switched off: every detection that passes the trigger counts")
        act = cfg.get("actuator", {})
        if cfg:
            if act.get("dry_run", True):
                add("deterrent", "Deterrent", "info", "dry run: events are logged, nothing plays",
                    "Set actuator.dry_run to no once the speaker is wired up and tested.")
            else:
                prog = str(act.get("command", "")).split()[:1]
                found = bool(prog) and (self.docker or bool(shutil.which(prog[0]) or (self.config_path.parent / prog[0]).exists()))
                add("deterrent", "Deterrent command", "ok" if found else "fail", str(act.get("command", "")) or "empty",
                    "" if found else "Set actuator.command to a program that exists on this machine.")
        od = self._output_dir()
        probe = od if od.exists() else self.root
        free = shutil.disk_usage(probe).free / 1e9
        add("disk", "Free disk space for events", "ok" if free >= 2 else ("warn" if free >= 0.5 else "fail"), f"{free:.1f} GB free",
            "" if free >= 2 else "Free some space or point output.dir at a larger disk.")
        return cfg

    def _detector_docker_checks(self, add: Callable) -> None:
        docker, compose = self._docker()
        add("docker", "Docker Engine", "ok" if docker else "fail", shutil.which("docker") or "not installed",
            "" if docker else "Install Docker Engine: https://docs.docker.com/engine/install/")
        add("compose", "Docker compose plugin", "ok" if compose else "fail", "present" if compose else "not found",
            "" if compose else "Install the docker-compose-plugin package.")
        self._common_checks(add)
        if not docker:
            return
        built = run(["docker", "image", "inspect", IMAGE, "--format", "{{.Size}}"], timeout=15)
        add("image", "Detector image", "ok" if built[0] == 0 else "fail",
            f"{IMAGE} ({int(built[1]) / 1e6:.0f} MB)" if built[0] == 0 and built[1].isdigit() else "not built",
            "" if built[0] == 0 else "Run the install step “Build the detector image”.")
        svc = self.service_status()
        add("service", "Detector container", "ok" if svc["active"] else "info",
            "running" if svc["active"] else ("created, stopped" if svc["installed"] else "not created"),
            "" if svc["active"] else "Start it from the Install menu.")

    def _stack_checks(self, role: str, add: Callable) -> None:
        docker, compose = self._docker()
        add("docker", "Docker Engine", "ok" if docker else "fail", shutil.which("docker") or "not installed",
            "" if docker else "Install Docker Engine: https://docs.docker.com/engine/install/")
        add("compose", "Docker compose plugin", "ok" if compose else "fail", "present" if compose else "not found",
            "" if compose else "Install the docker-compose-plugin package.")
        d = self._stack_dir(role)
        has = (d / "compose.yaml").exists()
        add("stack", "Stack folder", "ok" if has else "fail", str(d) if has else f"{d} has no compose.yaml",
            "" if has else "Run the install step that copies the stack files.")
        if role == "recorder":
            for name, title, fix in ((".env", "Generated passwords (.env)", "Run the install step “Generate passwords and start”."),
                                     ("pg.url", "Database address (pg.url)", "Run the install step “Database address”.")):
                p = d / name
                try:
                    ok, private = p.exists(), p.exists() and not stat.S_IMODE(p.stat().st_mode) & 0o077
                except OSError:
                    ok, private = False, False
                add(name.strip("."), title, "ok" if ok and private else ("warn" if ok else "fail"),
                    "present" if ok and private else ("present, but readable by other users" if ok else "missing"),
                    "" if ok and private else (f"chmod 600 {p}" if ok else fix))
            add("psql", "psql (loads the database tables)", "ok" if shutil.which("psql") else "fail",
                shutil.which("psql") or "not installed", "" if shutil.which("psql") else "Install postgresql-client.")
        else:
            aa = Path("/sys/module/apparmor/parameters/enabled")
            if aa.exists() and aa.read_text().strip() == "Y":
                add("apparmor", "AppArmor profile for Frigate", "info", "AppArmor is on",
                    "Load deploy/gpuserver/docker-frigate.apparmor as described in deploy/gpuserver/README.md, or the container will not start.")
        if has and docker:
            rows = self.stack_status(role)
            if rows is None:
                add("containers", "Containers", "warn", "docker could not be asked (permission?)",
                    "Add your user to the docker group or run the console with sudo.")
            else:
                up = sum(r["state"] == "running" for r in rows)
                bad = [r["name"] for r in rows if r["state"] not in ("running", "exited", "created")]
                add("containers", "Containers", "ok" if up and not bad else ("fail" if bad else "info"),
                    f"{up} of {len(rows)} running" + (f"; unhealthy: {', '.join(bad)}" if bad else ""),
                    "" if up and not bad else "Start the stack from the Install menu.")
        free = shutil.disk_usage(d if d.exists() else self.root).free / 1e9
        add("disk", "Free disk space", "ok" if free >= 20 else ("warn" if free >= 5 else "fail"), f"{free:.0f} GB free",
            "" if free >= 20 else "Recordings need room: free space or shorten retention.")

    # ------------------------------------------------------------------ install steps
    def steps(self, role: str | None = None) -> list[Step]:
        role = role or self.role
        steps = {"detector": self._detector_steps, "recorder": self._recorder_steps, "gpu": self._gpu_steps}[role]()
        prefix = str(self.root) + "/"
        for st in steps:
            st.shown = [" ".join(c).replace(prefix, "") for c in st.commands]
        return steps

    def _pip(self, *pkgs: str, extra: list[str] | None = None) -> list[str]:
        if shutil.which("uv"):
            return ["uv", "pip", "install", "--python", str(self.venv_py), *(extra or []), *pkgs]
        return [str(self.venv_py), "-m", "pip", "install", *(extra or []), *pkgs]

    def _detector_docker_steps(self) -> list[Step]:
        docker, compose = self._docker()
        mp = self._model_path()
        svc = self.service_status()
        built = docker and run(["docker", "image", "inspect", IMAGE], timeout=15)[0] == 0
        dc = ["docker", "compose"]
        return [
            Step("docker", "Docker Engine and the compose plugin", "The detector runs as a container; no Python is installed on this machine.",
                 docker and compose,
                 manual="Install Docker Engine and docker-compose-plugin (https://docs.docker.com/engine/install/), then come back."),
            Step("config", "Create the configuration", "Copies config.example.toml to config.toml. Your file is never overwritten.",
                 self.config_path.exists()),
            Step("stream", "Camera stream address", "Stored in the private .env file (mode 600); passed to the container, never built into the image.",
                 bool(self._env().get("SG_RTSP_URL")), needs="SG_RTSP_URL",
                 needs_help="Camera stream address, e.g. rtsp://user:password@camera:554/ch1 (use the low-resolution sub-stream)"),
            Step("build", "Build the detector image", f"Builds {IMAGE} from the Dockerfile (about 900 MB, a few minutes the first time).",
                 built, [dc + ["build", "detector"]]),
            Step("model", "Export the detector model",
                 "Builds a one-off tools image (PyTorch, about 2 GB) and writes the model to models/. The weights are AGPL-3.0. "
                 "Skip it if you copy a model file into models/ yourself.",
                 None if mp is None else mp.exists(), [dc + ["--profile", "tools", "run", "--rm", "--build", "export"]]),
            Step("start", "Start the detector container", "Starts it now and after every reboot (restart: unless-stopped).",
                 svc["active"], [dc + ["up", "-d", "detector"]]),
        ]

    def _detector_steps(self) -> list[Step]:
        if self.docker:
            return self._detector_docker_steps()
        py = str(self.venv_py)
        miss = self._missing_modules(["onnxruntime", "cv2", "numpy", "requests", "paho"])
        mp = self._model_path()
        svc = self.service_status()
        venv_cmd = ["uv", "venv", "-p", "3.13", ".venv"] if shutil.which("uv") else [sys.executable, "-m", "venv", ".venv"]
        return [
            Step("venv", "Create the Python environment", "A private environment in .venv keeps these packages apart from the system's.",
                 self.venv_py.exists(), [venv_cmd]),
            Step("packages", "Install the packages", "The detector runtime: ONNX Runtime, OpenCV, NumPy, requests, MQTT client (about 250 MB).",
                 None if miss is None else not miss,
                 [self._pip("onnxruntime", "opencv-python-headless", "requests", "numpy", "paho-mqtt")]),
            Step("config", "Create the configuration", "Copies config.example.toml to config.toml. Your file is never overwritten.",
                 self.config_path.exists()),
            Step("model", "Export the detector model",
                 "Downloads YOLOv8s (Open Images V7) and converts it for the CPU. Needs PyTorch for the conversion: about 1.5 GB "
                 "of downloads, several minutes. The weights are AGPL-3.0.",
                 None if mp is None else mp.exists(),
                 [self._pip("torch", "torchvision", extra=["--index-url", "https://download.pytorch.org/whl/cpu"]),
                  self._pip("ultralytics", "onnx"), [py, "scripts/export_model.py", "s"]]),
            Step("stream", "Camera stream address", "Stored in the private .env file (mode 600), never in config.toml.",
                 bool(self._env().get("SG_RTSP_URL")), needs="SG_RTSP_URL",
                 needs_help="Camera stream address, e.g. rtsp://user:password@camera:554/ch1 (use the low-resolution sub-stream)"),
            Step("sound", "Generate the deterrent sound", "A sweep tone in sounds/deter.wav. The deterrent stays in dry run until you switch it on.",
                 (self.root / "sounds" / "deter.wav").exists(),
                 [[py, "scripts/make_tone.py", "sounds/deter.wav", "--lo", "4000", "--hi", "12000"]]),
            Step("service", "Install the background service", f"A systemd user service ({UNIT}) that restarts the detector if it stops.",
                 svc["installed"], [["systemctl", "--user", "daemon-reload"]]),
            Step("start", "Start the detector now, and at every login", "Enables and starts the service.",
                 svc["active"], [["systemctl", "--user", "enable", "--now", UNIT]]),
        ]

    def _recorder_steps(self) -> list[Step]:
        d = self._stack_dir("recorder")
        docker, compose = self._docker()
        rows = self.stack_status("recorder") if docker else None
        return [
            Step("docker", "Docker Engine and the compose plugin", "The recorder runs as containers.", docker and compose,
                 manual="Install Docker Engine and docker-compose-plugin (https://docs.docker.com/engine/install/), then come back."),
            Step("stackdir", f"Copy the stack files to {d}",
                 "Copies deploy/recorder there. Existing passwords, the database address and the message broker password file are kept.",
                 (d / "compose.yaml").exists()),
            Step("database", "Database address", "The PostgreSQL connection string, stored in pg.url (mode 600) in the stack folder.",
                 (d / "pg.url").exists(), needs="PG_URL",
                 needs_help="PostgreSQL connection string (postgresql://user:password@host/database?sslmode=require)"),
            Step("setup", "Generate passwords, load the tables, build and start",
                 "Runs the stack's setup.sh: it generates every password (none is shown), loads the database tables, "
                 "builds the images and starts the containers. Safe to run again.",
                 (d / ".env").exists() and (d / "mosquitto" / "passwd").exists(), [["bash", "setup.sh"]]),
            Step("up", "Start the containers", "Starts anything that is stopped.",
                 None if rows is None else bool(rows) and all(r["state"] == "running" for r in rows if r["state"] != "exited"),
                 [["docker", "compose", "up", "-d"]]),
        ]

    def _gpu_steps(self) -> list[Step]:
        d = self._stack_dir("gpu")
        docker, compose = self._docker()
        rows = self.stack_status("gpu") if docker else None
        return [
            Step("docker", "Docker Engine and the compose plugin", "Frigate and the listener run as containers.", docker and compose,
                 manual="Install Docker Engine and docker-compose-plugin (https://docs.docker.com/engine/install/), then come back."),
            Step("frigate", "Camera list for Frigate", "deploy/gpuserver/config.yml lists the cameras, zones and detector hardware.",
                 None, manual="Edit deploy/gpuserver/config.yml for your cameras and GPU. deploy/gpuserver/README.md covers AppArmor and storage."),
            Step("up", "Start Frigate and the listener", "Builds the listener image and starts both containers.",
                 None if rows is None else bool(rows) and all(r["state"] == "running" for r in rows),
                 [["docker", "compose", "up", "-d", "--build"]]),
        ]

    def run_step(self, role: str, step_id: str, emit: Callable[[str], None], dry_run: bool = False,
                 answer: str | None = None) -> int:
        """Run one install step, sending output lines to emit. Returns 0 on success."""
        step = next((s for s in self.steps(role) if s.id == step_id), None)
        if step is None:
            raise ConfigError(f"no step {step_id!r} for role {role!r}")
        if step.manual:
            emit(step.manual)
            return 0 if step.done else 3
        cwd = self._stack_dir(role) if role != "detector" else self.root
        if dry_run:
            if step.needs:
                emit(f"would ask for: {step.needs_help}")
            for line in self._builtin_plan(role, step_id):
                emit("would " + line)
            for c in step.shown:
                emit(f"would run in {cwd}: {c}")
            return 0
        if step.needs:
            if not answer:
                raise ConfigError(f"this step needs a value: {step.needs_help}")
            if step.needs == "PG_URL":
                if not re.match(r"^postgres(ql)?://\S+$", answer.strip()):
                    raise ConfigError("the address starts with postgresql:// and has no spaces")
                if not cwd.is_dir():
                    raise ConfigError(f"{cwd} does not exist yet: copy the stack files first")
                _atomic_write(cwd / "pg.url", answer.strip() + "\n", 0o600)
                emit("stored in pg.url (private)")
            else:
                self.env_set(step.needs, answer)
                emit(f"{step.needs} stored in .env (private)")
            return 0
        rc = self._builtin(role, step_id, emit)
        if rc:
            return rc
        for c in step.commands:
            emit("$ " + " ".join(c).replace(str(self.root) + "/", ""))
            rc = self._stream(c, cwd, emit)
            if rc:
                emit(f"failed with exit code {rc}")
                return rc
        return 0

    def _unit_text(self) -> str:
        return (f"[Unit]\nDescription=squirrel-guard detector\nAfter=network-online.target\n\n[Service]\n"
                f"WorkingDirectory={self.config_path.parent}\n"
                f"ExecStart={self.venv_py} -m squirrel_guard.main -c {self.config_path}\n"
                f"Environment=PYTHONPATH={self.root}\nRestart=on-failure\nRestartSec=10\n\n[Install]\nWantedBy=default.target\n")

    def _builtin_plan(self, role: str, step_id: str) -> list[str]:
        if (role, step_id) == ("detector", "config"):
            return [f"copy config.example.toml to {self.config_path}"]
        if (role, step_id) == ("detector", "service"):
            return [f"write {self._unit_path()}"]
        if (role, step_id) == ("recorder", "stackdir"):
            return [f"copy {self.root / 'deploy' / 'recorder'} to {self._stack_dir(role)} (keeping .env, pg.url, mosquitto/passwd)"]
        return []

    def _builtin(self, role: str, step_id: str, emit: Callable[[str], None]) -> int:
        if role == "detector" and self.docker and step_id in ("build", "model", "start"):
            if step_id == "start" and not self.config_path.exists():
                emit("create the configuration first (a missing config.toml would be mounted as an empty folder)")
                return 2
            for d in ("models", "sounds", "data"):                 # owned by you, not created by docker as root
                (self.root / d).mkdir(exist_ok=True)
        elif (role, step_id) == ("detector", "config"):
            emit("created config.toml" if self.create_config() else "config.toml already exists: left as it is")
        elif (role, step_id) == ("detector", "sound"):
            (self.root / "sounds").mkdir(exist_ok=True)
        elif (role, step_id) == ("detector", "service"):
            if not self.venv_py.exists():
                emit("create the Python environment first")
                return 2
            _atomic_write(self._unit_path(), self._unit_text(), 0o644)
            emit(f"wrote {self._unit_path()}")
        elif (role, step_id) == ("recorder", "stackdir"):
            src, dst = self.root / "deploy" / "recorder", self._stack_dir(role)
            try:
                dst.mkdir(parents=True, exist_ok=True)
                keep = {".env", "pg.url", "passwd"}
                n = 0
                for p in sorted(src.rglob("*")):
                    rel = p.relative_to(src)
                    if "__pycache__" in rel.parts or p.is_symlink():
                        continue
                    t = dst / rel
                    if p.is_dir():
                        t.mkdir(exist_ok=True)
                    elif not (t.exists() and p.name in keep):
                        shutil.copy2(p, t)
                        n += 1
                emit(f"copied {n} files to {dst}")
            except PermissionError:
                user = os.environ.get("USER", "$USER")
                emit(f"no permission to write {dst}. Create it once as root, then run this step again:")
                emit(f"  sudo install -d -o {user} -g {user} {dst}")
                return 13
        return 0

    @staticmethod
    def _stream(argv: list[str], cwd: Path, emit: Callable[[str], None]) -> int:
        try:
            p = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 stdin=subprocess.DEVNULL, errors="replace", env=_compose_env())
        except FileNotFoundError:
            emit(f"{argv[0]}: not installed")
            return 127
        except OSError as e:
            emit(str(e))
            return 126
        assert p.stdout
        for line in p.stdout:
            emit(scrub(line.rstrip("\n")))
        return p.wait()

    # ------------------------------------------------------------------ service, logs, events
    def service_status(self) -> dict:
        if self.docker:
            rows = [r for r in (self.stack_status("detector") or []) if r["name"] == "detector"]
            active = any(r["state"] == "running" for r in rows)
            return {"installed": bool(rows), "active": active, "enabled": bool(rows), "unit": "container detector"}
        installed = self._unit_path().exists()
        active = enabled = False
        if installed and shutil.which("systemctl"):
            active = run(["systemctl", "--user", "is-active", UNIT], timeout=5)[1] == "active"
            enabled = run(["systemctl", "--user", "is-enabled", UNIT], timeout=5)[1] == "enabled"
        return {"installed": installed, "active": active, "enabled": enabled, "unit": UNIT}

    def service(self, action: str) -> tuple[int, str]:
        if action not in ("start", "stop", "restart", "enable", "disable"):
            raise ConfigError("choose start, stop, restart, enable or disable")
        if self.docker:
            if action in ("enable", "disable"):
                raise ConfigError("the container restarts by itself (restart: unless-stopped); use start or stop")
            if not self.config_path.exists():
                raise ConfigError("create the configuration first (Install menu)")
            argv = {"start": ["up", "-d", "detector"], "stop": ["stop", "detector"], "restart": ["restart", "detector"]}[action]
            return run(["docker", "compose", *argv], cwd=self.root, timeout=120, env=_compose_env())
        if not self._unit_path().exists():
            raise ConfigError("the service is not installed yet: run the install step “Install the background service”")
        return run(["systemctl", "--user", action, UNIT], timeout=30)

    def logs(self, n: int = 60) -> list[str]:
        if self.docker:
            rc, out = run(["docker", "compose", "logs", "--no-color", "--no-log-prefix", "-t", "--tail", str(max(1, min(n, 500))),
                           "detector"], cwd=self.root, timeout=15, env=_compose_env())
            return [scrub(l) for l in out.splitlines()] if rc == 0 and out else ["no log available"]
        rc, out = run(["journalctl", "--user", "-u", UNIT, "-n", str(max(1, min(n, 500))), "--no-pager", "-o", "short-iso"], timeout=10)
        return [scrub(l) for l in out.splitlines()] if rc == 0 else [out or "no log available"]

    def events(self, n: int = 30) -> list[dict]:
        """Newest first. Reads only the tail of events.jsonl."""
        p = self._output_dir() / "events.jsonl"
        try:
            with open(p, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 256 * 1024))
                tail = f.read().decode("utf-8", "replace").splitlines()
        except OSError:
            return []
        out = []
        for line in reversed(tail):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict):
                out.append({k: d.get(k) for k in ("time", "label", "score", "verifier", "confirmed", "action", "detect_ms", "snapshot")})
            if len(out) >= n:
                break
        return out

    def snapshot_path(self, name: str) -> Path | None:
        """An event snapshot by file name; refuses anything that is not a plain .jpg name in the events folder."""
        if not re.match(r"^[A-Za-z0-9_.\-]+\.jpg$", name):
            return None
        p = self._output_dir() / name
        return p if p.is_file() else None

    # ------------------------------------------------------------------ summary
    def status(self) -> dict:
        role = self.role
        checks = self.checks(role)
        steps = self.steps(role)
        ev = self.events(1)
        count = lambda s: sum(c.status == s for c in checks)
        try:
            dry = bool(self.config().get("actuator", {}).get("dry_run", True))
        except ConfigError:
            dry = True
        return {
            "role": role, "role_title": ROLES[role][0], "root": str(self.root),
            "runtime": self.settings()["runtime"], "runtimes": RUNTIMES,
            "checks": {"ok": count("ok"), "warn": count("warn"), "fail": count("fail"), "info": count("info")},
            "steps_done": sum(s.done is True for s in steps), "steps_total": len(steps),
            "next_step": next((s.title for s in steps if s.done is False), None),
            "service": self.service_status() if role == "detector" else None,
            "containers": self.stack_status(role) if role != "detector" else None,
            "dry_run": dry if role == "detector" else None,
            "last_event": ev[0] if ev else None,
            "time": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }

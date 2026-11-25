"""Web GUI for squirrel-guard: `sg gui`. Standard library only.

It can change settings, store secrets and run install steps, so it is locked down:
- it listens on 127.0.0.1 only (use `ssh -L 8760:127.0.0.1:8760 host` to reach another machine's console);
- a random token, printed once in the start URL, becomes an HttpOnly SameSite=Strict cookie; every request needs it;
- the Host header must be this loopback address (stops DNS rebinding); changes also need the token in a header
  and, when the browser sends one, a matching Origin (stops other sites from posting to it);
- it runs only the fixed install steps of core.py, and never sends a secret value back to the browser.
"""
from __future__ import annotations

import hmac
import json
import mimetypes
import secrets
import threading
import urllib.parse
import webbrowser
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .core import ROLES, ConfigError, Site

WEB = Path(__file__).with_name("web")
COOKIE = "sg_console"
MAX_BODY = 64 * 1024


class Job:
    def __init__(self, title: str):
        self.title, self.lines, self.done, self.rc = title, [], False, None


class State:
    def __init__(self, site: Site, token: str):
        self.site, self.token = site, token
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.running: str | None = None
        self.ui_dir = site.root / "deploy" / "recorder" / "persons-ui" / "static"      # the shared design system

    def start_job(self, role: str, step_ids: list[str], dry_run: bool) -> str:
        with self.lock:
            if self.running and not self.jobs[self.running].done:
                raise ConfigError("another step is still running: wait for it to finish")
            jid = secrets.token_hex(6)
            job = self.jobs[jid] = Job(", ".join(step_ids))
            self.running = jid
            for old in list(self.jobs)[:-8]:
                del self.jobs[old]

        def work() -> None:
            rc = 0
            try:
                for sid in step_ids:
                    step = next(s for s in self.site.steps(role) if s.id == sid)
                    job.lines.append(f"== {step.title} ==")
                    if step.needs and not dry_run:
                        job.lines.append("needs a value: enter it in the field on this step, then run the rest again")
                        rc = 3
                        break
                    rc = self.site.run_step(role, sid, job.lines.append, dry_run=dry_run)
                    if rc:
                        job.lines.append("This step did not finish. Later steps were not run.")
                        break
            except Exception as e:                                # shown to the operator, never a stack trace page
                job.lines.append(f"error: {e}")
                rc = 1
            job.rc, job.done = rc, True
        threading.Thread(target=work, daemon=True).start()
        return jid


def make_handler(state: State, port: int):
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    origins = {"http://" + h for h in hosts}
    site = state.site

    class Handler(BaseHTTPRequestHandler):
        server_version = "sg-console"
        sys_version = ""

        def log_message(self, *a) -> None:                        # quiet, and the start URL carries the token
            pass

        # -------------------------------------------------------------- plumbing
        def send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def json(self, obj, code: int = 200) -> None:
            self.send(code, json.dumps(obj).encode(), "application/json")

        def deny(self, code: int, msg: str) -> None:
            self.send(code, (msg + "\n").encode(), "text/plain; charset=utf-8")

        def authed(self) -> bool:
            for part in self.headers.get("Cookie", "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == COOKIE and hmac.compare_digest(v, state.token):
                    return True
            return False

        def file(self, path: Path, ctype: str | None = None) -> None:
            try:
                data = path.read_bytes()
            except OSError:
                return self.deny(404, "not found")
            self.send(200, data, ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream")

        # -------------------------------------------------------------- GET
        def do_GET(self) -> None:
            if self.headers.get("Host") not in hosts:
                return self.deny(403, "open the console at the address `sg gui` printed")
            url = urllib.parse.urlsplit(self.path)
            q = urllib.parse.parse_qs(url.query)
            if url.path == "/" and "t" in q:
                if hmac.compare_digest(q["t"][0], state.token):
                    return self.send(303, b"", "text/plain", {
                        "Location": "/", "Set-Cookie": f"{COOKIE}={state.token}; HttpOnly; SameSite=Strict; Path=/"})
                return self.deny(403, "that link has expired: run `sg gui` again and use the new one")
            if not self.authed():
                return self.deny(403, "open the console with the full address `sg gui` printed (it carries a one-time key)")
            p = url.path
            try:
                if p == "/":
                    html = (WEB / "index.html").read_text().replace("{{token}}", state.token)
                    return self.send(200, html.encode(), "text/html; charset=utf-8")
                if p in ("/console.css", "/console.js"):
                    return self.file(WEB / p[1:])
                if p in ("/ui/ui.css", "/ui/ui.js"):
                    return self.file(state.ui_dir / p[4:])
                if p.startswith("/snapshot/"):
                    snap = site.snapshot_path(p[len("/snapshot/"):])
                    return self.file(snap, "image/jpeg") if snap else self.deny(404, "not found")
                if p == "/api/status":
                    st = site.status()
                    st["roles"] = {k: {"title": t, "what": w} for k, (t, w) in ROLES.items()}
                    st["settings"] = site.settings()
                    return self.json(st)
                if p == "/api/steps":
                    return self.json({"role": site.role, "steps": [asdict(s) for s in site.steps()]})
                if p == "/api/config":
                    return self.json({"path": str(site.config_path), "exists": site.config_path.exists(),
                                      "fields": [f.public() for f in site.fields()]})
                if p == "/api/env":
                    return self.json(site.env_status())
                if p == "/api/endpoints":
                    from . import endpoints as ep
                    return self.json({"role": site.role, "endpoints": ep.listing(site)})
                if p == "/api/checks":
                    return self.json([asdict(c) for c in site.checks()])
                if p == "/api/service":
                    return self.json({"role": site.role, "service": site.service_status(), "logs": site.logs(80) if site.role == "detector" else [],
                                      "containers": site.stack_status(site.role) if site.role != "detector" else None})
                if p == "/api/events":
                    return self.json(site.events(60))
                if p == "/api/job":
                    job = state.jobs.get(q.get("id", [""])[0])
                    if not job:
                        return self.json({"error": "that run is no longer kept"}, 404)
                    start = int(q.get("from", ["0"])[0]) if q.get("from", ["0"])[0].isdigit() else 0
                    return self.json({"lines": job.lines[start:], "next": len(job.lines), "done": job.done, "rc": job.rc})
            except ConfigError as e:
                return self.json({"error": str(e)}, 400)
            self.deny(404, "not found")

        do_HEAD = do_GET

        # -------------------------------------------------------------- POST
        def do_POST(self) -> None:
            origin = self.headers.get("Origin")
            if (self.headers.get("Host") not in hosts or not self.authed() or (origin and origin not in origins)
                    or not hmac.compare_digest(self.headers.get("X-SG-Token", ""), state.token)):
                return self.deny(403, "not allowed")
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 <= n <= MAX_BODY:
                    return self.deny(413, "too large")
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                return self.deny(400, "bad request")
            s = lambda k: body.get(k) if isinstance(body.get(k), str) else ""
            p = urllib.parse.urlsplit(self.path).path
            try:
                if p == "/api/config":
                    return self.json({"value": site.set_value(s("id"), s("value"))})
                if p == "/api/env":
                    site.env_set(s("name"), s("value") if body.get("value") is not None else None)
                    return self.json({"ok": True})
                if p == "/api/settings":
                    site.set_setting(s("key"), s("value"))
                    return self.json({"ok": True})
                if p == "/api/endpoint":
                    from . import endpoints as ep
                    eid, what = s("id"), s("what")
                    if what == "address":
                        ep.set_address(site, site.role, eid, s("value"))
                    elif what == "model":
                        ep.set_model(site, site.role, eid, s("value"))
                    elif what == "key":
                        ep.set_api_key(site, site.role, eid, s("value") if body.get("value") is not None else None)
                    elif what != "test":
                        raise ConfigError("unknown endpoint action")
                    return self.json({"ok": True, "test": ep.test(site, eid) if what in ("test", "address", "key") else None})
                if p == "/api/service":
                    rc, out = site.service(s("action"))
                    return self.json({"ok": rc == 0, "output": out}, 200 if rc == 0 else 500)
                if p == "/api/run":
                    role, dry = site.role, bool(body.get("dry_run"))
                    ids = [x for x in body.get("steps", []) if isinstance(x, str)][:20]
                    known = {st.id: st for st in site.steps(role)}
                    if not ids or any(i not in known for i in ids):
                        raise ConfigError("unknown install step")
                    if len(ids) == 1 and known[ids[0]].needs and not dry:
                        out: list[str] = []
                        rc = site.run_step(role, ids[0], out.append, answer=s("answer"))
                        return self.json({"ok": rc == 0, "lines": out})
                    return self.json({"job": state.start_job(role, ids, dry)})
            except ConfigError as e:
                return self.json({"error": str(e)}, 400)
            self.deny(404, "not found")

    return Handler


def make_server(site: Site, port: int = 8760, token: str | None = None) -> tuple[ThreadingHTTPServer, str]:
    """Bind the first free port from `port` upward. Returns (server, start URL)."""
    token = token or secrets.token_urlsafe(24)
    state = State(site, token)
    last: OSError | None = None
    for p in range(port, port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), make_handler(state, p))
        except OSError as e:
            last = e
            continue
        srv.daemon_threads = True
        return srv, f"http://127.0.0.1:{p}/?t={token}"
    raise ConfigError(f"no free port between {port} and {port + 19}: {last}")


def serve(site: Site, port: int = 8760, open_browser: bool = True) -> int:
    srv, url = make_server(site, port)
    print(f"squirrel-guard console: {url}\n(this machine only; the key in the address is valid until you stop it with Ctrl-C)", flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.server_close()
    return 0

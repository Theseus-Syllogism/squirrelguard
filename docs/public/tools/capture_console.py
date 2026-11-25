#!/usr/bin/env python3
"""Screenshots of the console and of the evidence pages for the public guide, from example data only.

    python3 docs/public/tools/capture_console.py [--pages DIR]

Console: a throwaway copy of the checkout (example configuration, nothing installed), a stand-in model server and
a stand-in broker port on this machine, then the web console's pages in headless Chromium. The addresses shown are
rewritten in the page to documentation examples before capture (127.0.0.1:8080, 192.0.2.10:1883, /opt/squirrel-guard).

Evidence pages: `SG_SCREENSHOT_DIR=DIR pytest tests/test_evidence_export.py` saves the review pages as self-contained
HTML built from test data (invented names); pass that folder with --pages to capture them in the light theme.

Needs node with playwright-core (NODE_PATH) and a Chromium (CHROME), like capture_screens.cjs.
"""
import json
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE.parent / "img" / "screens"


class Model(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = {"/health": {"status": "ok"}, "/v1/models": {"data": [{"id": "qwen-vl-q5"}]},
                "/props": {"modalities": {"vision": True}}}.get(self.path)
        data = json.dumps(body or {"error": "not found"}).encode()
        self.send_response(200 if body else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    pages = sys.argv[sys.argv.index("--pages") + 1] if "--pages" in sys.argv else ""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="sg-docs-"))
    root = tmp / "squirrel-guard"
    root.mkdir()
    shutil.copytree(REPO / "squirrel_guard", root / "squirrel_guard", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(REPO / "config.example.toml", root)
    (root / "deploy" / "recorder" / "persons-ui").mkdir(parents=True)
    shutil.copytree(REPO / "deploy" / "recorder" / "persons-ui" / "static", root / "deploy" / "recorder" / "persons-ui" / "static")
    sys.path.insert(0, str(root))
    from squirrel_guard.console import endpoints, web
    from squirrel_guard.console.core import Site

    model = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    threading.Thread(target=model.serve_forever, daemon=True).start()
    broker = socket.socket()
    broker.bind(("127.0.0.1", 0))
    broker.listen(8)
    mport, bport = model.server_address[1], broker.getsockname()[1]

    site = Site(root=root, home=tmp / "home")
    site.create_config()
    endpoints.set_address(site, "detector", "verifier", f"http://127.0.0.1:{mport}")
    endpoints.set_address(site, "detector", "mqtt", f"127.0.0.1:{bport}")
    site.env_set("SG_RTSP_URL", f"rtsp://viewer:example@127.0.0.1:{bport}/ch1")
    srv, url = web.make_server(site, port=18820)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    me = str(pathlib.Path.home())                       # tool paths on the build machine are not for publication
    swaps = {f"127.0.0.1:{mport}": "127.0.0.1:8080", f"127.0.0.1:{bport}": "192.0.2.10:1883",
             str(root): "/opt/squirrel-guard", str(tmp / "home"): "/home/operator",
             me + "/.local/bin": "/usr/bin", me: "/home/operator"}
    try:
        subprocess.run(["node", str(HERE / "capture_console.cjs"), url, str(OUT), json.dumps(swaps), pages], check=True)
    finally:
        srv.shutdown()
        model.shutdown()
        broker.close()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

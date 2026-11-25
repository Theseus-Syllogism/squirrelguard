"""`sg`: the squirrel-guard command. Every action of the terminal UI and the web GUI is also a plain command here,
so it can be scripted and used over a slow SSH line.

  sg                 open the terminal UI (or print help when not on a terminal)
  sg setup           installation and configuration menu
  sg status          one-screen summary            sg check     preflight checks (exit 1 if any fail)
  sg config ...      list / get / set / path       sg env ...   list / set / unset secrets (values are never shown)
  sg endpoints ...   list / test / set / key / model: model servers, Frigate, broker, timestamp service
  sg service ...     status|start|stop|restart|logs
  sg events          recent detections             sg gui       web GUI on this machine only
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys

from .core import ROLES, RUNTIMES, ConfigError, Site

MARK = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL", "info": "info"}
COLOR = {"ok": "32", "warn": "33", "fail": "31", "info": "36"}


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() and not os.environ.get("NO_COLOR") else text


def _show(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return "" if value is None else str(value)


def cmd_status(site: Site, a) -> int:
    st = site.status()
    if a.json:
        print(json.dumps(st, indent=1))
        return 0
    c = st["checks"]
    print(f"squirrel-guard  ·  {st['role_title']}  ·  {st['root']}")
    print(f"  install   {st['steps_done']} of {st['steps_total']} steps done" + (f"; next: {st['next_step']}" if st["next_step"] else ""))
    print(f"  checks    {c['ok']} ok, {c['warn']} warnings, {c['fail']} failed")
    if st["service"] is not None:
        s = st["service"]
        print("  runs as   " + RUNTIMES[st["runtime"]])
        print("  service   " + (_c("running", "32") if s["active"] else ("stopped" if s["installed"] else "not installed")))
        print("  deterrent " + ("dry run (logs only)" if st["dry_run"] else _c("armed", "33")))
    if st["containers"] is not None:
        up = sum(r["state"] == "running" for r in st["containers"])
        print(f"  containers {up} of {len(st['containers'])} running")
    e = st["last_event"]
    print("  last event " + (f"{e['time']}  {e['label']} {e['score']}  -> {e['action']}" if e else "none recorded"))
    return 0


def cmd_check(site: Site, a) -> int:
    checks = site.checks(a.role)
    if a.json:
        print(json.dumps([c.__dict__ for c in checks], indent=1))
    else:
        for c in checks:
            print(f"  {_c(MARK[c.status], COLOR[c.status])}  {c.title}: {c.detail}")
            if c.fix and c.status in ("warn", "fail"):
                print(f"        -> {c.fix}")
    return 1 if any(c.status == "fail" for c in checks) else 0


def cmd_config(site: Site, a) -> int:
    if a.action == "path":
        print(site.config_path)
    elif a.action == "list":
        sec = None
        for f in site.fields():
            if a.key and not f.id.startswith(a.key):
                continue
            if f.section != sec:
                sec = f.section
                print(f"\n[{sec}]")
            v = f.value if f.line >= 0 else f.default
            note = "" if f.line >= 0 else "  (default)"
            print(f"  {f.key:<24} {_show(v)}{note}")
    elif a.action == "get":
        if not a.key:
            raise ConfigError("usage: sg config get SECTION.KEY")
        f = site.field(a.key)
        print(_show(f.value if f.line >= 0 else f.default))
    elif a.action == "set":
        if not a.key or a.value is None:
            raise ConfigError("usage: sg config set SECTION.KEY VALUE")
        v = site.set_value(a.key, a.value)
        print(f"{a.key} = {_show(v)}  (saved; previous file kept as {site.config_path.name}.bak)")
    return 0


def cmd_env(site: Site, a) -> int:
    if a.action == "list":
        for e in site.env_status():
            print(f"  {_c('set    ', '32') if e['set'] else 'not set'}  {e['name']:<16} {e['help']}")
    elif a.action == "set":
        if not a.name:
            raise ConfigError("usage: sg env set NAME   (the value is asked for, hidden)")
        value = getpass.getpass(f"{a.name} (hidden): ") if sys.stdin.isatty() else sys.stdin.readline()
        site.env_set(a.name, value)
        print(f"{a.name} stored in {site.env_path} (private)")
    elif a.action == "unset":
        if not a.name:
            raise ConfigError("usage: sg env unset NAME")
        site.env_set(a.name, None)
        print(f"{a.name} removed")
    return 0


def _ep_line(e: dict) -> str:
    addr = e["address"] or "-"
    note = {"default": "  (default)", "fallback": "  (" + e["fallback"] + ")", "none": "  (not set: " + (e["fallback"] or "unused") + ")"}
    return f"{addr}{note.get(e['source'], '')}"


def cmd_endpoints(site: Site, a) -> int:
    from . import endpoints as ep
    role = a.role or site.role
    if a.action == "list":
        rows = ep.listing(site, role)
        if a.json:
            print(json.dumps(rows, indent=1))
            return 0
        if not rows:
            print(f"no outside addresses for the {role} role")
        for e in rows:
            print(f"\n{e['id']}  {e['title']}\n    address  {_ep_line(e)}")
            if e["model_key"]:
                print(f"    model    {e['model'] or '(not set: the server picks)'}")
            if e["api_key"]:
                print(f"    API key  {_c('set', '32') if e['api_key_set'] else 'not set'}  ({e['api_key']})")
            print(f"    stored   {e['stored_in']}")
        return 0
    if a.action == "test":
        rows = [ep.test(site, a.id, role)] if a.id else ep.test_all(site, role)
        if a.json:
            print(json.dumps(rows, indent=1))
        else:
            for r in rows:
                print(f"  {_c(MARK[r['status']], COLOR[r['status']])}  {r['title']}: {r['detail']}")
                if r.get("fix") and r["status"] in ("warn", "fail"):
                    print(f"        -> {r['fix']}")
        return 1 if any(r["status"] == "fail" for r in rows) else 0
    if not a.id:
        raise ConfigError(f"usage: sg endpoints {a.action} ID" + (" VALUE" if a.action in ("set", "model") else ""))
    if a.action == "set":
        if a.value is None:
            raise ConfigError("usage: sg endpoints set ID ADDRESS   (\"\" clears it)")
        v = ep.set_address(site, role, a.id, a.value)
        print(f"{a.id}: {v or 'cleared'}  (saved)")
        if v:
            r = ep.test(site, a.id, role)
            print(f"  {_c(MARK[r['status']], COLOR[r['status']])}  {r['detail']}")
    elif a.action == "model":
        v = ep.set_model(site, role, a.id, a.value or "")
        print(f"{a.id} model: {v or 'cleared (the server picks)'}  (saved)")
    elif a.action == "key":
        if a.value == "-":
            ep.set_api_key(site, role, a.id, None)
            print(f"{a.id}: API key removed")
            return 0
        if a.value is not None:
            raise ConfigError("the key is asked for, hidden; do not put it on the command line (`-` removes it)")
        value = getpass.getpass(f"API key for {a.id} (hidden): ") if sys.stdin.isatty() else sys.stdin.readline()
        ep.set_api_key(site, role, a.id, value)
        print(f"{a.id}: API key stored (private)")
    return 0


def cmd_service(site: Site, a) -> int:
    if a.action == "status":
        s = site.service_status()
        print("running" if s["active"] else ("stopped" if s["installed"] else "not installed"),
              "(starts automatically)" if s["enabled"] else "")
        return 0 if s["active"] else 3
    if a.action == "logs":
        print("\n".join(site.logs(a.lines)))
        return 0
    rc, out = site.service(a.action)
    print(out or f"{a.action}: done")
    return rc


def cmd_events(site: Site, a) -> int:
    ev = site.events(a.lines)
    if a.json:
        print(json.dumps(ev, indent=1))
        return 0
    if not ev:
        print("no events recorded yet")
    for e in ev:
        print(f"  {e['time']}  {str(e['label']):<10} {e['score']!s:<6} check: {_show(e['verifier']) or '-':<9} -> {e['action']}")
    return 0


def _ask(prompt: str, default: str = "") -> str:
    try:
        return input(prompt).strip() or default
    except EOFError:
        return default


def plain_setup(site: Site, a) -> int:
    """The install menu as plain questions: works without curses, over serial lines and in scripts (--yes)."""
    role = a.role or site.role
    if not a.role and not a.yes and sys.stdin.isatty():
        print("What does this machine do?")
        for i, (k, (title, what)) in enumerate(ROLES.items(), 1):
            print(f"  {i}. {title}{'  (current)' if k == role else ''}\n     {what}")
        pick = _ask(f"Choose 1-{len(ROLES)} [keep current]: ")
        if pick.isdigit() and 1 <= int(pick) <= len(ROLES):
            role = list(ROLES)[int(pick) - 1]
    if role != site.role and not a.dry_run:
        site.set_setting("role", role)
    print(f"\nInstalling: {ROLES[role][0]}" + ("  (dry run: nothing is changed)" if a.dry_run else ""))
    failed = 0
    for n, s in enumerate(site.steps(role), 1):
        s = next(x for x in site.steps(role) if x.id == s.id)      # state may have changed after earlier steps
        head = f"\n[{n}] {s.title}"
        if s.done and not a.dry_run:
            print(f"{head}: {_c('already done', '32')}")
            continue
        print(f"{head}\n    {s.why}")
        if s.manual:
            print(f"    To do by hand: {s.manual}")
            continue
        answer = None
        if not a.dry_run:
            for c in s.shown:
                print("    $ " + c)
            if not a.yes and _ask("    Do this now? [Y/n/q] ", "y").lower()[:1] in ("n", "q"):
                print("    skipped")
                continue
            if s.needs:
                if not sys.stdin.isatty():
                    print("    skipped: needs a value typed at a terminal (or use `sg env set`)")
                    continue
                answer = getpass.getpass(f"    {s.needs_help}\n    (hidden) > ")
        try:
            rc = site.run_step(role, s.id, lambda line: print("    | " + line, flush=True), dry_run=a.dry_run, answer=answer)
        except ConfigError as e:
            rc = 2
            print("    " + _c(str(e), "31"))
        if rc:
            failed += 1
            print("    " + _c("this step did not finish", "31"))
            if not a.yes and _ask("    Continue with the next step? [y/N] ", "n").lower()[:1] != "y":
                break
    if not a.dry_run:
        print("\nChecks:")
        cmd_check(site, argparse.Namespace(role=role, json=False))
    return 1 if failed else 0


def cmd_setup(site: Site, a) -> int:
    if a.plain or a.yes or a.dry_run or not (sys.stdin.isatty() and sys.stdout.isatty()):
        return plain_setup(site, a)
    from . import tui
    return tui.main(site, start="install")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sg", description="squirrel-guard: install, configure and run.",
                                 epilog="Run `sg` alone for the terminal UI, `sg gui` for the web GUI.")
    ap.add_argument("-c", "--config", help="configuration file (default: config.toml beside this checkout)")
    sub = ap.add_subparsers(dest="cmd", metavar="COMMAND")
    roles = list(ROLES)
    p = sub.add_parser("status", help="one-screen summary"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("check", help="preflight checks; exit code 1 if any fail")
    p.add_argument("--role", choices=roles); p.add_argument("--json", action="store_true")
    p = sub.add_parser("setup", help="installation and configuration menu")
    p.add_argument("--role", choices=roles); p.add_argument("--dry-run", action="store_true", help="show what each step would do")
    p.add_argument("--yes", action="store_true", help="run every step without asking (steps that need a secret are skipped)")
    p.add_argument("--plain", action="store_true", help="plain questions instead of the full-screen menu")
    p = sub.add_parser("config", help="show or change settings")
    p.add_argument("action", choices=["list", "get", "set", "path"]); p.add_argument("key", nargs="?"); p.add_argument("value", nargs="?")
    p = sub.add_parser("env", help="secrets: list, set, unset (values are never shown)")
    p.add_argument("action", choices=["list", "set", "unset"]); p.add_argument("name", nargs="?")
    p = sub.add_parser("endpoints", help="addresses of model servers, Frigate, the broker, the timestamp service",
                       description="list | test [ID] | set ID ADDRESS | key ID (asked for, hidden; `-` removes) | model ID NAME")
    p.add_argument("action", choices=["list", "test", "set", "key", "model"]); p.add_argument("id", nargs="?")
    p.add_argument("value", nargs="?"); p.add_argument("--role", choices=roles); p.add_argument("--json", action="store_true")
    p = sub.add_parser("service", help="the detector's background service")
    p.add_argument("action", choices=["status", "start", "stop", "restart", "enable", "disable", "logs"])
    p.add_argument("-n", "--lines", type=int, default=40)
    p = sub.add_parser("events", help="recent detections"); p.add_argument("-n", "--lines", type=int, default=20)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("role", help="show or set what this machine does"); p.add_argument("name", nargs="?", choices=roles)
    p = sub.add_parser("runtime", help="how the detector runs: venv (Python + systemd) or docker (container)")
    p.add_argument("name", nargs="?", choices=["venv", "docker"])
    sub.add_parser("tui", help="terminal UI")
    p = sub.add_parser("gui", help="web GUI, reachable from this machine only")
    p.add_argument("--port", type=int, default=8760); p.add_argument("--no-open", action="store_true", help="do not open a browser")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    site = Site(root=os.environ.get("SG_ROOT"), config=a.config)   # SG_ROOT: another checkout (used by the tests)
    try:
        if a.cmd in (None, "tui"):
            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                if a.cmd == "tui":
                    print("the terminal UI needs a terminal; try `sg status` or `sg setup --plain`", file=sys.stderr)
                    return 2
                ap.print_help()
                return 0
            from . import tui
            return tui.main(site)
        if a.cmd == "gui":
            from . import web
            return web.serve(site, port=a.port, open_browser=not a.no_open)
        if a.cmd == "role":
            if a.name:
                site.set_setting("role", a.name)
            print(f"{site.role}: {ROLES[site.role][0]}")
            return 0
        if a.cmd == "runtime":
            if a.name:
                site.set_setting("runtime", a.name)
            print(f"{site.settings()['runtime']}: {RUNTIMES[site.settings()['runtime']]}")
            return 0
        return {"status": cmd_status, "check": cmd_check, "setup": cmd_setup, "config": cmd_config, "env": cmd_env,
                "service": cmd_service, "events": cmd_events, "endpoints": cmd_endpoints}[a.cmd](site, a)
    except ConfigError as e:
        print(f"sg: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())

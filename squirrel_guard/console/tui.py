"""Terminal UI for squirrel-guard (curses, standard library only).

Layout: a menu on the left, the chosen page on the right, the selected row's explanation underneath, key hints at
the bottom. Up/Down or j/k move, Enter or Right opens, Left or Esc goes back to the menu, q quits, ? shows help.

Long-running work (install steps) and secret entry leave the full-screen view and use the plain terminal, so the
output can be scrolled back and a secret is never drawn on the screen.
"""
from __future__ import annotations

import curses
import getpass
import locale
import textwrap

from .core import ROLES, RUNTIMES, ConfigError, Site

MENU = [("overview", "Overview"), ("install", "Install"), ("config", "Configuration"), ("endpoints", "Endpoints"),
        ("secrets", "Secrets"),
        ("checks", "Checks"), ("service", "Service"), ("events", "Events")]
HELP = """Keys
  Up/Down, j/k   move            Enter, Right   open or change the selected row
  Left, Esc      back to menu    Tab            switch between menu and page
  r              refresh         q              quit
Page keys are listed on the bottom line of each page.

Every action here is also a command: sg status, sg check, sg setup, sg config set, sg endpoints, sg env set,
sg service, sg events.
Secrets are typed hidden and are never displayed."""


class Row:
    def __init__(self, text: str, detail: str = "", tone: str = "", key=None, heading: bool = False):
        self.text, self.detail, self.tone, self.key, self.heading = text, detail, tone, key, heading


class App:
    def __init__(self, scr, site: Site, start: str = "overview"):
        self.scr, self.site = scr, site
        self.page = next((i for i, (k, _) in enumerate(MENU) if k == start), 0)
        self.focus = "page" if start != "overview" else "menu"
        self.sel, self.top = 0, 0
        self.rows: list[Row] = []
        self.msg, self.msg_tone = "", ""
        self.dry_run = False
        self.ep_results: dict[str, dict] = {}       # endpoint id -> last test result
        self.utf = "utf" in (locale.getpreferredencoding(False) or "").lower()
        curses.curs_set(0)
        self.pair = {}
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            for i, (name, fg) in enumerate([("ok", curses.COLOR_GREEN), ("warn", curses.COLOR_YELLOW), ("fail", curses.COLOR_RED),
                                            ("info", curses.COLOR_CYAN), ("dim", curses.COLOR_BLUE)], 1):
                curses.init_pair(i, fg, -1)
                self.pair[name] = curses.color_pair(i)
        self.load()

    # ------------------------------------------------------------------ data
    def mark(self, tone: str) -> str:
        if self.utf:
            return {"ok": "✓", "warn": "!", "fail": "✗", "info": "·", "todo": "○"}.get(tone, " ")
        return {"ok": "+", "warn": "!", "fail": "x", "info": ".", "todo": "o"}.get(tone, " ")

    def load(self) -> None:
        key = MENU[self.page][0]
        try:
            self.rows = getattr(self, "rows_" + key)()
        except ConfigError as e:
            self.rows = [Row(str(e), tone="fail")]
        self.sel = min(self.sel, max(0, len(self.rows) - 1))
        self._skip_headings(1)

    def _skip_headings(self, step: int) -> None:
        n = len(self.rows)
        for _ in range(n):
            if 0 <= self.sel < n and self.rows[self.sel].heading:
                self.sel = (self.sel + step) % n
            else:
                break

    def rows_overview(self) -> list[Row]:
        st = self.site.status()
        c = st["checks"]
        rows = [Row(f"This machine: {st['role_title']}", ROLES[st["role"]][1] + "  Enter changes the role.", key="role"),
                Row(f"Install: {st['steps_done']} of {st['steps_total']} steps done",
                    f"Next: {st['next_step']}" if st["next_step"] else "Nothing left to install.",
                    "ok" if not st["next_step"] else "warn", key="install"),
                Row(f"Checks: {c['ok']} ok, {c['warn']} warnings, {c['fail']} failed", "Enter opens the checks.",
                    "fail" if c["fail"] else ("warn" if c["warn"] else "ok"), key="checks")]
        if st["service"] is not None:
            s = st["service"]
            rows.append(Row("Service: " + ("running" if s["active"] else ("stopped" if s["installed"] else "not installed")),
                            "The detector's background service. Enter opens the service page.", "ok" if s["active"] else "info", key="service"))
            rows.append(Row("Deterrent: " + ("dry run (events are logged, nothing plays)" if st["dry_run"] else "ARMED"),
                            "Change actuator.dry_run under Configuration.", "info" if st["dry_run"] else "warn", key="config"))
        if st["containers"] is not None:
            up = sum(r["state"] == "running" for r in st["containers"])
            rows.append(Row(f"Containers: {up} of {len(st['containers'])} running",
                            ", ".join(f"{r['name']} ({r['state']})" for r in st["containers"]) or "none", "ok" if up else "info"))
        e = st["last_event"]
        rows.append(Row("Last event: " + (f"{e['time']}  {e['label']} {e['score']} -> {e['action']}" if e else "none recorded"),
                        "Enter opens the event list.", key="events"))
        rows.append(Row(f"Folder: {st['root']}", "This checkout."))
        return rows

    def rows_install(self) -> list[Row]:
        role = self.site.role
        how = f"  ({RUNTIMES[self.site.settings()['runtime']]})" if role == "detector" else ""
        rows = [Row(f"{ROLES[role][0]}{how}" + ("   [dry run: nothing is changed]" if self.dry_run else ""), heading=True)]
        for n, s in enumerate(self.site.steps(role), 1):
            tone = "ok" if s.done else ("info" if s.done is None else "todo")
            detail = s.why
            if s.manual:
                detail += "  By hand: " + s.manual
            elif s.commands:
                detail += "  Runs: " + " ; ".join(s.shown)
            rows.append(Row(f"{self.mark(tone)} {n}. {s.title}", detail, tone if tone != "todo" else "", key=s.id))
        return rows

    def rows_config(self) -> list[Row]:
        rows, sec = [], None
        for f in self.site.fields():
            if f.section != sec:
                sec = f.section
                rows.append(Row(f"[{sec}]", heading=True))
            v = f.value if f.line >= 0 else f.default
            shown = ("yes" if v else "no") if isinstance(v, bool) else (", ".join(map(str, v)) if isinstance(v, list) else str(v))
            note = "" if f.line >= 0 else " (default)"
            help_ = (f.help or "No description.") + ("" if f.editable else "  Change this one in a text editor.")
            rows.append(Row(f"  {f.key:<24} {shown}{note}", help_, key=f.id))
        if not rows:
            rows.append(Row("No configuration yet. Run Install > Create the configuration.", tone="warn"))
        return rows

    def rows_endpoints(self) -> list[Row]:
        from . import endpoints as ep
        rows = []
        for e in ep.listing(self.site):
            r = self.ep_results.get(e["id"])
            rows.append(Row(e["title"], heading=True))
            note = {"default": " (default)", "fallback": f" ({e['fallback']})", "none": f" (not set: {e['fallback'] or 'unused'})"}
            result = f"  Last test: {r['detail']}" + (f"  What to do: {r['fix']}" if r.get("fix") and r["status"] != "ok" else "") if r else ""
            rows.append(Row(f"{self.mark(r['status']) if r else ' '}   address  {e['address'] or '-'}{note.get(e['source'], '')}",
                            f"{e['help']}  Stored in {e['stored_in']}." + result, r["status"] if r else "", key=(e["id"], "address")))
            if e["model_key"]:
                rows.append(Row(f"    model    {e['model'] or '(not set: the server picks)'}",
                                "Model name to ask for. llama.cpp ignores it; model routers and other servers need it. "
                                "Test lists the server's models.", key=(e["id"], "model")))
            if e["api_key"]:
                rows.append(Row(f"    API key  {'set' if e['api_key_set'] else 'not set'}",
                                f"Only if the server was started with a key (llama.cpp --api-key). Stored as {e['api_key']} "
                                "in the private secrets file; never shown. x removes it.", "ok" if e["api_key_set"] else "",
                                key=(e["id"], "key")))
        if not rows:
            rows.append(Row("This role talks to no outside servers that can be set here.", tone="info"))
        return rows

    def test_endpoint(self, eid: str) -> None:
        from . import endpoints as ep
        self.say(f"Testing {eid}...", "info")
        self.draw()
        r = ep.test(self.site, eid)
        self.ep_results[eid] = r
        self.say(f"{eid}: {r['detail']}", {"ok": "ok", "warn": "warn", "fail": "fail"}.get(r["status"], "info"))

    def rows_secrets(self) -> list[Row]:
        return [Row(f"{self.mark('ok' if e['set'] else 'todo')} {e['name']:<16} {'set' if e['set'] else 'not set'}",
                    (e["help"] or "Extra entry in .env.") + "  Values are never shown.", "ok" if e["set"] else "", key=e["name"])
                for e in self.site.env_status()]

    def rows_checks(self) -> list[Row]:
        return [Row(f"{self.mark(c.status)} {c.title}: {c.detail}", c.fix or "Nothing to do.", c.status) for c in self.site.checks()]

    def rows_service(self) -> list[Row]:
        if self.site.role != "detector":
            rows = self.site.stack_status(self.site.role)
            if rows is None:
                return [Row("The stack is not installed here, or docker cannot be asked.", tone="warn")]
            return [Row(f"{self.mark('ok' if r['state'] == 'running' else 'warn')} {r['name']:<18} {r['state']:<10} {r['status']}",
                        "Start or rebuild the stack from the Install page.", "ok" if r["state"] == "running" else "warn") for r in rows]
        s = self.site.service_status()
        state = "running" if s["active"] else ("stopped" if s["installed"] else "not installed")
        rows = [Row(f"Detector service: {state}" + ("  (starts at login)" if s["enabled"] else ""),
                    "s start, t stop, x restart. Install it from the Install page.", "ok" if s["active"] else "info"),
                Row("Recent log", heading=True)]
        return rows + [Row("  " + line) for line in self.site.logs(40)]

    def rows_events(self) -> list[Row]:
        ev = self.site.events(60)
        if not ev:
            return [Row("No events recorded yet.", "Events appear here once the detector has seen something.")]
        return [Row(f"{str(e['time'])[:19]}  {str(e['label']):<10} {e['score']!s:<6} -> {e['action']}",
                    f"Second opinion: {e['verifier'] or 'none'}; confirmed: {'yes' if e['confirmed'] else 'no'}; "
                    f"detection took {e['detect_ms']} ms; snapshot {e['snapshot']}",
                    "ok" if e["confirmed"] else "") for e in ev]

    # ------------------------------------------------------------------ drawing
    def put(self, y: int, x: int, text: str, attr: int = 0, width: int | None = None) -> None:
        h, w = self.scr.getmaxyx()
        if y < 0 or y >= h or x >= w:
            return
        text = text[: max(0, (width if width is not None else w - x))]
        try:
            self.scr.addstr(y, x, text, attr)
        except curses.error:
            pass                                             # writing the bottom-right cell raises; harmless

    def draw(self) -> None:
        self.scr.erase()
        h, w = self.scr.getmaxyx()
        if h < 12 or w < 60:
            self.put(0, 0, "Window too small: needs 60 x 12.")
            self.scr.refresh()
            return
        mw = 18
        title = f" squirrel-guard  ·  {ROLES[self.site.role][0]} " if self.utf else f" squirrel-guard  -  {ROLES[self.site.role][0]} "
        self.put(0, 0, title.ljust(w), curses.A_REVERSE | curses.A_BOLD)
        for i, (_, label) in enumerate(MENU):
            attr = curses.A_BOLD if i == self.page else 0
            if i == self.page and self.focus == "menu":
                attr |= curses.A_REVERSE
            self.put(2 + i, 0, f" {'>' if i == self.page else ' '} {label}".ljust(mw), attr)
        for y in range(1, h - 1):
            self.put(y, mw, "|" if not self.utf else "│", self.pair.get("dim", 0))
        detail_h = 4
        body_h = h - 3 - detail_h
        x0, bw = mw + 2, w - mw - 3
        if self.sel < self.top:
            self.top = self.sel
        if self.sel >= self.top + body_h:
            self.top = self.sel - body_h + 1
        if self.top and self.top < len(self.rows) and self.rows[self.top - 1].heading and self.sel == self.top:
            self.top -= 1                                    # keep a section heading visible above its first row
        for n, row in enumerate(self.rows[self.top: self.top + body_h]):
            i = self.top + n
            attr = self.pair.get(row.tone, 0)
            if row.heading:
                attr = curses.A_BOLD | curses.A_UNDERLINE
            if i == self.sel and self.focus == "page" and not row.heading:
                attr = curses.A_REVERSE
            self.put(1 + n, x0, row.text.ljust(bw) if i == self.sel else row.text, attr, bw)
        if len(self.rows) > body_h:
            self.put(1, w - 1, "^" if self.top else " ")
            self.put(body_h, w - 1, "v" if self.top + body_h < len(self.rows) else " ")
        dy = h - 1 - detail_h
        self.put(dy, mw + 1, ("─" if self.utf else "-") * (w - mw - 1), self.pair.get("dim", 0))
        detail = self.rows[self.sel].detail if self.rows and self.focus == "page" else ""
        if self.msg:
            self.put(dy + 1, x0, self.msg, self.pair.get(self.msg_tone, 0) | curses.A_BOLD, bw)
            lines = textwrap.wrap(detail, bw)[: detail_h - 2]
            off = 2
        else:
            lines = textwrap.wrap(detail, bw)[: detail_h - 1]
            off = 1
        for n, line in enumerate(lines):
            self.put(dy + off + n, x0, line)
        self.put(h - 1, 0, (" " + self.hints()).ljust(w - 1), curses.A_REVERSE)
        self.scr.refresh()

    def hints(self) -> str:
        if self.focus == "menu":
            return "Up/Down choose   Enter open   ? help   q quit"
        page = MENU[self.page][0]
        extra = {"overview": "Enter open", "install": "Enter run step   a run all remaining   d dry run   o role   m venv/docker   p stack folder",
                 "config": "Enter change", "endpoints": "Enter change   t test   a test all   x remove key",
                 "secrets": "Enter set   x remove", "checks": "r run again",
                 "service": "s start   t stop   x restart   r refresh", "events": "r refresh"}[page]
        return f"{extra}   Left back   ? help   q quit"

    # ------------------------------------------------------------------ interaction helpers
    def say(self, text: str, tone: str = "ok") -> None:
        self.msg, self.msg_tone = text, tone

    def prompt(self, label: str, initial: str = "") -> str | None:
        """One-line input on the bottom of the screen. Esc cancels (returns None)."""
        h, w = self.scr.getmaxyx()
        buf, pos = list(initial), len(initial)
        curses.curs_set(1)
        try:
            while True:
                self.put(h - 2, 0, " " * (w - 1))
                self.put(h - 2, 0, (label + ": ")[: w - 10], curses.A_BOLD)
                x0 = min(len(label) + 2, w - 10)
                room = w - x0 - 1
                start = max(0, pos - room + 1)
                self.put(h - 2, x0, "".join(buf)[start: start + room])
                self.put(h - 1, 0, " Enter save   Esc cancel   Ctrl-U clear".ljust(w - 1), curses.A_REVERSE)
                self.scr.move(h - 2, min(w - 1, x0 + pos - start))
                self.scr.refresh()
                try:
                    ch = self.scr.get_wch()
                except curses.error:
                    continue
                if ch in ("\n", "\r") or ch == curses.KEY_ENTER:
                    return "".join(buf)
                if ch == "\x1b":
                    return None
                if ch in (curses.KEY_BACKSPACE, "\x7f", "\b"):
                    if pos:
                        pos -= 1
                        buf.pop(pos)
                elif ch == curses.KEY_DC:
                    if pos < len(buf):
                        buf.pop(pos)
                elif ch == curses.KEY_LEFT:
                    pos = max(0, pos - 1)
                elif ch == curses.KEY_RIGHT:
                    pos = min(len(buf), pos + 1)
                elif ch in (curses.KEY_HOME, "\x01"):
                    pos = 0
                elif ch in (curses.KEY_END, "\x05"):
                    pos = len(buf)
                elif ch == "\x15":
                    buf, pos = [], 0
                elif isinstance(ch, str) and ch.isprintable():
                    buf.insert(pos, ch)
                    pos += 1
        finally:
            curses.curs_set(0)

    def confirm(self, question: str) -> bool:
        h, w = self.scr.getmaxyx()
        self.put(h - 2, 0, " " * (w - 1))
        self.put(h - 2, 0, (question + " [y/N] ")[: w - 1], curses.A_BOLD | self.pair.get("warn", 0))
        self.scr.refresh()
        while True:
            try:
                ch = self.scr.get_wch()
            except curses.error:
                continue
            return ch in ("y", "Y")

    def shell(self, fn) -> None:
        """Leave the full-screen view, run fn() on the plain terminal, wait for Enter, come back."""
        curses.def_prog_mode()
        curses.endwin()
        try:
            fn()
        except ConfigError as e:
            print(f"\n{e}")
        except KeyboardInterrupt:
            print("\ninterrupted")
        try:
            input("\nPress Enter to return to the menu ")
        except (EOFError, KeyboardInterrupt):
            pass
        curses.reset_prog_mode()
        self.scr.clear()
        self.load()

    # ------------------------------------------------------------------ actions
    def run_steps(self, ids: list[str]) -> None:
        role = self.site.role

        def go() -> None:
            for sid in ids:
                step = next(s for s in self.site.steps(role) if s.id == sid)
                print(f"\n== {step.title} ==\n{step.why}")
                answer = None
                if step.needs and not self.dry_run:
                    answer = getpass.getpass(f"{step.needs_help}\n(hidden) > ")
                    if not answer.strip():
                        print("nothing entered: skipped")
                        continue
                rc = self.site.run_step(role, sid, lambda line: print("  " + line, flush=True), dry_run=self.dry_run, answer=answer)
                if rc:
                    print(f"\nThis step did not finish (code {rc}). Later steps were not run.")
                    return
            print("\nDone." if not self.dry_run else "\nDry run: nothing was changed.")
        self.shell(go)

    def activate(self) -> None:
        page = MENU[self.page][0]
        row = self.rows[self.sel] if self.rows else None
        if not row or row.heading:
            return
        if page == "overview":
            if row.key == "role":
                self.choose_role()
            elif row.key:
                self.goto(row.key)
        elif page == "install" and row.key:
            step = next(s for s in self.site.steps() if s.id == row.key)
            if step.manual:
                self.say("This step is done by hand: see the text below.", "info")
            elif step.done and not self.dry_run and not self.confirm(f"“{step.title}” is already done. Run it again?"):
                self.say("Left as it is.", "info")
            else:
                self.run_steps([row.key])
        elif page == "config" and row.key:
            f = self.site.field(row.key)
            if not f.editable:
                self.say("This setting spans several lines: change it in a text editor.", "warn")
                return
            v = f.value if f.line >= 0 else f.default
            if f.kind == "bool":
                new = "no" if v else "yes"
            else:
                shown = ", ".join(map(str, v)) if isinstance(v, list) else str(v)
                new = self.prompt(f.id, shown)
                if new is None:
                    self.say("Not changed.", "info")
                    return
            try:
                self.site.set_value(f.id, new)
                self.say(f"Saved {f.id}. Restart the service for it to take effect.", "ok")
            except ConfigError as e:
                self.say(str(e), "fail")
            self.load()
        elif page == "endpoints" and row.key:
            from . import endpoints as ep
            eid, what = row.key
            try:
                if what == "key":
                    def go() -> None:
                        print(f"\nAPI key for {eid}\nLeave empty to keep the current key.")
                        value = getpass.getpass("(hidden) > ")
                        if value.strip():
                            ep.set_api_key(self.site, self.site.role, eid, value)
                            print("Stored (private file, mode 600).")
                        else:
                            print("Not changed.")
                    self.shell(go)
                    return
                cur = next(e for e in ep.listing(self.site) if e["id"] == eid)
                new = self.prompt(f"{eid} {what} (empty clears)", (cur["address"] if cur["source"] == "set" else "") if what == "address" else cur["model"])
                if new is None:
                    self.say("Not changed.", "info")
                    return
                if what == "address":
                    ep.set_address(self.site, self.site.role, eid, new)
                    self.test_endpoint(eid) if new.strip() else self.say("Address cleared.", "ok")
                else:
                    ep.set_model(self.site, self.site.role, eid, new)
                    self.say("Model saved. Restart the service or containers for it to take effect.", "ok")
            except ConfigError as e:
                self.say(str(e), "fail")
            self.load()
        elif page == "secrets" and row.key:
            name = row.key

            def go() -> None:
                help_ = next((e["help"] for e in self.site.env_status() if e["name"] == name), "")
                print(f"\n{name}\n{help_}\nLeave empty to keep the current value.")
                value = getpass.getpass("(hidden) > ")
                if value.strip():
                    self.site.env_set(name, value)
                    print("Stored (private file, mode 600).")
                else:
                    print("Not changed.")
            self.shell(go)

    def choose_role(self) -> None:
        keys = list(ROLES)
        nxt = keys[(keys.index(self.site.role) + 1) % len(keys)]
        if self.confirm(f"Change this machine's role to “{ROLES[nxt][0]}”?"):
            self.site.set_setting("role", nxt)
            self.say(f"Role: {ROLES[nxt][0]}.", "ok")
        self.load()

    def goto(self, key: str) -> None:
        self.page = next(i for i, (k, _) in enumerate(MENU) if k == key)
        self.sel, self.top, self.focus = 0, 0, "page"
        self.load()

    def page_key(self, ch) -> None:
        page = MENU[self.page][0]
        row = self.rows[self.sel] if self.rows else None
        try:
            if page == "install":
                if ch == "d":
                    self.dry_run = not self.dry_run
                    self.say("Dry run on: steps only show what they would do." if self.dry_run else "Dry run off.", "info")
                elif ch == "a":
                    todo = [s.id for s in self.site.steps() if not s.done and not s.manual]
                    if not todo:
                        self.say("Nothing left to run.", "ok")
                    elif self.dry_run or self.confirm(f"Run the {len(todo)} remaining steps in order?"):
                        self.run_steps(todo)
                elif ch == "o":
                    self.choose_role()
                elif ch == "m" and self.site.role == "detector":
                    new = "venv" if self.site.docker else "docker"
                    self.site.set_setting("runtime", new)
                    self.say(f"The detector now runs as: {RUNTIMES[new]}.", "ok")
                elif ch == "p" and self.site.role == "recorder":
                    new = self.prompt("Stack folder", self.site.settings()["recorder_dir"])
                    if new:
                        self.site.set_setting("recorder_dir", new)
                        self.say("Stack folder saved.", "ok")
            elif page == "endpoints" and ch in ("t", "a", "x"):
                from . import endpoints as ep
                if ch == "a":
                    for e in ep.listing(self.site):
                        self.test_endpoint(e["id"])
                    bad = [r for r in self.ep_results.values() if r["status"] == "fail"]
                    self.say(f"{len(bad)} failed: see the rows marked {self.mark('fail')}." if bad else "Tested: nothing failed.", "fail" if bad else "ok")
                elif row and row.key and ch == "t":
                    self.test_endpoint(row.key[0])
                elif row and row.key and ch == "x" and row.key[1] == "key":
                    if self.confirm(f"Remove the API key for {row.key[0]}?"):
                        ep.set_api_key(self.site, self.site.role, row.key[0], None)
                        self.say("API key removed.", "ok")
            elif page == "secrets" and ch == "x" and row and row.key:
                if self.confirm(f"Remove {row.key} from the secrets file?"):
                    self.site.env_set(row.key, None)
                    self.say(f"{row.key} removed.", "ok")
            elif page == "service" and ch in ("s", "t", "x") and self.site.role == "detector":
                action = {"s": "start", "t": "stop", "x": "restart"}[ch]
                if action == "start" or self.confirm(f"{action.capitalize()} the detector service?"):
                    rc, out = self.site.service(action)
                    self.say(f"{action}: done." if rc == 0 else (out.splitlines() or ["failed"])[-1], "ok" if rc == 0 else "fail")
        except ConfigError as e:
            self.say(str(e), "fail")
        self.load()

    def show_help(self) -> None:
        self.scr.erase()
        for n, line in enumerate(HELP.splitlines()):
            self.put(1 + n, 2, line)
        self.put(self.scr.getmaxyx()[0] - 1, 0, " Press any key ".ljust(self.scr.getmaxyx()[1] - 1), curses.A_REVERSE)
        self.scr.refresh()
        try:
            self.scr.get_wch()
        except curses.error:
            pass

    # ------------------------------------------------------------------ main loop
    def loop(self) -> int:
        while True:
            self.draw()
            try:
                ch = self.scr.get_wch()
            except curses.error:
                continue
            except KeyboardInterrupt:
                return 130
            self.msg = ""
            if ch == curses.KEY_RESIZE:
                continue
            if ch in ("q", "Q"):
                return 0
            if ch == "?":
                self.show_help()
            elif ch == "\t":
                self.focus = "page" if self.focus == "menu" else "menu"
            elif ch == "r":
                self.load()
                self.say("Refreshed.", "info")
            elif self.focus == "menu":
                if ch in (curses.KEY_UP, "k"):
                    self.page = (self.page - 1) % len(MENU)
                    self.sel = self.top = 0
                    self.load()
                elif ch in (curses.KEY_DOWN, "j"):
                    self.page = (self.page + 1) % len(MENU)
                    self.sel = self.top = 0
                    self.load()
                elif ch in ("\n", "\r", curses.KEY_ENTER, curses.KEY_RIGHT, "l"):
                    self.focus = "page"
                    self._skip_headings(1)
                elif isinstance(ch, str) and ch.isdigit() and 1 <= int(ch) <= len(MENU):
                    self.page, self.sel, self.top = int(ch) - 1, 0, 0
                    self.load()
            else:
                n = len(self.rows)
                if ch in (curses.KEY_UP, "k") and n:
                    self.sel = (self.sel - 1) % n
                    self._skip_headings(-1)
                elif ch in (curses.KEY_DOWN, "j") and n:
                    self.sel = (self.sel + 1) % n
                    self._skip_headings(1)
                elif ch == curses.KEY_NPAGE and n:
                    self.sel = min(n - 1, self.sel + 10)
                    self._skip_headings(1)
                elif ch == curses.KEY_PPAGE and n:
                    self.sel = max(0, self.sel - 10)
                    self._skip_headings(1)
                elif ch in (curses.KEY_LEFT, "h", "\x1b"):
                    self.focus = "menu"
                elif ch in ("\n", "\r", curses.KEY_ENTER, curses.KEY_RIGHT, "l"):
                    self.activate()
                elif isinstance(ch, str):
                    self.page_key(ch)


def main(site: Site, start: str = "overview") -> int:
    locale.setlocale(locale.LC_ALL, "")
    import os
    os.environ.setdefault("ESCDELAY", "25")                  # Esc reacts at once instead of after a second
    return curses.wrapper(lambda scr: App(scr, site, start).loop())

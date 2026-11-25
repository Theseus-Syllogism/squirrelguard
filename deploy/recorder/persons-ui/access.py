"""Roles and the audit log for the persons UI.

Roles, lowest to highest:
  viewer        read the pages and search
  investigator  also edit identities and vehicles, review matches, export evidence (and see their own exports)
  supervisor    also see every export, the custody log and the audit log

Single sign-on (TRUST_PROXY_USER=1): the role comes from the Remote-Groups header nginx copies from Authelia.
ROLE_GROUPS maps groups to roles, e.g. "supervisor=admins,sg-supervisor;investigator=sg-investigator;viewer=sg-viewer".
A signed-in user in none of these groups is refused. Without SSO (one shared password) the account has BASIC_ROLE.

Audit log: every page or API request (not images or static files) and every refusal is appended to audit_log, a hash
chain like the custody log, with the user, role, action, target and source address.
"""
from __future__ import annotations

import os
from functools import wraps

from flask import Response, g, request

import evidence_export as evx

RANK = {"viewer": 1, "investigator": 2, "supervisor": 3}
DEFAULT_GROUPS = "supervisor=admins,sg-supervisor;investigator=sg-investigator;viewer=sg-viewer"
AUDIT_FIELDS = ("ts", "actor", "role", "action", "target", "detail", "remote")
# requests that are not worth a log line: pictures, scripts, styles (the page that shows them is logged)
QUIET = {"ui_static", "crop", "vcrop", "face_img", "face_template_img", "evidence_file", "replay", "static", "act"}


def group_map(spec: str | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (spec or os.environ.get("ROLE_GROUPS") or DEFAULT_GROUPS).split(";"):
        role, _, groups = part.partition("=")
        role = role.strip()
        if role in RANK:
            for grp in (x.strip() for x in groups.split(",")):
                if grp and RANK[role] > RANK.get(out.get(grp, ""), 0):
                    out[grp] = role                                 # a group listed twice keeps the higher role
    return out


def role_for(groups_header: str | None, sso: bool) -> str | None:
    if not sso:
        r = os.environ.get("BASIC_ROLE", "supervisor")
        return r if r in RANK else "viewer"
    gm = group_map()
    best = None
    for grp in (groups_header or "").split(","):
        r = gm.get(grp.strip())
        if r and (best is None or RANK[r] > RANK[best]):
            best = r
    return best


def at_least(role: str) -> bool:
    return RANK.get(getattr(g, "role", None) or "", 0) >= RANK[role]


def require(role: str):
    """Route decorator: refuse (403) below `role`. The refusal is logged by the after_request hook."""
    def deco(fn):
        @wraps(fn)
        def inner(*a, **k):
            if not at_least(role):
                g.audit_action = "denied"
                return Response(f"This needs the {role} role; you are signed in as {g.get('role') or 'no role'}.\n", 403,
                                {"Content-Type": "text/plain; charset=utf-8"})
            return fn(*a, **k)
        return inner
    return deco


def append(c, table: str, fields: dict, keys: tuple[str, ...], lock: int) -> str:
    """Append one entry to a hash-chained table. One writer at a time per table (advisory lock)."""
    with c.transaction():
        c.execute("SELECT pg_advisory_xact_lock(%s)", (lock,))
        last = c.execute(f"SELECT entry_hash FROM {table} ORDER BY id DESC LIMIT 1").fetchone()
        prev = last[0] if last else evx.GENESIS
        h = evx.chain_hash(prev, {k: fields[k] for k in keys})
        cols = list(keys) + ["prev_hash", "entry_hash"]
        c.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})",
                  [fields[k] for k in keys] + [prev, h])
    return h


def audit(c, action: str, target: str = "", detail: str = "") -> str:
    a = request.authorization
    actor = request.headers.get("Remote-User") or (a.username if a else "") or "unknown"
    return append(c, "audit_log", {"ts": evx.stamp(evx.utcnow()), "actor": actor, "role": g.get("role") or "",
                                   "action": action, "target": target[:300], "detail": detail[:1000],
                                   "remote": request.remote_addr or ""}, AUDIT_FIELDS, 752002)


def request_action() -> tuple[str, str, str]:
    """(action, target, detail) describing the current request for the audit log."""
    ep = request.endpoint or "unknown"
    action = g.get("audit_action") or (("view:" if request.method in ("GET", "HEAD") else "change:") + ep)
    target = request.path
    detail = ""
    if request.args:
        detail = "&".join(f"{k}={v}" for k, v in request.args.items(multi=True) if k != "done")
    if request.method == "POST":
        keys = sorted(k for k in request.form.keys() if k not in ("csrf",))
        detail = (detail + " " if detail else "") + "fields: " + ",".join(keys)
    return action, target, detail

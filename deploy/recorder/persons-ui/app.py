"""Persons review UI: identities (re-ID clusters) and their sightings. Rename / mark household / merge / delete.

Private visitor log. Age and gender shown here are APPARENT guesses from the LLM, with their confidence.
"""
import collections
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import psycopg
from flask import Flask, Response, abort, g, redirect, render_template_string, request, send_from_directory, url_for

E = os.environ
CROPS = Path(E.get("CROP_DIR", "/data/snapshots/persons"))
VCROPS = CROPS.parent / "vehicles"
EVID = CROPS.parent / "evidence"
FACES = CROPS.parent / "faces"          # per-sighting face evidence crops (written by the bridge)
ENROLL_KINDS = ("household", "known")   # the only identities a face may be enrolled for (and named by)
FACE_MIN_FACES = int(E.get("FACE_MIN_FACES", "3"))
FACE_MIN_QUALITY = float(E.get("FACE_MIN_QUALITY", "0.10"))
BAND_COLOURS = {"strong": "#1f9d55", "likely": "#4c9a2a", "possible": "#c98a12", "unlikely": "#d64545"}
NEAR = "interval '2 minutes'"  # person <-> vehicle co-occurrence window
app = Flask(__name__)
import access  # noqa: E402  (roles + audit log)
import mimetypes  # noqa: E402  (python:slim has no /etc/mime.types: .webp would go out as octet-stream)
mimetypes.add_type("image/webp", ".webp")
TZ = dt.datetime.now().astimezone().tzinfo


def db():
    return psycopg.connect(E["PG_URL"], autocommit=True)


@app.before_request
def auth():
    if request.path.startswith("/act/"):   # signed one-tap links from notifications: the signature is the auth
        return None
    if E.get("TRUST_PROXY_USER") == "1":
        # single sign-on: nginx + Authelia in front, this app listens on 127.0.0.1 only (HOST below), so the
        # Remote-User header can only come from nginx, which overwrites any client-sent copy
        if not request.headers.get("Remote-User"):
            return Response("sign-in required (open https://persons.cam.example.org)", 401)
    else:
        a = request.authorization
        if not a or a.username != E["PERSONS_UI_USER"] or a.password != E["PERSONS_UI_PASS"]:
            return Response("login required", 401, {"WWW-Authenticate": 'Basic realm="persons"'})
    g.role = access.role_for(request.headers.get("Remote-Groups"), E.get("TRUST_PROXY_USER") == "1")
    if g.role is None:
        g.audit_action = "denied"
        return Response("Your account has no role for this site. Ask a supervisor to add you to a group "
                        "(sg-viewer, sg-investigator or sg-supervisor).\n", 403, {"Content-Type": "text/plain; charset=utf-8"})
    if request.method == "POST" and not access.at_least("investigator"):   # every change needs investigator or above
        g.audit_action = "denied"
        return Response("Changes need the investigator role; you are a viewer.\n", 403, {"Content-Type": "text/plain; charset=utf-8"})


@app.after_request
def audit_request(resp):
    """Audit log: every page and API request that reached sign-in, including refusals (pictures and scripts are not)."""
    if request.endpoint in access.QUIET or request.path.startswith(("/ui/", "/act/")) or resp.status_code in (304, 401, 404):
        return resp
    if resp.status_code >= 400 and resp.status_code != 403:
        return resp
    try:
        with db() as c:
            action, target, detail = access.request_action()
            access.audit(c, action if resp.status_code != 403 else "denied", target, detail)
    except Exception as e:                       # the page still works; the gap itself is reported in the server log
        app.logger.error("audit log write failed: %s", e)
    return resp


def name(iid, label):
    return label or f"Unknown #{iid}"


def when(t):
    return t.astimezone(TZ).strftime("%Y-%m-%d %H:%M") if t else ""


def summary(rows):
    """rows of (attributes, age_low, age_high, gender, gender_conf) -> typical description."""
    ages = [(l, h) for _, l, h, _, _ in rows if isinstance(l, int) and isinstance(h, int)]
    g = collections.defaultdict(list)
    for _, _, _, gen, conf in rows:
        if gen:
            g[gen].append(conf or 0)
    gender = max(g, key=lambda k: (len(g[k]), np.mean(g[k]))) if g else None
    last = next((a for a, *_ in rows if a), {}) or {}
    clothes = ", ".join(x for x in [
        " ".join(filter(None, [last.get("upper_color"), last.get("upper_clothing")])),
        " ".join(filter(None, [last.get("lower_color"), last.get("lower_clothing")]))] if x and "none" not in x)
    return {
        "age": f"{int(np.median([l for l, _ in ages]))}–{int(np.median([h for _, h in ages]))}" if ages else "?",
        "gender": f"{gender} ({np.mean(g[gender]):.0%})" if gender else "?",
        "clothes": clothes or "?",
    }


# --- PULC attributes (squirrel_guard/pulc.py; columns sightings.pulc / vehicle_sightings.pulc) -----------------------
# search filter name -> (PULC label, threshold on the visit-mean probability). Sleeves compare the pair.
PULC_FILTERS = {"backpack": ("Backpack", 0.5), "hat": ("Hat", 0.5), "long_sleeves": ("LongSleeve", None),
                "short_sleeves": ("ShortSleeve", None), "shorts": ("Shorts", 0.5), "long_coat": ("LongCoat", 0.5),
                "skirt_or_dress": ("Skirt&Dress", 0.5), "handbag": ("HandBag", 0.5), "shoulder_bag": ("ShoulderBag", 0.5)}


def pulc_by(c, sql: str, args) -> dict:
    """{key: pulc} from a (key, pulc) query; {} when the column does not exist yet (schema not applied)."""
    try:
        with c.transaction():
            return {k: v for k, v in c.execute(sql, args).fetchall() if v}
    except Exception:
        return {}


def pulc_sql_filter(names: list) -> tuple[str, list]:
    """SQL condition on s.pulc for the chosen filters (all must hold)."""
    conds = []
    for n in names:
        lab, thr = PULC_FILTERS[n]
        if thr is None:
            other = "ShortSleeve" if lab == "LongSleeve" else "LongSleeve"
            conds.append(f"(s.pulc->'attributes'->>'{lab}')::real > (s.pulc->'attributes'->>'{other}')::real")
        else:
            conds.append(f"(s.pulc->'attributes'->>'{lab}')::real >= {thr}")
    return " AND ".join(conds) or "true", []


def pulc_matches(pv: dict | None, names: list) -> bool:
    a = (pv or {}).get("attributes") or {}
    for n in names:
        lab, thr = PULC_FILTERS[n]
        if lab not in a:
            return False
        if thr is None:
            other = "ShortSleeve" if lab == "LongSleeve" else "LongSleeve"
            if not a[lab] > a.get(other, 1.0):
                return False
        elif a[lab] < thr:
            return False
    return True


def pulc_line(pv: dict | None) -> str:
    """'apparent: backpack, long sleeves; mostly back view' (or why there is none)."""
    if not pv:
        return ""
    if not pv.get("crops"):
        return f"PULC: {pv.get('skipped', 'no usable crop')}"
    tags = ", ".join(t.replace("_", " ") for t in pv.get("tags") or []) or "no clear attributes"
    o = pv.get("orientation") or {}
    k = max(o, key=o.get) if o else None
    return f"apparent: {tags}" + (f"; mostly {k} view" if k else "")


SECTIONS = [("index", "People", ("index", "identity", "sighting")), ("vehicles", "Vehicles", ("vehicles", "vehicle")),
            ("alerts", "Alerts", ("alerts",)), ("reports", "Reports", ("reports", "report")),
            ("exports", "Evidence", ("exports", "export_view", "export_new", "export_job")), ("audit_page", "Audit", ("audit_page",))]
SECTION_ROLE = {"exports": "investigator", "audit_page": "supervisor"}
PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark"><title>Persons</title>
<script>try{var t=localStorage.getItem("sg-replay-theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}</script>
<link rel="stylesheet" href="{{ url_for('ui_static', f='ui.css') }}?v={{ ui_ver }}"><script src="{{ url_for('ui_static', f='ui.js') }}?v={{ ui_ver }}" defer></script>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%232563eb' stroke-width='2'%3E%3Ccircle cx='12' cy='8' r='4'/%3E%3Cpath d='M4 21a8 8 0 0 1 16 0'/%3E%3C/svg%3E">
</head><body><a class="skip" href="#main">Skip to content</a>
<header class="app-header"><div class="in">
 <a class="brand" href="{{ url_for('index') }}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></svg>Persons <span>· visitor log</span></a>
 <nav class="nav" aria-label="Sections">{% for ep, label, eps in sections %}<a href="{{ url_for(ep) }}"{% if request.endpoint in eps %} aria-current="page"{% endif %}>{{ label }}</a>{% endfor %}</nav>
 <form class="searchbox" role="search" action="{{ url_for('search') }}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
  <input id="q" name="q" type="search" placeholder="Search: e.g. carried a box to the porch last week" value="{{ request.args.get('q', '') }}" aria-label="Search sightings"><kbd>/</kbd></form>
 {% if who %}<span class="whoami" title="Signed in as {{ who }}, role {{ role }}">{{ who }} <span class="k">{{ role }}</span></span>{% endif %}
 <button id="theme-toggle" class="icon-btn" type="button" aria-label="Theme"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg></button>
</div></header>
<main id="main"><p class="page-note"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7.5v.5"/></svg>Apparent age and gender are model guesses, not facts. Names come only from people you enrolled.</p>
{{ body|safe }}</main></body></html>"""
UI_DIR = Path(__file__).with_name("static")
UI_VER = str(int(max((f.stat().st_mtime for f in UI_DIR.glob("ui.*")), default=0)))


@app.route("/ui/<path:f>")
def ui_static(f):
    """Shared design system (ui.css, ui.js); cached by the browser, busted by ?v=<mtime>."""
    if f not in ("ui.css", "ui.js"):
        abort(404)
    return send_from_directory(UI_DIR, f, max_age=86400)


@app.context_processor
def _shell():
    a = request.authorization
    who = request.headers.get("Remote-User") or (a.username if a else "")
    return {"sections": [x for x in SECTIONS if access.at_least(SECTION_ROLE.get(x[0], "viewer"))], "ui_ver": UI_VER,
            "who": who, "role": g.get("role") or ""}


@app.route("/")
def index():
    with db() as c:
        ids = c.execute("""SELECT i.id, i.label, i.kind, i.first_seen, i.last_seen, count(s.id),
                                  (SELECT crop_file FROM sightings x WHERE x.identity_id = i.id ORDER BY ts_start DESC LIMIT 1),
                                  count(*) FILTER (WHERE s.match_kind = 'new_possible'),
                                  percentile_cont(0.5) WITHIN GROUP (ORDER BY s.height_cm), count(s.height_cm)
                           FROM identities i LEFT JOIN sightings s ON s.identity_id = i.id
                           GROUP BY i.id ORDER BY i.last_seen DESC""").fetchall()
        rows = {}
        for (iid,) in [(r[0],) for r in ids]:
            rows[iid] = c.execute("SELECT attributes, age_low, age_high, gender, gender_conf FROM sightings "
                                  "WHERE identity_id = %s ORDER BY ts_start DESC LIMIT 20", (iid,)).fetchall()
    body = render_template_string("""<h2>Identities ({{ ids|length }})</h2><table>
<tr><th></th><th>Identity</th><th>Sightings</th><th>Seen</th><th>Apparent age / gender</th><th>Height</th><th>Typical clothing</th></tr>
{% for iid, label, kind, first, last, n, crop, poss, hmed, hn in ids %}{% set s = sums[iid] %}
<tr><td>{% if crop %}<img class="t" src="{{ url_for('crop', f=crop) }}">{% endif %}</td>
<td><a href="{{ url_for('identity', iid=iid) }}">{{ name(iid, label) }}</a> <span class="k {{ kind }}">{{ kind }}</span>
{% if poss %}<div class="flag">possible match flagged</div>{% endif %}</td>
<td>{{ n }}</td><td><small>first {{ when(first) }}<br>last {{ when(last) }}</small></td>
<td>{{ s.age }}<br><small>{{ s.gender }}</small></td>
<td>{% if hn %}{{ '%.0f'|format(hmed) }} cm<br><small>{{ hn }} sighting{{ 's' if hn > 1 }}</small>{% else %}<small>-</small>{% endif %}</td>
<td>{{ s.clothes }}</td></tr>{% endfor %}</table>""",
        ids=ids, sums={k: summary(v) for k, v in rows.items()}, name=name, when=when)
    return render_template_string(PAGE, body=body)


@app.route("/identity/<int:iid>")
def identity(iid):
    with db() as c:
        idt = c.execute("SELECT id, label, kind, notes, first_seen, last_seen, n FROM identities WHERE id = %s", (iid,)).fetchone()
        if not idt:
            abort(404)
        ss = c.execute("""SELECT id, ts_start, ts_end, camera, crop_file, attributes, age_low, age_high, gender,
                                 gender_conf, match_sim, match_kind, possible_id, possible_sim, frigate_event,
                                 height_cm, height_iqr_cm, gait, explain
                          FROM sightings WHERE identity_id = %s ORDER BY ts_start DESC LIMIT 200""", (iid,)).fetchall()
        hs = c.execute("""SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY height_cm),
                                 percentile_cont(0.25) WITHIN GROUP (ORDER BY height_cm),
                                 percentile_cont(0.75) WITHIN GROUP (ORDER BY height_cm), count(height_cm),
                                 percentile_cont(0.5) WITHIN GROUP (ORDER BY (gait->>'cadence_spm')::real)
                                     FILTER (WHERE (gait->>'usable')::boolean),
                                 count(*) FILTER (WHERE (gait->>'usable')::boolean)
                          FROM sightings WHERE identity_id = %s""", (iid,)).fetchone()
        others = c.execute("SELECT id, label FROM identities WHERE id <> %s ORDER BY last_seen DESC", (iid,)).fetchall()
        templates = face_templates(c, iid)
        models_html = face_models_html(c, iid)
        pulcs = pulc_by(c, "SELECT id, pulc FROM sightings WHERE identity_id = %s AND pulc IS NOT NULL", (iid,))
        near = c.execute(f"""SELECT v.id, v.label, v.plate, v.color, v.body_type, count(*) FROM sightings s
                            JOIN vehicle_sightings vs ON vs.ts_start BETWEEN s.ts_start - {NEAR} AND COALESCE(s.ts_end, s.ts_start) + {NEAR}
                            JOIN vehicles v ON v.id = vs.vehicle_id WHERE s.identity_id = %s
                            GROUP BY v.id ORDER BY count(*) DESC LIMIT 10""", (iid,)).fetchall()
        driven = occupant_vehicles(c, iid)
        gprof = gait_profile(c, iid)
    walk_html = render_template_string("""{% if p.walks %}<section aria-labelledby="walk-h"><h3 id="walk-h">Walk profile
<small>{{ p.walks }} measured walk{{ 's' if p.walks != 1 }}</small></h3>
<table><tr><th>Measurement</th><th>Usual</th><th>Spread</th><th>Walks</th></tr>{% for k, f in p.features.items() %}
<tr><td>{{ label[k][0] }}</td><td>{{ fmt(k, f.median) }}</td><td>{{ fmt(k, f.spread) if f.spread is not none else '–' }}</td><td>{{ f.n }}</td></tr>{% endfor %}</table>
{{ chart|safe }}<p class="muted">Gait can support or weaken other evidence; it does not identify anyone.</p></section>{% endif %}""",
        p=gprof, label=gaitlab.LABEL, fmt=gaitlab._fmt,
        chart=gaitlab.svg_cycle({"knee_cycle": {k: v["mean"] for k, v in gprof["knee_cycle"].items()}}, gprof) if gprof["knee_cycle"] else "")
    frigate = E.get("FRIGATE_URL", "https://frigate.cam.example.org")
    body = render_template_string("""<h2>{{ name(i[0], i[1]) }} <span class="k {{ i[2] }}">{{ i[2] }}</span></h2>
{% if i[2] == 'unknown' and not i[1] %}<p><small>Unnamed: grouped automatically by face and/or appearance. It is never named
or trusted automatically; give it a name or mark it household/known (and enrol a face, with consent) below.</small></p>{% endif %}
{% if driven %}<h3>In vehicles (face seen inside)</h3><table><tr><th>Vehicle</th><th>Times</th><th>As</th><th>Last</th><th>Best face match</th><th></th></tr>
{% for d in driven %}<tr><td><a href="{{ url_for('vehicle', vid=d.vid) }}">{{ d.vname }}</a></td><td>{{ d.n }}</td>
<td><small>{{ d.kinds }}</small></td><td><small>{{ when(d.last) }}</small></td><td>{{ '%.2f'|format(d.sim) if d.sim is not none else '-' }}</td>
<td>{% if d.crop %}<img src="{{ url_for('face_img', f=d.crop) }}" style="height:48px;border-radius:4px">{% endif %}</td></tr>{% endfor %}</table>{% endif %}
<div class="stat-row">
 <div class="stat"><b>{{ ss|length }}</b><span>sightings</span></div>
 <div class="stat"><b>{{ when(i[4]) }}</b><span>first seen</span></div>
 <div class="stat"><b>{{ when(i[5]) }}</b><span>last seen</span></div>
 <div class="stat">{% if hs[3] %}<b>{{ '%.0f'|format(hs[0]) }} cm</b><span>height, middle half {{ '%.0f'|format(hs[1]) }}&ndash;{{ '%.0f'|format(hs[2]) }} cm ({{ hs[3] }} sighting{{ 's' if hs[3] > 1 }})</span>{% else %}<b>&ndash;</b><span>height: needs a calibrated camera</span>{% endif %}</div>
 <div class="stat">{% if hs[5] %}<b>{{ '%.0f'|format(hs[4]) }}/min</b><span>step rate ({{ hs[5] }} side-on walk{{ 's' if hs[5] > 1 }})</span>{% else %}<b>&ndash;</b><span>step rate: no clear side-on walk yet</span>{% endif %}</div>
</div>
{% if near %}<p>Seen with vehicles: {% for v in near %}<a href="{{ url_for('vehicle', vid=v[0]) }}">{{ v[1] or ('#%d'|format(v[0])) }} {{ v[2] or '' }} {{ v[3] or '' }} {{ v[4] or '' }}</a> ({{ v[5] }}x){% if not loop.last %}, {% endif %}{% endfor %}</p>{% endif %}
<div class="actions" aria-label="Edit identity"><form method="post" action="{{ url_for('edit', iid=i[0]) }}">
 <input name="label" placeholder="Name" aria-label="Name" value="{{ i[1] or '' }}">
 <select name="kind" aria-label="Kind">{% for k in ['unknown','watch','household','known'] %}<option {{ 'selected' if k==i[2] }}>{{ k }}</option>{% endfor %}</select>
 <input name="notes" placeholder="Notes" aria-label="Notes" size="36" value="{{ i[3] or '' }}"> <button class="primary">Save</button></form>
<span class="sep" aria-hidden="true"></span><form method="post" action="{{ url_for('merge', iid=i[0]) }}" data-confirm-title="Merge identities?" data-confirm="All sightings of this identity move to {target}, and this identity is removed. It cannot be undone." data-confirm-ok="Merge" data-confirm-kind="warn" data-confirm-icon="merge">Merge into
 <select name="into" required aria-label="Merge into"><option value="" disabled selected>Choose identity…</option>{% for o in others %}<option value="{{ o[0] }}">{{ name(o[0], o[1]) }}</option>{% endfor %}</select>
 <button class="warn">Merge</button></form>
<span class="sep" aria-hidden="true"></span><a class="btn" href="{{ url_for('export_new', identity=i[0]) }}">Export evidence package</a>
<span class="sep" aria-hidden="true"></span><form method="post" action="{{ url_for('delete', iid=i[0]) }}" data-confirm-title="Delete this identity?" data-confirm="This deletes the identity, all its sightings and any enrolled faces. It cannot be undone." data-confirm-ok="Delete identity">
 <button class="danger">Delete</button></form></div>
{{ faces_html|safe }}
{{ walk_html|safe }}
<div class="grid">{% for s in ss %}{% set a = s[5] or {} %}<div class="card">
{% if s[4] %}<img src="{{ url_for('crop', f=s[4]) }}">{% endif %}
<b>{{ when(s[1]) }}</b> &middot; {{ s[3] }}<br>
{{ a.get('upper_color','') }} {{ a.get('upper_clothing','') }}; {{ a.get('lower_color','') }} {{ a.get('lower_clothing','') }}<br>
<small>apparent {{ s[6] }}–{{ s[7] }}, {{ s[8] }} ({{ '%.0f%%'|format((s[9] or 0)*100) }})<br>
{{ a.get('activity','') }}<br>match {{ s[11] }} {{ '%.2f'|format(s[10] or 0) }}
{% if s[15] %}<br>height {{ '%.0f'|format(s[15]) }} cm <small>&plusmn;{{ '%.0f'|format((s[16] or 0) / 2) }}</small>
{% if hs[3] and hs[3] >= 3 and (s[15] - hs[0])|abs > 12 %}<span class="flag">differs from usual {{ '%.0f'|format(hs[0]) }} cm</span>{% endif %}{% endif %}
{% set g = s[17] or {} %}{% if g.get('usable') %}<br>walk {{ g.cadence_spm }} steps/min{% if g.step_m %}, step {{ g.step_m }} m, {{ g.speed_mps }} m/s{% endif %}{% endif %}
{% if s[12] %}<br><span class="flag">possibly <a href="{{ url_for('identity', iid=s[12]) }}">#{{ s[12] }}</a> ({{ '%.2f'|format(s[13]) }})</span>{% endif %}
{% if pulcs.get(s[0]) %}<br>{{ pulc_line(pulcs[s[0]]) }}{% endif %}
{% set x = s[18] or {} %}{% if x.band %}<br><span class="band" style="background:{{ colours.get(x.band, '#666') }}">{{ x.band }} {{ x.pct }}%</span>{% endif %}
<br><a href="{{ url_for('sighting', sid=s[0]) }}">Why?</a> &middot; <a href="{{ frigate }}/explore?event_id={{ s[14] }}">Frigate</a></small></div>{% endfor %}</div>""",
        i=idt, ss=ss, hs=hs, others=others, near=near, name=name, when=when, frigate=frigate, colours=BAND_COLOURS,
        driven=driven, walk_html=walk_html,
        faces_html=templates_html(idt, templates) + models_html, pulcs=pulcs, pulc_line=pulc_line)
    return render_template_string(PAGE, body=body)


@app.post("/identity/<int:iid>/edit")
def edit(iid):
    kind = request.form.get("kind", "unknown")
    if kind not in ("unknown", "watch", "household", "known"):
        abort(400)
    with db() as c:
        c.execute("UPDATE identities SET label = NULLIF(%s, ''), kind = %s, notes = NULLIF(%s, '') WHERE id = %s",
                  (request.form.get("label", "").strip()[:80], kind, request.form.get("notes", "").strip()[:500], iid))
        if kind not in ENROLL_KINDS:
            drop_face_templates(c, iid)   # enrolled faces exist only for household/known people
    return redirect(url_for("identity", iid=iid, done="saved"))


@app.post("/identity/<int:iid>/merge")
def merge(iid):
    into = int(request.form["into"])
    if into == iid:
        abort(400)
    with db() as c, c.transaction():
        c.execute("UPDATE sightings SET identity_id = %s, match_kind = 'merged', reviewed = true WHERE identity_id = %s", (into, iid))
        embs = [np.asarray(e, np.float32) for (e,) in c.execute("SELECT embedding FROM sightings WHERE identity_id = %s", (into,)).fetchall()]
        if embs:
            cen = np.mean(embs, 0); cen /= np.linalg.norm(cen) + 1e-12
        else:          # only vehicle-occupant faces on both: no body fingerprint, keep the zero centroid
            cen = np.asarray(c.execute("SELECT centroid FROM identities WHERE id = %s", (into,)).fetchone()[0], np.float32)
        c.execute("""UPDATE identities SET centroid = %s, n = %s,
                       first_seen = LEAST(first_seen, (SELECT first_seen FROM identities WHERE id = %s)),
                       last_seen = GREATEST(last_seen, (SELECT last_seen FROM identities WHERE id = %s)) WHERE id = %s""",
                  ([float(x) for x in cen], len(embs), iid, iid, into))
        try:
            with c.transaction():
                c.execute("UPDATE occupant_faces SET identity_id = %s WHERE identity_id = %s", (into, iid))
                c.execute("UPDATE vehicle_sightings SET driver_identity_id = %s WHERE driver_identity_id = %s", (into, iid))
        except psycopg.errors.UndefinedTable:
            pass
        try:
            with c.transaction():
                c.execute("UPDATE face_templates SET identity_id = %s WHERE identity_id = %s", (into, iid))
                if c.execute("SELECT kind FROM identities WHERE id = %s", (into,)).fetchone()[0] not in ENROLL_KINDS:
                    drop_face_templates(c, into)
        except psycopg.errors.UndefinedTable:
            pass
        c.execute("DELETE FROM identities WHERE id = %s", (iid,))
    return redirect(url_for("identity", iid=into, done="merged"))


@app.post("/identity/<int:iid>/delete")
def delete(iid):
    with db() as c, c.transaction():
        files = [f for (f,) in c.execute("SELECT crop_file FROM sightings WHERE identity_id = %s", (iid,)).fetchall() if f]
        c.execute("DELETE FROM sightings WHERE identity_id = %s", (iid,))
        c.execute("DELETE FROM identities WHERE id = %s", (iid,))
    for f in files:
        (CROPS / f).unlink(missing_ok=True)
    return redirect(url_for("index", done="deleted"))


@app.route("/alerts")
def alerts():
    with db() as c:
        rows = c.execute("SELECT ts, kind, subject, message, sighting, sent FROM alerts ORDER BY ts DESC LIMIT 300").fetchall()
    def link(subject):
        k, _, i = subject.partition(":")
        return url_for("identity", iid=int(i)) if k == "person" else url_for("vehicle", vid=int(i))
    body = render_template_string("""<h2>Alerts ({{ rows|length }})</h2><table>
<tr><th>When</th><th>Kind</th><th>Subject</th><th>Message</th><th>Sent</th></tr>
{% for r in rows %}<tr><td><small>{{ when(r[0]) }}</small></td><td>{{ r[1] }}</td><td><a href="{{ link(r[2]) }}">{{ r[2] }}</a></td>
<td>{{ r[3] }}</td><td>{{ 'yes' if r[5] else 'no' }}</td></tr>{% endfor %}</table>""", rows=rows, when=when, link=link)
    return render_template_string(PAGE, body=body)


@app.route("/reports")
def reports():
    with db() as c:
        rows = c.execute("SELECT id, period, ts, left(text, 160) FROM reports ORDER BY ts DESC LIMIT 100").fetchall()
    body = render_template_string("""<h2>Reports</h2><table><tr><th>When</th><th>Period</th><th>Summary</th></tr>
{% for r in rows %}<tr><td><a href="{{ url_for('report', rid=r[0]) }}">{{ when(r[2]) }}</a></td><td>{{ r[1] }}</td>
<td><small>{{ r[3] }}&hellip;</small></td></tr>{% endfor %}</table>""", rows=rows, when=when)
    return render_template_string(PAGE, body=body)


@app.route("/report/<int:rid>")
def report(rid):
    with db() as c:
        r = c.execute("SELECT period, ts, since, until, facts, text FROM reports WHERE id = %s", (rid,)).fetchone()
    if not r:
        abort(404)
    body = render_template_string("""<h2>{{ r[0]|capitalize }} report &middot; {{ when(r[1]) }}</h2>
<p><small>{{ when(r[2]) }} &rarr; {{ when(r[3]) }} &middot; written by the local LLM from the facts below only</small></p>
<pre style="white-space:pre-wrap;background:#fff;padding:12px;border-radius:6px">{{ r[5] }}</pre>
<details><summary>Facts</summary><pre style="background:#fff;padding:12px">{{ facts }}</pre></details>""",
        r=r, when=when, facts=json.dumps(r[4], indent=1, default=str))
    return render_template_string(PAGE, body=body)


def vname(vid, label, plate):
    return label or (f"Vehicle #{vid}" + (f" ({plate})" if plate else ""))


def occupant_vehicles(c, iid: int) -> list:
    """Vehicles in which this identity's face was seen (occupant_faces), newest first."""
    try:
        with c.transaction():
            rows = c.execute("""SELECT v.id, v.label, v.plate, count(*), max(o.ts), max(o.sim),
                                       string_agg(DISTINCT o.match_kind, ', '),
                                       (array_agg(o.crop_file ORDER BY o.ts DESC) FILTER (WHERE o.crop_file IS NOT NULL))[1]
                                FROM occupant_faces o JOIN vehicle_sightings vs ON vs.id = o.vehicle_sighting_id
                                JOIN vehicles v ON v.id = vs.vehicle_id WHERE o.identity_id = %s
                                GROUP BY v.id ORDER BY max(o.ts) DESC LIMIT 20""", (iid,)).fetchall()
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
        return []
    kinds = {"face_template": "enrolled face", "face_unknown": "same unnamed face", "face_new": "first seen"}
    return [{"vid": r[0], "vname": vname(r[0], r[1], r[2]), "n": r[3], "last": r[4], "sim": r[5],
             "kinds": ", ".join(kinds.get(k.strip(), k) for k in (r[6] or "").split(",") if k.strip()), "crop": r[7]}
            for r in rows]


def driver_line(occ: dict | None, drv: tuple | None) -> dict | None:
    """Vehicle sighting -> what to show about its occupants. drv = (identity id, label, kind, driver_kind, sim, crop).
    Names appear only for enrolled household/known identities; tracked identities show as Unknown #N."""
    occ = occ or {}
    n = occ.get("occupants_detected")
    if drv and drv[0] is not None:
        iid, label, kind, dk, sim, crop = drv
        if dk == "face_template" and kind in ENROLL_KINDS:
            text = f"Driver: {name(iid, label)} (enrolled, face match {sim:.2f})"
        elif dk == "face_unknown":
            text = f"Driver: {name(iid, label)} (unnamed, same face as before {sim:.2f})"
        else:
            text = f"Driver: {name(iid, label)} (unnamed, face first seen here)"
        others = max(0, len([o for o in occ.get("occupants") or [] if o.get("identity")]) - 1)
        return {"text": text + (f" + {others} other occupant(s)" if others else ""), "iid": iid, "crop": crop}
    if n:
        why = sorted({r for o in occ.get("occupants") or [] for r in ((o.get("match") or {}).get("reasons") or [])
                      if r != "FACE_NO_MATCH"})
        return {"text": f"Occupant(s) detected ({n}), not recognised" + (f" ({', '.join(why)})" if why else ""),
                "iid": None, "crop": None}
    if occ:
        return {"text": "No occupant face visible", "iid": None, "crop": None}
    return None


@app.route("/vehicles")
def vehicles():
    with db() as c:
        vs = c.execute("""SELECT v.id, v.label, v.kind, v.plate, v.body_type, v.color, v.make_model, v.first_seen, v.last_seen,
                                 count(s.id), (SELECT crop_file FROM vehicle_sightings x WHERE x.vehicle_id = v.id ORDER BY ts_start DESC LIMIT 1),
                                 count(*) FILTER (WHERE s.match_kind = 'new_possible')
                          FROM vehicles v LEFT JOIN vehicle_sightings s ON s.vehicle_id = v.id
                          GROUP BY v.id ORDER BY v.last_seen DESC""").fetchall()
    body = render_template_string("""<h2>Vehicles ({{ vs|length }})</h2><table>
<tr><th></th><th>Vehicle</th><th>Plate</th><th>Look</th><th>Sightings</th><th>Seen</th></tr>
{% for v in vs %}<tr><td>{% if v[10] %}<img class="t" src="{{ url_for('vcrop', f=v[10]) }}">{% endif %}</td>
<td><a href="{{ url_for('vehicle', vid=v[0]) }}">{{ vname(v[0], v[1], None) }}</a> <span class="k {{ v[2] }}">{{ v[2] }}</span>
{% if v[11] %}<div class="flag">possible match flagged</div>{% endif %}</td>
<td><b>{{ v[3] or '—' }}</b></td><td>{{ v[5] or '' }} {{ v[4] or '' }}<br><small>{{ v[6] or '' }} (apparent)</small></td>
<td>{{ v[9] }}</td><td><small>first {{ when(v[7]) }}<br>last {{ when(v[8]) }}</small></td></tr>{% endfor %}</table>""",
        vs=vs, vname=vname, when=when)
    return render_template_string(PAGE, body=body)


@app.route("/vehicle/<int:vid>")
def vehicle(vid):
    with db() as c:
        v = c.execute("SELECT id, label, kind, notes, plate, body_type, color, make_model, first_seen, last_seen FROM vehicles WHERE id = %s", (vid,)).fetchone()
        if not v:
            abort(404)
        ss = c.execute("""SELECT ts_start, camera, crop_file, attributes, plate, plate_score, match_kind, match_sim,
                                 possible_id, possible_sim, frigate_event, id FROM vehicle_sightings
                          WHERE vehicle_id = %s ORDER BY ts_start DESC LIMIT 200""", (vid,)).fetchall()
        drivers = {}
        try:
            with c.transaction():
                for sid_, occ, did, dl, dkind, dk, dsim, crop in c.execute(
                        """SELECT vs.id, vs.occupant_face, vs.driver_identity_id, i.label, i.kind, vs.driver_kind, vs.driver_sim,
                                  (SELECT o.crop_file FROM occupant_faces o WHERE o.vehicle_sighting_id = vs.id
                                   AND o.identity_id = vs.driver_identity_id LIMIT 1)
                           FROM vehicle_sightings vs LEFT JOIN identities i ON i.id = vs.driver_identity_id
                           WHERE vs.vehicle_id = %s""", (vid,)).fetchall():
                    drivers[sid_] = driver_line(occ, (did, dl, dkind, dk, dsim or 0.0, crop) if did else None)
        except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
            pass
        others = c.execute("SELECT id, label, plate FROM vehicles WHERE id <> %s ORDER BY last_seen DESC", (vid,)).fetchall()
        reads = vehicle_readings(c, vid)
        vpulc = pulc_by(c, "SELECT frigate_event, pulc FROM vehicle_sightings WHERE vehicle_id = %s AND pulc IS NOT NULL",
                        (vid,))
        near = c.execute(f"""SELECT i.id, i.label, count(*) FROM vehicle_sightings vs
                            JOIN sightings s ON s.ts_start BETWEEN vs.ts_start - {NEAR} AND COALESCE(vs.ts_end, vs.ts_start) + {NEAR}
                            JOIN identities i ON i.id = s.identity_id WHERE vs.vehicle_id = %s
                            GROUP BY i.id ORDER BY count(*) DESC LIMIT 10""", (vid,)).fetchall()
    frigate = E.get("FRIGATE_URL", "https://frigate.cam.example.org")
    body = render_template_string("""<h2>{{ vname(v[0], v[1], v[4]) }} <span class="k {{ v[2] }}">{{ v[2] }}</span></h2>
<div class="stat-row">
 <div class="stat"><b>{{ v[4] or '&ndash;'|safe }}</b><span>plate{{ '' if v[4] else ' not read' }}</span></div>
 <div class="stat"><b>{{ ((v[6] or '') ~ ' ' ~ (v[5] or '')).strip() or '&ndash;'|safe }}</b><span>{{ v[7] or 'vehicle' }} (apparent)</span></div>
 <div class="stat"><b>{{ ss|length }}</b><span>sightings</span></div>
 <div class="stat"><b>{{ when(v[8]) }}</b><span>first seen</span></div>
 <div class="stat"><b>{{ when(v[9]) }}</b><span>last seen</span></div>
</div>
{% if near %}<p>Seen with people: {% for p in near %}<a href="{{ url_for('identity', iid=p[0]) }}">{{ name(p[0], p[1]) }}</a> ({{ p[2] }}x){% if not loop.last %}, {% endif %}{% endfor %}</p>{% endif %}
<div class="actions" aria-label="Edit vehicle"><form method="post" action="{{ url_for('vedit', vid=v[0]) }}">
 <input name="label" placeholder="Name" aria-label="Name" value="{{ v[1] or '' }}">
 <select name="kind" aria-label="Kind">{% for k in ['unknown','watch','household','known'] %}<option {{ 'selected' if k==v[2] }}>{{ k }}</option>{% endfor %}</select>
 <input name="notes" placeholder="Notes" aria-label="Notes" size="36" value="{{ v[3] or '' }}"> <button class="primary">Save</button></form>
<span class="sep" aria-hidden="true"></span><form method="post" action="{{ url_for('vmerge', vid=v[0]) }}" data-confirm-title="Merge vehicles?" data-confirm="All sightings of this vehicle move to {target}, and this vehicle is removed. It cannot be undone." data-confirm-ok="Merge" data-confirm-kind="warn" data-confirm-icon="merge">Merge into
 <select name="into" required aria-label="Merge into"><option value="" disabled selected>Choose vehicle…</option>{% for o in others %}<option value="{{ o[0] }}">{{ vname(o[0], o[1], o[2]) }}</option>{% endfor %}</select>
 <button class="warn">Merge</button></form>
<span class="sep" aria-hidden="true"></span><form method="post" action="{{ url_for('vdelete', vid=v[0]) }}" data-confirm-title="Delete this vehicle?" data-confirm="This deletes the vehicle and all its sightings. It cannot be undone." data-confirm-ok="Delete vehicle"><button class="danger">Delete</button></form></div>
<div class="grid">{% for s in ss %}{% set a = s[3] or {} %}<div class="card">
{% if s[2] %}<img src="{{ url_for('vcrop', f=s[2]) }}">{% endif %}
<b>{{ when(s[0]) }}</b> &middot; {{ s[1] }}<br>{{ a.get('color','') }} {{ a.get('body_type','') }} {{ a.get('apparent_make','') }} {{ a.get('apparent_model','') }}<br>
<small>plate {{ s[4] or '—' }}{% if s[5] %} ({{ '%.0f%%'|format(s[5]*100) }}){% endif %}<br>{{ ', '.join(a.get('distinguishing_features', [])[:4]) }}<br>
{% set pv = vpulc.get(s[10]) %}{% if pv %}PULC: {{ pv.color or '?' }} {{ pv.type or '?' }} <small>({% for k, v in pv.colors.items() %}{{ k }} {{ '%.0f'|format(v * 100) }}%{% if not loop.last %}, {% endif %}{% endfor %})</small><br>{% endif %}
{% set dr = drivers.get(s[11]) %}{% if dr %}{% if dr.crop %}<img src="{{ url_for('face_img', f=dr.crop) }}" style="height:40px;width:auto;border-radius:4px;vertical-align:middle"> {% endif %}{% if dr.iid %}<a href="{{ url_for('identity', iid=dr.iid) }}">{{ dr.text }}</a>{% else %}{{ dr.text }}{% endif %}<br>{% endif %}
{% set rd = reads.get(s[11]) %}{% if rd %}{{ rd|safe }}{% endif %}
{{ a.get('activity','') }} &middot; match {{ s[6] }} {{ '%.2f'|format(s[7] or 0) }}
{% if s[8] %}<br><span class="flag">possibly <a href="{{ url_for('vehicle', vid=s[8]) }}">#{{ s[8] }}</a> ({{ '%.2f'|format(s[9]) }})</span>{% endif %}
<br><a href="{{ frigate }}/explore?event_id={{ s[10] }}">Frigate</a></small></div>{% endfor %}</div>""",
        v=v, ss=ss, others=others, near=near, vname=vname, name=name, when=when, frigate=frigate, vpulc=vpulc,
        drivers=drivers, reads=reads)
    return render_template_string(PAGE, body=body)


import gaitlab  # noqa: E402  (copy of squirrel_guard/gaitlab.py)


def gait_profile(c, iid, exclude_sid=None) -> dict:
    rows = c.execute("SELECT gait FROM sightings WHERE identity_id = %s AND gait IS NOT NULL AND id <> %s ORDER BY ts_start DESC LIMIT 60",
                     (iid, exclude_sid or -1)).fetchall()
    return gaitlab.profile([r[0] for r in rows])


GAIT_PANEL = """<section aria-labelledby="gait-h"><h3 id="gait-h">Walk</h3>
{% if not g or not g.get('walking') %}<p class="muted">No measurable walk in this visit{% if g and g.reason %} ({{ g.reason }}){% endif %}.</p>
{% else %}<p>{{ g.view }} view, {{ g.seconds }} s, {{ g.strides }} strides, measurement quality {{ '%.0f'|format((g.quality or 0) * 100) }}%.
{% if not g.usable %}<span class="flag">too irregular to compare</span>{% endif %}</p>
{% if cmp and cmp['rows'] %}<p><b>{{ cmp.text }}</b> <small>({{ cmp.profile_walks }} earlier walk{{ 's' if cmp.profile_walks != 1 }} of this identity)</small></p>
<table><tr><th>Measurement</th><th>This walk</th><th>Usual</th><th>Comparison</th></tr>
{% for r in cmp['rows'] %}<tr><td>{{ r.label }}</td><td>{{ r.this_text }}</td><td>{{ r.usual_text }} <small>({{ r.n }})</small></td>
<td>{% if r.verdict == 'different' %}<span class="flag">{{ r.verdict }}</span>{% else %}{{ r.verdict }}{% endif %}</td></tr>{% endfor %}</table>
{% elif cmp %}<p class="muted">{{ cmp.text }}</p>{% endif %}
{{ chart|safe }}{% endif %}</section>"""


def vehicle_readings(c, vid) -> dict:
    """Per vehicle sighting: own plate reading (each character shaded by confidence), measured colour (or why there
    is none), direction and standing time, and why it was filed under this vehicle. {} before the columns exist."""
    from markupsafe import escape
    out = {}
    try:
        with c.transaction():
            rows = c.execute("SELECT id, plate_read, colour, motion, explain FROM vehicle_sightings WHERE vehicle_id = %s", (vid,)).fetchall()
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
        return out
    for sid_, pr, col, mv, ex in rows:
        bits = []
        if pr:
            txt = pr.get("text") or pr.get("candidate") or ""
            chars = "".join(f'<span title="{escape(ch)}: {cf:.0%}" style="opacity:{max(0.35, cf):.2f};{"text-decoration:underline dotted" if cf < 0.6 else ""}">{escape(ch)}</span>'
                            for ch, cf in zip(txt, pr.get("char_conf") or []))
            state = "" if pr.get("readable") else ' <span class="flag">unsure, not used</span>'
            bits.append(f'own plate reading <code>{chars}</code> ({pr.get("conf", 0):.0%}, {pr.get("reads", 0)} frame{"s" if pr.get("reads") != 1 else ""}){state}')
        if col:
            bits.append("colour: unknown (night/infrared)" if col.get("night") else
                        f'measured colour {escape(col.get("name") or "?")} ({(col.get("confidence") or 0):.0%}){" " + escape(col["note"]) if col.get("note") else ""}')
        if mv:
            bits.append(f'{escape(mv["direction"])}' + (f', stood {mv["stopped_s"]:.0f} s' if mv.get("stopped_s") else ""))
        if ex:
            bits.append(f'filed: <span class="band" style="background:{BAND_COLOURS.get(ex.get("band"), "#666")}">{escape(ex.get("band"))} {ex.get("pct")}%</span> '
                        + "; ".join(escape(r["text"]) for r in ex.get("reasons", [])))
        if bits:
            out[sid_] = "<small>" + "<br>".join(bits) + "</small><br>"
    return out


@app.post("/vehicle/<int:vid>/edit")
def vedit(vid):
    kind = request.form.get("kind", "unknown")
    if kind not in ("unknown", "watch", "household", "known"):
        abort(400)
    with db() as c:
        c.execute("UPDATE vehicles SET label = NULLIF(%s, ''), kind = %s, notes = NULLIF(%s, '') WHERE id = %s",
                  (request.form.get("label", "").strip()[:80], kind, request.form.get("notes", "").strip()[:500], vid))
    return redirect(url_for("vehicle", vid=vid, done="saved"))


@app.post("/vehicle/<int:vid>/merge")
def vmerge(vid):
    into = int(request.form["into"])
    if into == vid:
        abort(400)
    with db() as c, c.transaction():
        c.execute("UPDATE vehicle_sightings SET vehicle_id = %s, match_kind = 'merged' WHERE vehicle_id = %s", (into, vid))
        embs = [np.asarray(e, np.float32) for (e,) in c.execute("SELECT embedding FROM vehicle_sightings WHERE vehicle_id = %s", (into,)).fetchall()]
        cen = np.mean(embs, 0); cen /= np.linalg.norm(cen) + 1e-12
        src_plate = c.execute("SELECT plate FROM vehicles WHERE id = %s", (vid,)).fetchone()[0]
        c.execute("DELETE FROM vehicles WHERE id = %s", (vid,))
        c.execute("""UPDATE vehicles SET centroid = %s, n = %s, plate = COALESCE(plate, %s) WHERE id = %s""",
                  ([float(x) for x in cen], len(embs), src_plate, into))
    return redirect(url_for("vehicle", vid=into, done="merged"))


@app.post("/vehicle/<int:vid>/delete")
def vdelete(vid):
    with db() as c, c.transaction():
        files = [f for (f,) in c.execute("SELECT crop_file FROM vehicle_sightings WHERE vehicle_id = %s", (vid,)).fetchall() if f]
        c.execute("DELETE FROM vehicle_sightings WHERE vehicle_id = %s", (vid,))
        c.execute("DELETE FROM vehicles WHERE id = %s", (vid,))
    for f in files:
        (VCROPS / f).unlink(missing_ok=True)
    return redirect(url_for("vehicles", done="deleted"))


@app.route("/vcrop/<path:f>")
def vcrop(f):
    if "/" in f or not f.endswith(".jpg"):
        abort(404)
    return send_from_directory(VCROPS, f)


@app.route("/sighting/<int:sid>")
def sighting(sid):
    """Why? page: the readable explanation, visual evidence and the decision trail for one sighting."""
    with db() as c:
        s = c.execute("""SELECT s.id, s.frigate_event, s.ts_start, s.ts_end, s.camera, s.identity_id, i.label, i.kind,
                                s.explain, s.evidence_card, s.evidence_gif, s.match_strip, s.crop_file, s.trajectory_file, s.routine
                         FROM sightings s JOIN identities i ON i.id = s.identity_id WHERE s.id = %s""", (sid,)).fetchone()
        if not s:
            abort(404)
        face_html = sighting_face_html(c, sid, s[5], s[6], s[7]) + sighting_geometry_html(c, sid)
        rv_html = review_html(c, sid)
        gw = (_col(c, "SELECT gait FROM sightings WHERE id = %s", (sid,)) or [(None,)])[0][0]
        gprof = gait_profile(c, s[5], sid) if gw else None
        gait_html = render_template_string(GAIT_PANEL, g=gw, cmp=gaitlab.compare(gw, gprof) if gw and gw.get("usable") and gprof and gprof["walks"] else None,
                                           chart=gaitlab.svg_cycle(gw, gprof) if gw else "") if gw else ""
        trail = c.execute("""SELECT ts, stage, band, pct, reasons, message FROM decisions WHERE frigate_event = %s
                             ORDER BY ts""", (s[1],)).fetchall()
        pv = pulc_by(c, "SELECT id, pulc FROM sightings WHERE id = %s", (sid,)).get(sid)
        fz = _col(c, "SELECT fused_trajectory_file FROM sightings WHERE id = %s", (sid,))
    fused = fz[0][0] if fz and fz[0][0] else None
    frigate = E.get("FRIGATE_URL", "https://frigate.cam.example.org")
    body = render_template_string("""{% set x = s[8] or {} %}
<h2>{{ name(s[5], s[6]) }} <span class="k {{ s[7] }}">{{ s[7] }}</span> <small>{{ when(s[2]) }} &middot; {{ s[4] }}</small></h2>
{{ rv_html|safe }}
{% if x %}<p><b>{{ x.headline }}</b> <span class="band" style="background:{{ colours.get(x.band, '#666') }}">{{ x.band }} {{ x.pct }}%</span>
{% if x.estimated %}<small>estimated</small>{% endif %}</p>
<ul>{% for e in x.evidence %}<li>{{ e }}</li>{% endfor %}</ul>
{% if x.did %}<p>Did: {{ x.did|join('; ') }}</p>{% endif %}
{% if x.body %}<p>Looked: {{ x.body|join('; ') }} <small>(estimates)</small></p>{% endif %}
<p class="codes">{% for r in x.reasons %}<code>{{ r }}</code>{% endfor %}</p>{% endif %}
<p><a href="{{ url_for('identity', iid=s[5]) }}">Identity page</a> &middot; <a href="{{ frigate }}/explore?event_id={{ s[1] }}">Open in Frigate</a>
{% if can_export %} &middot; <a href="{{ url_for('export_new', sids=s[0]) }}"><b>Export evidence package</b></a>{% endif %}
{% if fused %} &middot; <a href="{{ url_for('replay') }}?data={{ url_for('evidence_file', f=fused) }}&scene=projection&cutouts=trail"><b>3D replay, all cameras</b></a>{% endif %}
{% if s[13] %} &middot; <a href="{{ url_for('replay') }}?data={{ url_for('evidence_file', f=s[13]) }}&scene=projection&cutouts=trail"><b>3D replay{% if fused %} ({{ s[4] }} only){% endif %}</b></a>{% else %} &middot; <small>3D replay after camera calibration (<a href="{{ url_for('replay') }}?data=demo/demo.json&scene=projection&cutouts=trail">demo</a>)</small>{% endif %}</p>
{% set rt = s[14] %}{% if rt %}<h3>Routine {% if rt.unusual %}<span class="band" style="background:#c0392b">unusual</span>{% elif rt.ready %}<span class="band" style="background:#2e7d32">ordinary</span>{% endif %}</h3>
{% if rt.note %}<p><small>{{ rt.note }}</small></p>{% endif %}
<p><small>{{ rt.bits }} bits of surprise against {{ rt.history }} earlier visits (alert threshold {{ rt.threshold }}; every 1 bit = twice as rare)</small></p>
<table><tr><th>Part</th><th>Bits</th><th>What was compared</th></tr>{% for p in rt.parts|sort(attribute='bits', reverse=True) %}<tr><td>{{ p.code }}</td><td>{{ p.bits }}</td><td><small>{{ p.text }}</small></td></tr>{% endfor %}</table>{% endif %}
{{ face_html|safe }}
{{ gait_html|safe }}
{% if pv %}<h3>Apparent attributes <small>(PULC, PA100K-trained; estimates, weak evidence only)</small></h3>
{% if pv.crops %}<p>{{ pulc_line(pv) }} <small>({{ pv.crops }} full-body crop{{ 's' if pv.crops != 1 }} of {{ pv.considered or pv.crops }};
crop disagreement {{ '%.2f'|format(pv.spread or 0) }})</small></p>
<p><small>View: {% for k, v in (pv.orientation or {}).items() %}{{ k }} {{ '%.0f'|format(v * 100) }}%{% if not loop.last %}, {% endif %}{% endfor %}
{% if pv.apparent %} &middot; apparent gender: female {{ '%.0f'|format((pv.apparent.gender_female or 0) * 100) }}%
{% if pv.apparent.age %} &middot; apparent age: {% for k, v in pv.apparent.age.items() %}{{ k.replace('_', '-') }} {{ '%.0f'|format(v * 100) }}%{% if not loop.last %}, {% endif %}{% endfor %}{% endif %}{% endif %}</small></p>
<table><tr><th>Attribute</th><th>Mean p</th></tr>{% for k, v in pv.attributes|dictsort(by='value', reverse=True) %}{% if v >= 0.2 %}<tr><td>{{ k }}</td><td>{{ '%.2f'|format(v) }}</td></tr>{% endif %}{% endfor %}</table>
{% else %}<p><small>{{ pulc_line(pv) }} ({{ pv.considered or 0 }} crop{{ 's' if pv.considered != 1 }} looked at)</small></p>{% endif %}{% endif %}
<div class="ev">{% if s[9] %}<img src="{{ url_for('evidence_file', f=s[9]) }}" alt="evidence card">{% endif %}
{% if s[11] %}<p><small>This visit next to the most similar earlier sightings it was compared with:</small></p><img src="{{ url_for('evidence_file', f=s[11]) }}" alt="match strip" style="max-width:760px">{% endif %}
{% if s[10] %}<img src="{{ url_for('evidence_file', f=s[10]) }}" alt="visit flip-book" style="max-width:420px">{% endif %}
{% if not (s[9] or s[11] or s[10]) and s[12] %}<img src="{{ url_for('crop', f=s[12]) }}" style="max-width:240px">{% endif %}</div>
<h3>Decision trail</h3><table><tr><th>When</th><th>Stage</th><th>Band</th><th>Reasons</th><th>Message</th></tr>
{% for t, st, b, p, rs, m in trail %}<tr><td><small>{{ when(t) }}</small></td><td>{{ st }}</td>
<td>{% if b %}<span class="band" style="background:{{ colours.get(b, '#666') }}">{{ b }}{% if p is not none %} {{ p }}%{% endif %}</span>{% endif %}</td>
<td class="codes">{% for r in rs %}<code>{{ r }}</code>{% endfor %}</td><td><small>{{ m or '' }}</small></td></tr>{% endfor %}</table>""",
        s=s, trail=trail, name=name, when=when, frigate=frigate, colours=BAND_COLOURS, face_html=face_html, pv=pv,
        pulc_line=pulc_line, fused=fused, rv_html=rv_html, can_export=access.at_least("investigator"), gait_html=gait_html)
    return render_template_string(PAGE, body=body)


# -- face templates: enrolment of household/known people only (docs/faceid.md) ----------------------------------------
def face_templates(c, iid: int) -> list:
    try:
        with c.transaction():
            return c.execute("""SELECT id, created, n_faces, quality, liveness, source_event, model, note,
                                       crop_jpeg IS NOT NULL
                                FROM face_templates WHERE identity_id = %s ORDER BY created DESC""", (iid,)).fetchall()
    except psycopg.errors.UndefinedTable:
        return []


def drop_face_templates(c, iid: int) -> None:
    """Delete an identity's enrolled faces (it is no longer household/known, or consent was withdrawn)."""
    try:
        with c.transaction():
            n = c.execute("DELETE FROM face_templates WHERE identity_id = %s", (iid,)).rowcount
            if n:
                c.execute("INSERT INTO decisions (stage, subject, reasons, message) VALUES ('action', %s, %s, %s)",
                          (f"person:{iid}", ["USER_FACE_REMOVED"], f"{n} enrolled face template(s) removed"))
    except psycopg.errors.UndefinedTable:
        pass


def face_models_html(c, iid: int) -> str:
    """Identity page: the prior 3D face models (one fitted face per visit), aligned and compared with the identity's
    mean face, plus the 3D viewer. Explanatory evidence only, within this identity."""
    try:
        ix = _model_index(c, iid)
    except Exception:
        return ""
    ms = ix["models"]
    if not ms:
        return ""
    rms = [m["vs_mean"]["rms_pct"] for m in ms if m.get("vs_mean")]
    return render_template_string("""<h3 id="face-models">Face models (3D) <small>{{ ms|length }} fitted face{{ 's' if ms|length != 1 }}</small></h3>
<p><small>One face shape is fitted per visit (many frames, lens-corrected). Models are aligned (rotation, scale, position) and
compared point by point in units of the eye spacing{% if med is not none %}; this identity's models agree within
{{ med }} % of the eye spacing (median){% endif %}. Explanatory evidence only: identity is decided by the face embedding,
never by shape; distance, angle, lighting and expression also move these numbers.</small></p>
<iframe src="{{ url_for('replay') }}face3d.html?identity={{ iid }}" title="3D face models of this identity"
 style="width:100%;max-width:760px;height:420px;border:1px solid #8884;border-radius:8px"></iframe>
<table><tr><th>Visit</th><th>Camera</th><th>Frames</th><th>Fit error</th><th>vs mean (RMS / p90)</th><th>Scale</th><th></th></tr>
{% for m in ms %}<tr><td>{{ when(m.ts) }}</td><td>{{ m.camera }}</td><td>{{ m.fit.n_frames }}</td>
<td>{{ m.fit.reproj_px_med }} px</td><td>{% if m.vs_mean %}{{ m.vs_mean.rms_pct }} % / {{ m.vs_mean.p90_pct }} %{% else %}-{% endif %}</td>
<td>{{ 'mm (2 cameras)' if (m.fit.multiview or {}).scale_used else 'relative' }}</td>
<td><a href="{{ url_for('sighting', sid=m.sighting) }}">Why?</a> &middot;
<a href="{{ url_for('replay') }}face3d.html?identity={{ iid }}&event={{ m.event|urlencode }}">3D</a></td></tr>{% endfor %}</table>""",
        ms=ms, iid=iid, med=round(float(np.median(rms)), 2) if rms else None, when=when)


def templates_html(idt, templates) -> str:
    if idt[2] not in ENROLL_KINDS and not templates:
        return ""
    return render_template_string("""<h3>Enrolled faces</h3>
{% if not ts %}<p><small>None. Open one of this person's visits (Why?) and use <i>Use this visit's face</i>; only
household/known people can be enrolled, and only with their consent.</small></p>{% endif %}
<div class="grid">{% for t in ts %}<div class="card">{% if t[8] %}<img src="{{ url_for('face_template_img', tid=t[0]) }}" style="width:120px">{% endif %}<br>
<small>enrolled {{ when(t[1]) }} from <code>{{ t[5] }}</code><br>{{ t[2] }} faces, quality {{ '%.2f'|format(t[3] or 0) }},
liveness {{ '%.2f'|format(t[4] or 0) }}<br>model {{ t[6] }}{% if t[7] %}<br>{{ t[7] }}{% endif %}</small>
<form method="post" action="{{ url_for('face_template_delete', tid=t[0]) }}" data-confirm-title="Remove this enrolled face?" data-confirm="The person will no longer be recognised from this face sample." data-confirm-ok="Remove face"><button class="danger">Remove</button></form>
</div>{% endfor %}</div>""", ts=templates, when=when)


def enroll_problems(meta: dict) -> list:
    """Same gates as the bridge's matching (facematch.py): liveness, face count, quality."""
    out = []
    if not (meta.get("liveness") or {}).get("ok"):
        out.append("liveness did not pass")
    if int(meta.get("n_faces") or 0) < FACE_MIN_FACES:
        out.append(f"fewer than {FACE_MIN_FACES} usable face frames")
    if float(meta.get("quality") or 0) < FACE_MIN_QUALITY:
        out.append("face quality too low")
    return out


def sighting_face_html(c, sid, iid, label, kind) -> str:
    try:
        with c.transaction():
            r = c.execute("SELECT face_vec IS NOT NULL, face_meta FROM sightings WHERE id = %s", (sid,)).fetchone()
    except psycopg.errors.UndefinedColumn:
        return ""
    if not r or not r[1]:
        return ""
    has_vec, m = r
    targets = c.execute("SELECT id, label, kind FROM identities WHERE kind IN ('household', 'known') "
                        "ORDER BY (id = %s) DESC, label", (iid,)).fetchall()
    return render_template_string("""<h3>Face</h3>{% set lv = m.liveness or {} %}{% set d = m.match or {} %}{% set ap = m.apparent or {} %}
<p>{% for f in m.crop_files or [] %}<img src="{{ url_for('face_img', f=f) }}" style="height:120px;border-radius:4px;margin-right:6px">{% endfor %}</p>
{% if has_vec %}<p><small>{{ m.n_faces }} usable face frames ({{ m.n_rejected or 0 }} rejected as disagreeing, {{ m.faces_seen }} seen) &middot;
quality {{ m.quality }} &middot; liveness {{ lv.score }} ({{ 'passed' if lv.ok else 'NOT passed' }}) &middot; face ~{{ m.size_px }} px &middot;
consistency {{ m.consistency }}</small></p>{% else %}<p><small>No usable face ({{ m.reason }}; largest {{ m.max_size_px }} px).</small></p>{% endif %}
{% if d.status %}<p>Face decision: <b>{{ d.status }}</b>{% if d.sim is not none %} (best enrolled {{ '%.2f'|format(d.sim) }}{% if d.second_sim is not none %}, next {{ '%.2f'|format(d.second_sim) }}{% endif %}){% endif %}
<span class="codes">{% for r in d.reasons or [] %}<code>{{ r }}</code>{% endfor %}</span></p>{% endif %}
{% set tr = m.track or {} %}{% if tr.status %}<p>Unnamed-face tracking: <b>{{ tr.status }}</b>{% if tr.sim is not none %} (best unnamed visitor {{ '%.2f'|format(tr.sim) }}{% if tr.second_sim is not none %}, next {{ '%.2f'|format(tr.second_sim) }}{% endif %}){% endif %}
<span class="codes">{% for r in tr.reasons or [] %}<code>{{ r }}</code>{% endfor %}</span><br><small>Automatic grouping of faces that are NOT enrolled: it never names anyone and never makes anyone trusted.</small></p>{% endif %}
{% if ap.expression %}<p><small>Apparent expression: {{ ap.expression }} &mdash; a description only, never used for alerts or identity.</small></p>{% endif %}
{% if ap.age %}<p><small>Apparent age {{ ap.age.low }}&ndash;{{ ap.age.high }}{% if ap.gender %}, apparently {{ ap.gender.value }} ({{ '%.0f'|format(ap.gender.conf * 100) }}%){% endif %} (estimates)</small></p>{% endif %}
{% if has_vec %}{% if targets and not problems %}<form method="post" action="{{ url_for('face_enroll', sid=sid) }}">
Use this visit's face for <select name="identity">{% for t in targets %}<option value="{{ t[0] }}">{{ name(t[0], t[1]) }} ({{ t[2] }})</option>{% endfor %}</select>
<label><input type="checkbox" name="consent" value="yes" required> this person agreed to be recognised by face</label>
<button class="primary">Enroll</button></form>{% elif problems %}<p><small>Not usable for enrolment: {{ problems|join(', ') }}.</small></p>
{% else %}<p><small>Enrolment needs a household or known identity (mark this person household/known first).</small></p>{% endif %}{% endif %}""",
        m=m, has_vec=has_vec, targets=targets, problems=enroll_problems(m), sid=sid, name=name)


def sighting_geometry_html(c, sid) -> str:
    """Face geometry (facegeom: per-frame proportions + relative depth, aggregated) and the fitted face (facefit: one
    shape over the visit, with a 3D view by detail level). Explanatory evidence only, never identity."""
    try:
        with c.transaction():
            r = c.execute("SELECT frigate_event, face_geom, face_fit, identity_id FROM sightings WHERE id = %s",
                          (sid,)).fetchone()
    except psycopg.errors.UndefinedColumn:
        return ""
    if not r or not (r[1] or r[2]):
        return ""
    ev, g, ff, iid = r
    out = []
    if g:
        import facegeom
        files = g.get("files") or {}
        urls = {k: url_for("evidence_file", f=v) for k, v in files.items() if k in ("overlay", "depth")}
        gm = dict(g)
        if files.get("mesh") and (EVID / files["mesh"]).is_file():
            try:
                gm["mesh"] = json.loads((EVID / files["mesh"]).read_text())
            except ValueError:
                pass
        out.append("<h3>Face geometry <small>(proportions and relative depth; explanation only)</small></h3>"
                   + facegeom.render_html(gm, g.get("compare"), urls, show_all=request.args.get("all") == "1",
                                          uid=f"fg{sid}"))
    if ff and ff.get("n_frames"):
        feats = ff.get("features") or {}
        metric = ff.get("metric") or {}
        mv = ff.get("multiview") or {}
        out.append(render_template_string("""<h3>Fitted face <small>(one shape over the whole visit; explanation only)</small></h3>
<p><small>{{ f.n_frames }} frames{% if f.yaw_range %}, yaw {{ f.yaw_range[0] }}&deg; to {{ f.yaw_range[1] }}&deg;{% endif %}
{% if f.reproj_px_med is not none %} &middot; fit error {{ f.reproj_px_med }} px median ({{ f.reproj_ipd_pct_med }} % of the eye spacing){% endif %}
{% if f.vertex_sd_ipd %} &middot; shape uncertainty {{ f.vertex_sd_ipd.median }} (median, eye-spacing units){% endif %}
&middot; {{ 'metric scale from two cameras' if mv.scale_used else 'relative scale (one camera)' }}</small></p>
{% if f.file %}<p><a href="{{ url_for('replay') }}face3d.html?identity={{ iid }}&event={{ ev|urlencode }}"><b>3D face</b></a>
(<a href="{{ url_for('identity', iid=iid) }}#face-models">all of this person's face models</a>)
<small>(detail levels 250 to 14,000 triangles; drag to rotate)</small></p>{% endif %}
<table><tr><th>Feature</th><th>Value</th><th>Uncertainty (sd)</th>{% if metric %}<th>mm</th>{% endif %}</tr>
{% for k, v in feats.items() %}{% if v.value is not none %}<tr><td>{{ k.replace('_', ' ') }}</td><td>{{ '%.3f'|format(v.value) }}</td>
<td>{{ '%.3f'|format(v.sd) if v.sd is not none else '' }}</td>{% if metric %}<td>{{ (metric.get(k) or {}).get('mm', '') }}</td>{% endif %}</tr>{% endif %}{% endfor %}</table>
{% set ex = f.apparent_expression or {} %}{% if ex %}<p><small>Apparent expression (description only): share of frames where
{% for k, v in ex.items() %}{% if k != 'note' and v is not none %}{{ k }} {{ '%.0f'|format(v * 100) }}%{% if not loop.last %}, {% endif %}{% endif %}{% endfor %}
moved from rest; those frames were down-weighted.</small></p>{% endif %}
<p><small>{{ f.note }} Values are divided by the eye spacing; depth is the model's relative estimate, not millimetres{% if metric %}
(millimetres: {{ metric.note }}){% endif %}.</small></p>""", f=ff, feats=feats, metric=metric, mv=mv, iid=iid, ev=ev))
    return "".join(out)


@app.post("/sighting/<int:sid>/enroll")
def face_enroll(sid):
    """Use this visit's face template for a household/known identity (consent box required)."""
    if request.form.get("consent") != "yes":
        abort(400, "consent required")
    target = int(request.form["identity"])
    with db() as c, c.transaction():
        s = c.execute("SELECT frigate_event, face_vec, face_meta FROM sightings WHERE id = %s", (sid,)).fetchone()
        i = c.execute("SELECT id, label, kind FROM identities WHERE id = %s", (target,)).fetchone()
        if not s or s[1] is None or not i:
            abort(404)
        if i[2] not in ENROLL_KINDS:
            abort(400, "only household/known identities can be enrolled")
        m = s[2] or {}
        if enroll_problems(m):
            abort(400, "; ".join(enroll_problems(m)))
        crop = None
        files = m.get("crop_files") or []
        if files and (FACES / files[0]).is_file():
            crop = (FACES / files[0]).read_bytes()   # kept IN the row: deleted with the template
        tid = c.execute("""INSERT INTO face_templates (identity_id, vec, model, aggregator, n_faces, quality, liveness,
                                                       source_event, crop_jpeg, note)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                        (target, s[1], m.get("model"), m.get("aggregator"), m.get("n_faces"), m.get("quality"),
                         (m.get("liveness") or {}).get("score"), s[0], crop,
                         "consent confirmed by " + request.headers.get("Remote-User", "persons-ui"))).fetchone()[0]
        v2 = _col(c, "SELECT face_vec_v2 FROM sightings WHERE id = %s", (sid,))
        if v2 and v2[0][0] is not None and (m.get("v2") or {}).get("model"):   # AdaFace template of the same visit
            c.execute("""INSERT INTO face_templates (identity_id, vec, model, aggregator, n_faces, quality, liveness,
                                                     source_event, note) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                      (target, v2[0][0], m["v2"]["model"], "faceid_v2", m["v2"].get("n_faces"), m.get("quality"),
                       (m.get("liveness") or {}).get("score"), s[0], "AdaFace template of the same consented visit"))
        c.execute("INSERT INTO decisions (stage, frigate_event, subject, reasons, message) VALUES ('action', %s, %s, %s, %s)",
                  (s[0], f"person:{target}", ["USER_FACE_ENROLL"], f"face template #{tid} enrolled for {name(i[0], i[1])}"))
    return redirect(url_for("identity", iid=target, done="enrolled"))


@app.post("/face_template/<int:tid>/delete")
def face_template_delete(tid):
    with db() as c:
        r = c.execute("DELETE FROM face_templates WHERE id = %s RETURNING identity_id", (tid,)).fetchone()
        if not r:
            abort(404)
        c.execute("INSERT INTO decisions (stage, subject, reasons, message) VALUES ('action', %s, %s, %s)",
                  (f"person:{r[0]}", ["USER_FACE_REMOVED"], f"face template #{tid} deleted"))
    return redirect(url_for("identity", iid=r[0], done="removed"))


@app.route("/face_template/<int:tid>.jpg")
def face_template_img(tid):
    with db() as c:
        r = c.execute("SELECT crop_jpeg FROM face_templates WHERE id = %s", (tid,)).fetchone()
    if not r or r[0] is None:
        abort(404)
    return Response(bytes(r[0]), mimetype="image/jpeg", headers={"Cache-Control": "private, no-store"})


@app.route("/face/<path:f>")
def face_img(f):
    if "/" in f or not f.endswith(".jpg") or "_face" not in f:
        abort(404)
    return send_from_directory(FACES, f)

# -- natural-language search (Frigate semantic search: Jina CLIP thumbnails + local-Qwen descriptions) ----------------
SEARCH_TIME = [  # (phrase regex, days back from now, days back to); stripped from the query text
    (r"\btoday\b", 0, None), (r"\byesterday\b", 1, 0), (r"\blast night\b", 1, 0), (r"\bthis week\b", 7, None),
    (r"\blast week\b", 7, None), (r"\b(?:in the )?(?:last|past) (\d+) days?\b", "n", None),
    (r"\b(?:last|this) month\b", 31, None)]


def _time_filter(q: str):
    """'... last week' -> (query without the phrase, after_ts, before_ts, phrase) in local time."""
    import re
    now = dt.datetime.now(dt.timezone.utc).astimezone()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for pat, back, to in SEARCH_TIME:
        m = re.search(pat, q, re.I)
        if not m:
            continue
        days = int(m.group(1)) if back == "n" else back
        after = midnight - dt.timedelta(days=days)
        before = midnight - dt.timedelta(days=to) if to is not None else None
        if "night" in pat:
            after, before = midnight - dt.timedelta(hours=6), midnight + dt.timedelta(hours=6)
        rest = (q[:m.start()] + q[m.end():]).strip(" ,.")
        return rest, after.timestamp(), before.timestamp() if before else None, m.group(0)
    return q, None, None, None


def _frigate(path: str, params: dict):
    """Frigate API call as the signed-in user, over the private link (gpuserver :8971 only accepts the recorder host)."""
    import ssl
    import urllib.parse
    import urllib.request
    base = E.get("FRIGATE_API", "https://192.0.2.30:8971")
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(f"{base}{path}?{urllib.parse.urlencode(params)}", headers={
        "X-Proxy-Secret": E.get("FRIGATE_PROXY_SECRET", ""), "Remote-User": request.headers.get("Remote-User", "persons-ui"),
        "Remote-Groups": "admins"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        return json.loads(r.read())


@app.route("/search")
def search():
    q = (request.args.get("q") or "").strip()
    attrs = [a for a in request.args.getlist("attr") if a in PULC_FILTERS]
    body = "<h2>Search</h2><p><small>Finds visits by what they look like and by the written descriptions the local Qwen " \
           "makes of each person/vehicle event. Time phrases (today, yesterday, last night, last week, last 3 days, " \
           "last month) narrow the dates. Results are similarity, not certainty.</small></p>"
    body += render_template_string("""<form action="{{ url_for('search') }}"><input type="hidden" name="q" value="{{ q }}">
<small>Apparent attributes (PULC, full-body views only):</small> {% for f in filters %}<label><input type="checkbox" name="attr"
value="{{ f }}" {{ 'checked' if f in attrs }}> {{ f.replace('_', ' ') }}</label> {% endfor %}<button class="primary">Filter</button></form>""",
                                   q=q, attrs=attrs, filters=list(PULC_FILTERS))
    if not q and attrs:
        return render_template_string(PAGE, body=body + pulc_search(attrs))
    if not q:
        return render_template_string(PAGE, body=body)
    text, after, before, phrase = _time_filter(q)
    params = {"query": text or q, "search_type": "thumbnail,description", "limit": 40, "timezone": "America/Denver"}
    if after:
        params["after"] = after
    if before:
        params["before"] = before
    try:
        hits = _frigate("/api/events/search", params)
    except Exception as e:                                   # Frigate down / search disabled: say so plainly
        return render_template_string(PAGE, body=body + (
            "<div class='notice warn-box'><b>Text search is not available right now.</b> It uses the NVR's semantic search, "
            f"which did not answer ({type(e).__name__}). The attribute filters above still work.</div>"))
    ids = [h["id"] for h in hits] if isinstance(hits, list) else []
    people, cars, hit_pulc = {}, {}, {}
    if ids:
        with db() as c:
            for r in c.execute("""SELECT s.frigate_event, s.id, s.identity_id, i.label, i.kind, s.objects, s.routine
                                  FROM sightings s JOIN identities i ON i.id = s.identity_id
                                  WHERE s.frigate_event = ANY(%s)""", (ids,)).fetchall():
                people[r[0]] = r[1:]
            hit_pulc = pulc_by(c, "SELECT frigate_event, pulc FROM sightings WHERE frigate_event = ANY(%s)", (ids,)) \
                if attrs else {}
            for r in c.execute("""SELECT vs.frigate_event, vs.vehicle_id, v.label, v.kind, vs.plate FROM vehicle_sightings vs
                                  JOIN vehicles v ON v.id = vs.vehicle_id WHERE vs.frigate_event = ANY(%s)""", (ids,)).fetchall():
                cars[r[0]] = r[1:]
    frigate = E.get("FRIGATE_URL", "https://frigate.cam.example.org")
    rows = []
    for h in hits if isinstance(hits, list) else []:
        if attrs and not pulc_matches(hit_pulc.get(h["id"]), attrs):   # attribute filters: people only
            continue
        d = h.get("data") or {}
        dist = h.get("search_distance")
        rows.append({"id": h["id"], "label": h.get("label"), "t": dt.datetime.fromtimestamp(h["start_time"], dt.timezone.utc),
                     "desc": d.get("description") or "", "source": h.get("search_source"),
                     "match": None if dist is None else max(0, min(100, round((1.2 - dist) / 0.6 * 100))),
                     "p": people.get(h["id"]), "v": cars.get(h["id"])})
    body += render_template_string("""<p><b>{{ rows|length }}</b> result{{ '' if rows|length == 1 else 's' }} for
<q>{{ text or q }}</q>{% for a in attrs %} <span class="k">{{ a.replace('_', ' ') }}</span>{% endfor %}{% if phrase %} <span class="k">{{ phrase }}</span>{% endif %}
<small>(best first; "match" is relative within this search)</small></p>
<div class="grid">{% for r in rows %}<div class="card">
<a href="{{ frigate }}/explore?event_id={{ r.id }}"><img src="{{ frigate }}/api/events/{{ r.id }}/thumbnail.webp" alt="{{ r.label }}" loading="lazy"></a>
<b>{{ r.label }}</b> <small>{{ when(r.t) }}</small>{% if r.match is not none %} <span class="k">match {{ r.match }}</span>{% endif %}
{% if r.p %}<br><a href="{{ url_for('identity', iid=r.p[1]) }}">{{ name(r.p[1], r.p[2]) }}</a> <span class="k {{ r.p[3] }}">{{ r.p[3] }}</span>
 &middot; <a href="{{ url_for('sighting', sid=r.p[0]) }}">Why?</a>
{% if r.p[5] and r.p[5].unusual %}<br><span class="flag">unusual visit</span>{% endif %}
{% if r.p[4] %}<br><small>objects: {% for o in r.p[4] %}{{ o.label }}{% if o.carried %} (carried){% endif %}{% if not loop.last %}, {% endif %}{% endfor %}</small>{% endif %}{% endif %}
{% if r.v %}<br><a href="{{ url_for('vehicle', vid=r.v[0]) }}">{{ r.v[1] or ('Vehicle #%d'|format(r.v[0])) }}</a> <span class="k {{ r.v[2] }}">{{ r.v[2] }}</span>{% endif %}
{% if r.desc %}<p><small>{{ r.desc }}</small></p>{% endif %}
</div>{% endfor %}</div>""", rows=rows, q=q, text=text, phrase=phrase, frigate=frigate, when=when, name=name,
                                   attrs=attrs)
    return render_template_string(PAGE, body=body)


def pulc_search(attrs: list) -> str:
    """Attribute-only search: the latest 60 person sightings whose PULC visit summary has all chosen attributes."""
    cond, _ = pulc_sql_filter(attrs)
    try:
        with db() as c:
            rows = c.execute(f"""SELECT s.id, s.identity_id, i.label, i.kind, s.ts_start, s.camera, s.crop_file, s.pulc
                                 FROM sightings s JOIN identities i ON i.id = s.identity_id
                                 WHERE s.pulc IS NOT NULL AND (s.pulc->>'crops')::int > 0 AND {cond}
                                 ORDER BY s.ts_start DESC LIMIT 60""").fetchall()
    except Exception as e:
        return f"<p class='flag'>Attribute search unavailable ({type(e).__name__}; apply persons_schema.sql)</p>"
    return render_template_string("""<p><b>{{ rows|length }}</b> sighting{{ '' if rows|length == 1 else 's' }} with
{% for a in attrs %}<span class="k">{{ a.replace('_', ' ') }}</span> {% endfor %}<small>(apparent, newest first, max 60)</small></p>
<div class="grid">{% for r in rows %}<div class="card">{% if r[6] %}<img src="{{ url_for('crop', f=r[6]) }}">{% endif %}
<a href="{{ url_for('identity', iid=r[1]) }}">{{ name(r[1], r[2]) }}</a> <span class="k {{ r[3] }}">{{ r[3] }}</span>
<small>{{ when(r[4]) }} &middot; {{ r[5] }}</small><br><small>{{ pulc_line(r[7]) }}</small>
<br><a href="{{ url_for('sighting', sid=r[0]) }}">Why?</a></div>{% endfor %}</div>""",
                                  rows=rows, attrs=attrs, name=name, when=when, pulc_line=pulc_line)


# -- JSON API for the replay viewer's investigation pane (docs/replay-pane.md) -----------------------------------------
# Same sign-in as the pages (the before_request hook above covers /api/*). Read-only, local data only: no outside
# lookups; Frigate links are URLs for the signed-in browser, never fetched here.
import re  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))   # related.py sits next to this file (also when loaded by path)
import related  # noqa: E402

EVENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")   # Frigate event ids: <epoch>.<frac>-<random>
RELATED_DAYS = int(E.get("PANE_RELATED_DAYS", "90"))           # candidate window (unknowns are kept 90 days anyway)
RELATED_CANDIDATES = int(E.get("PANE_RELATED_CANDIDATES", "600"))
RELATED_TOP = int(E.get("PANE_RELATED_TOP", "12"))
VEHICLE_WINDOW_MIN = 10
PANE_FACE_LINKS = E.get("PANE_FACE_LINKS", "1") != "0"   # visit-to-visit face similarity as evidence (never a name)


def _iso(t):
    return t.isoformat() if t else None


def _frigate_links(eid: str) -> dict:
    f = E.get("FRIGATE_URL", "https://frigate.cam.example.org")
    return {"clip": f"{f}/api/events/{eid}/clip.mp4", "thumbnail": f"{f}/api/events/{eid}/thumbnail.webp",
            "explore": f"{f}/explore?event_id={eid}"}


def _sighting_links(sid, eid, crop_file=None, traj=None) -> dict:
    out = {"why": url_for("sighting", sid=sid), **_frigate_links(eid)}
    if crop_file:
        out["crop"] = url_for("crop", f=crop_file)
    if traj:
        out["replay_data"] = url_for("evidence_file", f=traj)
    return out


def _json(obj, status=200):
    return Response(json.dumps(obj, default=lambda o: o.isoformat() if hasattr(o, "isoformat") else float(o)),
                    status, mimetype="application/json", headers={"Cache-Control": "private, no-store"})


def _col(c, sql: str, args) -> list:
    """Query that may hit a column a not-yet-migrated schema lacks: [] instead of a 500."""
    try:
        with c.transaction():
            return c.execute(sql, args).fetchall()
    except (psycopg.errors.UndefinedColumn, psycopg.errors.UndefinedTable):
        return []


def _visit_face_fit(c, s: dict) -> dict | None:
    """The visit's fitted face (3D) for the replay pane: summary + GLB + the identity's prior-model index endpoint."""
    f = (_col(c, "SELECT face_fit FROM sightings WHERE id = %s", (s["id"],)) or [(None,)])[0][0]
    if not f or not f.get("n_frames"):
        return None
    out = {k: f.get(k) for k in ("n_frames", "reproj_px_med", "reproj_ipd_pct_med", "yaw_range", "vertex_sd_ipd",
                                  "apparent_expression")}
    out["metric"] = bool((f.get("multiview") or {}).get("scale_used"))
    out["glb"] = url_for("evidence_file", f=f["file"]) if f.get("file") and (EVID / f["file"]).is_file() else None
    out["models"] = url_for("api_identity_faces", iid=s["identity_id"])
    out["identity"] = s["identity_id"]
    return out


def _face_summary(c, s: dict, kind: str) -> dict | None:
    m, f = s.get("face_meta") or {}, s.get("face") or {}
    if not m and not f:
        return None
    d = m.get("match") or {}
    lv = m.get("liveness") or {}
    ap = m.get("apparent") or {}
    out = {"usable": s.get("has_face_vec", False), "n_faces": m.get("n_faces"), "faces_seen": m.get("faces_seen"),
           "quality": m.get("quality"), "size_px": m.get("size_px"), "max_size_px": m.get("max_size_px"),
           "consistency": m.get("consistency"), "reason": m.get("reason"),
           "liveness": {"ok": lv.get("ok"), "score": lv.get("score")} if lv else None,
           "crops": [url_for("face_img", f=x) for x in (m.get("crop_files") or [])[:4]],
           "decision": {"status": d.get("status"), "sim": d.get("sim"), "second_sim": d.get("second_sim"),
                        "reasons": [{"code": r, "text": REASON_TEXT.get(r, r)} for r in d.get("reasons") or []],
                        "threshold": (d.get("thresholds") or {}).get("sim")} if d else None,
           # apparent expression: a description only, never used for identity or alerts (docs/faceid.md)
           "expression": ap.get("expression"),
           "frigate_face": {"name": f.get("name"), "score": f.get("score")} if f.get("name") else None}
    # the enrolled face it matched (household/known only, consented): shown only for an actual gated match
    if d.get("status") == "match" and d.get("template") and kind in ENROLL_KINDS:
        out["enrolled_crop"] = url_for("face_template_img", tid=int(d["template"]))
    return out


REASON_TEXT = {  # short copy of bridge/explain.py REASON_TEXT for the codes the pane shows (codes stay the source)
    "FACE_MATCH": "face recognised (enrolled person)", "FACE_SEEN": "face recognised, no identity with that label",
    "FACE_TEMPLATE_MATCH": "face matches an enrolled face template (liveness passed)",
    "FACE_NO_MATCH": "face seen, not an enrolled person", "FACE_LIVENESS_FAIL": "face liveness check failed: not used",
    "FACE_TOO_SMALL": "face too small to compare", "FACE_TOO_FEW": "too few usable face frames",
    "FACE_LOW_QUALITY": "face quality too low", "FACE_INCONSISTENT": "faces in this visit disagree",
    "APPEARANCE_STRONG": "clothing and build closely match", "APPEARANCE_LIKELY": "clothing and build match",
    "APPEARANCE_POSSIBLE": "clothing and build are somewhat similar", "NEW_PERSON": "nobody on file looks similar",
    "HEIGHT_MATCH": "height matches", "HEIGHT_CONSISTENT": "height is consistent", "HEIGHT_MISMATCH": "height differs",
    "GAIT_MATCH": "walking rhythm matches", "GAIT_DIFFERS": "walking rhythm differs",
    "GAIT_SIG_MATCH": "gait signature is close", "GAIT_SIG_DIFFERS": "gait signature differs",
    "NIGHT": "at night", "REPEAT_24H": "seen repeatedly today", "LINGER": "stayed a long time",
    "ZONE": "entered a watched zone", "WATCHLIST": "on the watch list", "UNKNOWN": "not a known person",
    "ACTION_FALL": "a possible fall", "WATCHED_OBJECT": "carrying an object on the watch list",
    "PULC_CONSISTENT": "apparent clothing/bag attributes agree with earlier visits (weak)",
    "PULC_CONTRADICTS": "apparent clothing/bag attributes differ from earlier visits (weak)",
    "PULC_APPARENT_GENDER_DIFFERS": "apparent gender differs from earlier visits (estimate, weak)",
    "PULC_APPARENT_AGE_DIFFERS": "apparent age band differs from earlier visits (estimate, weak)",
}


def _reason(code: str) -> dict:
    if code.startswith("CARRYING_"):
        return {"code": code, "text": "carrying " + code[9:].lower().replace("_", " ")}
    if code.startswith("ACTION_") and code not in REASON_TEXT:
        return {"code": code, "text": "action: " + code[7:].lower().replace("_", " ")}
    return {"code": code, "text": REASON_TEXT.get(code, code.lower().replace("_", " "))}


VISIT_COLS = ["id", "frigate_event", "camera", "ts_start", "ts_end", "score", "identity_id", "label", "kind", "first_seen",
              "last_seen", "match_sim", "match_kind", "possible_id", "possible_sim", "reviewed", "attributes", "age_low",
              "age_high", "gender", "gender_conf", "crop_file", "height_cm", "height_iqr_cm", "gait", "explain",
              "evidence_card", "evidence_gif", "match_strip", "trajectory_file", "behavior", "embedding"]
VISIT_OPT = ["objects", "actions", "routine", "face", "face_meta", "pulc", "face_vec", "gait_vec"]   # newer columns


def _load_visit(c, where: str, arg) -> dict | None:
    cols = ", ".join(("i." if k in ("label", "kind", "first_seen", "last_seen") else "s.") + k for k in VISIT_COLS)
    r = c.execute(f"SELECT {cols} FROM sightings s JOIN identities i ON i.id = s.identity_id WHERE {where}",
                  (arg,)).fetchone()
    if not r:
        return None
    s = dict(zip(VISIT_COLS, r))
    for k in VISIT_OPT:
        s[k] = (_col(c, f"SELECT {k} FROM sightings WHERE id = %s", (s["id"],)) or [(None,)])[0][0]
    s["has_face_vec"] = s.get("face_vec") is not None
    return s


def _related_sightings(c, s: dict) -> list:
    """Ranked earlier sightings (same identity + re-ID / face / gait neighbours) within RELATED_DAYS."""
    rows = c.execute("""SELECT s.id, s.frigate_event, s.identity_id, i.label, i.kind, s.ts_start, s.ts_end, s.camera,
                               s.crop_file, s.trajectory_file, s.reviewed, s.explain->>'band', (s.explain->>'pct')::int,
                               s.embedding
                        FROM sightings s JOIN identities i ON i.id = s.identity_id
                        WHERE s.id <> %s AND s.ts_start < %s AND s.ts_start >= %s - make_interval(days => %s)
                        ORDER BY s.ts_start DESC LIMIT %s""",
                     (s["id"], s["ts_start"], s["ts_start"], RELATED_DAYS, RELATED_CANDIDATES)).fetchall()
    keys = ["id", "event", "identity_id", "label", "kind", "ts_start", "ts_end", "camera", "crop_file", "trajectory_file",
            "reviewed", "band", "pct", "embedding"]
    cands = [dict(zip(keys, r)) for r in rows]
    if not cands:
        return []
    ids = [d["id"] for d in cands]
    if PANE_FACE_LINKS and s.get("face_vec") is not None:   # only visits that both have a face template (512-d)
        fv = dict(_col(c, "SELECT id, face_vec FROM sightings WHERE id = ANY(%s) AND face_vec IS NOT NULL", (ids,)))
        for d in cands:
            d["face_vec"] = fv.get(d["id"])
    # gait signatures are 4096-d: fetch them for a short list only (best by appearance + same identity)
    pre = related.rank(s, cands, top_k=max(40, RELATED_TOP * 3))
    if s.get("gait_vec") is not None:
        short = [d["id"] for d in pre]
        gv = dict(_col(c, "SELECT id, gait_vec FROM sightings WHERE id = ANY(%s) AND gait_vec IS NOT NULL", (short,)))
        if gv:
            by = {d["id"]: d for d in cands}
            for i in short:
                by[i]["gait_vec"] = gv.get(i)
            pre = related.rank(s, [by[i] for i in short], top_k=RELATED_TOP)
    return pre[:RELATED_TOP]


def _identity_brief(c, iid) -> dict | None:
    r = c.execute("""SELECT i.id, i.label, i.kind, i.first_seen, i.last_seen, count(s.id),
                            (SELECT crop_file FROM sightings x WHERE x.identity_id = i.id ORDER BY ts_start DESC LIMIT 1)
                     FROM identities i LEFT JOIN sightings s ON s.identity_id = i.id WHERE i.id = %s GROUP BY i.id""",
                  (iid,)).fetchone()
    if not r:
        return None
    return {"id": r[0], "name": name(r[0], r[1]), "label": r[1], "kind": r[2], "first_seen": _iso(r[3]),
            "last_seen": _iso(r[4]), "sightings": r[5], "url": url_for("identity", iid=r[0]),
            "crop": url_for("crop", f=r[6]) if r[6] else None, "api": url_for("api_identity", iid=r[0])}


@app.route("/api/visit/<eid>")
def api_visit(eid):
    """One visit (by Frigate event id) + ranked related items with the evidence for every link."""
    if not EVENT_RE.match(eid):
        abort(404)
    with db() as c:
        s = _load_visit(c, "s.frigate_event = %s", eid)
        if not s:
            return _json({"error": "no stored visit for this event", "event": eid}, 404)
        kind = s["kind"]
        x = s.get("explain") or {}
        idt = _identity_brief(c, s["identity_id"])
        rel = _related_sightings(c, s)
        poss = _identity_brief(c, s["possible_id"]) if s.get("possible_id") else None
        t0, t1 = s["ts_start"], s["ts_end"] or s["ts_start"]
        vrows = c.execute("""SELECT vs.id, vs.frigate_event, vs.vehicle_id, v.label, v.kind, v.plate, v.color, v.body_type,
                                    v.make_model, vs.ts_start, vs.camera, vs.crop_file,
                                    extract(epoch FROM vs.ts_start - %s)
                             FROM vehicle_sightings vs JOIN vehicles v ON v.id = vs.vehicle_id
                             WHERE vs.ts_start BETWEEN %s - make_interval(mins => %s) AND %s + make_interval(mins => %s)
                             ORDER BY abs(extract(epoch FROM vs.ts_start - %s)) LIMIT 8""",
                          (t0, t0, VEHICLE_WINDOW_MIN, t1, VEHICLE_WINDOW_MIN, t0)).fetchall()
        together = dict(c.execute(f"""SELECT vs.vehicle_id, count(DISTINCT s.id) FROM sightings s
                                      JOIN vehicle_sightings vs ON vs.ts_start BETWEEN s.ts_start - {NEAR}
                                           AND COALESCE(s.ts_end, s.ts_start) + {NEAR}
                                      WHERE s.identity_id = %s AND s.id <> %s AND vs.vehicle_id = ANY(%s)
                                      GROUP BY vs.vehicle_id""",
                                  (s["identity_id"], s["id"], [v[2] for v in vrows])).fetchall()) if vrows else {}
        alerts_ = c.execute("SELECT ts, kind, subject, message, sent FROM alerts WHERE sighting = %s ORDER BY ts",
                            (eid,)).fetchall()
        trail = c.execute("""SELECT ts, stage, band, pct, reasons, message FROM decisions WHERE frigate_event = %s
                             ORDER BY ts""", (eid,)).fetchall()
        face = _face_summary(c, s, kind)
        face_fit = _visit_face_fit(c, s)
        fused = _col(c, "SELECT fused_trajectory_file FROM sightings WHERE id = %s", (s["id"],))
    fused = fused[0][0] if fused and fused[0][0] else None
    a = s.get("attributes") or {}
    pv = s.get("pulc") or {}
    o = pv.get("orientation") or {}
    g = s.get("gait") or {}
    b = s.get("behavior") or {}
    rt = s.get("routine")
    out = {
        "event": eid, "sighting": s["id"], "camera": s["camera"], "start": _iso(t0), "end": _iso(s["ts_end"]),
        "duration_s": b.get("duration_s") or ((s["ts_end"] - t0).total_seconds() if s["ts_end"] else None),
        "identity": idt, "match": {"kind": s["match_kind"], "sim": s["match_sim"], "reviewed": s["reviewed"]},
        "explain": {"headline": x.get("headline"), "band": x.get("band"), "pct": x.get("pct"),
                    "estimated": x.get("estimated", True), "evidence": x.get("evidence") or [],
                    "did": x.get("did") or [], "body": x.get("body") or [],
                    "reasons": [_reason(r) for r in x.get("reasons") or []], "scores": x.get("scores") or {}},
        "face": face,
        "face_fit": face_fit,
        "attributes": {
            "described": {k: a.get(k) for k in ("upper_color", "upper_clothing", "lower_color", "lower_clothing",
                                                 "activity", "carrying", "accessories") if a.get(k)},
            "apparent_age": [s["age_low"], s["age_high"]] if s["age_low"] and s["age_high"] else None,
            "apparent_gender": {"value": s["gender"], "conf": s["gender_conf"]} if s["gender"] and s["gender"] != "unclear" else None,
            "pulc": {"tags": pv.get("tags") or [], "view": max(o, key=o.get) if o else None,
                     "line": pulc_line(pv)} if pv else None},
        "objects": s.get("objects") or [], "actions": s.get("actions") or [],
        "routine": {k: rt.get(k) for k in ("bits", "threshold", "history", "ready", "unusual", "note")} | {
            "parts": sorted(rt.get("parts") or [], key=lambda p: -(p.get("bits") or 0))[:4]} if rt else None,
        "height": {"cm": s["height_cm"], "iqr_cm": s["height_iqr_cm"]} if s["height_cm"] else None,
        "gait": {k: g.get(k) for k in ("usable", "cadence_spm", "step_m", "speed_mps")} if g else None,
        "gait_signature": s.get("gait_vec") is not None,
        "behavior": {k: b.get(k) for k in ("duration_s", "path_length_m", "speed_mean_mps", "entered_zones") if b.get(k)},
        "evidence": {k: url_for("evidence_file", f=s[c_]) for k, c_ in (("card", "evidence_card"), ("gif", "evidence_gif"),
                                                                         ("strip", "match_strip")) if s.get(c_)},
        "links": _sighting_links(s["id"], eid, s["crop_file"], s["trajectory_file"])
                 | {"identity": url_for("identity", iid=s["identity_id"])}
                 | ({"replay_fused": url_for("evidence_file", f=fused)} if fused else {}),
        "alerts": [{"ts": _iso(t), "kind": k, "subject": sub, "message": m, "sent": sent} for t, k, sub, m, sent in alerts_],
        "decisions": [{"ts": _iso(t), "stage": st, "band": bd, "pct": p, "reasons": rs or [], "message": m}
                      for t, st, bd, p, rs, m in trail],
        "related": [],
        "window_days": RELATED_DAYS,
    }
    rel_out = out["related"]
    if idt:
        rel_out.append({"type": "identity", "id": idt["id"], "name": idt["name"], "kind": idt["kind"], "crop": idt["crop"],
                        "links": {"identity": idt["url"], "api": idt["api"]}, "sightings": idt["sightings"],
                        "first_seen": idt["first_seen"], "last_seen": idt["last_seen"],
                        "strength": {"p": (x.get("pct") or 50) / 100, "pct": x.get("pct"), "band": x.get("band"),
                                     "edges": [{"kind": "assigned", "value": s["match_sim"], "effect": "base",
                                                "text": f"filed under this identity ({s['match_kind']}"
                                                        + (f", re-ID {s['match_sim']:.2f}" if s["match_sim"] else "") + ")"}]}})
    if poss:
        p = related.appearance_prob(float(s["possible_sim"] or 0))
        rel_out.append({"type": "possible_identity", "id": poss["id"], "name": poss["name"], "kind": poss["kind"],
                        "crop": poss["crop"], "links": {"identity": poss["url"], "api": poss["api"]},
                        "sightings": poss["sightings"], "first_seen": poss["first_seen"], "last_seen": poss["last_seen"],
                        "strength": {"p": round(p, 4), "pct": related.pct(p), "band": related.band_of(p), "edges": [
                            {"kind": "possible", "value": s["possible_sim"], "effect": "base",
                             "text": f"possibly the same person (re-ID {float(s['possible_sim'] or 0):.2f}); kept separate until confirmed"}]}})
    for d in rel:
        rel_out.append({"type": "sighting", "id": d["id"], "event": d["event"], "identity_id": d["identity_id"],
                        "name": name(d["identity_id"], d["label"]), "kind": d["kind"], "start": _iso(d["ts_start"]),
                        "camera": d["camera"], "band": d["band"], "pct": d["pct"], "crop": url_for("crop", f=d["crop_file"])
                        if d["crop_file"] else None, "reid_cos": d["reid_cos"], "face_cos": d["face_cos"],
                        "gait_dist": d["gait_dist"], "strength": d["strength"],
                        "links": _sighting_links(d["id"], d["event"], d["crop_file"], d["trajectory_file"])})
    for (vsid, veid, vid, vlabel, vkind, plate, colr, body, mm, vts, vcam, vcrop, dts) in vrows:
        rel_out.append({"type": "vehicle", "id": vid, "sighting": vsid, "event": veid, "name": vname(vid, vlabel, plate),
                        "kind": vkind, "plate": plate, "look": " ".join(filter(None, [colr, body, mm])) or None,
                        "start": _iso(vts), "camera": vcam, "crop": url_for("vcrop", f=vcrop) if vcrop else None,
                        "strength": related.vehicle_strength(float(dts), int(together.get(vid, 0)), VEHICLE_WINDOW_MIN * 60),
                        "links": {"vehicle": url_for("vehicle", vid=vid), **_frigate_links(veid)}})
    return _json(out)


# -- prior 3D face models (facemodels.py): an identity's fitted faces, aligned; explanation only -----------------------
def _face_models(c, iid: int) -> list[dict]:
    """The identity's visits with a fitted face whose 3D file exists, newest first."""
    rows = _col(c, """SELECT id, frigate_event, ts_start, camera, face_fit FROM sightings
                      WHERE identity_id = %s AND face_fit IS NOT NULL ORDER BY ts_start DESC LIMIT 60""", (iid,))
    out = []
    for sid, ev, ts, cam, ff in rows:
        f = ff or {}
        if f.get("file") and (EVID / f["file"]).is_file() and f.get("n_frames"):
            out.append({"sighting": sid, "event": ev, "ts": ts, "camera": cam, "fit": f, "path": EVID / f["file"]})
    return out


def _model_index(c, iid: int) -> dict:
    import facemodels as fm
    ms = _face_models(c, iid)
    shapes, keep = [], []
    for m in ms:
        try:
            V, T = fm.read_lod(m["path"])
            shapes.append(V)
            keep.append((m, T))
        except Exception:      # unreadable/older file: listed without geometry
            continue
    mean = None
    if len(shapes) >= 2:
        mean, aligned = fm.mean_shape(shapes, [float(m["fit"].get("n_frames") or 1) for m, _ in keep])
        for (m, _), A in zip(keep, aligned):
            m["vs_mean"] = fm.summary(np.linalg.norm(A - mean, axis=1))
    return {"models": [m for m, _ in keep], "mean": mean, "topology": keep[0][1] if keep else None}


@app.route("/api/identity/<int:iid>/faces")
def api_identity_faces(iid):
    """An identity's prior 3D face models (one fitted face per visit), how each deviates from the identity's mean face,
    and how consistent the models are. Comparison stays within this identity: explanation only, never identification."""
    with db() as c:
        idt = c.execute("SELECT id, label, kind FROM identities WHERE id = %s", (iid,)).fetchone()
        if not idt:
            abort(404)
        ix = _model_index(c, iid)
    ms = ix["models"]
    feats = {}
    for m in ms:
        for k, v in ((m["fit"].get("features") or {}).items()):
            if isinstance(v, dict) and v.get("value") is not None:
                feats.setdefault(k, []).append(float(v["value"]))
    spread = {k: {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v)), 4), "n": len(v)}
              for k, v in feats.items() if len(v) >= 2}
    rms = [m["vs_mean"]["rms_pct"] for m in ms if m.get("vs_mean")]
    return {"identity": {"id": idt[0], "name": name(idt[0], idt[1]), "kind": idt[2]},
            "models": [{"sighting": m["sighting"], "event": m["event"], "ts": m["ts"].isoformat(), "camera": m["camera"],
                        "n_frames": m["fit"].get("n_frames"), "reproj_px": m["fit"].get("reproj_px_med"),
                        "metric": bool((m["fit"].get("multiview") or {}).get("scale_used")),
                        "glb": url_for("evidence_file", f=m["fit"]["file"]), "vs_mean": m.get("vs_mean"),
                        "why": url_for("sighting", sid=m["sighting"])} for m in ms],
            "mean": {"glb": url_for("api_identity_face_mean", iid=iid), "n": len(ms),
                     "consistency_rms_pct_median": round(float(np.median(rms)), 2) if rms else None} if ix["mean"] is not None else None,
            "feature_spread": spread,
            "note": ("Prior 3D models of this identity only, aligned (rotation, scale, position) and compared point by point "
                     "in units of the eye spacing. Explanatory evidence: identity is decided by the face embedding, never by "
                     "shape; differences also come from distance, angle, lighting and expression.")}


@app.route("/api/identity/<int:iid>/face_mean.glb")
def api_identity_face_mean(iid):
    import facemodels as fm
    with db() as c:
        ix = _model_index(c, iid)
    if ix["mean"] is None:
        abort(404)
    return Response(fm.write_glb(ix["mean"], ix["topology"], name=f"identity {iid} mean face"),
                    mimetype="model/gltf-binary", headers={"Cache-Control": "no-store"})


@app.route("/api/face/<eid>/deviation")
def api_face_deviation(eid):
    """Per-point deviation (1/1000 of the eye spacing, LOD1 vertex order) of a visit's fitted face from its identity's
    mean face (?ref=mean, default) or from another visit of the SAME identity (?ref=<event>)."""
    import facemodels as fm
    if not EVENT_RE.match(eid):
        abort(404)
    ref = request.args.get("ref", "mean")
    with db() as c:
        r = c.execute("SELECT identity_id, face_fit FROM sightings WHERE frigate_event = %s", (eid,)).fetchone()
        if not r or not (r[1] or {}).get("file"):
            abort(404)
        iid = r[0]
        A, _ = fm.read_lod(EVID / r[1]["file"])
        if ref == "mean":
            ix = _model_index(c, iid)
            if ix["mean"] is None:
                return {"available": False, "why": "the identity needs at least two fitted faces for a mean"}
            B = ix["mean"]
        else:
            if not EVENT_RE.match(ref):
                abort(404)
            q = c.execute("SELECT identity_id, face_fit FROM sightings WHERE frigate_event = %s", (ref,)).fetchone()
            if not q or q[0] != iid or not (q[1] or {}).get("file"):
                abort(404)        # comparisons stay within one identity
            B, _ = fm.read_lod(EVID / q[1]["file"])
    d = fm.deviation(A, B)
    return {"available": True, "event": eid, "ref": ref, "per_point_milli_ipd": np.round(d * 1000).astype(int).tolist(),
            **fm.summary(d)}


@app.route("/api/identity/<int:iid>")
def api_identity(iid):
    """An identity with its visits (newest first, max 200) and the vehicles seen around them."""
    with db() as c:
        idt = _identity_brief(c, iid)
        if not idt:
            return _json({"error": "no such identity"}, 404)
        rows = c.execute("""SELECT id, frigate_event, ts_start, ts_end, camera, crop_file, trajectory_file, match_kind,
                                   match_sim, explain->>'band', (explain->>'pct')::int, height_cm,
                                   (routine->>'unusual')::boolean
                            FROM sightings WHERE identity_id = %s ORDER BY ts_start DESC LIMIT 200""",
                         (iid,)).fetchall()
        hs = c.execute("""SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY height_cm), count(height_cm),
                                 percentile_cont(0.5) WITHIN GROUP (ORDER BY (gait->>'cadence_spm')::real)
                                     FILTER (WHERE (gait->>'usable')::boolean)
                          FROM sightings WHERE identity_id = %s""", (iid,)).fetchone()
        near = c.execute(f"""SELECT v.id, v.label, v.plate, v.color, v.body_type, count(*) FROM sightings s
                            JOIN vehicle_sightings vs ON vs.ts_start BETWEEN s.ts_start - {NEAR} AND COALESCE(s.ts_end, s.ts_start) + {NEAR}
                            JOIN vehicles v ON v.id = vs.vehicle_id WHERE s.identity_id = %s
                            GROUP BY v.id ORDER BY count(*) DESC LIMIT 10""", (iid,)).fetchall()
        n_templates = len(face_templates(c, iid))
        driven = occupant_vehicles(c, iid)
    return _json({
        "identity": idt, "enrolled_faces": n_templates,
        "height_cm_median": hs[0], "height_n": hs[1], "cadence_spm_median": hs[2],
        "visits": [{"sighting": r[0], "event": r[1], "start": _iso(r[2]), "end": _iso(r[3]), "camera": r[4],
                    "crop": url_for("crop", f=r[5]) if r[5] else None, "match_kind": r[7], "match_sim": r[8],
                    "band": r[9], "pct": r[10], "height_cm": r[11], "unusual": bool(r[12]),
                    "links": _sighting_links(r[0], r[1], r[5], r[6])} for r in rows],
        "vehicles": [{"id": v[0], "name": vname(v[0], v[1], v[2]), "plate": v[2],
                      "look": " ".join(filter(None, [v[3], v[4]])) or None, "times": v[5],
                      "url": url_for("vehicle", vid=v[0])} for v in near],
        "in_vehicles": [{"id": d["vid"], "name": d["vname"], "times": d["n"], "as": d["kinds"], "last": _iso(d["last"]),
                         "best_sim": d["sim"], "url": url_for("vehicle", vid=d["vid"])} for d in driven]})


@app.route("/replay/")
@app.route("/replay/<path:f>")
def replay(f="index.html"):
    """3D trajectory replay viewer (static files from deploy/recorder/replay, mounted at /app/replay): the page,
    its scripts, scene.json + splat files, and the demo/ dataset."""
    parts = f.split("/")
    clip = len(parts) >= 3 and parts[:2] == ["scenes", "clip"]  # per-clip scenes (scripts/scene_worker.py)
    objects = (((len(parts) == 3 and parts[0] in ("demo", "scenes")) or (clip and len(parts) == 4))
               and parts[-2].endswith("_objects") and f.endswith(".glb"))   # 3D object models (scripts/rig_objects.py)
    if ".." in parts or not (objects or (
            (len(parts) == 1 or (len(parts) == 2 and parts[0] in ("demo", "scenes")) or (clip and len(parts) == 3))
            and f.endswith((".html", ".js", ".json", ".webp", ".jpg", ".ply", ".splat", ".ksplat", ".bin")))):  # .bin = body3d mesh
        abort(404)
    return send_from_directory("/app/replay", f)


def scene_slug(frame: str) -> str:
    """Calibration frame id (scripts/multicam_rig.py) -> file-name-safe scene name under replay/scenes/."""
    return re.sub(r"[^A-Za-z0-9]+", "-", str(frame)).strip("-")[:120]


def attach_scene(tr: dict, stem: str | None = None) -> dict:
    """Room splat + 3D objects for a replay. First the clip's own scene (replay/scenes/clip/<stem>.json, built from
    that visit's recordings by scripts/scene_worker.py and aligned to the installed calibration), else the scene of
    the replay's calibration frame (replay/scenes/<slug>.json, scripts/publish_scene.py). A clip manifest with
    status "camera_moved" is not attached: the frame scene is used and the replay is flagged."""
    warn = None
    if stem:
        cm = Path("/app/replay/scenes/clip") / f"{stem}.json"
        if re.fullmatch(r"[A-Za-z0-9._-]+", stem) and cm.is_file():
            try:
                m = json.loads(cm.read_text())
            except ValueError:
                m = {}
            if m.get("status") == "ok" and m.get("splat"):
                a = tr.setdefault("assets", {})
                for k in ("splat", "objects"):
                    if m.get(k) and not a.get(k) and str(m[k]).startswith("/replay/scenes/clip/"):
                        a[k] = m[k]
                tr["scene"] = {"frame": m.get("frame"), "built": m.get("built"), "note": m.get("note"),
                               "source": "clip", "reused": m.get("reused"), "residual": m.get("residual")}
                return tr
            if m.get("status") == "camera_moved":
                warn = m.get("note") or "cameras moved since the calibration"
    fr = (tr.get("fusion") or {}).get("frame") or (tr.get("calibration") or {}).get("frame") or tr.get("calibration_frame")
    if not fr:
        return tr
    man = Path("/app/replay/scenes") / f"{scene_slug(fr)}.json"
    if not man.is_file():
        return tr
    try:
        m = json.loads(man.read_text())
    except ValueError:
        return tr
    a = tr.setdefault("assets", {})
    for k in ("splat", "objects"):
        if m.get(k) and not a.get(k):
            a[k] = "/replay/scenes/" + m[k]
    tr["scene"] = {"frame": fr, "built": m.get("built"), "note": m.get("note"), "source": "frame"}
    if warn:
        tr["scene"]["warning"] = warn
    return tr


@app.route("/evidence/<path:f>")
def evidence_file(f):
    if f.endswith("_traj.json") and "/" not in f and (EVID / f).is_file():
        try:
            return _json(attach_scene(json.loads((EVID / f).read_text()), f[:-len("_traj.json")]))
        except ValueError:
            pass
    if "/" in f or not f.endswith((".jpg", ".gif", "_traj.json", "_replay_atlas.webp", "_replay_ground.json",
                                    "_facegeom_mesh.gltf", "_facefit.glb", "_body.bin", "_body_faces.bin")):
        abort(404)
    return send_from_directory(EVID, f)


def _act_check(token: str):
    """<identity>.<action>.<expiry>.<sig> signed by the bridge (key derived from PERSONS_UI_PASS)."""
    import hashlib, hmac, time as _t
    try:
        iid, action, exp, sig = token.split(".")
        iid, exp = int(iid), int(exp)
    except ValueError:
        abort(404)
    key = hashlib.sha256(b"camstack-act|" + E["PERSONS_UI_PASS"].encode()).digest()
    good = hmac.new(key, f"{iid}.{action}.{exp}".encode(), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(good, sig) or action not in ("known", "watch") or exp < _t.time():
        abort(403)
    return iid, action


@app.route("/act/<token>", methods=["GET", "POST"])
def act(token):
    iid, action = _act_check(token)
    words = {"known": "Mark as OK (known person: no more alerts)", "watch": "Add to the watch list (always alert)"}
    with db() as c:
        idt = c.execute("SELECT id, label, kind FROM identities WHERE id = %s", (iid,)).fetchone()
        if not idt:
            abort(404)
        if request.method == "POST":
            c.execute("UPDATE identities SET kind = %s WHERE id = %s", (action, iid))
            c.execute("""INSERT INTO decisions (stage, subject, reasons, message) VALUES ('action', %s, %s, %s)""",
                      (f"person:{iid}", [f"USER_{action.upper()}"], f"{name(idt[0], idt[1])} set to {action} from a notification link"))
            from markupsafe import escape
            body = (f'<h2>Done</h2><p>{escape(name(idt[0], idt[1]))} is now <span class="k {escape(action)}">{escape(action)}</span>.</p>'
                    '<p><small>You can close this page. Undo from the Persons page if this was a mistake.</small></p>')
        else:
            body = render_template_string("""<h2>{{ n }}</h2><p>Currently <span class="k {{ k }}">{{ k }}</span>.</p>
<form method="post"><button class="primary big">{{ w }}</button></form>""", n=name(idt[0], idt[1]), k=idt[2], w=words[action])
    # /act/ is reached without single sign-on (the signed token is the auth), so /ui/ui.css is not reachable: inline it
    css = (UI_DIR / "ui.css").read_text()
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark"><title>Persons</title><style>{css}</style></head>
<body><main class="act-page">{body}</main><script>document.querySelector("form")?.addEventListener("submit", e => {{
const b = e.target.querySelector("button"); b.classList.add("is-busy"); setTimeout(() => b.disabled = true, 0); }});</script></body></html>"""


# -- evidence packages, custody, review, audit (evidence_export.py, access.py, docs/evidence-export.md) -------------
import hashlib  # noqa: E402
import evidence_export as evx  # noqa: E402
import tsa as tsalib  # noqa: E402
import threading  # noqa: E402

STORE = Path(E.get("EVIDENCE_STORE", "/data/evidence-store"))
KEYS = Path(E.get("EVIDENCE_KEYS", "/data/evidence-keys"))
RECORDINGS = Path(E.get("RECORDINGS", "/recordings"))
LEDGER = Path(E.get("LEDGER", "/ledger"))
TSA_URL = E.get("TSA_URL", "").strip()
CAMERA_CLOCK_NOTE = E.get("CAMERA_CLOCK_NOTE", "")
PURPOSES = ["Criminal investigation", "Incident report", "Insurance claim", "Request by the person recorded", "Court order or subpoena", "Other"]
SOFTWARE_VERSION = E.get("SG_VERSION", "unknown")
MAX_ITEMS = 25
SEGMENT_PAD_S = 5
CUSTODY_KEYS = ("export_id", "ts", "actor", "action", "detail", "remote")
_signer = None
JOBS: dict = {}
JOB_LOCK = threading.Lock()


def signer() -> evx.Signer:
    global _signer
    if _signer is None:
        _signer = evx.Signer(KEYS)
    return _signer


def _actor() -> str:
    a = request.authorization
    return request.headers.get("Remote-User") or (a.username if a else "") or "unknown"


def custody(c, export_id: int, action: str, detail: str = "", actor: str | None = None, remote: str | None = None) -> str:
    return access.append(c, "custody_log", {"export_id": export_id, "ts": evx.stamp(evx.utcnow()), "actor": actor or _actor(),
                                            "action": action, "detail": detail,
                                            "remote": request.remote_addr or "" if remote is None else remote}, CUSTODY_KEYS, 752001)


def _chain(c, table="custody_log", export_id=None, limit=None) -> list[dict]:
    keys = ["id"] + list(CUSTODY_KEYS if table == "custody_log" else access.AUDIT_FIELDS) + ["prev_hash", "entry_hash"]
    q = f"SELECT {', '.join(keys)} FROM {table} " + ("WHERE export_id = %s " if export_id else "") + "ORDER BY id"
    return [dict(zip(keys, r)) for r in c.execute(q, (export_id,) if export_id else ()).fetchall()]


# ---------------------------------------------------------------- human review of a match (item 7)
def reviews(c, sid: int) -> dict:
    rows = c.execute("SELECT ts, reviewer, role, outcome, note FROM match_reviews WHERE sighting_id = %s ORDER BY id", (sid,)).fetchall()
    hist = [{"ts": _iso(t), "reviewer": r, "role": ro, "outcome": o, "note": n} for t, r, ro, o, n in rows]
    return {"latest": hist[-1] if hist else None, "history": hist}


def review_html(c, sid: int) -> str:
    rv = reviews(c, sid)
    return render_template_string("""<section aria-labelledby="rv-h"><h3 id="rv-h">Human review</h3>
{% set l = rv.latest %}{% if not l %}<div class="unconfirmed">NOT CONFIRMED BY A PERSON. This match is an unverified lead until someone reviews it.</div>
{% elif l.outcome == 'confirmed' %}<div class="confirmed">Confirmed by {{ l.reviewer }} ({{ l.role }}) on {{ when_(l.ts) }}{% if l.note %}: {{ l.note }}{% endif %}</div>
{% else %}<div class="unconfirmed">{{ 'REJECTED' if l.outcome == 'rejected' else 'REVIEWED, NOT DECIDED' }} by {{ l.reviewer }} on {{ when_(l.ts) }}{% if l.note %}: {{ l.note }}{% endif %}</div>{% endif %}
{% if can %}<form method="post" action="{{ url_for('review', sid=sid) }}" class="actions" data-confirm-title="Record your review?" data-confirm="Your name, the time and your decision are added to this sighting's review history and the audit log. They cannot be edited later; a new review replaces the decision, not the history." data-confirm-kind="warn" data-confirm-icon="info">
 <select name="outcome" required aria-label="Decision"><option value="" disabled selected>Your decision…</option>
 <option value="confirmed">Confirmed: same person</option><option value="rejected">Rejected: not the same person</option><option value="unsure">Unsure</option></select>
 <input name="note" maxlength="300" size="40" placeholder="What you compared (required to reject)" aria-label="Note">
 <button class="primary">Record review</button></form>{% endif %}
{% if rv.history|length > 1 %}<details><summary>Review history ({{ rv.history|length }})</summary><table><tr><th>When</th><th>Reviewer</th><th>Decision</th><th>Note</th></tr>
{% for r in rv.history|reverse %}<tr><td>{{ when_(r.ts) }}</td><td>{{ r.reviewer }} <small>{{ r.role }}</small></td><td>{{ r.outcome }}</td><td>{{ r.note }}</td></tr>{% endfor %}</table></details>{% endif %}</section>""",
                                  rv=rv, sid=sid, can=access.at_least("investigator"),
                                  when_=lambda t: when(dt.datetime.fromisoformat(t)) if t else "")


@app.route("/sighting/<int:sid>/review", methods=["POST"])
@access.require("investigator")
def review(sid):
    outcome, note = request.form.get("outcome", ""), request.form.get("note", "").strip()[:300]
    if outcome not in ("confirmed", "rejected", "unsure"):
        abort(400)
    if outcome == "rejected" and not note:
        return Response("Say what you compared when rejecting a match (the note is required).\n", 400, {"Content-Type": "text/plain; charset=utf-8"})
    with db() as c:
        r = c.execute("SELECT frigate_event FROM sightings WHERE id = %s", (sid,)).fetchone()
        if not r:
            abort(404)
        c.execute("INSERT INTO match_reviews (sighting_id, frigate_event, reviewer, role, outcome, note) VALUES (%s, %s, %s, %s, %s, %s)",
                  (sid, r[0], _actor(), g.role, outcome, note))
    g.audit_action = f"review:{outcome}"
    return redirect(url_for("sighting", sid=sid, done="reviewed"))


# ---------------------------------------------------------------- continuous recording segments + capture-time seals (item 5)
def _segment_start(name: str) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(name[:19], "%Y-%m-%d_%H-%M-%S").astimezone()    # recorder writes local time (TZ)
    except ValueError:
        return None


def _ledger(camera: str) -> list[dict]:
    p = LEDGER / f"{camera}.jsonl"
    try:
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    except (OSError, ValueError):
        return []


def segments_for(camera: str, start: dt.datetime, end: dt.datetime | None, seg_s: int = 600) -> dict:
    """Segments of `camera` that overlap the event, each checked against the ledger. Re-hashes every segment."""
    end = end or start
    lo, hi = start - dt.timedelta(seconds=SEGMENT_PAD_S), end + dt.timedelta(seconds=SEGMENT_PAD_S)
    led = _ledger(camera)
    by = {d["file"]: d for d in led}
    out = []
    d = RECORDINGS / camera
    names = sorted(p.name for p in d.glob("*.mkv")) if d.is_dir() and re.match(r"^[A-Za-z0-9_-]+$", camera) else []
    starts = [(n, _segment_start(n)) for n in names]
    starts = [(n, t) for n, t in starts if t]
    for i, (n, t) in enumerate(starts):
        t_end = starts[i + 1][1] if i + 1 < len(starts) else t + dt.timedelta(seconds=seg_s)
        if t < hi and t_end > lo:
            out.append((n, t))
    for n, t in [(f.split("/")[-1], _segment_start(f.split("/")[-1])) for f in by if f.startswith(camera + "/")]:
        if t and lo - dt.timedelta(seconds=seg_s) < t < hi and n not in {x for x, _ in out} and t + dt.timedelta(seconds=seg_s) > lo:
            out.append((n, t))                                       # sealed, but already removed by retention
    segs = []
    for n, t in sorted(out, key=lambda x: x[1]):
        rel, p = f"{camera}/{n}", RECORDINGS / camera / n
        e = by.get(rel)
        g_ = {"file": rel, "path": p, "start": _iso(t), "included": False}
        if not p.exists():
            g_.update(status="gone", status_text="removed by retention; its capture-time hash is in the ledger" if e else "removed")
        elif not e:
            g_.update(status="unsealed", status_text="not sealed yet (still recording, or the sealer is not running)")
        else:
            digest, _ = evx.sha256_file(p)
            ok = digest == e["sha256"]
            g_.update(status="sealed" if ok else "differs", sha256=digest, sealed_utc=e["sealed_utc"], seq=e["seq"],
                      status_text="matches its capture-time seal" if ok else "DOES NOT MATCH its capture-time seal")
        segs.append(g_)
    seqs = [x["seq"] for x in segs if x.get("seq")]
    excerpt, anchors = [], []
    if seqs:
        aps = [json.loads(l) for l in (LEDGER / "anchors" / f"{camera}.jsonl").read_text().splitlines() if l.strip()] \
            if (LEDGER / "anchors" / f"{camera}.jsonl").exists() else []
        anchor = next((a for a in aps if a["seq"] >= max(seqs)), None)
        last = anchor["seq"] if anchor else led[-1]["seq"]
        excerpt = [x for x in led if min(seqs) <= x["seq"] <= last]
        if anchor:
            anchors = [anchor]
        for x in segs:
            if x.get("seq"):
                x["anchor"] = f"{anchor['date']} (head {anchor['head'][:12]}…)" if anchor else "not anchored yet (anchors are made daily)"
        if excerpt and evx.verify_ledger(excerpt) is not None:
            for x in segs:
                x.update(status="differs", status_text="the ledger around this segment is broken")
    return {"segments": segs, "ledger": excerpt, "anchors": anchors}


# ---------------------------------------------------------------- export: choose items, build in the background (items 1-5, 8)
def _resolve_items(c, args) -> list[int]:
    if args.get("sids"):
        ids = [int(x) for x in re.findall(r"\d+", args["sids"])][:MAX_ITEMS]
    elif args.get("identity", "").isdigit():
        ids = [r[0] for r in c.execute("SELECT id FROM sightings WHERE identity_id = %s ORDER BY ts_start DESC LIMIT %s",
                                       (int(args["identity"]), MAX_ITEMS)).fetchall()]
    elif args.get("camera") and args.get("from") and args.get("to"):
        try:
            lo, hi = dt.datetime.fromisoformat(args["from"]).astimezone(), dt.datetime.fromisoformat(args["to"]).astimezone()
        except ValueError:
            return []
        ids = [r[0] for r in c.execute("SELECT id FROM sightings WHERE camera = %s AND ts_start BETWEEN %s AND %s ORDER BY ts_start LIMIT %s",
                                       (args["camera"], lo, hi, MAX_ITEMS)).fetchall()]
    else:
        return []
    rows = c.execute("SELECT id FROM sightings WHERE id = ANY(%s)", (ids,)).fetchall()
    have = {r[0] for r in rows}
    return [i for i in ids if i in have]


EXPORT_FORM = """<h2>Export evidence package</h2>
<div class="notice">One package for one case. It holds, for each sighting, the recorder's clip and snapshot unmodified,
the continuous recording around it with its capture-time seal, the pictures and explanation this system produced, the
human review, and a printable report. Every file is hashed, the manifest is signed{% if tsa %} and timestamped by an outside
service{% endif %}. Creating, downloading and checking it are written to the custody log under your name (<b>{{ actor }}</b>, {{ role }}).</div>
{% if err %}<div class="notice warn-box" role="alert">{{ err }}</div>{% endif %}
{% if not rows %}<div class="notice warn-box">No sightings selected. Open a sighting or an identity and choose “Export”.</div>{% else %}
<form method="post" class="block">
<table><tr><th>Include</th><th>When</th><th>Camera</th><th>Filed under</th><th>Review</th></tr>
{% for r in rows %}<tr><td><input type="checkbox" name="sid" value="{{ r.id }}" checked aria-label="Include sighting {{ r.id }}"></td>
<td><a href="{{ url_for('sighting', sid=r.id) }}">{{ when(r.ts) }}</a></td><td>{{ r.camera }}</td><td>{{ r.name }}</td>
<td>{% if r.review == 'confirmed' %}<span class="k known">confirmed</span>{% elif r.review %}<span class="flag">{{ r.review }}</span>{% else %}<span class="flag">not reviewed</span>{% endif %}</td></tr>{% endfor %}</table>
<p><label>Case or incident reference<br><input name="case_ref" required maxlength="80" value="{{ f.case_ref }}" placeholder="e.g. 2026-004512" size="30"></label></p>
<p><label>Purpose<br><select name="purpose" required>{% for p in purposes %}<option{% if p == f.purpose %} selected{% endif %}>{{ p }}</option>{% endfor %}</select></label></p>
<p><label>Note for the record (optional)<br><input name="note" maxlength="200" value="{{ f.note }}" size="60"></label></p>
<p><label><input type="checkbox" name="segments" checked> Include the continuous recording segments (10-minute files, sealed at capture; larger package)</label></p>
<p><label><input type="checkbox" name="ack" required> I am authorised to export these recordings for the purpose above.</label></p>
<div class="actions"><button class="primary">Create package</button> <a class="btn ghost" href="{{ back }}">Cancel</a></div></form>{% endif %}"""


@app.route("/exports/new", methods=["GET", "POST"])
@access.require("investigator")
def export_new():
    with db() as c:
        ids = _resolve_items(c, request.args)
        rows = []
        for sid in ids:
            r = c.execute("""SELECT s.id, s.ts_start, s.camera, s.identity_id, i.label FROM sightings s JOIN identities i ON i.id = s.identity_id
                             WHERE s.id = %s""", (sid,)).fetchone()
            rv = reviews(c, sid)["latest"]
            rows.append({"id": r[0], "ts": r[1], "camera": r[2], "name": name(r[3], r[4]), "review": rv["outcome"] if rv else None})
    f = {k: request.form.get(k, "").strip() for k in ("case_ref", "purpose", "note")}
    err = None
    if request.method == "POST":
        chosen = [int(x) for x in request.form.getlist("sid") if x.isdigit() and int(x) in ids]
        if not chosen:
            err = "Tick at least one sighting."
        elif not f["case_ref"] or f["purpose"] not in PURPOSES or not request.form.get("ack"):
            err = "Enter a case reference, choose a purpose and confirm that you are authorised."
        elif not STORE.is_dir() or not os.access(STORE, os.W_OK):
            err = "The evidence store is not set up on this installation (see docs/evidence-export.md)."
        else:
            try:
                signer()
            except Exception as e:
                err = f"The signing key is not available ({type(e).__name__}); see docs/evidence-export.md."
        if not err:
            jid = _start_export(chosen, f, bool(request.form.get("segments")))
            return redirect(url_for("export_job", jid=jid))
    back = request.referrer if request.referrer and request.referrer.startswith(request.host_url) else url_for("index")
    body = render_template_string(EXPORT_FORM, rows=rows, err=err, f=f, purposes=PURPOSES, actor=_actor(), role=g.role,
                                  tsa=bool(TSA_URL), when=when, back=back)
    return render_template_string(PAGE, body=body), (400 if err else 200)


@app.route("/sighting/<int:sid>/export")
def export_one(sid):
    return redirect(url_for("export_new", sids=str(sid)))


def _start_export(sids: list[int], f: dict, with_segments: bool) -> str:
    import secrets
    jid = secrets.token_hex(8)
    who = {"actor": _actor(), "role": g.role, "remote": request.remote_addr or ""}
    visits = {}
    with db() as c:                                              # the explanations need the request context (url_for)
        for sid in sids:
            eid = c.execute("SELECT frigate_event FROM sightings WHERE id = %s", (sid,)).fetchone()[0]
            visits[sid] = json.loads(api_visit(eid).get_data())
    with JOB_LOCK:
        JOBS[jid] = {"state": "running", "step": "starting", "owner": who["actor"], "xid": None, "error": None}
        for old in list(JOBS)[:-50]:
            del JOBS[old]
    threading.Thread(target=_export_job, args=(jid, sids, f, with_segments, who, visits), daemon=True).start()
    return jid


def _export_job(jid, sids, f, with_segments, who, visits):
    job = JOBS[jid]
    try:
        job["xid"] = _export_build(sids, f, with_segments, who, visits, lambda s: job.update(step=s))
        job["state"] = "done"
    except Exception as e:
        app.logger.exception("export failed")
        job.update(state="failed", error=f"{type(e).__name__}: {e}")


def _export_build(sids, f, with_segments, who, visits, step) -> int:
    import shutil
    import tempfile
    now = evx.utcnow()
    probe, rec_offsets = None, []
    if TSA_URL:
        step("measuring the clock against the outside time service")
        try:
            probe = tsalib.stamp(hashlib.sha256(f"clock probe {now.isoformat()}".encode()).hexdigest(), TSA_URL)
        except tsalib.TSAError:
            probe = None
    items, has_original, last_anchor = [], False, None
    with tempfile.TemporaryDirectory(dir=STORE, prefix=".tmp-") as tmp:
        tmp = Path(tmp)
        with db() as c:
            for n, sid in enumerate(sids, 1):
                s = c.execute("""SELECT s.id, s.frigate_event, s.ts_start, s.ts_end, s.camera, s.identity_id, i.label,
                                        s.crop_file, s.evidence_card, s.evidence_gif, s.match_strip, s.trajectory_file, s.gait
                                 FROM sightings s JOIN identities i ON i.id = s.identity_id WHERE s.id = %s""", (sid,)).fetchone()
                eid, d = s[1], tmp / f"{n:02d}"
                (d / "original" / "segments").mkdir(parents=True)
                (d / "derived").mkdir()
                files, missing = [], []
                step(f"item {n} of {len(sids)}: retrieving the clip")
                for fname, path, what in (("clip.mp4", f"/api/events/{eid}/clip.mp4", "Event recorder clip, as retrieved"),
                                          ("snapshot.jpg", f"/api/events/{eid}/snapshot.jpg", "Event recorder snapshot, as retrieved")):
                    try:
                        _, _, off = _frigate_file(path, d / "original" / fname, who["actor"])
                        if off is not None:
                            rec_offsets.append(off)
                        files.append({"path": f"original/{fname}", "src": d / "original" / fname, "role": "original", "description": what,
                                      "source": f"event recorder, event {eid}, retrieved {evx.stamp(evx.utcnow())}"})
                    except Exception as e:                       # retention may have removed it: say so in the package
                        missing.append(f"original/{fname}: the event recorder did not supply it ({type(e).__name__})")
                step(f"item {n} of {len(sids)}: checking the continuous recording against its seals")
                seg = segments_for(s[4], s[2], s[3])
                for x in seg["segments"]:
                    if with_segments and x["status"] in ("sealed", "unsealed", "differs") and x["path"].exists():
                        dst = d / "original" / "segments" / x["path"].name
                        shutil.copyfile(x["path"], dst)
                        if evx.sha256_file(dst)[0] != x.get("sha256", evx.sha256_file(x["path"])[0]):
                            raise RuntimeError(f"{x['file']} changed while it was being copied")
                        x["included"] = True
                        files.append({"path": f"original/segments/{x['path'].name}", "src": dst, "role": "original",
                                      "description": f"Continuous recording segment ({x['status_text']})", "source": f"continuous recorder, {x['file']}"})
                    if x["status"] == "gone":
                        missing.append(f"segment {x['file']}: removed by retention before this export")
                if not seg["segments"]:
                    missing.append("no continuous recording segment covers this event (recorder off, or another camera name)")
                if seg["anchors"]:
                    last_anchor = seg["anchors"][-1]
                for col, folder, out, what in ((s[7], CROPS, "crop.jpg", "Crop of the person from the recording"),
                                               (s[8], EVID, "evidence_card.jpg", "Evidence card: key frame, path and facts"),
                                               (s[9], EVID, "flipbook.gif", "Flip-book of the visit"),
                                               (s[10], EVID, "match_strip.jpg", "This visit beside the sightings it was compared with"),
                                               (s[11], EVID, "trajectory.json", "Tracked path of the person")):
                    if col and "/" not in col and (folder / col).is_file():
                        shutil.copyfile(folder / col, d / "derived" / out)
                        files.append({"path": f"derived/{out}", "src": d / "derived" / out, "role": "derived", "description": what, "source": col})
                has_original = has_original or any(x["path"] == "original/clip.mp4" for x in files)
                v = visits[sid]
                v["gait"] = s[12]
                if s[12] and s[12].get("usable"):
                    gp = gait_profile(c, s[5], sid)
                    if gp["walks"]:
                        v["gait_compare"] = gaitlab.compare(s[12], gp)
                for x in seg["segments"]:
                    x.pop("path", None)
                items.append({"n": n, "subject": {"sighting": sid, "event": eid, "camera": s[4], "start": _iso(s[2]), "end": _iso(s[3]),
                                                  "identity": name(s[5], s[6])},
                              "visit": v, "review": reviews(c, sid), "files": files, "missing": missing,
                              "segments": seg["segments"], "ledger": seg["ledger"], "anchors": seg["anchors"]})
            step("signing and timestamping")
            head = c.execute("SELECT entry_hash FROM custody_log ORDER BY id DESC LIMIT 1").fetchone()
            xid = c.execute("SELECT nextval('evidence_exports_id_seq')").fetchone()[0]
            export_id = f"EX-{now:%Y%m%d}-{xid:05d}"
            clock = evx.clock_record(probe, round(sum(rec_offsets) / len(rec_offsets)) if rec_offsets else None, last_anchor, CAMERA_CLOCK_NOTE)
            meta = {"export_id": export_id, "case_ref": f["case_ref"], "purpose": f["purpose"], "note": f["note"],
                    "exported_by": who["actor"], "exported_role": who["role"], "created_utc": evx.stamp(now),
                    "created_local": now.astimezone(TZ).isoformat(timespec="seconds"), "system": {"version": SOFTWARE_VERSION},
                    "custody_prev": head[0] if head else evx.GENESIS, "clock": clock,
                    "timestamp_note": f"manifest.tsr from {TSA_URL}" if TSA_URL else "none configured"}
            fname = f"{export_id}.zip"
            res = evx.build_package(STORE / fname, items, meta, signer(), (lambda h: tsalib.stamp(h, TSA_URL)) if TSA_URL else None)
            try:
                from psycopg.types.json import Jsonb
                first = items[0]["subject"]
                c.execute("""INSERT INTO evidence_exports (id, created, sighting_id, frigate_event, camera, items, export_ref, case_ref, purpose,
                                 exported_by, exported_role, package_file, package_sha256, package_bytes, manifest_sha256, manifest,
                                 has_original, signing_fingerprint, tsa, clock)
                             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                          (xid, now, first["sighting"], first["event"], first["camera"], Jsonb([it["subject"] | {"n": it["n"]} for it in items]),
                           export_id, f["case_ref"], f["purpose"], who["actor"], who["role"], fname, res["package_sha256"], res["package_bytes"],
                           res["manifest_sha256"], Jsonb(res["manifest"]), has_original, signer().fingerprint,
                           Jsonb(res["timestamp"]) if res["timestamp"] else None, Jsonb(clock)))
                ts_note = (f"; outside timestamp {res['timestamp']['gen_time']} serial {res['timestamp']['serial']}" if res["timestamp"]
                           else (f"; outside timestamp FAILED: {res['timestamp_error']}" if res["timestamp_error"] else ""))
                custody(c, xid, "created", f"{len(items)} item(s); package sha256 {res['package_sha256']}; manifest sha256 "
                                           f"{res['manifest_sha256']}; case {f['case_ref']}; purpose {f['purpose']}{ts_note}",
                        actor=who["actor"], remote=who["remote"])
            except Exception:
                (STORE / fname).unlink(missing_ok=True)          # no package without its record
                raise
    return xid


@app.route("/exports/job/<jid>")
@access.require("investigator")
def export_job(jid):
    job = JOBS.get(jid)
    if not job or job["owner"] != _actor():
        abort(404)
    if job["state"] == "done":
        return redirect(url_for("export_view", xid=job["xid"], done="exported"))
    body = render_template_string("""<h2>Building the evidence package</h2>{% if job.state == 'failed' %}
<div class="notice warn-box" role="alert"><b>The package was not created.</b> {{ job.error }}. Nothing was stored.
<a href="{{ url_for('exports') }}">Back to the evidence list</a></div>{% else %}<meta http-equiv="refresh" content="2">
<p role="status"><span class="spin" aria-hidden="true" style="display:inline-block;width:14px;height:14px;border-radius:50%;border:2px solid var(--line-2);border-top-color:var(--accent);animation:sgspin .7s linear infinite;vertical-align:-2px"></span>
&nbsp;{{ job.step }}…</p><p class="muted">Large recordings take a while to hash and copy. This page updates by itself.</p>
<style>@keyframes sgspin{to{transform:rotate(360deg)}}</style>{% endif %}""", job=job)
    return render_template_string(PAGE, body=body)


def _export_row(c, xid):
    keys = ["id", "created", "sighting_id", "frigate_event", "camera", "items", "export_ref", "case_ref", "purpose", "exported_by",
            "exported_role", "package_file", "package_sha256", "package_bytes", "manifest_sha256", "manifest", "has_original",
            "signing_fingerprint", "tsa", "clock"]
    r = c.execute(f"SELECT {', '.join(keys)} FROM evidence_exports WHERE id = %s", (xid,)).fetchone()
    return dict(zip(keys, r)) if r else None


def _may_see(x) -> bool:
    return access.at_least("supervisor") or x["exported_by"] == _actor()


@app.route("/exports")
@access.require("investigator")
def exports():
    sup = access.at_least("supervisor")
    with db() as c:
        rows = c.execute("""SELECT id, created, export_ref, case_ref, purpose, exported_by, jsonb_array_length(items), has_original, package_bytes,
                                   (SELECT count(*) FROM custody_log l WHERE l.export_id = e.id AND l.action = 'downloaded'), tsa IS NOT NULL
                            FROM evidence_exports e WHERE %s OR exported_by = %s ORDER BY id DESC LIMIT 300""", (sup, _actor())).fetchall()
        chain = evx.verify_chain(_chain(c)) if sup else None
    body = render_template_string("""<h2>Evidence packages <small>{{ 'all' if sup else 'yours' }}, {{ rows|length }} shown</small></h2>
{% if chain %}{% if chain.ok %}<div class="notice">Custody log intact: {{ chain.entries }} entr{{ 'y' if chain.entries == 1 else 'ies' }}.
 Head hash <code>{{ chain.head }}</code> &middot; <a href="{{ url_for('custody_json') }}">Download the log (JSON)</a></div>
{% else %}<div class="notice warn-box" role="alert"><b>The custody log has been altered.</b> The chain breaks at entry {{ chain.broken_at }}.
 <a href="{{ url_for('custody_json') }}">Download the log (JSON)</a></div>{% endif %}{% endif %}
<p class="muted">To export, open a sighting or an identity and choose “Export evidence package”. Signing key
<code>{{ fp }}</code> &middot; <a href="{{ url_for('public_key') }}">public key</a>.</p>
<table><tr><th>Export</th><th>Created</th><th>Case</th><th>Purpose</th><th>By</th><th>Items</th><th>Original clip</th><th>Timestamp</th><th>Size</th><th>Downloads</th></tr>
{% for id, t, ref, case, purpose, by, nitems, orig, size, dl, ts in rows %}<tr><td><a href="{{ url_for('export_view', xid=id) }}"><b>{{ ref }}</b></a></td>
<td>{{ when(t) }}</td><td>{{ case }}</td><td>{{ purpose }}</td><td>{{ by }}</td><td>{{ nitems }}</td>
<td>{% if orig %}included{% else %}<span class="flag">not available</span>{% endif %}</td><td>{{ 'outside' if ts else '–' }}</td>
<td>{{ '%.1f'|format(size / 1e6) }} MB</td><td>{{ dl }}</td></tr>{% endfor %}</table>""",
                                  rows=rows, chain=chain, when=when, sup=sup, fp=_fingerprint())
    return render_template_string(PAGE, body=body)


def _fingerprint() -> str:
    try:
        return signer().fingerprint
    except Exception:
        return "not available"


@app.route("/exports/public-key.pem")
def public_key():
    return Response(signer().public_pem, 200, {"Content-Type": "application/x-pem-file",
                                               "Content-Disposition": "attachment; filename=squirrel-guard-signing-key.pem"})


@app.route("/export/<int:xid>")
@access.require("investigator")
def export_view(xid):
    with db() as c:
        x = _export_row(c, xid)
        if not x or not _may_see(x):
            abort(404)
        log = _chain(c, export_id=xid)
        chain = evx.verify_chain(_chain(c))
        rel = c.execute("SELECT id, created, created_by, package_bytes, summary FROM release_copies WHERE export_id = %s ORDER BY id", (xid,)).fetchall()
    m = x["manifest"]
    body = render_template_string("""<h2>{{ x.export_ref }} <small>case {{ x.case_ref }} &middot; {{ x.purpose }}</small></h2>
{% if not x.has_original %}<div class="notice warn-box" role="alert"><b>No original event clip is in this package.</b>
 {% for it in m['items'] %}{% for t in it.missing %}{{ t }}. {% endfor %}{% endfor %}</div>{% endif %}
{% set unrev = m['items']|selectattr('review', 'none')|list|length %}{% if unrev %}<div class="unconfirmed">{{ unrev }} of {{ m['items']|length }} item{{ 's' if m['items']|length != 1 }} NOT confirmed by a person.</div>{% endif %}
{% if not chain.ok %}<div class="notice warn-box" role="alert"><b>The custody log has been altered</b> (chain breaks at entry {{ chain.broken_at }}).</div>{% endif %}
<div class="stat-row"><div class="stat"><b>{{ m['items']|length }}</b><span>item{{ 's' if m['items']|length != 1 }}</span></div>
<div class="stat"><b>{{ '%.1f'|format(x.package_bytes / 1e6) }} MB</b><span>package, {{ m.files|length }} files each hashed</span></div>
<div class="stat"><b>{{ x.exported_by }}</b><span>{{ x.exported_role }}, {{ when(x.created) }}</span></div>
<div class="stat"><b>{{ 'Outside' if x.tsa else 'None' }}</b><span>timestamp{% if x.tsa %} {{ x.tsa.gen_time }}{% endif %}</span></div></div>
<div class="actions"><a class="btn primary" href="{{ url_for('export_download', xid=x.id) }}">Download package</a>
<form method="post" action="{{ url_for('export_verify', xid=x.id) }}"><button>Check integrity now</button></form>
<form method="post" action="{{ url_for('release_new', xid=x.id) }}" data-confirm-title="Make a release copy?" data-confirm="A second package with every face and number plate blurred is made from this one, for disclosure. It is linked to this package by hash; the original stays as it is." data-confirm-kind="warn" data-confirm-icon="info"><button>Make release copy (redacted)</button></form></div>
<h3>Clocks</h3><table>{% for k, label in [('export_host_text', 'Export host vs outside time'), ('recorder_text', 'Event recorder'), ('sealer_text', 'Continuous recorder'), ('camera_text', 'Cameras')] %}
<tr><th>{{ label }}</th><td>{{ (x.clock or {}).get(k, '') }}</td></tr>{% endfor %}</table>
<h3>Items</h3><table><tr><th>#</th><th>When</th><th>Camera</th><th>Filed under</th><th>Review</th><th>Continuous recording</th></tr>
{% for it in m['items'] %}<tr><td>{{ it.n }}</td><td><a href="{{ url_for('sighting', sid=it.subject.sighting) }}">{{ iso_when(it.subject.start) }}</a></td><td>{{ it.subject.camera }}</td><td>{{ it.subject.identity }}</td>
<td>{% if it.review and it.review.outcome == 'confirmed' %}<span class="k known">confirmed</span> {{ it.review.reviewer }}{% elif it.review %}<span class="flag">{{ it.review.outcome }}</span>{% else %}<span class="flag">not reviewed</span>{% endif %}</td>
<td>{% for s in it.segments %}<div><code>{{ s.file.split('/')[-1] }}</code> <span class="{{ 'seg-ok' if s.status == 'sealed' else ('seg-bad' if s.status == 'differs' else 'seg-none') }}">{{ s.status_text }}</span>{% if s.included %} (included){% endif %}</div>{% else %}<small>none</small>{% endfor %}</td></tr>{% endfor %}</table>
<h3>Hashes and signature</h3>
<table><tr><th>Package file</th><td><code>{{ x.package_sha256 }}</code></td></tr><tr><th>manifest.json</th><td><code>{{ x.manifest_sha256 }}</code></td></tr>
<tr><th>Signing key</th><td><code>{{ x.signing_fingerprint }}</code> (Ed25519)</td></tr>
{% if x.tsa %}<tr><th>Outside timestamp</th><td>{{ x.tsa.gen_time }} from {{ x.tsa.url }}, serial {{ x.tsa.serial }}</td></tr>{% endif %}</table>
<details><summary>Files in the package ({{ m.files|length }})</summary><table><tr><th>File</th><th>Role</th><th>Bytes</th><th>SHA-256</th></tr>
{% for f in m.files %}<tr><td><code>{{ f.path }}</code></td><td>{{ f.role }}</td><td>{{ '{:,}'.format(f.bytes) }}</td><td><code style="word-break:break-all">{{ f.sha256 }}</code></td></tr>{% endfor %}</table></details>
{% if rel %}<h3>Release copies</h3><table><tr><th>Made</th><th>By</th><th>Blurred</th><th>Size</th><th></th></tr>
{% for rid, t, by, size, sm in rel %}<tr><td>{{ when(t) }}</td><td>{{ by }}</td><td>{{ sm.faces }} face{{ 's' if sm.faces != 1 }}, {{ sm.plates }} plate{{ 's' if sm.plates != 1 }} in {{ sm.files }} file{{ 's' if sm.files != 1 }}</td>
<td>{{ '%.1f'|format(size / 1e6) }} MB</td><td><a href="{{ url_for('release_download', rid=rid) }}">Download</a></td></tr>{% endfor %}</table>{% endif %}
<h3>Chain of custody <small>every entry's hash covers the one before it</small></h3>
<table><tr><th>When (UTC)</th><th>Who</th><th>What</th><th>Detail</th><th>From</th><th>Entry hash</th></tr>
{% for l in log %}<tr><td><small>{{ l.ts }}</small></td><td>{{ l.actor }}</td><td>{% if l.action == 'verify_failed' %}<span class="flag">{{ actions[l.action] }}</span>{% else %}{{ actions.get(l.action, l.action) }}{% endif %}</td>
<td><small style="word-break:break-all">{{ l.detail }}</small></td><td><small>{{ l.remote }}</small></td><td><code title="{{ l.entry_hash }}">{{ l.entry_hash[:16] }}&hellip;</code></td></tr>{% endfor %}</table>""",
                                  x=x, m=m, log=log, chain=chain, when=when, actions=evx.ACTIONS, rel=rel,
                                  iso_when=lambda t: when(dt.datetime.fromisoformat(t)) if t else "")
    return render_template_string(PAGE, body=body)


@app.route("/export/<int:xid>/download")
@access.require("investigator")
def export_download(xid):
    with db() as c:
        x = _export_row(c, xid)
        if not x or not _may_see(x) or not (STORE / x["package_file"]).is_file():
            abort(404)
        custody(c, xid, "downloaded", f"package sha256 {x['package_sha256']}")
    return send_from_directory(STORE, x["package_file"], as_attachment=True, max_age=0)


@app.route("/export/<int:xid>/verify", methods=["POST"])
@access.require("investigator")
def export_verify(xid):
    with db() as c:
        x = _export_row(c, xid)
        if not x or not _may_see(x):
            abort(404)
        problems = evx.verify_package(STORE / x["package_file"], x["package_sha256"], x["manifest_sha256"], signer().public_pem,
                                      tsalib.verify_openssl if x["tsa"] else None)
        custody(c, xid, "verify_failed" if problems else "verified",
                "; ".join(problems) or "stored package matches its recorded hashes, signature" + (" and outside timestamp" if x["tsa"] else ""))
    return redirect(url_for("export_view", xid=xid, done="verify_failed" if problems else "verified"))


@app.route("/exports/custody.json")
@access.require("supervisor")
def custody_json():
    with db() as c:
        rows = _chain(c)
    return _json({"format": "squirrel-guard-custody/1", "exported_utc": evx.stamp(evx.utcnow()), "check": evx.verify_chain(rows),
                  "hash_rule": "entry_hash = sha256(prev_hash + '\\n' + canonical JSON of {action, actor, detail, export_id, remote, ts}); "
                               "first prev_hash is 64 zeros", "entries": rows})


# ---------------------------------------------------------------- audit log (item 6)
@app.route("/audit")
@access.require("supervisor")
def audit_page():
    actor, action = request.args.get("actor", "").strip(), request.args.get("action", "").strip()
    with db() as c:
        rows = c.execute("""SELECT id, ts, actor, role, action, target, detail, remote, entry_hash FROM audit_log
                            WHERE (%s = '' OR actor = %s) AND (%s = '' OR action LIKE %s) ORDER BY id DESC LIMIT 500""",
                         (actor, actor, action, action + "%")).fetchall()
        chain = evx.verify_chain(_chain(c, "audit_log"), access.AUDIT_FIELDS)
        actors = [r[0] for r in c.execute("SELECT DISTINCT actor FROM audit_log ORDER BY 1").fetchall()]
    body = render_template_string("""<h2>Audit log <small>who looked at what</small></h2>
{% if chain.ok %}<div class="notice">Audit log intact: {{ chain.entries }} entries. Head hash <code>{{ chain.head }}</code> &middot;
<a href="{{ url_for('audit_json') }}">Download (JSON)</a></div>{% else %}<div class="notice warn-box" role="alert"><b>The audit log has been altered.</b>
The chain breaks at entry {{ chain.broken_at }}. <a href="{{ url_for('audit_json') }}">Download (JSON)</a></div>{% endif %}
<form class="actions" method="get"><select name="actor" aria-label="User"><option value="">Everyone</option>{% for a in actors %}<option{% if a == actor %} selected{% endif %}>{{ a }}</option>{% endfor %}</select>
<select name="action" aria-label="Kind"><option value="">Everything</option>{% for k, label in [('view:search', 'Searches'), ('view:', 'Views'), ('change:', 'Changes'), ('review', 'Reviews'), ('denied', 'Refusals')] %}<option value="{{ k }}"{% if k == action %} selected{% endif %}>{{ label }}</option>{% endfor %}</select>
<button>Filter</button></form>
<table><tr><th>When (UTC)</th><th>Who</th><th>Role</th><th>What</th><th>Target</th><th>Detail</th><th>From</th></tr>
{% for id, ts, a, ro, ac, tg, de, rm, h in rows %}<tr><td><small>{{ ts }}</small></td><td>{{ a }}</td><td>{{ ro }}</td>
<td>{% if ac == 'denied' %}<span class="flag">refused</span>{% else %}{{ ac }}{% endif %}</td><td><code>{{ tg }}</code></td><td><small>{{ de }}</small></td><td><small>{{ rm }}</small></td></tr>{% endfor %}</table>""",
                                  rows=rows, chain=chain, actors=actors, actor=actor, action=action)
    return render_template_string(PAGE, body=body)


@app.route("/audit.json")
@access.require("supervisor")
def audit_json():
    with db() as c:
        rows = _chain(c, "audit_log")
    return _json({"format": "squirrel-guard-audit/1", "exported_utc": evx.stamp(evx.utcnow()),
                  "check": evx.verify_chain(rows, access.AUDIT_FIELDS),
                  "hash_rule": "entry_hash = sha256(prev_hash + '\\n' + canonical JSON of {action, actor, detail, remote, role, target, ts})",
                  "entries": rows})


def _frigate_file(path: str, dest: Path, actor: str) -> tuple[str, int, float | None]:
    """Download one event-recorder file unchanged. -> (sha256, bytes, recorder clock minus ours in seconds, or None)."""
    import email.utils
    import ssl
    import time as _t
    import urllib.request
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(E.get("FRIGATE_API", "https://192.0.2.30:8971") + path, headers={
        "X-Proxy-Secret": E.get("FRIGATE_PROXY_SECRET", ""), "Remote-User": actor, "Remote-Groups": "admins"})
    t0 = _t.time()
    with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
        t1 = _t.time()
        off = None
        if r.headers.get("Date"):
            try:
                off = round(email.utils.parsedate_to_datetime(r.headers["Date"]).timestamp() - (t0 + t1) / 2, 1)
            except (TypeError, ValueError):
                off = None
        digest, n = evx.save_stream(r, dest)
    return digest, n, off


# ---------------------------------------------------------------- redacted release copies (item 9)
RELEASES: dict = {}


@app.route("/export/<int:xid>/release", methods=["POST"])
@access.require("investigator")
def release_new(xid):
    with db() as c:
        x = _export_row(c, xid)
        if not x or not _may_see(x):
            abort(404)
    who = {"actor": _actor(), "remote": request.remote_addr or ""}
    if RELEASES.get(xid) == "running":
        return redirect(url_for("export_view", xid=xid, done="release_running"))
    RELEASES[xid] = "running"

    def work():
        import tempfile
        try:
            import redact
            now = evx.utcnow()
            with tempfile.TemporaryDirectory(dir=STORE, prefix=".tmp-") as tmp, db() as c:
                rid = c.execute("SELECT nextval('release_copies_id_seq')").fetchone()[0]
                fname = f"{x['export_ref']}-R{rid}.zip"
                res = redact.make_release(STORE / x["package_file"], STORE / fname,
                                          {"release_id": f"{x['export_ref']}-R{rid}", "created_utc": evx.stamp(now), "created_by": who["actor"]},
                                          signer(), Path(tmp))
                from psycopg.types.json import Jsonb
                c.execute("""INSERT INTO release_copies (id, export_id, created, created_by, package_file, package_sha256, package_bytes,
                                                         manifest_sha256, summary) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                          (rid, xid, now, who["actor"], fname, res["package_sha256"], res["package_bytes"], res["manifest_sha256"], Jsonb(res["summary"])))
                s = res["summary"]
                custody(c, xid, "release_created", f"release {fname}: {s['faces']} face and {s['plates']} plate detections blurred in "
                                                   f"{s['files']} files; package sha256 {res['package_sha256']}", actor=who["actor"], remote=who["remote"])
            RELEASES[xid] = "done"
        except Exception as e:
            app.logger.exception("release copy failed")
            RELEASES[xid] = f"failed: {type(e).__name__}: {e}"
    threading.Thread(target=work, daemon=True).start()
    return redirect(url_for("export_view", xid=xid, done="release_started"))


@app.route("/release/<int:rid>/download")
@access.require("investigator")
def release_download(rid):
    with db() as c:
        r = c.execute("SELECT export_id, package_file, package_sha256 FROM release_copies WHERE id = %s", (rid,)).fetchone()
        x = _export_row(c, r[0]) if r else None
        if not r or not _may_see(x) or not (STORE / r[1]).is_file():
            abort(404)
        custody(c, r[0], "release_downloaded", f"release {r[1]} sha256 {r[2]}")
    return send_from_directory(STORE, r[1], as_attachment=True, max_age=0)


@app.route("/crop/<path:f>")
def crop(f):
    if "/" in f or not f.endswith(".jpg"):
        abort(404)
    return send_from_directory(CROPS, f)


if __name__ == "__main__":
    from waitress import serve
    serve(app, host=E.get("HOST", "0.0.0.0"), port=int(E.get("PORT", "8092")), trusted_proxy="127.0.0.1",
          trusted_proxy_headers={"x-forwarded-proto", "x-forwarded-host", "x-forwarded-for"}, clear_untrusted_proxy_headers=True)

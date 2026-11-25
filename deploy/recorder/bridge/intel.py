"""Phase 3/4: behaviour updates, alert rules and LLM-written reports (facts from the database only).

Alerts apply to identities/vehicles of kind 'unknown' or 'watch'; 'household' and 'known' never alert.
"""
import datetime as dt
import html
import json
import logging
import os
import time
from zoneinfo import ZoneInfo

import requests

import routine

log = logging.getLogger("camstack-intel")
E = os.environ
TZ = ZoneInfo(E.get("LOCAL_TZ", "America/Denver"))
ALERT_KINDS = ("unknown", "watch")
NIGHT = (E.get("NIGHT_START", "21:30"), E.get("NIGHT_END", "07:00"))
LINGER_MIN = float(E.get("LINGER_MIN", "5"))
PARKED_MIN = float(E.get("PARKED_MIN", "30"))
REPEAT_N = int(E.get("REPEAT_N", "3"))
RECURRING_DAYS = int(E.get("RECURRING_DAYS", "3"))
ALERT_ZONES = {z.strip() for z in E.get("ALERT_ZONES", "").split(",") if z.strip()}
UI = E.get("PERSONS_UI_URL", "https://persons.cam.example.org")
LLM = E.get("REPORT_LLM_URL", "http://192.0.2.30:5922")
STARTED = dt.datetime.now(dt.timezone.utc)  # no back-filled alerts for sightings older than this process


def is_night(t: dt.datetime) -> bool:
    hm = t.astimezone(TZ).strftime("%H:%M")
    s, e = NIGHT
    return (hm >= s or hm < e) if s > e else (s <= hm < e)


# ---------------------------------------------------------------- alerts
def raise_alert(conn, ntfy, kind, subject, key, message, sighting=None, crop_path=None, click=None):
    row = conn.execute("INSERT INTO alerts (kind, subject, dedupe_key, message, sighting) VALUES (%s,%s,%s,%s,%s) "
                       "ON CONFLICT (dedupe_key) DO NOTHING RETURNING id", (kind, subject, key, message, sighting)).fetchone()
    if not row:
        return
    try:
        ntfy(kind, message, crop_path, click)
        conn.execute("UPDATE alerts SET sent = true WHERE id = %s", (row[0],))
    except Exception as e:
        log.warning("alert ntfy failed: %s", e)
    log.info("ALERT %s %s: %s", kind, subject, message)
    import explain
    explain.record(conn, "alert", sighting, subject, band=None, reasons=[explain.ALERT_REASONS.get(kind, kind.upper())],
                   message=message, scores={})


def person_name(iid, label):
    return label or f"Unknown #{iid}"


def on_person_sighting(conn, ntfy, snap_dir, frigate_event):
    r = conn.execute("""SELECT s.identity_id, i.label, i.kind, s.ts_start, s.crop_file, s.attributes->>'description'
                        FROM sightings s JOIN identities i ON i.id = s.identity_id WHERE s.frigate_event = %s""",
                     (frigate_event,)).fetchone()
    if not r or r[2] not in ALERT_KINDS:
        return
    iid, label, kind, ts, crop, desc = r
    crop_path = snap_dir / "persons" / crop if crop else None
    click = f"{UI}/identity/{iid}"
    who = person_name(iid, label) + (" (watch list)" if kind == "watch" else "")
    local = ts.astimezone(TZ)
    if is_night(ts):
        raise_alert(conn, ntfy, "night_unknown", f"person:{iid}", f"night:{frigate_event}",
                    f"{who} at {local:%H:%M} (night). {desc or ''}", frigate_event, crop_path, click)
    n24 = conn.execute("SELECT count(*) FROM sightings WHERE identity_id = %s AND ts_start > %s - interval '24 hours'",
                       (iid, ts)).fetchone()[0]
    if n24 >= REPEAT_N:
        raise_alert(conn, ntfy, "repeat_unknown", f"person:{iid}", f"repeat:{iid}:{local:%Y-%m-%d}",
                    f"{who} seen {n24} times in 24 h (latest {local:%H:%M}).", frigate_event, crop_path, click)


def on_vehicle_sighting(conn, ntfy, snap_dir, frigate_event):
    r = conn.execute("""SELECT vs.vehicle_id, v.label, v.kind, v.plate, v.color, v.body_type, vs.ts_start, vs.crop_file
                        FROM vehicle_sightings vs JOIN vehicles v ON v.id = vs.vehicle_id WHERE vs.frigate_event = %s""",
                     (frigate_event,)).fetchone()
    if not r or r[2] not in ALERT_KINDS:
        return
    vid, label, kind, plate, color, body, ts, crop = r
    days = conn.execute("SELECT count(DISTINCT (ts_start AT TIME ZONE %s)::date) FROM vehicle_sightings "
                        "WHERE vehicle_id = %s AND ts_start > %s - interval '7 days'", (str(TZ), vid, ts)).fetchone()[0]
    if days >= RECURRING_DAYS:
        what = label or f"Vehicle #{vid} {color or ''} {body or ''}".strip()
        raise_alert(conn, ntfy, "recurring_vehicle", f"vehicle:{vid}", f"recurring:{vid}:{ts.astimezone(TZ):%Y-%m-%d}",
                    f"{what}{' plate ' + plate if plate else ''} seen on {days} different days this week.",
                    frigate_event, snap_dir / "vehicles" / crop if crop else None, f"{UI}/vehicle/{vid}")


FALL_MIN = float(E.get("FALL_ALERT_MIN", "0.6"))
# Open-vocabulary objects (profiler "objects"): alert when an unknown/watch-list person CARRIES one of these labels.
# Empty = off. Labels as exported (scripts/openvocab_vocab.txt), e.g. "ladder,pry bar,bolt cutters,gun,knife,gas can".
WATCH_OBJECTS = {o.strip().lower() for o in E.get("WATCH_OBJECTS", "").split(",") if o.strip()}
WATCH_OBJECT_MIN = float(E.get("WATCH_OBJECT_MIN", "0.3"))   # max per-frame score over the visit


def watched_objects(objects) -> list:
    """Carried objects from the watch list, best first (the profiler already applied its min-frames rule)."""
    return [o for o in objects or [] if o.get("carried") and (o.get("label") or "").lower() in WATCH_OBJECTS
            and float(o.get("score") or 0) >= WATCH_OBJECT_MIN]


def on_behavior(conn, ntfy, snap_dir, kind, frigate_event, b):
    fall = next((a for a in b.get("actions") or [] if a.get("tag") == "fall" and a.get("p", 0) >= FALL_MIN), None)
    if kind == "person" and fall:   # safety: alert for ANYONE, household included
        r = conn.execute("""SELECT s.identity_id, i.label, s.crop_file FROM sightings s JOIN identities i
                            ON i.id = s.identity_id WHERE s.frigate_event = %s""", (frigate_event,)).fetchone()
        if r:
            raise_alert(conn, ntfy, "possible_fall", f"person:{r[0]}", f"fall:{frigate_event}",
                        f"{person_name(r[0], r[1])} may have fallen ({fall['p'] * 100:.0f}%, {fall['t']:.0f} s into the "
                        f"visit). Check the camera.", frigate_event, snap_dir / "persons" / r[2] if r[2] else None,
                        f"{UI}/identity/{r[0]}")
    if kind == "person":
        try:
            rr = score_routine(conn, frigate_event, b)
        except Exception:                                     # the routine model must never break alerting
            log.exception("routine score failed for %s", frigate_event)
            rr = None
        if rr and rr["unusual"]:
            r = conn.execute("""SELECT s.identity_id, i.label, i.kind, s.crop_file FROM sightings s JOIN identities i
                                ON i.id = s.identity_id WHERE s.frigate_event = %s""", (frigate_event,)).fetchone()
            if r and r[2] in ALERT_KINDS:
                raise_alert(conn, ntfy, "unusual_routine", f"person:{r[0]}", f"routine:{frigate_event}",
                            f"Unusual visit by {person_name(r[0], r[1])}: " + "; ".join(rr["reasons"][:3]) + ".",
                            frigate_event, snap_dir / "persons" / r[3] if r[3] else None, f"{UI}/identity/{r[0]}")
    if kind == "person" and ALERT_ZONES & set(b.get("entered_zones") or []):
        r = conn.execute("""SELECT s.identity_id, i.label, i.kind, s.crop_file FROM sightings s JOIN identities i
                            ON i.id = s.identity_id WHERE s.frigate_event = %s""", (frigate_event,)).fetchone()
        if r and r[2] in ALERT_KINDS:
            zones = ", ".join(sorted(ALERT_ZONES & set(b["entered_zones"])))
            raise_alert(conn, ntfy, "zone_unknown", f"person:{r[0]}", f"zone:{frigate_event}",
                        f"{person_name(r[0], r[1])} entered {zones}.", frigate_event,
                        snap_dir / "persons" / r[3] if r[3] else None, f"{UI}/identity/{r[0]}")
    carried = watched_objects(b.get("objects")) if kind == "person" and WATCH_OBJECTS else []
    if carried:
        r = conn.execute("""SELECT s.identity_id, i.label, i.kind, s.crop_file FROM sightings s JOIN identities i
                            ON i.id = s.identity_id WHERE s.frigate_event = %s""", (frigate_event,)).fetchone()
        if r and r[2] in ALERT_KINDS:
            what = ", ".join(f"{o['label']} ({o['score'] * 100:.0f}%, {o['frames']} frames)" for o in carried[:3])
            raise_alert(conn, ntfy, "watched_object", f"person:{r[0]}", f"object:{frigate_event}",
                        f"{person_name(r[0], r[1])} appears to be carrying: {what}. Detector guess; check the clip.",
                        frigate_event, snap_dir / "persons" / r[3] if r[3] else None, f"{UI}/identity/{r[0]}")


def score_routine(conn, frigate_event: str, b: dict) -> dict | None:
    """Score one person visit against the last 90 days of visits (routine.Routine) and store it on the sighting."""
    row = conn.execute("SELECT ts_start FROM sightings WHERE frigate_event = %s", (frigate_event,)).fetchone()
    if not row:
        return None
    hist = [{"ts": t, "behavior": hb} for t, hb in conn.execute(
        """SELECT ts_start, behavior FROM sightings WHERE behavior IS NOT NULL AND frigate_event <> %s
           AND ts_start > now() - interval '90 days'""", (frigate_event,)).fetchall()]
    res = routine.Routine(hist, TZ).score(row[0], b)
    conn.execute("UPDATE sightings SET routine = %s WHERE frigate_event = %s", (json.dumps(res), frigate_event))
    return res


def periodic_checks(conn, ntfy, snap_dir):
    """Lingering people / parked vehicles, including events that are still open (stationary objects never end)."""
    for iid, label, kind, ev, crop, mins in conn.execute(
            """SELECT s.identity_id, i.label, i.kind, s.frigate_event, s.crop_file,
                      EXTRACT(epoch FROM COALESCE(s.ts_end, now()) - s.ts_start) / 60
               FROM sightings s JOIN identities i ON i.id = s.identity_id
               WHERE i.kind = ANY(%s) AND s.ts_start > GREATEST(now() - interval '1 day', %s)""",
            (list(ALERT_KINDS), STARTED)).fetchall():
        if mins >= LINGER_MIN:
            raise_alert(conn, ntfy, "linger_unknown", f"person:{iid}", f"linger:{ev}",
                        f"{person_name(iid, label)} has stayed about {mins:.0f} min.", ev,
                        snap_dir / "persons" / crop if crop else None, f"{UI}/identity/{iid}")
    for vid, label, kind, plate, color, body, ev, crop, mins in conn.execute(
            """SELECT vs.vehicle_id, v.label, v.kind, v.plate, v.color, v.body_type, vs.frigate_event, vs.crop_file,
                      EXTRACT(epoch FROM COALESCE(vs.ts_end, now()) - vs.ts_start) / 60
               FROM vehicle_sightings vs JOIN vehicles v ON v.id = vs.vehicle_id
               WHERE v.kind = ANY(%s) AND vs.ts_start > GREATEST(now() - interval '2 days', %s)""",
            (list(ALERT_KINDS), STARTED)).fetchall():
        if mins >= PARKED_MIN:
            what = label or f"Vehicle #{vid} {color or ''} {body or ''}".strip()
            raise_alert(conn, ntfy, "parked_vehicle", f"vehicle:{vid}", f"parked:{ev}",
                        f"{what}{' plate ' + plate if plate else ''} parked about {mins:.0f} min.", ev,
                        snap_dir / "vehicles" / crop if crop else None, f"{UI}/vehicle/{vid}")


# ---------------------------------------------------------------- reports
def gather_facts(conn, since, until):
    q = lambda sql, *a: conn.execute(sql, a).fetchall()
    f = {"period": {"from": since.astimezone(TZ).isoformat(timespec="minutes"),
                    "to": until.astimezone(TZ).isoformat(timespec="minutes")}}
    f["people"] = dict(zip(("sightings", "distinct_identities", "night_sightings"), q(
        """SELECT count(*), count(DISTINCT identity_id),
                  count(*) FILTER (WHERE (ts_start AT TIME ZONE %s)::time >= %s::time OR (ts_start AT TIME ZONE %s)::time < %s::time)
           FROM sightings WHERE ts_start >= %s AND ts_start < %s""", str(TZ), NIGHT[0], str(TZ), NIGHT[1], since, until)[0]))
    f["people"]["new_identities"] = q("SELECT count(*) FROM identities WHERE first_seen >= %s AND first_seen < %s", since, until)[0][0]
    f["people"]["by_kind"] = {k: n for k, n in q("""SELECT i.kind, count(*) FROM sightings s JOIN identities i ON i.id = s.identity_id
                                                     WHERE s.ts_start >= %s AND s.ts_start < %s GROUP BY 1""", since, until)}
    f["people"]["top_unconfirmed"] = [
        {"identity": person_name(i, l), "kind": k, "sightings": n, "first": a.astimezone(TZ).strftime("%m-%d %H:%M"),
         "last": b.astimezone(TZ).strftime("%m-%d %H:%M"), "apparent": d,
         **({"estimated_height_cm": round(h)} if h else {})}
        for i, l, k, n, a, b, d, h in q("""SELECT i.id, i.label, i.kind, count(*), min(s.ts_start), max(s.ts_start),
                                          (array_agg(s.attributes->>'description' ORDER BY s.ts_start DESC))[1],
                                          (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY x.height_cm)
                                             FROM sightings x WHERE x.identity_id = i.id)
                                        FROM sightings s JOIN identities i ON i.id = s.identity_id
                                        WHERE s.ts_start >= %s AND s.ts_start < %s AND i.kind = ANY(%s)
                                        GROUP BY i.id ORDER BY count(*) DESC LIMIT 8""", since, until, list(ALERT_KINDS))]
    f["vehicles"] = dict(zip(("sightings", "distinct_vehicles", "with_plate"), q(
        "SELECT count(*), count(DISTINCT vehicle_id), count(*) FILTER (WHERE plate IS NOT NULL) FROM vehicle_sightings "
        "WHERE ts_start >= %s AND ts_start < %s", since, until)[0]))
    f["vehicles"]["new_vehicles"] = q("SELECT count(*) FROM vehicles WHERE first_seen >= %s AND first_seen < %s", since, until)[0][0]
    f["vehicles"]["top_unconfirmed"] = [
        {"vehicle": l or f"Vehicle #{i}", "plate": p, "look": f"{c or ''} {bt or ''} {mm or ''}".strip(), "sightings": n}
        for i, l, p, c, bt, mm, n in q("""SELECT v.id, v.label, v.plate, v.color, v.body_type, v.make_model, count(*)
                                          FROM vehicle_sightings vs JOIN vehicles v ON v.id = vs.vehicle_id
                                          WHERE vs.ts_start >= %s AND vs.ts_start < %s AND v.kind = ANY(%s)
                                          GROUP BY v.id ORDER BY count(*) DESC LIMIT 8""", since, until, list(ALERT_KINDS))]
    f["squirrels"] = dict(zip(("events", "confirmed", "deterrent_fired", "quiet_hours"), q(
        """SELECT count(*), count(*) FILTER (WHERE confirmed), count(*) FILTER (WHERE action = 'fired'),
                  count(*) FILTER (WHERE action = 'quiet') FROM events
           WHERE lower(label) = 'squirrel' AND camera NOT LIKE '%%\\_test' AND ts >= %s AND ts < %s""", since, until)[0]))
    f["alerts"] = {k: n for k, n in q("SELECT kind, count(*) FROM alerts WHERE ts >= %s AND ts < %s GROUP BY 1", since, until)}
    return f


REPORT_PROMPT = (
    "Write a short security log summary for a homeowner from the JSON facts below. Use ONLY these facts; do not "
    "speculate about intent, identity or anything not in the data. 'apparent' descriptions are model guesses, say so "
    "if you mention them. Structure: one-sentence overview, then 'People', 'Vehicles', 'Squirrels', 'Alerts' "
    "(skip a section if empty), then 'Worth reviewing' with at most 3 concrete items. Plain text, under 250 words.\n\nFACTS:\n")


def write_report(facts):
    body = {"temperature": 0.2, "max_tokens": 700, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": REPORT_PROMPT + json.dumps(facts, indent=1, default=str)}]}
    try:
        r = requests.post(LLM.rstrip("/") + "/v1/chat/completions", json=body, timeout=180)
        r.raise_for_status()
        return (r.json()["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        log.warning("report LLM failed: %s", e)
        return None


def run_report(db, period, send_mail_fn, ntfy):
    until = dt.datetime.now(dt.timezone.utc)
    since = until - (dt.timedelta(days=7) if period == "weekly" else dt.timedelta(days=1))
    with db() as conn:
        facts = gather_facts(conn, since, until)
    text = write_report(facts) or "(LLM unavailable; facts only)"
    from psycopg.types.json import Jsonb
    with db() as conn:
        rid = conn.execute("INSERT INTO reports (period, since, until, facts, text) VALUES (%s,%s,%s,%s,%s) RETURNING id",
                           (period, since, until, Jsonb(facts), text)).fetchone()[0]
    title = f"{period.capitalize()} security report ({until.astimezone(TZ):%Y-%m-%d})"
    try:
        ntfy("report", f"{title}: {text[:300]}", None, f"{UI}/report/{rid}")
    except Exception as e:
        log.warning("report ntfy failed: %s", e)
    if send_mail_fn:
        try:
            send_mail_fn(title, text, facts, f"{UI}/report/{rid}")
        except Exception as e:
            log.warning("report email failed: %s", e)
    log.info("REPORT %s #%s written", period, rid)
    return rid


def scheduler(db, send_mail_fn, ntfy, snap_dir):
    """Every 5 min: lingering/parked checks. Daily report at REPORT_AT (Mondays also weekly)."""
    at = E.get("REPORT_AT", "07:00")
    last_day = None
    while True:
        try:
            with db() as conn:
                periodic_checks(conn, ntfy, snap_dir)
            now = dt.datetime.now(TZ)
            if now.strftime("%H:%M") >= at and last_day != now.date():
                with db() as conn:
                    done = conn.execute("SELECT 1 FROM reports WHERE period = 'daily' AND (ts AT TIME ZONE %s)::date = %s",
                                        (str(TZ), now.date())).fetchone()
                if not done:
                    run_report(db, "daily", send_mail_fn, ntfy)
                    if now.weekday() == 0:
                        run_report(db, "weekly", send_mail_fn, ntfy)
                last_day = now.date()
        except Exception:
            log.exception("scheduler pass failed")
        time.sleep(300)


def report_html(title, text, facts, link):
    p, v, s = facts.get("people", {}), facts.get("vehicles", {}), facts.get("squirrels", {})
    rows = [("People sightings", p.get("sightings")), ("Distinct people", p.get("distinct_identities")),
            ("New people", p.get("new_identities")), ("Night sightings", p.get("night_sightings")),
            ("Vehicle sightings", v.get("sightings")), ("Distinct vehicles", v.get("distinct_vehicles")),
            ("Plates read", v.get("with_plate")), ("Squirrels confirmed", s.get("confirmed")),
            ("Deterrent fired", s.get("deterrent_fired")), ("Alerts", sum((facts.get("alerts") or {}).values()))]
    table = "".join(f"<tr><td style='padding:3px 12px 3px 0;color:#666'>{html.escape(k)}</td><td><b>{v}</b></td></tr>" for k, v in rows)
    return (f"<div style='font-family:system-ui,sans-serif;max-width:720px'><h2>{html.escape(title)}</h2>"
            f"<pre style='white-space:pre-wrap;font-family:inherit'>{html.escape(text)}</pre>"
            f"<table style='margin:12px 0'>{table}</table>"
            f"<p><a href='{html.escape(link)}' style='background:#1a73e8;color:#fff;padding:9px 16px;border-radius:6px;"
            f"text-decoration:none'>Open report</a></p><p style='color:#888;font-size:12px'>Written by the local LLM from "
            f"stored facts only; apparent age/gender are model guesses.</p></div>")

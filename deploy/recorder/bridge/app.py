"""camstack bridge: MQTT squirrelguard/+/event -> Postgres row + snapshot file + ntfy push + optional email.

Notifications go out only for confirmed events, and are rate-limited per channel so a squirrel that
hangs around does not produce a flood. Email is off until SMTP_HOST is set (Proton Mail Bridge:
127.0.0.1:1025, STARTTLS, Bridge-generated password, self-signed cert -> SMTP_TLS_VERIFY=0).
"""
from __future__ import annotations

import vehmatch  # noqa: E402  (explained vehicle matching; copy of squirrel_guard/vehmatch.py)
import base64
import json
import logging
import os
import smtplib
import ssl
import time
from email.message import EmailMessage
from pathlib import Path

import paho.mqtt.client as mqtt
import psycopg
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("camstack-bridge")
E = os.environ


def mail_to() -> str:
    """SMTP_TO may hold several addresses separated by commas, semicolons or spaces."""
    import re as _re
    return ", ".join(a for a in _re.split(r"[,;\s]+", E.get("SMTP_TO", "")) if a)

SNAP_DIR = Path(E.get("SNAPSHOT_DIR", "/data/snapshots"))
NTFY_URL = E.get("NTFY_URL", "http://127.0.0.1:8090/squirrels")
NTFY_BASE = NTFY_URL.rsplit("/", 1)[0]
# ntfy topic per label (people and squirrels separately, so the phone can treat them differently)
TOPICS = {"squirrel": "squirrels", "person": "people"}
TITLES = {"squirrel": ("Squirrel", "chipmunk"), "person": ("Person", "bust_in_silhouette")}
NTFY_EVERY = float(E.get("NTFY_MIN_INTERVAL_S", "120"))
MAIL_EVERY = float(E.get("SMTP_MIN_INTERVAL_S", "1800"))
NOTIFY_ACTIONS = set(E.get("NOTIFY_ACTIONS", "fired,dry-run").split(","))
_last = {"ntfy": 0.0, "mail": 0.0}


def db():
    return psycopg.connect(E["PG_URL"], autocommit=True)


def due(channel: str, every: float) -> bool:
    now = time.monotonic()
    if now - _last[channel] < every:
        return False
    _last[channel] = now
    return True


def topic_url(label: str) -> str:
    return f"{NTFY_BASE}/{TOPICS.get((label or '').lower(), 'squirrels')}"


def ntfy_auth(headers: dict):
    # Bearer token, not Basic auth: the recorder host's inline Suricata drops plaintext HTTP Basic credentials
    # (ET INFO 2006380), and container-to-container traffic here passes through its NFQUEUE.
    if E.get("NTFY_PUB_TOKEN"):
        headers["Authorization"] = f"Bearer {E['NTFY_PUB_TOKEN']}"
        return None
    return (E["NTFY_PUB_USER"], E["NTFY_PUB_PASS"])


def describe(ev: dict) -> str:
    if (ev.get("label") or "").lower() != "squirrel":
        return f"{ev['label']} {ev['score']:.2f} on {ev['camera']}\n{ev.get('time', ev.get('ts'))}"
    v = ev.get("verifier") or "verifier offline"
    mode = " (dry run, no sound)" if ev.get("action") == "dry-run" else ""
    return (f"{ev['label']} {ev['score']:.2f} on {ev['camera']}, VLM: {v}, action: {ev['action']}{mode}\n"
            f"{ev.get('time', ev.get('ts'))}")


def send_ntfy(ev: dict, jpeg: bytes | None) -> None:
    label = (ev.get("label") or "").lower()
    name, tag = TITLES.get(label, (ev.get("label") or "Object", "eyes"))
    headers = {"Title": f"{name} at {ev['camera']}", "Tags": tag, "Priority": "default",
               "Message": describe(ev).replace("\n", " | ")}
    if ev.get("click"):
        headers["Click"] = ev["click"]
    auth = ntfy_auth(headers)
    url = topic_url(label)
    if jpeg:
        headers["Filename"] = ev.get("snapshot", "snapshot.jpg")
        r = requests.put(url, data=jpeg, headers=headers, auth=auth, timeout=15)
    else:
        r = requests.post(url, data=headers.pop("Message").encode(), headers=headers, auth=auth, timeout=15)
    r.raise_for_status()


def send_clip(meta: dict, mp4: bytes) -> None:
    label = (meta.get("label") or "").lower()
    name, tag = TITLES.get(label, (meta.get("label") or "Object", "eyes"))
    headers = {"Title": f"Clip: {name} at {meta['camera']}", "Tags": f"{tag},movie_camera",
               "Message": "Event clip with detection boxes", "Filename": meta.get("filename", "clip.mp4")}
    if meta.get("click"):
        headers["Click"] = meta["click"]
    auth = ntfy_auth(headers)
    r = requests.put(topic_url(label), data=mp4, headers=headers, auth=auth, timeout=60)
    r.raise_for_status()


def send_mail(ev: dict, jpeg: bytes | None) -> None:
    """HTML email: annotated snapshot embedded inline (box, label + score, timestamp), details, Open-in-Frigate link."""
    import html
    from email.utils import make_msgid
    label = (ev.get("label") or "object").lower()
    name = TITLES.get(label, (ev.get("label") or "Object", ""))[0]
    when = ev.get("time", ev.get("ts")) or ""
    msg = EmailMessage()
    msg["Subject"] = f"{name} at {ev['camera']} ({when[:19].replace('T', ' ')})"
    msg["From"] = E["SMTP_FROM"]
    msg["To"] = mail_to()
    link = ev.get("click")
    msg.set_content(describe(ev) + (f"\n\nOpen in Frigate: {link}" if link else ""))
    rows = [("Camera", ev["camera"]), ("Detected", f"{name} ({ev['score']:.0%})"), ("Time", when[:19].replace("T", " "))]
    if label == "squirrel":
        rows.append(("LLM check", ev.get("verifier") or "verifier offline"))
        rows.append(("Action", ev.get("action", "")))
    cid = make_msgid(domain="camstack")
    table = "".join(f'<tr><td style="padding:4px 12px 4px 0;color:#666">{html.escape(k)}</td>'
                    f'<td style="padding:4px 0"><b>{html.escape(str(v))}</b></td></tr>' for k, v in rows)
    button = (f'<p style="margin:18px 0"><a href="{html.escape(link)}" style="background:#1a73e8;color:#fff;'
              f'padding:10px 18px;border-radius:6px;text-decoration:none;font-weight:600">Open in Frigate</a></p>'
              f'<p style="color:#888;font-size:12px">Link works on the home network: {html.escape(link)}</p>') if link else ""
    img = (f'<img src="cid:{cid[1:-1]}" alt="snapshot" style="max-width:100%;border-radius:6px;display:block">'
           if jpeg else "<p><i>No snapshot available.</i></p>")
    body = (f'<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:720px">'
            f'<h2 style="margin:0 0 12px">{html.escape(name)} at {html.escape(ev["camera"])}</h2>'
            f'{img}<table style="margin-top:14px;font-size:14px">{table}</table>{button}</div>')
    msg.add_alternative(body, subtype="html")
    if jpeg:
        msg.get_payload()[1].add_related(jpeg, maintype="image", subtype="jpeg", cid=cid,
                                         filename=ev.get("snapshot", "snapshot.jpg"))
    ctx = ssl.create_default_context()
    if E.get("SMTP_TLS_VERIFY", "1") == "0":  # Proton Bridge on localhost uses a self-signed cert
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    with smtplib.SMTP(E["SMTP_HOST"], int(E.get("SMTP_PORT", "1025")), timeout=30) as s:
        s.starttls(context=ctx)
        s.login(E["SMTP_USER"], E["SMTP_PASS"])
        s.send_message(msg)

def handle(payload: bytes) -> None:
    ev = json.loads(payload)
    b64 = ev.pop("snapshot_jpeg_b64", None)
    jpeg = base64.b64decode(b64) if b64 else None
    if jpeg:
        d = SNAP_DIR / ev["camera"]
        d.mkdir(parents=True, exist_ok=True)
        (d / ev["snapshot"]).write_bytes(jpeg)
    with db() as conn:
        conn.execute(
            "INSERT INTO events (ts, camera, label, score, box, verifier, confirmed, action, detect_ms, snapshot)"
            " VALUES (COALESCE(%s::timestamptz, now()), %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (ev.get("time"), ev["camera"], ev["label"], ev["score"], ev.get("box"), ev.get("verifier"),
             ev["confirmed"], ev["action"], ev.get("detect_ms"), ev.get("snapshot")))
    log.info("stored %s %s %.2f confirmed=%s action=%s", ev["camera"], ev["label"], ev["score"],
             ev["confirmed"], ev["action"])
    if "notify" in ev:  # Frigate-mode listener decided (per-label spacing)
        if not (ev["notify"] and ev["confirmed"]):
            return
        ntfy_ok = True
    else:              # laptop mode: bridge-side spacing
        if not (ev["confirmed"] and ev["action"] in NOTIFY_ACTIONS):
            return
        ntfy_ok = due("ntfy", NTFY_EVERY)
    if ntfy_ok:
        try:
            send_ntfy(ev, jpeg)
            log.info("ntfy sent")
        except Exception as e:
            log.warning("ntfy failed: %s", e)
    if E.get("SMTP_HOST") and due("mail", MAIL_EVERY):
        try:
            send_mail(ev, jpeg)
            log.info("email sent to %d recipient(s)", len(mail_to().split(", ")))
        except Exception as e:
            log.warning("email failed: %s", e)


def handle_clip(payload: bytes) -> None:
    meta = json.loads(payload)
    mp4 = base64.b64decode(meta.pop("clip_mp4_b64"))
    d = SNAP_DIR / meta["camera"]
    d.mkdir(parents=True, exist_ok=True)
    (d / meta.get("filename", "clip.mp4")).write_bytes(mp4)
    try:
        send_clip(meta, mp4)
        log.info("clip sent: %s %s (%.1f MB)", meta["camera"], meta.get("frigate_event"), len(mp4) / 1e6)
    except Exception as e:
        log.warning("clip ntfy failed: %s", e)


def handle_deter(payload: bytes) -> None:
    """Play a deterrent sound on the camera. The key only runs /etc/deter/deter.sh there (forced command)."""
    import subprocess
    req = json.loads(payload)
    cam = req.get("camera", "frontyard")
    target = E.get(f"DETER_TARGET_{cam.upper()}") or E.get("DETER_TARGET")
    if not target:
        log.warning("deter requested for %s but no DETER_TARGET set", cam)
        return
    sound = "".join(ch for ch in str(req.get("sound", "random")) if ch.isalnum() or ch == "_") or "random"
    vol = max(0, min(100, int(req.get("volume", 80))))
    cmd = ["ssh", "-i", "/deter/id_ed25519", "-o", "UserKnownHostsFile=/deter/known_hosts",
           "-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", target, f"{sound} {vol}"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    if p.returncode == 0:
        log.info("deterrent on %s: %s", cam, p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "ok")
    else:
        log.warning("deterrent on %s failed (%d): %s", cam, p.returncode, (p.stderr or p.stdout).strip()[-200:])


MATCH_SIM = float(E.get("PERSON_MATCH_SIM", "0.70"))        # same identity at or above
POSSIBLE_SIM = float(E.get("PERSON_POSSIBLE_SIM", "0.55"))  # flag for review between POSSIBLE and MATCH


def face_identity(conn, face) -> int | None:
    """Identity whose label equals the Frigate face name (case-insensitive), if exactly one exists."""
    if not face or not face.get("name"):
        return None
    r = conn.execute("SELECT id FROM identities WHERE lower(label) = lower(%s)", (face["name"],)).fetchall()
    return r[0][0] if len(r) == 1 else None


def apply_late_face(conn, frigate_event: str, face: dict) -> None:
    """A face recognised after the sighting was stored (Frigate names faces during the visit): record it and, when
    the sighting had founded a brand-new identity of its own, move it to the face's identity and drop the empty one."""
    from psycopg.types.json import Jsonb
    r = conn.execute("SELECT identity_id, match_kind, face FROM sightings WHERE frigate_event = %s", (frigate_event,)).fetchone()
    if not r or r[2]:
        return
    conn.execute("UPDATE sightings SET face = %s WHERE frigate_event = %s", (Jsonb(face), frigate_event))
    fid = face_identity(conn, face)
    if fid is None or fid == r[0]:
        return
    n = conn.execute("SELECT count(*) FROM sightings WHERE identity_id = %s", (r[0],)).fetchone()[0]
    if r[1] in ("new", "new_possible") and n == 1:
        conn.execute("UPDATE sightings SET identity_id = %s, match_kind = 'face', match_sim = %s WHERE frigate_event = %s",
                     (fid, float(face.get("score") or 0.9), frigate_event))
        if not occupant_refs(conn, r[0]):
            conn.execute("DELETE FROM identities WHERE id = %s", (r[0],))
        log.info("late face: %s moved to identity %s (%s)", frigate_event, fid, face.get("name"))


# -- face templates (squirrel_guard/faceid.py -> facematch.py) ------------------------------------------------------
FACE_DIR = SNAP_DIR / "faces"


def enrolled_templates(conn) -> list[dict]:
    """Enrolled face templates of household/known identities (the only people a face can name)."""
    try:
        with conn.transaction():   # savepoint: a bridge newer than the schema must not abort the sighting
            rows = conn.execute("""SELECT t.id, t.identity_id, i.label, i.kind, t.vec, t.model FROM face_templates t
                                   JOIN identities i ON i.id = t.identity_id
                                   WHERE i.kind IN ('household', 'known')""").fetchall()
    except Exception as e:
        log.warning("face templates unavailable (apply persons_schema.sql): %s", e)
        return []
    return [{"template_id": r[0], "identity_id": r[1], "label": r[2], "kind": r[3], "vec": r[4], "model": r[5]}
            for r in rows]


def face_decision(conn, fid: dict | None) -> dict | None:
    """The visit's face template vs the enrolled ones (liveness/quality/count gates inside facematch.decide)."""
    if not fid:
        return None
    return facematch.decide_any(fid, enrolled_templates(conn))


def unknown_gallery(conn, exclude_event: str | None = None) -> list[dict]:
    """Recent usable visit templates of UNNAMED unknown identities (kind 'unknown', no label): person sightings
    (face_vec / face_vec_v2) and vehicle occupant faces, the last FACE_UNKNOWN_LAST_N per identity, within the
    retention window. Each visit gives one InspireFace and (when present) one AdaFace entry. Only visits that passed
    the same gates as an enrolled match (facematch.gate_reasons) are used."""
    if not facematch.FACE_AUTO_TRACK:
        return []
    meta_keys = "jsonb_build_object('n_faces', m->'n_faces', 'quality', m->'quality', 'size_px', m->'size_px', " \
                "'consistency', m->'consistency', 'liveness', m->'liveness')"
    try:
        with conn.transaction():
            rows = conn.execute(f"""
                SELECT identity_id, vec, vec_v2, model, v2_model, {meta_keys} FROM (
                  SELECT s.identity_id, s.face_vec AS vec, s.face_vec_v2 AS vec_v2, s.face_meta->>'model' AS model,
                         s.face_meta->'v2'->>'model' AS v2_model, s.face_meta AS m, s.ts_start AS ts
                  FROM sightings s JOIN identities i ON i.id = s.identity_id
                  WHERE i.kind = 'unknown' AND i.label IS NULL AND s.face_vec IS NOT NULL
                    AND s.frigate_event IS DISTINCT FROM %s AND s.ts_start > now() - make_interval(days => %s)
                  UNION ALL
                  SELECT o.identity_id, o.vec, o.vec_v2, o.model, o.v2_model, o.meta, o.ts
                  FROM occupant_faces o JOIN identities i ON i.id = o.identity_id
                  WHERE i.kind = 'unknown' AND i.label IS NULL AND o.vec IS NOT NULL
                    AND o.frigate_event IS DISTINCT FROM %s AND o.ts > now() - make_interval(days => %s)
                ) x ORDER BY ts DESC""", (exclude_event, RETAIN_DAYS, exclude_event, RETAIN_DAYS)).fetchall()
    except Exception as e:
        log.warning("unknown-face gallery unavailable (apply persons_schema.sql): %s", e)
        return []
    out, per = [], {}
    for iid, vec, vec2, model, v2m, meta in rows:
        if per.get(iid, 0) >= facematch.FACE_UNKNOWN_LAST_N or facematch.gate_reasons(meta or {}):
            continue
        per[iid] = per.get(iid, 0) + 1
        out.append({"identity_id": iid, "vec": vec, "model": model})
        if vec2 is not None and v2m:
            out.append({"identity_id": iid, "vec": vec2, "model": v2m})
    return out


def track_unknown(conn, fid: dict | None, fdec: dict | None, ev: str | None = None) -> dict | None:
    """Auto-tracking decision for a face that did not match an enrolled template (None when off/not applicable)."""
    if not facematch.FACE_AUTO_TRACK or not fid or (fdec and fdec.get("status") == "match"):
        return None
    if not facematch.usable(fid):
        return None
    return facematch.decide_unknown(fid, unknown_gallery(conn, ev))


def new_unnamed_identity(conn, ts, origin: str | None = None, centroid=None) -> int:
    """A new unnamed identity (kind unknown, no label). Occupant-only identities have no body fingerprint: their
    centroid is all zeros, so body re-ID (cosine) can never match them."""
    if centroid is None:
        dim = conn.execute("SELECT array_length(centroid, 1) FROM identities WHERE array_length(centroid, 1) > 0 "
                           "ORDER BY id DESC LIMIT 1").fetchone()
        centroid = [0.0] * int(dim[0] if dim and dim[0] else 512)
    iid = conn.execute("INSERT INTO identities (centroid, first_seen, last_seen) VALUES (%s, %s, %s) RETURNING id",
                       (centroid, ts, ts)).fetchone()[0]
    if origin:
        try:
            with conn.transaction():
                conn.execute("UPDATE identities SET origin = %s WHERE id = %s", (origin, iid))
        except Exception as e:
            log.warning("identity origin not stored (apply persons_schema.sql): %s", e)
    return iid


def store_faceid(conn, ev: str, fid: dict, dec: dict | None, track: dict | None = None) -> None:
    """Keep the visit template + summary on the sighting; evidence crops under snapshots/faces/<event>_face<k>.jpg.
    Nothing about race/ethnicity is ever in the message; the expression is stored as a description only."""
    from psycopg.types.json import Jsonb
    files = []
    crops = fid.get("crops_b64") or ([fid["best_crop_b64"]] if fid.get("best_crop_b64") else [])
    if crops:
        FACE_DIR.mkdir(parents=True, exist_ok=True)
        for f in FACE_DIR.glob(f"{ev}_face*.jpg"):        # replaced by a better template: drop the old crops
            f.unlink(missing_ok=True)
        for k, c in enumerate(crops[:3]):
            name = f"{ev}_face{k}.jpg"
            (FACE_DIR / name).write_bytes(base64.b64decode(c))
            files.append(name)
    meta = {k: v for k, v in fid.items() if k not in ("vec", "crops_b64", "best_crop_b64", "facegeom", "facefit")}
    v2 = fid.get("v2") or {}
    meta.pop("v2", None)
    if v2.get("vec") is not None:                     # AdaFace template: column face_vec_v2, its summary in face_meta
        meta["v2"] = {k: v for k, v in v2.items() if k != "vec"}
    meta["apparent"] = {k: v for k, v in (meta.get("apparent") or {}).items() if k in
                        ("expression", "expression_share", "expression_note", "age", "gender")}
    meta.update({"match": dec, "crop_files": files})
    if track is not None:                              # unknown-face auto-tracking (never a label)
        meta["track"] = facematch.public(track)
    vec = [float(x) for x in fid["vec"]] if fid.get("vec") is not None else None
    try:
        with conn.transaction():
            conn.execute("UPDATE sightings SET face_vec = %s, face_meta = %s WHERE frigate_event = %s",
                         (vec, Jsonb(meta), ev))
    except Exception as e:
        log.warning("face template not stored for %s (apply persons_schema.sql): %s", ev, e)
    if v2.get("vec") is not None:
        try:
            with conn.transaction():
                conn.execute("UPDATE sightings SET face_vec_v2 = %s WHERE frigate_event = %s",
                             ([float(x) for x in v2["vec"]], ev))
        except Exception as e:
            log.warning("v2 face template not stored for %s (apply persons_schema.sql): %s", ev, e)
    store_facegeom(conn, ev, fid, dec)
    store_facefit(conn, ev, fid)


# -- face geometry (squirrel_guard/facegeom.py -> facegeomcmp.py): explanatory evidence, never decides identity ------
def facegeom_files(ev: str) -> dict:
    return {"overlay": f"{ev}_facegeom_overlay.jpg", "depth": f"{ev}_facegeom_depth.jpg",
            "mesh": f"{ev}_facegeom_mesh.gltf"}


def store_facegeom(conn, ev: str, fid: dict, dec: dict | None) -> None:
    """The visit's face geometry -> sightings.face_geom (features, uncertainties, files, comparison with the
    enrolled geometry of the matched/candidate identity); images + glTF under snapshots/evidence/<event>_facegeom_*
    (deleted with the sighting by the retention janitor, orphans by facegeom_sweep)."""
    from psycopg.types.json import Jsonb
    g = fid.get("facegeom")
    if not g:
        return
    ev_dir = SNAP_DIR / "evidence"
    names = facegeom_files(ev)
    files = {}
    try:
        ev_dir.mkdir(parents=True, exist_ok=True)
        for k in names:
            (ev_dir / names[k]).unlink(missing_ok=True)      # replaced by a better template
        for k, key in (("overlay", "overlay_jpg_b64"), ("depth", "depth_jpg_b64")):
            if g.get(key):
                (ev_dir / names[k]).write_bytes(base64.b64decode(g[key]))
                files[k] = names[k]
        if g.get("mesh"):
            (ev_dir / names["mesh"]).write_text(json.dumps(g["mesh"], separators=(",", ":")))
            files["mesh"] = names["mesh"]
    except Exception as e:
        log.warning("face geometry images not written for %s: %s", ev, e)
    rec = {k: v for k, v in g.items() if k not in ("overlay_jpg_b64", "depth_jpg_b64", "mesh")}
    rec["files"] = files
    iid = (dec or {}).get("identity") or (dec or {}).get("candidate")
    if iid and g.get("n_faces"):
        enr = enrolled_geometry(conn, iid, exclude_event=ev)
        cmp = facegeomcmp.compare(rec, enr)
        if cmp:
            rec["compare"] = {**cmp, "identity": iid, "enrolled_visits": enr.get("visits")}
    try:
        with conn.transaction():
            conn.execute("UPDATE sightings SET face_geom = %s WHERE frigate_event = %s", (Jsonb(rec), ev))
    except Exception as e:
        log.warning("face geometry not stored for %s (apply persons_schema.sql): %s", ev, e)


def store_facefit(conn, ev: str, fid: dict) -> None:
    """The visit's fitted face (squirrel_guard/facefit.py: one shape over the visit, features with bootstrap sd,
    detail levels LOD0-3) -> sightings.face_fit (JSON without the mesh) + snapshots/evidence/<event>_facefit.glb.
    Explanatory evidence only; identity stays with the face embedding."""
    from psycopg.types.json import Jsonb
    f = fid.get("facefit")
    if not f:
        return
    rec = {k: v for k, v in f.items() if k != "glb_b64"}
    name = f"{ev}_facefit.glb"
    try:
        p = SNAP_DIR / "evidence" / name
        p.unlink(missing_ok=True)
        if f.get("glb_b64"):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(base64.b64decode(f["glb_b64"]))
            rec["file"] = name
    except Exception as e:
        log.warning("face fit mesh not written for %s: %s", ev, e)
    try:
        with conn.transaction():
            conn.execute("UPDATE sightings SET face_fit = %s WHERE frigate_event = %s", (Jsonb(rec), ev))
    except Exception as e:
        log.warning("face fit not stored for %s (apply persons_schema.sql): %s", ev, e)


def sync_enrolled_geometry(conn) -> None:
    """Enrolled face templates take the geometry of the visit they were enrolled from (kept on the template row, so
    it lives and dies with the template, not with the sighting's retention)."""
    try:
        with conn.transaction():
            n = conn.execute("""UPDATE face_templates t SET geom = s.face_geom FROM sightings s
                                WHERE t.geom IS NULL AND t.source_event = s.frigate_event
                                  AND s.face_geom IS NOT NULL AND (s.face_geom->>'n_faces')::int > 0""").rowcount
        if n:
            log.info("face geometry: %d enrolled template(s) took their visit's geometry", n)
    except Exception as e:
        log.warning("enrolled face geometry not synced (apply persons_schema.sql): %s", e)


def enrolled_geometry(conn, iid: int, exclude_event: str | None = None) -> dict | None:
    """Merged geometry of an identity's enrolled templates (None when none has geometry)."""
    sync_enrolled_geometry(conn)
    try:
        with conn.transaction():
            rows = conn.execute("""SELECT geom FROM face_templates WHERE identity_id = %s AND geom IS NOT NULL
                                   AND source_event IS DISTINCT FROM %s""", (iid, exclude_event)).fetchall()
    except Exception:
        return None
    return facegeomcmp.merge([r[0] for r in rows])


def facegeom_sweep(conn) -> None:
    """Geometry images/meshes whose sighting is gone (deleted in the persons UI or by retention)."""
    import time as _t
    d = SNAP_DIR / "evidence"
    if not d.is_dir():
        return
    files = [f for f in list(d.glob("*_facegeom_*")) + list(d.glob("*_facefit.glb"))
             if _t.time() - f.stat().st_mtime > 3600]
    if not files:
        return
    ev_of = lambda f: f.name.rsplit("_facegeom_", 1)[0] if "_facegeom_" in f.name else f.name[:-len("_facefit.glb")]  # noqa: E731
    evs = {ev_of(f) for f in files}
    have = {e for (e,) in conn.execute("SELECT frigate_event FROM sightings WHERE frigate_event = ANY(%s)",
                                       (list(evs),)).fetchall()}
    gone = [f for f in files if ev_of(f) not in have]
    for f in gone:
        f.unlink(missing_ok=True)
    if gone:
        log.info("retention: removed %d orphaned face geometry files", len(gone))


def apply_late_faceid(conn, ev: str, fid: dict) -> None:
    """End-of-visit face template (the whole clip, better than the ~10 s live sample): store it when it is better,
    and when it matches an enrolled identity, move the sighting there (face outranks re-ID)."""
    try:
        with conn.transaction():
            r = conn.execute("SELECT identity_id, match_kind, face_meta, ts_start FROM sightings WHERE frigate_event = %s",
                             (ev,)).fetchone()
    except Exception as e:
        log.warning("late face template skipped for %s: %s", ev, e)
        return
    if not r or not facematch.better(fid, r[2]):
        return
    dec = face_decision(conn, fid)
    tdec = track_unknown(conn, fid, dec, ev)
    store_faceid(conn, ev, fid, dec, tdec)
    if (not dec or dec["status"] != "match") and tdec and tdec["status"] == "match" and tdec["identity"] != r[0] \
            and r[1] in ("new", "new_possible", "face_new"):
        # the whole-visit face matches an unnamed identity seen before: a sighting that founded its own identity
        # moves there (and the empty identity goes); body re-ID and face decisions already made are left alone
        old = conn.execute("SELECT count(*), max(label), max(kind) FROM sightings s JOIN identities i ON i.id = s.identity_id "
                           "WHERE s.identity_id = %s", (r[0],)).fetchone()
        if old and old[0] == 1 and old[1] is None and old[2] == "unknown":
            conn.execute("UPDATE sightings SET identity_id = %s, match_kind = 'face_unknown', match_sim = %s "
                         "WHERE frigate_event = %s", (tdec["identity"], tdec["sim"], ev))
            conn.execute("UPDATE identities SET n = n + 1, last_seen = GREATEST(last_seen, %s) WHERE id = %s",
                         (r[3], tdec["identity"]))
            if not occupant_refs(conn, r[0]):
                conn.execute("DELETE FROM identities WHERE id = %s", (r[0],))
            log.info("late face track: %s moved to unnamed identity %s (sim %.3f)", ev, tdec["identity"], tdec["sim"])
        return
    if not dec or dec["status"] != "match" or dec["identity"] == r[0]:
        return
    if r[1] in ("face", "face_template"):         # already decided by a face: keep it, the explanation shows both
        log.info("late face template for %s points to %s, sighting stays with %s (%s)", ev, dec["identity"], r[0], r[1])
        return
    old = conn.execute("SELECT count(*), max(label), max(kind) FROM sightings s JOIN identities i ON i.id = s.identity_id "
                       "WHERE s.identity_id = %s", (r[0],)).fetchone()
    conn.execute("UPDATE sightings SET identity_id = %s, match_kind = 'face_template', match_sim = %s WHERE frigate_event = %s",
                 (dec["identity"], dec["sim"], ev))
    conn.execute("UPDATE identities SET n = n + 1, last_seen = GREATEST(last_seen, %s) WHERE id = %s", (r[3], dec["identity"]))
    if old and old[0] == 1 and old[1] is None and old[2] == "unknown" and not occupant_refs(conn, r[0]):
        conn.execute("DELETE FROM identities WHERE id = %s", (r[0],))
    log.info("late face template: %s moved to identity %s (sim %.3f)", ev, dec["identity"], dec["sim"])


def handle_sighting(payload: bytes) -> None:
    """Store a person sighting and match it to an identity by re-ID fingerprint (cosine to identity centroids)."""
    import datetime as _dt
    import numpy as np
    from psycopg.types.json import Jsonb
    s = json.loads(payload)
    v = np.asarray(s["embedding"], np.float32)
    v /= np.linalg.norm(v) + 1e-12
    a = s.get("attributes") or {}
    crop_file = None
    if s.get("crop_jpeg_b64"):
        d = SNAP_DIR / "persons"
        d.mkdir(parents=True, exist_ok=True)
        crop_file = f"{s['frigate_event']}.jpg"
        (d / crop_file).write_bytes(base64.b64decode(s["crop_jpeg_b64"]))
    ts = _dt.datetime.fromtimestamp(s["start"], _dt.timezone.utc) if s.get("start") else _dt.datetime.now(_dt.timezone.utc)
    te = _dt.datetime.fromtimestamp(s["end"], _dt.timezone.utc) if s.get("end") else None
    with db() as conn, conn.transaction():
        if conn.execute("SELECT 1 FROM sightings WHERE frigate_event = %s", (s["frigate_event"],)).fetchone():
            return
        face = s.get("face") or None
        face_iid = face_identity(conn, face)
        fdec = face_decision(conn, s.get("faceid"))          # our enrolled-template match (liveness-gated)
        ft_iid = fdec["identity"] if fdec and fdec["status"] == "match" else None
        tdec = track_unknown(conn, s.get("faceid"), fdec, s["frigate_event"])   # unnamed-face auto-tracking
        rows = conn.execute("SELECT id, centroid, n FROM identities").fetchall()
        best_id, best_sim = None, -1.0
        for iid, cen, n in rows:
            if len(cen) != len(v):
                continue
            sim = float(np.dot(v, np.asarray(cen, np.float32)))
            if sim > best_sim:
                best_id, best_sim = iid, sim
        possible_id = possible_sim = None
        best_kind = conn.execute("SELECT kind, label FROM identities WHERE id = %s", (best_id,)).fetchone() \
            if best_id is not None else None
        # Rule (docs/faceid-v2.md "Auto-tracking"): enrolled face > Frigate face name > unnamed face track > body re-ID.
        # A usable face that clearly disagrees (< FACE_UNKNOWN_SPLIT*) with the faces of the UNNAMED identity body
        # re-ID picked starts a new unnamed identity instead; re-ID picks of named/household/known identities and
        # re-ID picks of identities without faces are left as they were (the face is then added to that identity).
        face_split = best_id is not None and best_sim >= MATCH_SIM and best_kind and best_kind[0] == "unknown" \
            and best_kind[1] is None and facematch.face_disagrees(tdec, best_id)
        if ft_iid is not None:
            # visit face template matched an ENROLLED household/known template (liveness, quality, count gates
            # passed): outranks Frigate's face name and the re-ID fingerprint
            iid, kind, msim = ft_iid, "face_template", float(fdec["sim"])
            conn.execute("UPDATE identities SET n = n + 1, last_seen = GREATEST(last_seen, %s) WHERE id = %s", (ts, iid))
        elif face_iid is not None:
            # enrolled face recognised by Frigate and labelled the same as an identity: that decides who it is
            iid, kind, msim = face_iid, "face", float(face.get("score") or 0.9)
            conn.execute("UPDATE identities SET n = n + 1, last_seen = GREATEST(last_seen, %s) WHERE id = %s", (ts, iid))
        elif tdec and tdec["status"] == "match":
            # the face matches an UNNAMED unknown identity's earlier faces: same unnamed identity (never named)
            iid, kind, msim = tdec["identity"], "face_unknown", float(tdec["sim"])
            conn.execute("UPDATE identities SET n = n + 1, last_seen = GREATEST(last_seen, %s) WHERE id = %s", (ts, iid))
        elif best_id is not None and best_sim >= MATCH_SIM and not face_split:
            iid, kind, msim = best_id, "match", best_sim
            cen, n = next((np.asarray(c, np.float32), n) for i, c, n in rows if i == best_id)
            w = min(n, 50)  # cap the weight so the centroid can follow slow appearance changes
            newc = cen * w + v
            newc /= np.linalg.norm(newc) + 1e-12
            conn.execute("UPDATE identities SET centroid = %s, n = n + 1, last_seen = GREATEST(last_seen, %s) "
                         "WHERE id = %s", ([float(x) for x in newc], ts, iid))
        else:
            if face_split:
                kind = "face_new"            # body looked like #best_id, the face clearly did not: kept apart
                possible_id, possible_sim = best_id, best_sim
            elif best_id is not None and best_sim >= POSSIBLE_SIM:
                possible_id, possible_sim, kind = best_id, best_sim, "new_possible"
            else:
                kind = "new"
            iid = conn.execute("INSERT INTO identities (centroid, first_seen, last_seen) VALUES (%s, %s, %s) "
                               "RETURNING id", ([float(x) for x in v], ts, ts)).fetchone()[0]
            msim = 1.0
        closest = {"id": best_id, "sim": round(best_sim, 4)} if best_id is not None else None
        conn.execute(
            "INSERT INTO sightings (frigate_event, camera, ts_start, ts_end, score, crops, embedding, attributes, age_low,"
            " age_high, gender, gender_conf, crop_file, identity_id, match_sim, match_kind, possible_id, possible_sim,"
            " explain) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (s["frigate_event"], s["camera"], ts, te, s.get("score"), s.get("crops"), [float(x) for x in v],
             Jsonb(a) if a else None, a.get("apparent_age_low"), a.get("apparent_age_high"),
             a.get("apparent_gender"), a.get("gender_confidence"), crop_file, iid, msim, kind, possible_id, possible_sim,
             Jsonb({"closest": closest})))
        if face:
            conn.execute("UPDATE sightings SET face = %s WHERE frigate_event = %s", (Jsonb(face), s["frigate_event"]))
        if s.get("faceid"):
            store_faceid(conn, s["frigate_event"], s["faceid"], fdec, tdec)
        store_pulc(conn, "sightings", s["frigate_event"], s.get("pulc"))
    with db() as conn:
        explain_and_strip(conn, s["frigate_event"], "identity_match")
        intel.on_person_sighting(conn, visitors_ntfy, SNAP_DIR, s["frigate_event"])
    log.info("sighting %s -> identity %s (%s, sim %.2f%s)", s["frigate_event"], iid, kind, msim,
             f", possible #{possible_id} {possible_sim:.2f}" if possible_id else "")


VEH_MATCH_SIM = float(E.get("VEHICLE_MATCH_SIM", "0.80"))
VEH_POSSIBLE_SIM = float(E.get("VEHICLE_POSSIBLE_SIM", "0.65"))
RETAIN_DAYS = int(E.get("PERSON_RETAIN_DAYS", "90"))


def store_pulc(conn, table: str, frigate_event: str, pv) -> None:
    """PULC attribute summary (profiler 'pulc') -> <table>.pulc. In a savepoint: a bridge newer than the schema
    (persons_schema.sql not applied yet) must not lose the sighting."""
    from psycopg.types.json import Jsonb
    if not pv or table not in ("sightings", "vehicle_sightings"):
        return
    try:
        with conn.transaction():
            conn.execute(f"UPDATE {table} SET pulc = %s WHERE frigate_event = %s", (Jsonb(pv), frigate_event))
    except Exception as e:
        log.warning("pulc not stored for %s (apply persons_schema.sql): %s", frigate_event, e)


# PULC (VeRi-776) vehicle type -> the body_type words the LLM uses
PULC_BODY = {"sedan": "sedan", "suv": "suv", "van": "van", "hatchback": "hatchback", "mpv": "minivan",
             "pickup": "pickup", "bus": "bus", "truck": "truck", "estate": "wagon"}


def pulc_vehicle_defaults(body, color, pv):
    """Body type / colour for matching when the LLM gave none: PULC's top class if it cleared its threshold.
    The LLM's words win whenever present (they are richer: 'silver', 'dark green')."""
    pv = pv or {}
    if not body and pv.get("type"):
        body = PULC_BODY.get(pv["type"])
    if not color and pv.get("color"):
        color = {"golden": "gold"}.get(pv["color"], pv["color"])
    return body, color


def norm_plate(p):
    p = "".join(ch for ch in (p or "").upper() if ch.isalnum())
    return p or None


def _colors(a):
    return {w for w in str((a or {}).get("color", "")).lower().replace("/", " ").replace("-", " ").split() if w not in ("and", "with")}


def handle_vehicle(payload: bytes) -> None:
    """Store a vehicle sighting; match by plate first, else fingerprint + same body type and colour."""
    import datetime as _dt
    import numpy as np
    from psycopg.types.json import Jsonb
    s = json.loads(payload)
    if s.get("plate_update"):
        return vehicle_plate_update(s)
    v = np.asarray(s["embedding"], np.float32)
    v /= np.linalg.norm(v) + 1e-12
    a = s.get("attributes") or {}
    plate = norm_plate(s.get("plate"))
    crop_file = None
    if s.get("crop_jpeg_b64"):
        d = SNAP_DIR / "vehicles"
        d.mkdir(parents=True, exist_ok=True)
        crop_file = f"{s['frigate_event']}.jpg"
        (d / crop_file).write_bytes(base64.b64decode(s["crop_jpeg_b64"]))
    ts = _dt.datetime.fromtimestamp(s["start"], _dt.timezone.utc) if s.get("start") else _dt.datetime.now(_dt.timezone.utc)
    te = _dt.datetime.fromtimestamp(s["end"], _dt.timezone.utc) if s.get("end") else None
    body = str(a.get("body_type") or "").lower() or None
    color = str(a.get("color") or "").lower() or None
    mm = " ".join(x for x in [a.get("apparent_make"), a.get("apparent_model")] if x and x != "unclear") or None
    body, color = pulc_vehicle_defaults(body, color, s.get("pulc"))
    # own readings from the profiler (squirrel_guard/vehicle_read.py): plate with per-character confidence, colour that
    # knows when it is night (infrared has no colour: the LLM's and PULC's colour guesses are dropped then)
    pr, col = s.get("plate_read") or {}, s.get("colour") or {}
    own = norm_plate(pr.get("text"))
    plate = plate or own                                        # Frigate's reader first; ours when Frigate read nothing
    plate_conf = pr.get("char_conf") if plate and plate == own else None
    night = bool(col.get("night"))
    if night:
        color = None
    elif col.get("name") and (col.get("confidence") or 0) >= 0.5 and \
            vehmatch.basic_colour(color) != vehmatch.basic_colour(col["name"]):
        color = col["name"]                                     # measured, white-balanced colour beats a guess
    new = {"plate": plate, "plate_conf": plate_conf, "colour": color, "colour_night": night, "body": body}
    with db() as conn, conn.transaction():
        if conn.execute("SELECT 1 FROM vehicle_sightings WHERE frigate_event = %s", (s["frigate_event"],)).fetchone():
            return
        vid = kind = msim = possible_id = possible_sim = None
        scored = []                                             # (explanation, vehicle, appearance cos, plate score)
        for iid, cen, vb, vc, vp in conn.execute("SELECT id, centroid, body_type, color, plate FROM vehicles").fetchall():
            sim = float(np.dot(v, np.asarray(cen, np.float32)))
            ps = vehmatch.plate_similarity(plate, vp)
            if sim >= VEH_POSSIBLE_SIM or (ps and ps["score"] >= 0.7):
                scored.append((vehmatch.explain(new, {"plate": vp, "colour": vc, "body": vb}, sim), iid, sim, ps["score"] if ps else None))
        same = [x for x in scored if x[0]["decision"] == "same"]
        maybe = [x for x in scored if x[0]["decision"] == "possible"]
        explain = None
        if same:
            explain, vid, sim, pscore = max(same, key=lambda x: x[0]["points"])
            kind, msim = ("plate", pscore) if pscore is not None and pscore >= 0.85 else ("appearance", sim)
        elif maybe:
            explain, possible_id, possible_sim, _ = max(maybe, key=lambda x: x[0]["points"])
        if vid is not None:
            cen, n = conn.execute("SELECT centroid, n FROM vehicles WHERE id = %s", (vid,)).fetchone()
            newc = np.asarray(cen, np.float32) * min(n, 50) + v
            newc /= np.linalg.norm(newc) + 1e-12
            conn.execute("UPDATE vehicles SET centroid = %s, n = n + 1, last_seen = GREATEST(last_seen, %s), "
                         "plate = COALESCE(plate, %s), body_type = COALESCE(body_type, %s), color = COALESCE(color, %s), "
                         "make_model = COALESCE(make_model, %s) WHERE id = %s",
                         ([float(x) for x in newc], ts, plate, body, color, mm, vid))
        else:
            kind = "new_possible" if possible_id else "new"
            vid = conn.execute("INSERT INTO vehicles (plate, body_type, color, make_model, centroid, first_seen, last_seen) "
                               "VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                               (plate, body, color, mm, [float(x) for x in v], ts, ts)).fetchone()[0]
            msim = 1.0
        sid = conn.execute("INSERT INTO vehicle_sightings (frigate_event, camera, label, ts_start, ts_end, score, crops, plate, "
                     "plate_score, embedding, attributes, crop_file, vehicle_id, match_sim, match_kind, possible_id, possible_sim) "
                     "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                     (s["frigate_event"], s["camera"], s.get("label"), ts, te, s.get("score"), s.get("crops"), plate,
                      s.get("plate_score") if plate != own or not pr else pr.get("conf"), [float(x) for x in v],
                      Jsonb(a) if a else None, crop_file, vid, msim, kind, possible_id, possible_sim)).fetchone()[0]
        try:
            with conn.transaction():                            # newer columns: a missing one must not lose the sighting
                conn.execute("UPDATE vehicle_sightings SET plate_read = %s, colour = %s, motion = %s, explain = %s WHERE id = %s",
                             (Jsonb(pr) if pr else None, Jsonb(col) if col else None, Jsonb(s["motion"]) if s.get("motion") else None,
                              Jsonb(explain) if explain else None, sid))
        except Exception as e:
            log.warning("vehicle readings not stored for %s (apply persons_schema.sql): %s", s["frigate_event"], e)
        store_pulc(conn, "vehicle_sightings", s["frigate_event"], s.get("pulc"))
        store_occupants(conn, sid, s["frigate_event"], ts, s)
    with db() as conn:
        intel.on_vehicle_sighting(conn, visitors_ntfy, SNAP_DIR, s["frigate_event"])
    log.info("vehicle %s -> vehicle %s (%s, sim %.2f, plate %s%s, colour %s)", s["frigate_event"], vid, kind, msim, plate or "-",
             f" read {pr.get('conf')}" if plate and plate == own else "", "night" if night else (color or "-"))


# -- vehicle occupants (squirrel_guard/occupants.py): enrolled match first, then unnamed auto-tracking -------------
def occupant_refs(conn, iid) -> int:
    """Occupant faces filed under an identity (0 when the table does not exist yet)."""
    try:
        with conn.transaction():
            return conn.execute("SELECT count(*) FROM occupant_faces WHERE identity_id = %s", (iid,)).fetchone()[0]
    except Exception:
        return 0


def occupant_decisions(conn, s: dict, ts) -> dict | None:
    """Occupant face templates of a vehicle message -> per-occupant decisions + identities to link.
    Enrolled household/known templates first (facematch.decide_any: the gates and thresholds of people). Without an
    enrolled match and with FACE_AUTO_TRACK on, a USABLE template (same gates) joins an unnamed unknown identity
    (facematch.decide_unknown) or founds a new one; an unusable one is only counted (no vector, no crop, no
    identity). With auto-tracking off only enrolled matches are linked, and no vector is ever kept.
    Returns {"summary": stored JSON, "links": [(slot, identity, kind, sim, occupant template, track decision)]}."""
    occ = s.get("occupant_faces")
    if occ is None and s.get("occupant_face"):
        occ = [s["occupant_face"]]
    if occ is None and s.get("occupants_detected") is None:
        return None
    occ = [o for o in (occ or []) if isinstance(o, dict)][:4]
    enrolled = enrolled_templates(conn) if occ else []
    gallery = unknown_gallery(conn, s["frigate_event"]) if occ and facematch.FACE_AUTO_TRACK else []
    out, links, used = [], [], set()
    for k, o in enumerate(occ):
        d = facematch.decide_any(o, enrolled)
        e = {"slot": k, "n_faces": o.get("n_faces"), "size_px": o.get("size_px"), "quality": o.get("quality"),
             "liveness_ok": bool((o.get("liveness") or {}).get("ok")), "rel_x": o.get("rel_x"),
             "match": facematch.public(d, named=True)}
        if d["status"] == "match" and d["identity"] not in used:
            used.add(d["identity"])
            links.append((k, d["identity"], "face_template", float(d["sim"]), o, None))
            e["identity"], e["kind"] = d["identity"], "face_template"
        elif d["status"] != "match" and facematch.FACE_AUTO_TRACK and facematch.usable(o):
            t = facematch.decide_unknown(o, gallery)
            e["track"] = facematch.public(t)
            if t["status"] == "match" and t["identity"] not in used:
                used.add(t["identity"])
                links.append((k, t["identity"], "face_unknown", float(t["sim"]), o, t))
                e["identity"], e["kind"] = t["identity"], "face_unknown"
            elif t["status"] != "match":
                links.append((k, None, "face_new", None, o, t))      # identity created by the caller
                e["kind"] = "face_new"
        out.append(e)
    summary = {"occupants_detected": s.get("occupants_detected"), "templates": len(occ), "occupants": out,
               "auto_track": facematch.FACE_AUTO_TRACK,
               "stats": {k: (s.get("occupant_stats") or {}).get(k) for k in
                         ("faces_seen", "frames_with_vehicle", "max_size_px", "cpu_s", "reason")}}
    return {"summary": summary, "links": links}


def store_occupants(conn, sid: int, ev: str, ts, s: dict) -> None:
    """Store the occupant count + decisions on the vehicle sighting; link the driver (best enrolled match, else the
    first tracked unnamed identity); keep vectors and one crop ONLY for faces linked to an identity."""
    from psycopg.types.json import Jsonb
    try:
        with conn.transaction():
            r = occupant_decisions(conn, s, ts)
            if r is None:
                return
            summ, links = r["summary"], r["links"]
            stored = []
            for k, iid, kind, sim, o, t in links:
                if iid is None:
                    iid = new_unnamed_identity(conn, ts, origin="vehicle_occupant")
                    summ["occupants"][k]["identity"] = iid
                else:
                    conn.execute("UPDATE identities SET n = n + 1, last_seen = GREATEST(last_seen, %s) WHERE id = %s",
                                 (ts, iid))
                crop = None
                if o.get("best_crop_b64"):
                    FACE_DIR.mkdir(parents=True, exist_ok=True)
                    crop = f"{ev}_occ{k}_face0.jpg"
                    (FACE_DIR / crop).write_bytes(base64.b64decode(o["best_crop_b64"]))
                keep_vec = kind != "face_template"           # enrolled people match via face_templates, not these
                v2 = o.get("v2") or {}
                meta = {kk: o.get(kk) for kk in ("n_faces", "quality", "size_px", "consistency", "liveness", "model",
                                                 "rel_x", "weight_sum", "yaw_abs_med")}
                conn.execute("""INSERT INTO occupant_faces (vehicle_sighting_id, frigate_event, ts, slot, identity_id,
                                    match_kind, sim, vec, vec_v2, model, v2_model, meta, crop_file)
                                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                             (sid, ev, ts, k, iid, kind, sim,
                              [float(x) for x in o["vec"]] if keep_vec and o.get("vec") is not None else None,
                              [float(x) for x in v2["vec"]] if keep_vec and v2.get("vec") is not None else None,
                              o.get("model"), v2.get("model"), Jsonb(meta), crop))
                stored.append((k, iid, kind, sim))
            drv = next((x for x in stored if x[2] == "face_template"), None) or (stored[0] if stored else None)
            conn.execute("""UPDATE vehicle_sightings SET occupants = %s, occupant_face = %s, driver_identity_id = %s,
                                   driver_sim = %s, driver_kind = %s WHERE id = %s""",
                         (s.get("occupants_detected"), Jsonb(summ), drv[1] if drv else None, drv[3] if drv else None,
                          drv[2] if drv else None, sid))
            log.info("vehicle %s: %s occupant(s), %d template(s), driver %s", ev, s.get("occupants_detected"),
                     summ["templates"], f"#{drv[1]} ({drv[2]})" if drv else "-")
    except Exception as e:
        log.warning("occupant faces not stored for %s (apply persons_schema.sql?): %s", ev, e)


def vehicle_plate_update(s: dict) -> None:
    plate = norm_plate(s.get("plate"))
    if not plate:
        return
    with db() as conn, conn.transaction():
        row = conn.execute("SELECT id, vehicle_id FROM vehicle_sightings WHERE frigate_event = %s", (s["frigate_event"],)).fetchone()
        if not row:
            return
        sid, vid = row
        conn.execute("UPDATE vehicle_sightings SET plate = %s, plate_score = %s WHERE id = %s", (plate, s.get("plate_score"), sid))
        owner = conn.execute("SELECT id FROM vehicles WHERE plate = %s", (plate,)).fetchone()
        if owner and owner[0] != vid:  # plate belongs to a known vehicle: move this sighting there
            conn.execute("UPDATE vehicle_sightings SET vehicle_id = %s, match_kind = 'plate' WHERE id = %s", (owner[0], sid))
            if not conn.execute("SELECT 1 FROM vehicle_sightings WHERE vehicle_id = %s", (vid,)).fetchone():
                conn.execute("DELETE FROM vehicles WHERE id = %s AND label IS NULL", (vid,))
            log.info("plate %s: sighting %s moved to vehicle %s", plate, s["frigate_event"], owner[0])
        elif not owner:
            conn.execute("UPDATE vehicles SET plate = %s WHERE id = %s AND plate IS NULL", (plate, vid))
            log.info("plate %s attached to vehicle %s", plate, vid)


def occupant_retention(conn) -> None:
    """Occupant faces of anyone not enrolled (household/known) follow the 90-day retention row by row, even when the
    vehicle itself is kept; unnamed identities left without any sighting or occupant face are removed (their face
    templates were their sightings'). Crops go in face_crop_sweep."""
    try:
        with conn.transaction():
            n = conn.execute("""DELETE FROM occupant_faces o USING identities i WHERE i.id = o.identity_id
                                AND i.kind NOT IN ('household', 'known') AND o.ts < now() - make_interval(days => %s)""",
                             (RETAIN_DAYS,)).rowcount
            conn.execute("""UPDATE vehicle_sightings vs SET driver_identity_id = NULL, driver_sim = NULL, driver_kind = NULL
                            WHERE driver_identity_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM occupant_faces o
                            WHERE o.vehicle_sighting_id = vs.id AND o.identity_id = vs.driver_identity_id)""")
            e = conn.execute("""DELETE FROM identities i WHERE i.kind = 'unknown' AND i.label IS NULL
                                AND i.last_seen < now() - interval '1 hour'
                                AND NOT EXISTS (SELECT 1 FROM sightings s WHERE s.identity_id = i.id)
                                AND NOT EXISTS (SELECT 1 FROM occupant_faces o WHERE o.identity_id = i.id)
                                AND NOT EXISTS (SELECT 1 FROM face_templates t WHERE t.identity_id = i.id)""").rowcount
    except Exception as ex:
        log.warning("occupant retention skipped (apply persons_schema.sql): %s", ex)
        return
    if n or e:
        log.info("retention: removed %d occupant faces older than %d days, %d empty unnamed identities", n, RETAIN_DAYS, e)


def face_crop_sweep(conn) -> None:
    """Face evidence crops whose sighting is gone (identity deleted in the persons UI, which mounts the snapshots
    read-only, or by retention) are deleted here, so a face never outlives its sighting by more than an hour."""
    if not FACE_DIR.is_dir():
        return
    import time as _t
    files = [f for f in FACE_DIR.glob("*_face*.jpg") if _t.time() - f.stat().st_mtime > 3600]
    if not files:
        return
    occ = [f for f in files if "_occ" in f.name]                # vehicle occupant crops: kept by occupant_faces
    files = [f for f in files if "_occ" not in f.name]
    evs = {f.name.rsplit("_face", 1)[0] for f in files}
    have = {e for (e,) in conn.execute("SELECT frigate_event FROM sightings WHERE frigate_event = ANY(%s)",
                                       (list(evs),)).fetchall()} if evs else set()
    gone = [f for f in files if f.name.rsplit("_face", 1)[0] not in have]
    if occ:
        try:
            with conn.transaction():
                keep = {c for (c,) in conn.execute("SELECT crop_file FROM occupant_faces WHERE crop_file = ANY(%s)",
                                                   ([f.name for f in occ],)).fetchall()}
        except Exception:
            keep = {f.name for f in occ}                          # table missing: do not guess, keep them
        gone += [f for f in occ if f.name not in keep]
    for f in gone:
        f.unlink(missing_ok=True)
    if gone:
        log.info("retention: removed %d orphaned face crops", len(gone))


def retention_janitor() -> None:
    """Hourly: delete UNCONFIRMED people/vehicles (no name, kind unknown) not seen for RETAIN_DAYS, with crops."""
    import time as _t
    while True:
        try:
            with db() as conn, conn.transaction():
                for tbl, stbl, fk, sub in (("identities", "sightings", "identity_id", "persons"),
                                           ("vehicles", "vehicle_sightings", "vehicle_id", "vehicles")):
                    old = [r[0] for r in conn.execute(
                        f"SELECT id FROM {tbl} WHERE label IS NULL AND kind = 'unknown' "
                        f"AND last_seen < now() - make_interval(days => %s)", (RETAIN_DAYS,)).fetchall()]
                    if not old:
                        continue
                    files = [f for (f,) in conn.execute(f"SELECT crop_file FROM {stbl} WHERE {fk} = ANY(%s)", (old,)).fetchall() if f]
                    if stbl == "sightings":
                        for row in conn.execute("SELECT evidence_card, evidence_gif, match_strip, frigate_event "
                                                "FROM sightings WHERE identity_id = ANY(%s)", (old,)).fetchall():
                            for f in row[:3]:
                                if f:
                                    (SNAP_DIR / "evidence" / f).unlink(missing_ok=True)
                            for f in (SNAP_DIR / "evidence").glob(f"{row[3]}_*"):   # trajectory + replay assets
                                f.unlink(missing_ok=True)
                            for f in FACE_DIR.glob(f"{row[3]}_face*.jpg"):          # face evidence crops
                                f.unlink(missing_ok=True)
                    conn.execute(f"DELETE FROM {stbl} WHERE {fk} = ANY(%s)", (old,))
                    conn.execute(f"DELETE FROM {tbl} WHERE id = ANY(%s)", (old,))
                    for f in files:
                        (SNAP_DIR / sub / f).unlink(missing_ok=True)
                    log.info("retention: removed %d unconfirmed %s older than %d days", len(old), tbl, RETAIN_DAYS)
            with db() as conn:
                conn.execute("DELETE FROM decisions WHERE ts < now() - make_interval(days => %s)", (RETAIN_DAYS,))
                occupant_retention(conn)
                face_crop_sweep(conn)
                facegeom_sweep(conn)
                sync_enrolled_geometry(conn)
        except Exception:
            log.exception("retention janitor failed")
        _t.sleep(3600)


import intel  # noqa: E402  (alerts + reports)
import facematch  # noqa: E402  (enrolled face templates)
import facegeomcmp  # noqa: E402  (face geometry comparison, explanatory)
import explain  # noqa: E402  (readable scores + decision log)
import strip as match_strip  # noqa: E402

EVIDENCE_DIR = SNAP_DIR / "evidence"
UI_URL = E.get("PERSONS_UI_URL", "https://persons.cam.example.org")
FRIGATE_URL = E.get("FRIGATE_URL", "https://frigate.cam.example.org")
EVIDENCE_MAIL = E.get("EVIDENCE_MAIL", "alerts")   # alerts = only visits that raised an alert; all; off


def explain_and_strip(conn, ev: str, stage: str, strip: bool = True) -> dict | None:
    """Explain the person match, keep it on the sighting, log the decision; render the match strip once."""
    from psycopg.types.json import Jsonb
    try:
        ex = explain.explain_person(conn, ev)
        if not ex:
            return None
        conn.execute("UPDATE sightings SET explain = %s WHERE frigate_event = %s", (Jsonb(ex), ev))
        explain.record(conn, stage, ev, f"person:{ex['identity']}", ex)
        if strip:
            cid = ex["identity"] if "NEW_PERSON" not in ex["reasons"] and "APPEARANCE_POSSIBLE" not in ex["reasons"] \
                else ex["scores"].get("closest_id")
            jpg = match_strip.render(conn, SNAP_DIR, ev, cid, explain.band_of, explain.appearance_prob, intel.TZ)
            if jpg:
                EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
                (EVIDENCE_DIR / f"{ev}_strip.jpg").write_bytes(jpg)
                conn.execute("UPDATE sightings SET match_strip = %s WHERE frigate_event = %s", (f"{ev}_strip.jpg", ev))
        return ex
    except Exception:
        log.exception("explain failed for %s", ev)
        return None


def store_evidence(conn, ev: str, evd: dict) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    card = gif = None
    if evd.get("card_jpeg_b64"):
        card = f"{ev}_card.jpg"
        (EVIDENCE_DIR / card).write_bytes(base64.b64decode(evd["card_jpeg_b64"]))
    if evd.get("gif_b64"):
        gif = f"{ev}.gif"
        (EVIDENCE_DIR / gif).write_bytes(base64.b64decode(evd["gif_b64"]))
    conn.execute("UPDATE sightings SET evidence_card = COALESCE(%s, evidence_card), "
                 "evidence_gif = COALESCE(%s, evidence_gif) WHERE frigate_event = %s", (card, gif, ev))


def action_link(identity: int, action: str, days: int = 7) -> str:
    """Signed one-tap link (no password in the notification): /act/<token> on the people UI."""
    import hashlib, hmac, time as _t
    key = hashlib.sha256(b"camstack-act|" + E.get("PERSONS_UI_PASS", "").encode()).digest()
    payload = f"{identity}.{action}.{int(_t.time()) + days * 86400}"
    sig = hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{UI_URL}/act/{payload}.{sig}"


def evidence_followup(conn, ev: str, ex: dict | None) -> None:
    """After the visit ends: email + ntfy with the evidence card, GIF, match strip and the readable explanation,
    for visits that raised an alert (EVIDENCE_MAIL=all: every person visit)."""
    if not ex or EVIDENCE_MAIL == "off":
        return
    alerts = conn.execute("SELECT kind FROM alerts WHERE sighting = %s ORDER BY ts", (ev,)).fetchall()
    if EVIDENCE_MAIL == "alerts" and not alerts:
        return
    akinds = [k for (k,) in alerts]
    why = [explain.ALERT_REASONS.get(k, k.upper()) for k in akinds]
    row = conn.execute("SELECT evidence_card, evidence_gif, match_strip, crop_file FROM sightings WHERE frigate_event = %s",
                       (ev,)).fetchone()
    files = {k: (EVIDENCE_DIR / f if k != "crop" else SNAP_DIR / "persons" / f) if f else None
             for k, f in zip(("card", "gif", "strip", "crop"), row)}
    files = {k: p for k, p in files.items() if p and p.exists()}
    links = {"frigate": f"{FRIGATE_URL}/explore?event_id={ev}", "identity": f"{UI_URL}/identity/{ex['identity']}",
             "why": f"{UI_URL}/sighting/{ex['sighting']}",
             "ok": action_link(ex["identity"], "known"), "watch": action_link(ex["identity"], "watch")}
    explain.record(conn, "evidence_sent", ev, f"person:{ex['identity']}", ex, reasons=why + ex["reasons"],
                   message=f"Alerted ({', '.join(why) or 'evidence'}): {ex['sentence']}")
    title = ("Watch-list person" if ex["kind"] == "watch" else ex["who"]) + " - visit summary"
    try:
        evidence_ntfy(title, ex, why, files, links)
    except Exception as e:
        log.warning("evidence ntfy failed: %s", e)
    if E.get("SMTP_HOST"):
        try:
            evidence_mail(title, ex, why, files, links)
        except Exception as e:
            log.warning("evidence email failed: %s", e)


def evidence_ntfy(title, ex, why, files, links):
    att = files.get("gif") or files.get("card") or files.get("crop")
    reasons = ", ".join(explain.REASON_TEXT.get(w, w.lower()) for w in why)
    msg = (f"{ex['headline']}." + (f" Alerted: {reasons}." if reasons else "") +
           (" " + "; ".join(ex["did"]) + "." if ex["did"] else ""))
    headers = {"Title": title, "Tags": "mag", "Click": links["why"],
               "Actions": f"view, Why?, {links['why']}; view, Mark OK, {links['ok']}; view, Watch, {links['watch']}"}
    auth = ntfy_auth(headers)
    if att:
        headers["Filename"] = att.name
        headers["Message"] = msg
        r = requests.put(VISITORS_URL, data=att.read_bytes(), headers=headers, auth=auth, timeout=30)
    else:
        r = requests.post(VISITORS_URL, data=msg.encode(), headers=headers, auth=auth, timeout=15)
    r.raise_for_status()


BAND_COLOURS = {"strong": "#1f9d55", "likely": "#4c9a2a", "possible": "#c98a12", "unlikely": "#d64545"}


def evidence_mail(title, ex, why, files, links):
    import html as _h
    import ssl
    import smtplib
    from email.message import EmailMessage
    from email.utils import make_msgid
    esc = _h.escape
    reasons = [explain.REASON_TEXT.get(w, w.lower()) for w in why]
    text = [title, "", ex["sentence"]]
    if reasons:
        text.append("Why alerted: " + ", ".join(reasons) + f"  [{', '.join(why)}]")
    if ex["did"]:
        text.append("Did: " + "; ".join(ex["did"]))
    if ex["body"]:
        text.append("Looked: " + "; ".join(ex["body"]) + " (apparent age/gender are estimates)")
    text += ["", f"Why this match: {links['why']}", f"Frigate: {links['frigate']}",
             f"Mark as OK (stop alerts): {links['ok']}", f"Add to watch list: {links['watch']}"]
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = title, E["SMTP_FROM"], mail_to()
    msg.set_content("\n".join(text))
    cids = {k: make_msgid(domain="camstack.local") for k in files}
    chip = (f'<span style="background:{BAND_COLOURS.get(ex["band"], "#666")};color:#fff;border-radius:4px;'
            f'padding:1px 7px;font-size:13px">{esc(ex["band"])} {ex["pct"]}%</span>')
    def img(k, alt, w=None):
        return (f'<img src="cid:{cids[k][1:-1]}" alt="{alt}" style="max-width:100%;{f"width:{w}px;" if w else ""}'
                f'border-radius:6px;display:block;margin:10px 0">') if k in cids else ""
    def btn(href, label, bg="#1a73e8"):
        return (f'<a href="{esc(href)}" style="background:{bg};color:#fff;padding:8px 14px;border-radius:6px;'
                f'text-decoration:none;font-weight:600;margin:0 6px 6px 0;display:inline-block">{label}</a>')
    rows = [("Who", f"{esc(ex['headline'])} {chip}"),
            ("Evidence", "<br>".join(esc(p) for p in ex["evidence"])),
            ("Why alerted", esc(", ".join(reasons)) + (f' <code style="color:#888">{esc(" ".join(why))}</code>' if why else "")),
            ("Did", esc("; ".join(ex["did"]))), ("Looked", esc("; ".join(ex["body"])) + ' <span style="color:#888">(estimates)</span>')]
    table = "".join(f'<tr><td style="color:#666;padding:3px 14px 3px 0;vertical-align:top">{k}</td><td>{v}</td></tr>'
                    for k, v in rows if v and v != chip)
    body = (f'<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:760px">'
            f'<h2 style="margin:0 0 8px">{esc(title)}</h2>{img("card", "evidence card")}'
            f'<table style="font-size:14px;margin:8px 0">{table}</table>'
            f'<p style="margin:14px 0 4px;color:#666;font-size:13px">This visit next to the most similar earlier sightings:</p>'
            f'{img("strip", "match strip")}{img("gif", "visit flip-book", 360)}'
            f'<p style="margin-top:14px">{btn(links["why"], "Why this match?")}{btn(links["frigate"], "Open in Frigate")}'
            f'{btn(links["ok"], "Mark as OK", "#1f9d55")}{btn(links["watch"], "Watch list", "#d64545")}</p>'
            f'<p style="color:#888;font-size:12px">Percentages are estimates. Links work on the home network.</p></div>')
    msg.add_alternative(body, subtype="html")
    html_part = msg.get_payload()[1]
    for k, p in files.items():
        sub = "gif" if p.suffix == ".gif" else "jpeg"
        html_part.add_related(p.read_bytes(), maintype="image", subtype=sub, cid=cids[k], filename=p.name)
    ctx = ssl.create_default_context()
    if E.get("SMTP_TLS_VERIFY", "1") == "0":
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    with smtplib.SMTP(E["SMTP_HOST"], int(E.get("SMTP_PORT", "1025")), timeout=60) as sm:
        sm.starttls(context=ctx)
        sm.login(E["SMTP_USER"], E["SMTP_PASS"])
        sm.send_message(msg)
    log.info("evidence email sent for %s", ex.get("sighting"))

VISITORS_URL = f"{NTFY_BASE}/visitors"
ALERT_TITLES = {"possible_fall": "Possible fall", "night_unknown": "Unknown person at night", "repeat_unknown": "Repeat visitor",
                "linger_unknown": "Person lingering", "parked_vehicle": "Unknown vehicle parked",
                "recurring_vehicle": "Recurring vehicle", "zone_unknown": "Person in alert zone",
                "report": "Security report"}


def visitors_ntfy(kind, message, crop_path=None, click=None):
    headers = {"Title": ALERT_TITLES.get(kind, kind), "Tags": "rotating_light" if kind != "report" else "page_facing_up",
               "Priority": "high" if kind in ("night_unknown", "zone_unknown") else "default"}
    if click:
        headers["Click"] = click
    auth = ntfy_auth(headers)
    if crop_path and Path(crop_path).exists():
        headers["Filename"] = Path(crop_path).name
        headers["Message"] = message.replace("\n", " ")
        r = requests.put(VISITORS_URL, data=Path(crop_path).read_bytes(), headers=headers, auth=auth, timeout=15)
    else:
        r = requests.post(VISITORS_URL, data=message.encode(), headers=headers, auth=auth, timeout=15)
    r.raise_for_status()


def send_report_mail(title, text, facts, link):
    if not E.get("SMTP_HOST"):
        return
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = title, E["SMTP_FROM"], mail_to()
    msg.set_content(text + f"\n\nOpen report: {link}")
    msg.add_alternative(intel.report_html(title, text, facts, link), subtype="html")
    ctx = ssl.create_default_context()
    if E.get("SMTP_TLS_VERIFY", "1") == "0":
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    with smtplib.SMTP(E["SMTP_HOST"], int(E.get("SMTP_PORT", "1025")), timeout=30) as sm:
        sm.starttls(context=ctx)
        sm.login(E["SMTP_USER"], E["SMTP_PASS"])
        sm.send_message(msg)


def handle_behavior(payload: bytes) -> None:
    """End-of-event behaviour summary from the profiler: dwell, path, stillness, zones."""
    import datetime as _dt
    from psycopg.types.json import Jsonb
    b = json.loads(payload)
    tbl = "sightings" if b.get("kind") == "person" else "vehicle_sightings"
    end = _dt.datetime.fromtimestamp(b["end"], _dt.timezone.utc) if b.get("end") else None
    with db() as conn, conn.transaction():
        n = conn.execute(f"UPDATE {tbl} SET ts_end = COALESCE(%s, ts_end), behavior = %s WHERE frigate_event = %s",
                         (end, Jsonb(b.get("behavior") or {}), b["frigate_event"])).rowcount
        if b.get("kind") == "person" and b.get("face") and n:
            apply_late_face(conn, b["frigate_event"], b["face"])
        if b.get("kind") == "person" and b.get("faceid") and n:
            apply_late_faceid(conn, b["frigate_event"], b["faceid"])
        h, g = b.get("height") or {}, b.get("gait")
        if n and tbl == "sightings" and b.get("trajectory"):
            EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
            ev = b["frigate_event"]
            for name, data in (b["trajectory"].pop("replay_files", None) or {}).items():
                # only <event>_replay_{atlas.webp,bg.jpg,ground.json} and the rig-driven SMPL body
                # (<event>_body.bin, <event>_body_faces.bin; SMPL-derived, local only): never a path from the message
                if name in (f"{ev}_replay_atlas.webp", f"{ev}_replay_bg.jpg", f"{ev}_replay_ground.json",
                            f"{ev}_body.bin", f"{ev}_body_faces.bin"):
                    (EVIDENCE_DIR / name).write_bytes(base64.b64decode(data))
            tf = f"{b['frigate_event']}_traj.json"
            (EVIDENCE_DIR / tf).write_text(json.dumps(b["trajectory"]))
            conn.execute("UPDATE sightings SET trajectory_file = %s WHERE frigate_event = %s", (tf, b["frigate_event"]))
            m = b["trajectory"].get("metrics") or {}
            b.setdefault("behavior", {}).update({k: m.get(k) for k in ("path_length_m", "speed_mean_mps", "stops")})
        if n and tbl == "sightings" and b.get("gait_vec"):
            conn.execute("UPDATE sightings SET gait_vec = %s WHERE frigate_event = %s", (b["gait_vec"], b["frigate_event"]))
        if n and tbl == "sightings" and b.get("actions") is not None:
            conn.execute("UPDATE sightings SET actions = %s WHERE frigate_event = %s",
                         (Jsonb(b["actions"]), b["frigate_event"]))
        if n and tbl == "sightings" and b.get("objects") is not None:
            try:
                with conn.transaction():   # savepoint: a bridge newer than the schema must not lose the rest
                    conn.execute("UPDATE sightings SET objects = %s WHERE frigate_event = %s",
                                 (Jsonb(b["objects"]), b["frigate_event"]))
            except Exception as e:
                log.warning("objects not stored for %s (apply persons_schema.sql): %s", b["frigate_event"], e)
        if n and tbl == "sightings" and (h or g):  # keypoint analysis: height only with a calibrated camera
            conn.execute("""UPDATE sightings SET height_cm = %s, height_iqr_cm = %s, height_frames = %s, gait = %s
                            WHERE frigate_event = %s""",
                         (h.get("cm"), h.get("iqr_cm"), h.get("frames"), Jsonb(g) if g else None, b["frigate_event"]))
        if n and tbl == "sightings" and b.get("evidence"):
            store_evidence(conn, b["frigate_event"], b["evidence"])
        if n:
            intel.on_behavior(conn, visitors_ntfy, SNAP_DIR, b.get("kind"), b["frigate_event"],
                              dict(b.get("behavior") or {}, actions=b.get("actions") or [],
                                   objects=b.get("objects") or []))
    if n and tbl == "sightings":
        with db() as conn:
            ex = explain_and_strip(conn, b["frigate_event"], "motion", strip=False)
            evidence_followup(conn, b["frigate_event"], ex)
    log.info("behavior %s %s: %s%s", b.get("kind"), b["frigate_event"], "stored" if n else "no sighting (not profiled)",
             f" height={(b.get('height') or {}).get('cm')} gait={'usable' if (b.get('gait') or {}).get('usable') else '-'}" if n else "")


FUSION_FILES = ("_traj.json", "_replay_atlas.webp", "_replay_bg.jpg", "_replay_ground.json", "_body.bin",
                "_body_faces.bin")


def handle_fusion(payload: bytes) -> None:
    """Fused multi-camera replay from the profiler (squirrel_guard.fusion, docs/fusion.md): one rig over the
    overlapping events of 2+ cameras. Stored as <primary event>_fused_traj.json (+ assets) and linked from EVERY
    member's sighting (sightings.fused_trajectory_file); each sighting keeps its own per-camera replay as well."""
    import re
    b = json.loads(payload)
    fid, tr = str(b.get("fusion_id") or ""), b.get("trajectory")
    ok_id = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    members = [str(m.get("frigate_event")) for m in b.get("members") or [] if ok_id.match(str(m.get("frigate_event")))]
    if not ok_id.match(fid) or not fid.endswith("_fused") or not tr or len(members) < 2:
        log.warning("fusion message ignored (id %r, %d members)", fid[:90], len(members))
        return
    with db() as conn, conn.transaction():
        n = conn.execute("SELECT count(*) FROM sightings WHERE frigate_event = ANY(%s)", (members,)).fetchone()[0]
        if not n:
            log.info("fusion %s: no stored sighting among %s", fid, members)
            return
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        for name, data in (tr.pop("replay_files", None) or {}).items():
            if name.startswith(fid + "_") and name[len(fid):] in FUSION_FILES[1:]:   # never a path from the message
                (EVIDENCE_DIR / name).write_bytes(base64.b64decode(data))
        tf = f"{fid}_traj.json"
        (EVIDENCE_DIR / tf).write_text(json.dumps(tr))
        try:
            with conn.transaction():   # savepoint: a bridge newer than the schema must not fail the rest
                k = conn.execute("UPDATE sightings SET fused_trajectory_file = %s WHERE frigate_event = ANY(%s)",
                                 (tf, members)).rowcount
        except Exception as e:
            k = 0
            log.warning("fused replay %s not linked (apply persons_schema.sql): %s", tf, e)
    fu = tr.get("fusion") or {}
    log.info("fusion %s: %s, %d of %d sightings linked, stature %s", fid, "+".join(fu.get("cameras") or []), k,
             len(members), fu.get("stature_m"))


# Work happens OFF the MQTT network thread: slow HTTP/SMTP inside on_message stalled keepalive pings, the broker
# dropped the client before the QoS1 PUBACK, and the persistent session redelivered the same clip every ~2 min.
import collections
import queue
import threading

_work: queue.Queue = queue.Queue(maxsize=200)
_done: "collections.OrderedDict[str, None]" = collections.OrderedDict()  # dedupe keys of handled messages


def _key(topic: str, payload: bytes) -> str | None:
    try:
        d = json.loads(payload)
    except ValueError:
        return None
    if topic.endswith("/clip"):
        return "clip:" + str(d.get("frigate_event") or d.get("filename"))
    if topic.endswith("/deter"):
        return "deter:" + str(d.get("frigate_event"))
    if topic.endswith("/sighting"):
        return "sighting:" + str(d.get("frigate_event"))
    if topic.endswith("/behavior"):
        return "behavior:" + str(d.get("frigate_event"))
    if topic.endswith("/fusion"):
        return "fusion:" + str(d.get("fusion_id"))
    if topic.endswith("/vehicle"):
        return ("plate:" if d.get("plate_update") else "vehicle:") + str(d.get("frigate_event")) + str(d.get("plate") or "")
    return "event:" + str(d.get("camera")) + ":" + str(d.get("snapshot") or d.get("ts"))


def _worker():
    while True:
        topic, payload = _work.get()
        k = _key(topic, payload)
        if k and k in _done:
            log.info("duplicate %s ignored (redelivery)", k)
            continue
        try:
            if topic.endswith("/clip"):
                handle_clip(payload)
            elif topic.endswith("/deter"):
                handle_deter(payload)
            elif topic.endswith("/sighting"):
                handle_sighting(payload)
            elif topic.endswith("/vehicle"):
                handle_vehicle(payload)
            elif topic.endswith("/behavior"):
                handle_behavior(payload)
            elif topic.endswith("/fusion"):
                handle_fusion(payload)
            else:
                handle(payload)
        except Exception:
            log.exception("bad message on %s", topic)
        if k:
            _done[k] = None
            while len(_done) > 2000:
                _done.popitem(last=False)


def on_message(client, userdata, msg):
    try:
        _work.put_nowait((msg.topic, msg.payload))  # returns at once, so paho acks and keeps pinging
    except queue.Full:
        log.warning("work queue full, dropping message on %s", msg.topic)


def main():
    threading.Thread(target=_worker, daemon=True).start()
    threading.Thread(target=retention_janitor, daemon=True).start()
    threading.Thread(target=intel.scheduler, args=(db, send_report_mail, visitors_ntfy, SNAP_DIR), daemon=True).start()
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="camstack-bridge", clean_session=False)
    c.username_pw_set(E["MQTT_USER"], E["MQTT_PASS"])
    c.on_connect = lambda cl, ud, fl, rc, pr: (log.info("MQTT connected (%s)", rc),
                                               cl.subscribe([("squirrelguard/+/event", 1), ("squirrelguard/+/clip", 0), ("squirrelguard/+/deter", 1),
                                                                  ("squirrelguard/+/sighting", 1),
                                                                  ("squirrelguard/+/vehicle", 1),
                                                                  ("squirrelguard/+/behavior", 1),
                                                                  ("squirrelguard/+/fusion", 1)]))  # clips QoS 0, see publish.py
    c.on_message = on_message
    c.reconnect_delay_set(1, 60)
    log.info("email %s", "enabled" if E.get("SMTP_HOST") else "disabled (SMTP_HOST unset)")
    c.connect(E.get("MQTT_HOST", "127.0.0.1"), int(E.get("MQTT_PORT", "1883")), keepalive=60)
    c.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    main()

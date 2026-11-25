"""Evidence packages, signatures, trusted timestamps and hash-chained logs (no Flask, no database driver).

A package is one ZIP for one case export, holding one or more sightings ("items"):

  items/NN/original/           the recorder's files exactly as retrieved: clip.mp4, snapshot.jpg, and the continuous
                               recording segments (segments/*.mkv) that cover the event, each checked against the
                               capture-time ledger
  items/NN/derived/            pictures and data this system produced from the recording
  items/NN/record/visit.json   the explained match as stored; review.json = the people who confirmed or rejected it;
                               ledger.jsonl + anchors.jsonl = the capture-time hashes of the segments and their daily
                               outside timestamp
  report.html                  printable report
  manifest.json                every file with SHA-256, size, role and origin; who exported, when, why; clock record
  manifest.sig                 Ed25519 signature of manifest.json (raw 64 bytes)
  signing-key.pem              the installation's public key; compare its fingerprint with the published one
  manifest.tsr                 RFC 3161 timestamp of manifest.json from an outside service (when configured)
  SHA256SUMS, verify.py, README.txt

Logs (custody, audit) are hash chains: entry_hash = sha256(prev_hash + "\\n" + canonical JSON of the entry's fields).
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import html
import json
import os
import zipfile
from pathlib import Path

FORMAT = "squirrel-guard-evidence/2"
GENESIS = "0" * 64
ACTIONS = {"created": "Package created", "downloaded": "Package downloaded", "verified": "Integrity check passed",
           "verify_failed": "Integrity check FAILED", "release_created": "Release copy created",
           "release_downloaded": "Release copy downloaded"}
LIMITS = [
    "A match made by this system is an investigative lead. It is not an identification and must be confirmed by a person.",
    "Percentages and bands (strong, likely, possible, unlikely) are this system's own scores, not error rates measured "
    "on an independent test set.",
    "Apparent age, gender, clothing, height, gait and vehicle details are model estimates and can be wrong.",
    "Recording segments marked 'matches capture-time seal' are unchanged since they were sealed, a few minutes after "
    "recording. Nothing here proves what happened between capture and sealing, or that the camera showed the scene "
    "faithfully.",
    "clip.mp4 and snapshot.jpg come from the event recorder, which assembles the clip on request from its stored video; "
    "their hashes identify the copies retrieved for this package.",
    "Files in derived/ were produced by software from the recording and are not themselves the recording.",
]


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def stamp(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()


def sha256_file(path: Path) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def save_stream(src, dest: Path, limit: int = 4 << 30) -> tuple[str, int]:
    """Copy a readable stream to dest unchanged, hashing as it goes. -> (sha256, bytes)."""
    h, n = hashlib.sha256(), 0
    with open(dest, "wb") as f:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            n += len(chunk)
            if n > limit:
                raise ValueError("file larger than the export limit")
            h.update(chunk)
            f.write(chunk)
    return h.hexdigest(), n


# ---------------------------------------------------------------------------------------------- hash chains
def chain_hash(prev: str, fields: dict) -> str:
    return hashlib.sha256(prev.encode() + b"\n" + canonical(fields)).hexdigest()


def entry_hash(prev: str, export_id: int, ts: str, actor: str, action: str, detail: str, remote: str) -> str:
    """Custody log entry (kept for the field order of the first format; same rule as chain_hash)."""
    return chain_hash(prev, {"export_id": export_id, "ts": ts, "actor": actor, "action": action, "detail": detail,
                             "remote": remote})


def verify_chain(rows: list[dict], fields: tuple[str, ...] | None = None) -> dict:
    """rows oldest first, each with prev_hash and entry_hash. fields = the hashed keys (default: custody log)."""
    keys = fields or ("export_id", "ts", "actor", "action", "detail", "remote")
    prev = GENESIS
    for r in rows:
        want = chain_hash(prev, {k: ("" if r.get(k) is None and k == "remote" else r.get(k)) for k in keys})
        if r["prev_hash"] != prev or r["entry_hash"] != want:
            return {"ok": False, "entries": len(rows), "broken_at": r["id"], "head": prev}
        prev = r["entry_hash"]
    return {"ok": True, "entries": len(rows), "broken_at": None, "head": prev}


def verify_ledger(lines: list[dict]) -> int | None:
    """Capture-time ledger excerpt (recorder/seal.py format): index of the first line that does not follow from
    the one before it, or None. The first line's prev is taken as given."""
    if not lines:
        return None
    prev = lines[0].get("prev")
    for i, d in enumerate(lines):
        fields = {k: v for k, v in d.items() if k not in ("prev", "hash")}
        if d.get("prev") != prev or d.get("hash") != chain_hash(prev, fields):
            return i
        prev = d["hash"]
    return None


# ---------------------------------------------------------------------------------------------- signing
class Signer:
    """Ed25519 key pair kept in key_dir (private key mode 600, created on first use)."""

    def __init__(self, key_dir: Path):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        key_dir.mkdir(parents=True, exist_ok=True)
        p = key_dir / "signing-key.pem"
        if not p.exists():
            k = Ed25519PrivateKey.generate()
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                        serialization.NoEncryption()))
        self._key = serialization.load_pem_private_key(p.read_bytes(), None)
        pub = self._key.public_key()
        self.public_pem = pub.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        raw = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")

    def sign(self, data: bytes) -> bytes:
        return self._key.sign(data)


def check_signature(public_pem: bytes, data: bytes, sig: bytes) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    try:
        serialization.load_pem_public_key(public_pem).verify(sig, data)
        return True
    except (InvalidSignature, ValueError):
        return False


def fingerprint_of(public_pem: bytes) -> str:
    from cryptography.hazmat.primitives import serialization
    raw = serialization.load_pem_public_key(public_pem).public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return "SHA256:" + base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")


# ---------------------------------------------------------------------------------------------- report
def _e(v) -> str:
    return html.escape("" if v is None else str(v))


def _rows(pairs) -> str:
    return "".join(f"<tr><th>{_e(k)}</th><td>{v}</td></tr>" for k, v in pairs if v not in (None, ""))


def _review_html(rv: dict | None) -> str:
    if not rv or not rv.get("latest"):
        return ('<div class="unrev">NOT CONFIRMED BY A PERSON. This match has not been reviewed. Treat it as an '
                'unverified lead only.</div>')
    last = rv["latest"]
    word = {"confirmed": "CONFIRMED", "rejected": "REJECTED", "unsure": "REVIEWED, NOT DECIDED"}.get(last["outcome"], last["outcome"])
    cls = "rev ok" if last["outcome"] == "confirmed" else "unrev"
    hist = "".join(f"<tr><td>{_e(r['ts'])}</td><td>{_e(r['reviewer'])}</td><td>{_e(r['outcome'])}</td><td>{_e(r.get('note'))}</td></tr>"
                   for r in rv.get("history") or [])
    return (f'<div class="{cls}">{word} by {_e(last["reviewer"])} on {_e(last["ts"])}'
            f'{(": " + _e(last.get("note"))) if last.get("note") else ""}</div>'
            f'<table><tr><th>When</th><th>Reviewer</th><th>Outcome</th><th>Note</th></tr>{hist}</table>')


def _item_html(it: dict, files: list[dict]) -> str:
    v = it.get("visit") or {}
    x = v.get("explain") or {}
    s = it["subject"]
    pre = it["prefix"]
    reasons = "".join(f"<li><code>{_e(r.get('code'))}</code> {_e(r.get('text'))}</li>" for r in x.get("reasons") or [])
    evidence = "".join(f"<li>{_e(t)}</li>" for t in x.get("evidence") or [])
    decisions = "".join(f"<tr><td>{_e(d.get('ts'))}</td><td>{_e(d.get('stage'))}</td><td>{_e(d.get('band'))} "
                        f"{_e(d.get('pct')) + '%' if d.get('pct') is not None else ''}</td><td>{_e(', '.join(d.get('reasons') or []))}</td>"
                        f"<td>{_e(d.get('message'))}</td></tr>" for d in v.get("decisions") or [])
    related = ""
    for r in v.get("related") or []:
        st = r.get("strength") or {}
        edges = "; ".join(_e(e.get("text")) for e in st.get("edges") or [])
        related += (f"<tr><td>{_e(r.get('type'))}</td><td>{_e(r.get('name'))}</td><td>{_e(r.get('start') or '')}</td>"
                    f"<td>{_e(st.get('band'))} {_e(st.get('pct'))}%</td><td>{edges}</td></tr>")
    segs = "".join(f"<tr><td><code>{_e(g['file'])}</code></td><td>{_e(g['status_text'])}</td><td>{_e(g.get('sealed_utc'))}</td>"
                   f"<td>{_e(g.get('anchor'))}</td></tr>" for g in it.get("segments") or [])
    have = {f["path"] for f in files}
    imgs = "".join(f'<figure><img src="{pre}{p}" alt=""><figcaption>{_e(c)} (<code>{pre}{p}</code>)</figcaption></figure>'
                   for p, c in (("original/snapshot.jpg", "Recorder snapshot, original"), ("derived/crop.jpg", "Crop of the person, derived"),
                                ("derived/evidence_card.jpg", "Evidence card, derived"), ("derived/match_strip.jpg", "Compared sightings, derived"))
                   if pre + p in have)
    missing = "".join(f"<li>{_e(t)}</li>" for t in it.get("missing") or [])
    gait, gc = v.get("gait") or {}, v.get("gait_compare") or {}
    gait_html = ""
    if gait.get("walking"):
        gait_html = (f"<h4>Walk</h4><p>{_e(gait.get('view'))} view, {_e(gait.get('seconds'))} s, cadence {_e(gait.get('cadence_spm'))} "
                     f"steps/min.</p>")
        if gc.get("rows"):
            gait_html += (f"<p><b>{_e(gc.get('text'))}</b> ({_e(gc.get('profile_walks'))} earlier walks)</p><table><tr><th>Measurement</th>"
                          "<th>This walk</th><th>Usual</th><th>Comparison</th></tr>"
                          + "".join(f"<tr><td>{_e(r['label'])}</td><td>{_e(r['this_text'])}</td><td>{_e(r['usual_text'])}</td>"
                                    f"<td>{_e(r['verdict'])}</td></tr>" for r in gc["rows"]) + "</table>")
    return f"""<section class="item"><h3>Item {_e(it['n'])}: {_e(s.get('identity'))}, {_e(s.get('camera'))}, {_e(s.get('start'))}</h3>
{_review_html(it.get('review'))}
{f'<div class="warn"><b>Not included for this item:</b><ul>{missing}</ul></div>' if missing else ''}
<table>{_rows([("Recorder event", f"<code>{_e(s.get('event'))}</code>"), ("Start (recorder clock)", _e(s.get("start"))),
               ("End (recorder clock)", _e(s.get("end"))), ("Camera", _e(s.get("camera"))), ("Filed under", _e(s.get("identity"))),
               ("Files", f"<code>{_e(pre)}</code>")])}</table>
<h4>How the match was made</h4>
<p><b>{_e(x.get('headline') or 'No explanation was stored for this sighting.')}</b>
{f"Band: <b>{_e(x.get('band'))}</b>, score {_e(x.get('pct'))}%." if x.get('band') else ''}</p>
{f'<p>What was compared:</p><ul>{evidence}</ul>' if evidence else ''}
{f'<p>Reason codes:</p><ul>{reasons}</ul>' if reasons else ''}
{gait_html}
{f'<h4>Continuous recording covering this event</h4><table><tr><th>Segment</th><th>Check</th><th>Sealed (UTC)</th><th>Day anchor</th></tr>{segs}</table>' if segs else ''}
{f'<h4>Other records it was linked to</h4><table><tr><th>Kind</th><th>Name</th><th>When</th><th>Strength</th><th>Why</th></tr>{related}</table>' if related else ''}
{f'<h4>Decision trail</h4><table><tr><th>When</th><th>Stage</th><th>Band</th><th>Reasons</th><th>Message</th></tr>{decisions}</table>' if decisions else ''}
{imgs}</section>"""


def render_report(m: dict, items: list[dict]) -> str:
    """Self-contained printable HTML. Everything from the database is escaped."""
    files = "".join(f"<tr><td><code>{_e(f['path'])}</code></td><td>{_e(f['role'])}</td><td class=n>{f['bytes']:,}</td>"
                    f"<td><code class=h>{_e(f['sha256'])}</code></td></tr>" for f in m["files"])
    ck = m.get("clock") or {}
    clock = _rows([
        ("Export host vs outside time service", _e(ck.get("export_host_text"))),
        ("Event recorder vs export host", _e(ck.get("recorder_text"))),
        ("Continuous recorder (segment seals)", _e(ck.get("sealer_text"))),
        ("Cameras", _e(ck.get("camera_text")))])
    unrev = sum(1 for it in items if not (it.get("review") or {}).get("latest") or it["review"]["latest"]["outcome"] != "confirmed")
    sig = m.get("signature") or {}
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Evidence report {_e(m['export_id'])}</title>
<style>body{{font:14px/1.5 system-ui,Segoe UI,Helvetica,Arial,sans-serif;color:#111;max-width:980px;margin:24px auto;padding:0 20px;counter-reset:sec}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:17px;margin:30px 0 8px;border-bottom:1px solid #999;padding-bottom:4px;counter-increment:sec}}
h2::before{{content:counter(sec) ". "}}h3{{font-size:15px;margin:18px 0 6px}}h4{{font-size:13.5px;margin:12px 0 4px}}
table{{border-collapse:collapse;width:100%;margin:6px 0}}th,td{{border:1px solid #bbb;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#eee;white-space:nowrap}}td.n{{text-align:right;font-variant-numeric:tabular-nums}}code{{font:12px ui-monospace,Consolas,monospace}}
code.h{{word-break:break-all}}.warn{{border:2px solid #a40000;padding:8px 12px;margin:12px 0}}.lead{{border:1px solid #666;padding:8px 12px;background:#f4f4f4}}
.unrev{{border:3px solid #a40000;color:#a40000;font-weight:800;font-size:17px;padding:10px 14px;margin:10px 0;letter-spacing:.02em}}
.rev{{border:2px solid #1b6e2a;color:#1b6e2a;font-weight:700;font-size:15px;padding:8px 12px;margin:10px 0}}
.item{{border-top:2px solid #333;margin-top:22px}}figure{{margin:10px 0}}img{{max-width:100%;max-height:400px;border:1px solid #999}}
figcaption{{font-size:12px;color:#444}}@media print{{body{{margin:0;max-width:none}}h2,h3{{break-after:avoid}}tr,figure{{break-inside:avoid}}.item{{break-before:page}}}}</style></head><body>
<h1>Evidence report</h1><p>Export <b>{_e(m['export_id'])}</b> &middot; case reference <b>{_e(m['case_ref'])}</b> &middot; {len(items)} item{'s' if len(items) != 1 else ''} &middot; format {_e(m['format'])}</p>
<p class="lead"><b>Read this first.</b> {_e(LIMITS[0])}</p>
{f'<div class="unrev">{unrev} of {len(items)} item{"s" if len(items) != 1 else ""} NOT confirmed by a person.</div>' if unrev else ''}
<h2>Export record</h2><table>{_rows([
    ("Exported by", _e(m["exported_by"]) + f" ({_e(m.get('exported_role'))})"), ("Exported (UTC)", _e(m["created_utc"])),
    ("Exported (local)", _e(m["created_local"])), ("Purpose", _e(m["purpose"])), ("Note", _e(m.get("note"))),
    ("Software version", _e(m["system"].get("version"))),
    ("Signed with", f"Ed25519 key <code>{_e(sig.get('fingerprint'))}</code> (<code>signing-key.pem</code>)" if sig else ""),
    ("Outside timestamp", _e(m.get("timestamp_note"))),
    ("Custody log head before this export", f"<code class=h>{_e(m['custody_prev'])}</code>")])}</table>
<h2>Clocks</h2><table>{clock}</table>
<h2>Items</h2>{''.join(_item_html(it, m['files']) for it in items)}
<h2>Files and their SHA-256 hashes</h2>
<p>To check: unzip, then run <code>sha256sum -c SHA256SUMS</code> or <code>python3 verify.py</code>. The signature and the
outside timestamp are checked as described in <code>README.txt</code>.</p>
<table><tr><th>File</th><th>Role</th><th>Bytes</th><th>SHA-256</th></tr>{files}</table>
<h2>Limits of this evidence</h2><ul>{''.join(f'<li>{_e(t)}</li>' for t in LIMITS)}</ul>
<p style="font-size:12px;color:#444">The machine-readable records are under <code>items/*/record/</code>. The custody log (who created,
downloaded and checked this package) is kept by the issuing installation.</p></body></html>"""


VERIFY_PY = r'''#!/usr/bin/env python3
"""Check an unzipped squirrel-guard evidence package: run `python3 verify.py` in its folder. Needs only Python 3;
checks the signature too when the `cryptography` package is installed (otherwise it prints the openssl command)."""
import hashlib, json, pathlib, sys
here = pathlib.Path(__file__).resolve().parent
m = json.loads((here / "manifest.json").read_text(encoding="utf-8"))
bad = 0

def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

for f in m["files"]:
    p = here / f["path"]
    if not p.is_file():
        print("MISSING ", f["path"]); bad += 1; continue
    ok = sha(p) == f["sha256"] and p.stat().st_size == f["bytes"]
    print("ok      " if ok else "CHANGED ", f["path"]); bad += not ok
fixed = {"manifest.json", "manifest.sig", "manifest.tsr", "signing-key.pem", "SHA256SUMS", "verify.py", "README.txt"}
listed = {f["path"] for f in m["files"]} | fixed
for p in sorted(here.rglob("*")):
    rel = p.relative_to(here).as_posix()
    if p.is_file() and rel not in listed:
        print("EXTRA   ", rel); bad += 1

# capture-time ledger: each segment's hash is in the ledger, and the ledger excerpt is an unbroken chain
for it in m.get("items", []):
    led = here / it["prefix"] / "record" / "ledger.jsonl"
    if not led.is_file():
        continue
    lines = [json.loads(l) for l in led.read_text().splitlines() if l.strip()]
    prev = lines[0]["prev"] if lines else None
    for i, d in enumerate(lines):
        fields = {k: v for k, v in d.items() if k not in ("prev", "hash")}
        if d["prev"] != prev or d["hash"] != hashlib.sha256(prev.encode() + b"\n" + canon(fields)).hexdigest():
            print("LEDGER BROKEN", led.relative_to(here), "line", i + 1); bad += 1; break
        prev = d["hash"]
    by = {d["file"].split("/")[-1]: d for d in lines}
    for g in it.get("segments", []):
        if not g.get("included"):
            continue
        p = here / it["prefix"] / "original" / "segments" / g["file"].split("/")[-1]
        d = by.get(p.name)
        if not d or not p.is_file():
            print("SEGMENT NOT IN LEDGER", p.name); bad += 1
        elif sha(p) != d["sha256"]:
            print("SEGMENT DIFFERS FROM ITS CAPTURE-TIME SEAL", p.name); bad += 1
        else:
            print("sealed  ", p.relative_to(here), "matches its seal of", d["sealed_utc"])

mh = sha(here / "manifest.json")
print("manifest.json sha256:", mh)
sig = here / "manifest.sig"
if sig.is_file():
    try:
        from cryptography.hazmat.primitives import serialization
        key = serialization.load_pem_public_key((here / "signing-key.pem").read_bytes())
        key.verify(sig.read_bytes(), (here / "manifest.json").read_bytes())
        print("signature: valid, key", m.get("signature", {}).get("fingerprint"), "(compare with the issuer's published fingerprint)")
    except ImportError:
        print("signature: not checked here; run: openssl pkeyutl -verify -pubin -inkey signing-key.pem -rawin -in manifest.json -sigfile manifest.sig")
    except Exception:
        print("signature: INVALID"); bad += 1
if (here / "manifest.tsr").is_file():
    print("outside timestamp: openssl ts -verify -data manifest.json -in manifest.tsr -CAfile /etc/ssl/certs/ca-certificates.crt")
print("RESULT:", "all files match the manifest" if not bad else "%d problem(s)" % bad)
sys.exit(1 if bad else 0)
'''

README = """squirrel-guard evidence package {export_id}
Case reference: {case_ref}

items/NN/original/  files from the recorders, unmodified (segments/ = continuous recording, sealed at capture time)
items/NN/derived/   produced by software from the recording
items/NN/record/    stored explanation (visit.json), human review (review.json), capture-time ledger excerpt
report.html         printable report: open it in a browser

Checks:
  sha256sum -c SHA256SUMS                       every file against the manifest
  python3 verify.py                             the same, plus extra files, ledger and segment seals, signature
  openssl pkeyutl -verify -pubin -inkey signing-key.pem -rawin -in manifest.json -sigfile manifest.sig
                                                Ed25519 signature; key fingerprint {fingerprint}
  openssl ts -verify -data manifest.json -in manifest.tsr -CAfile /etc/ssl/certs/ca-certificates.crt
                                                outside timestamp (only when manifest.tsr is present)
Compare the key fingerprint with the one the issuing agency publishes. The hash of manifest.json is also recorded
in the issuing installation's custody log.
"""


def build_package(dest: Path, items: list[dict], meta: dict, signer: Signer, timestamper=None) -> dict:
    """items: [{n, subject, visit, review, files: [{path (inside the item), src, role, description, source}],
    segments, ledger, anchors, missing}]. timestamper(sha256_hex) -> tsa.Stamp or raises. Returns the record."""
    entries, payload = [], []
    for it in items:
        it["prefix"] = f"items/{it['n']:02d}/"
        for f in it["files"]:
            digest, size = sha256_file(f["src"])
            entries.append({"path": it["prefix"] + f["path"], "role": f["role"], "sha256": digest, "bytes": size,
                            "description": f["description"], "source": f.get("source", "")})
            payload.append((it["prefix"] + f["path"], f["src"]))
        records = {"record/visit.json": it.get("visit") or {}, "record/review.json": it.get("review") or {"latest": None, "history": []}}
        blobs = {k: json.dumps(v, indent=1, ensure_ascii=False, default=str).encode() for k, v in records.items()}
        if it.get("ledger"):
            blobs["record/ledger.jsonl"] = "".join(json.dumps(d, sort_keys=True) + "\n" for d in it["ledger"]).encode()
        if it.get("anchors"):
            blobs["record/anchors.jsonl"] = "".join(json.dumps(d, sort_keys=True) + "\n" for d in it["anchors"]).encode()
        for k, b in blobs.items():
            entries.append({"path": it["prefix"] + k, "role": "record", "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b),
                            "description": k.split("/")[-1], "source": "database" if "json" in k and "ledger" not in k else "capture-time ledger"})
            payload.append((it["prefix"] + k, b))
    manifest = {"format": FORMAT, **meta, "signature": {"algorithm": "Ed25519", "fingerprint": signer.fingerprint},
                "items": [{"n": it["n"], "prefix": it["prefix"], "subject": it["subject"], "missing": it.get("missing") or [],
                           "review": (it.get("review") or {}).get("latest"),
                           "segments": [{k: g.get(k) for k in ("file", "status", "status_text", "included", "sha256", "sealed_utc", "anchor")}
                                        for g in it.get("segments") or []]} for it in items],
                "limitations": LIMITS, "files": entries}
    report = render_report(manifest, items).encode()
    manifest["files"].append({"path": "report.html", "role": "report", "sha256": hashlib.sha256(report).hexdigest(),
                              "bytes": len(report), "description": "Printable report", "source": "generated at export"})
    mbytes = json.dumps(manifest, indent=1, ensure_ascii=False, default=str).encode()
    msha = hashlib.sha256(mbytes).hexdigest()
    ts, ts_err = None, None
    if timestamper:
        try:
            ts = timestamper(msha)
        except Exception as e:                                    # recorded in the custody log; the package is still made
            ts_err = str(e)
    sums = "".join(f"{f['sha256']}  {f['path']}\n" for f in manifest["files"]) + f"{msha}  manifest.json\n"
    tmp = dest.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for path, src in payload:
            if isinstance(src, bytes):
                z.writestr(path, src)
            else:                                                 # video and pictures are already compressed
                z.write(src, path, compress_type=zipfile.ZIP_STORED)
        z.writestr("report.html", report)
        z.writestr("manifest.json", mbytes)
        z.writestr("manifest.sig", signer.sign(mbytes))
        z.writestr("signing-key.pem", signer.public_pem)
        if ts:
            z.writestr("manifest.tsr", ts.tsr)
        z.writestr("SHA256SUMS", sums)
        z.writestr("verify.py", VERIFY_PY)
        z.writestr("README.txt", README.format(fingerprint=signer.fingerprint, **meta))
    tmp.replace(dest)
    dest.chmod(0o440)
    psha, pbytes = sha256_file(dest)
    return {"manifest": manifest, "manifest_sha256": msha, "package_sha256": psha, "package_bytes": pbytes,
            "timestamp": ({"url": ts.url, "gen_time": stamp(ts.gen_time), "serial": ts.serial, "offset_s": ts.offset_s,
                           "round_trip_s": ts.round_trip_s} if ts else None), "timestamp_error": ts_err}


def _zhash(z: zipfile.ZipFile, name: str) -> str:
    h = hashlib.sha256()
    with z.open(name) as f:                                       # streamed: a segment can be larger than memory
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_package(path: Path, package_sha256: str, manifest_sha256: str, public_pem: bytes | None = None,
                   tsa_check=None) -> list[str]:
    """Problems found in a stored package ([] = intact): outer hash, manifest hash, signature, outside timestamp,
    every inner file. tsa_check(data, tsr) -> (ok|None, message), e.g. tsa.verify_openssl."""
    if not path.is_file():
        return ["the package file is missing from the store"]
    out = []
    if sha256_file(path)[0] != package_sha256:
        out.append("the package file does not match the hash recorded when it was created")
    try:
        with zipfile.ZipFile(path) as z:
            mbytes = z.read("manifest.json")
            names = set(z.namelist())
            if hashlib.sha256(mbytes).hexdigest() != manifest_sha256:
                out.append("manifest.json does not match the recorded hash")
            if public_pem is not None:
                if "manifest.sig" not in names or not check_signature(public_pem, mbytes, z.read("manifest.sig")):
                    out.append("the signature does not match this installation's key")
            if tsa_check and "manifest.tsr" in names:
                ok, msg = tsa_check(mbytes, z.read("manifest.tsr"))
                if ok is False:
                    out.append(f"the outside timestamp does not verify ({msg})")
            for f in json.loads(mbytes)["files"]:
                if f["path"] not in names:
                    out.append(f"{f['path']} is missing")
                elif _zhash(z, f["path"]) != f["sha256"]:
                    out.append(f"{f['path']} has been changed")
    except (zipfile.BadZipFile, KeyError, ValueError) as e:
        out.append(f"the package cannot be read ({type(e).__name__})")
    return out


# ---------------------------------------------------------------------------------------------- clocks
def _ahead(x: float, digits: int = 2) -> str:
    if abs(x) < 10 ** -digits:
        return "in step with"
    return f"{abs(x):.{digits}f} s {'ahead of' if x > 0 else 'behind'}"


def clock_record(export_probe=None, recorder_offset_s: float | None = None, sealer_anchor: dict | None = None,
                 camera_note: str = "") -> dict:
    """Plain-language clock statement for the manifest and report. export_probe: a tsa.Stamp (or None)."""
    out: dict = {}
    if export_probe is not None:
        u = export_probe.round_trip_s / 2
        out["export_host_offset_s"] = export_probe.offset_s
        out["export_host_text"] = (f"export host clock {_ahead(-export_probe.offset_s)} the outside time service "
                                   f"({export_probe.url}), uncertainty about ±{max(u, 0.01):.2f} s; measured at export")
    else:
        out["export_host_text"] = "not measured (no outside time service configured or it did not answer)"
    if recorder_offset_s is not None:
        out["recorder_offset_s"] = recorder_offset_s
        out["recorder_text"] = f"event recorder clock {_ahead(recorder_offset_s, 0)} the export host (HTTP Date header, ±1 s), measured at export"
    else:
        out["recorder_text"] = "not measured (the event recorder did not answer)"
    if sealer_anchor:
        out["sealer_offset_s"] = sealer_anchor.get("clock_offset_s")
        out["sealer_text"] = (f"continuous recorder clock {_ahead(-(sealer_anchor.get('clock_offset_s') or 0))} the outside time "
                              f"service on {sealer_anchor.get('date')} (daily ledger anchor)")
    else:
        out["sealer_text"] = "no daily anchor available for these segments"
    out["camera_text"] = camera_note or ("not measured: the cameras set their clocks by network time, but their offset "
                                         "is not recorded; the recorders time-stamp the video on arrival")
    return out

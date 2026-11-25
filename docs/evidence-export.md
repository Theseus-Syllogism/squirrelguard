# Evidence handling: roles, audit, review, packages, seals, release copies

What the review interface does so that a recording and the system's analysis of it can be defended later: who may
do what, a record of everything they did, a person's decision on every match that is used, and exports that anyone
can check without this software.

Code: `deploy/recorder/persons-ui/` (`access.py` roles and audit, `evidence_export.py` packages and chains, `tsa.py`
outside timestamps, `redact.py` release copies, routes in `app.py`) and `deploy/recorder/recorder/seal.py` (capture-time
seals). Tables: end of `deploy/recorder/persons_schema.sql`. Tests: `tests/test_evidence_export.py`.

## 1. Roles

| Role | May |
|---|---|
| viewer | read every page, search |
| investigator | also edit identities and vehicles, record reviews, export evidence, see and download **their own** exports |
| supervisor | also see every export, the custody log and the audit log (**Audit** in the top bar) |

With single sign-on the role comes from the user's Authelia groups (nginx passes `Remote-Groups`). Default mapping,
changeable with `ROLE_GROUPS` in the persons-ui environment:

```
supervisor=admins,sg-supervisor;investigator=sg-investigator;viewer=sg-viewer
```

A signed-in user in none of these groups is refused. Add an account with
`sudo /srv/camstack/authelia/user-password.sh NAME sg-investigator` (or `sg-viewer`, `sg-supervisor`): it can sign in
to the persons site only, with password and second factor. Without single sign-on (one shared password) the account
has `BASIC_ROLE` (default supervisor).

## 2. Audit log

Every page and API request that passes sign-in is appended to `audit_log`: time (UTC), user, role, what (`view:<page>`,
`change:<page>`, `review:<outcome>`, `denied`), the address and query (so searches are recorded with their words), and
the source address. Refusals are logged too. Pictures, scripts and styles are not (the page that shows them is).
**Audit** filters by user and kind and downloads the whole log as JSON.

## 3. Human review of a match

Every sighting page starts with its review state. Until a person decides, it reads **NOT CONFIRMED BY A PERSON** in
red. Investigators record *confirmed*, *rejected* (a note on what was compared is required) or *unsure*. Reviews are
never edited: a new one replaces the decision and the history stays. Exports carry the latest review and the full
history for every item, and the report prints unreviewed items in large red type.

## 4. Evidence packages

Open a sighting (one item) or an identity (all its sightings, up to 25) and choose **Export evidence package**. Tick
the items, give a case reference and a purpose, confirm you are authorised. The package is built in the background
(large recordings take a while) and lands on its own page.

```
items/NN/original/clip.mp4, snapshot.jpg      the event recorder's files as retrieved, unmodified
items/NN/original/segments/*.mkv              the continuous recording around the event (optional), see section 5
items/NN/derived/                             crop, evidence card, flip-book, match strip, track
items/NN/record/visit.json                    the explained match; also the walk comparison (docs/gait-analysis.md)
items/NN/record/review.json                   the review decision and history
items/NN/record/ledger.jsonl, anchors.jsonl   capture-time seals of the included segments
report.html                                   printable report
manifest.json                                 every file with SHA-256, size, role, origin; who, when, why; clocks
manifest.sig + signing-key.pem                Ed25519 signature of the manifest, and the public key
manifest.tsr                                  RFC 3161 timestamp of the manifest from an outside service
SHA256SUMS, verify.py, README.txt
```

Nothing is hidden: an event clip the recorder no longer had, a segment removed by retention, or a timestamp service
that did not answer are written into the package, its report and its page.

### Checking a package without this software

```
unzip EX-20261119-00012.zip -d pkg && cd pkg
sha256sum -c SHA256SUMS
python3 verify.py                     # files, extra files, ledger chain, segment seals, signature
openssl pkeyutl -verify -pubin -inkey signing-key.pem -rawin -in manifest.json -sigfile manifest.sig
openssl ts -verify -data manifest.json -in manifest.tsr -CAfile /etc/ssl/certs/ca-certificates.crt
```

Compare the key fingerprint (in `README.txt`, the report and on **Evidence**) with the one the agency publishes; the
public key is also at `/exports/public-key.pem`. The signing key is created on first export in `evidence-keys/` on
the recorder host (mode 600): back it up, keep it private, and do not replace it, or older packages can no longer be
traced to this installation.

### Outside timestamps

`TSA_URL` (default `http://timestamp.digicert.com`) receives the SHA-256 of `manifest.json`, nothing else, and returns
a token signed by the timestamp authority. It proves the manifest existed at that time without trusting this
installation's clock or database. Because every manifest also records the custody log's head hash at export, each
token anchors the custody log as well. Set `TSA_URL` empty to switch it off.

### Clocks

Every export records, in plain words: the export host's clock against the outside time service (from a separate
timestamp request, with its uncertainty), the event recorder's clock against the export host (HTTP `Date`, ±1 s), and
the continuous recorder's clock against the outside service from the most recent daily ledger anchor. Camera clocks
are not measured; `CAMERA_CLOCK_NOTE` can state how the cameras keep time (e.g. "NTP from the local gateway, checked
weekly").

## 5. Capture-time seals (continuous recording)

The `sealer` service hashes every closed 10-minute segment into a per-camera hash chain (`ledger/<camera>.jsonl`) a
few minutes after it is written, and once a day has the chain head timestamped by the outside service
(`ledger/anchors/<camera>.jsonl`). At export each segment around the event is hashed again and compared:

| Shown | Meaning |
|---|---|
| matches its capture-time seal | unchanged since it was sealed |
| DOES NOT MATCH its capture-time seal | changed after sealing: the package says so in red |
| not sealed yet | still recording, or the sealer was not running |
| removed by retention | the hash is in the ledger, the file is gone |

A broken ledger is never rewritten: the sealer stops sealing that camera and reports it every minute. The ledger lives
on the system disk (`/srv/camstack/ledger`), not on the video disk; add it to the nightly backup.

## 6. Release copies (redacted)

**Make release copy** on a package page writes a second, signed ZIP with every face and number plate the detectors
find blurred (OpenCV YuNet faces, open-image-models plate detector; both MIT-licensed code). It covers the event clip
and the stills, not the continuous segments, and has no sound. Its manifest names the source package's manifest hash,
and each redacted file names the hash of the file it came from. **Detectors miss things** (small or turned-away faces,
plates at steep angles): a person must check every frame before release, as its README says.

## 7. Custody and audit chains

Both logs use `entry_hash = sha256(prev_hash + "\n" + canonical JSON of the entry)`. All evidence tables refuse UPDATE,
DELETE and TRUNCATE (triggers); an edit that bypasses the triggers shows as a broken chain on **Evidence** and **Audit**.
The JSON downloads include the rule, so they can be re-checked independently.

## What this does not prove

- That the camera showed the scene faithfully, or anything about the minutes between capture and sealing.
- Who sat at the keyboard: the logs record the signed-in account.
- That a match is right: a confirmed review is a person's judgement, recorded; the report says so.

This is a technical record, not legal advice. Whether it meets a court's or an agency's evidence rules is for their
counsel and evidence custodian.

## Setup (recorder host)

`setup.sh` creates `evidence-store/`, `evidence-keys/` and `ledger/` (owner uid 1000) and applies the schema. The
`sealer` service and the new persons-ui image (OpenCV, ONNX Runtime, cryptography, openssl; about 250 MB more) come
up with `docker compose up -d --build`. Back up `evidence-store/`, `evidence-keys/` and `ledger/` with the database.

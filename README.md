# squirrel-guard

Local camera analytics that started as a garden squirrel deterrent: a detector finds an animal, a second (local)
vision model must agree, and only then does the camera's speaker make a noise. It grew into a private, fully local
analytics stack for a home or small site: recording, people and vehicle analysis for human review, and evidence
exports that a third party can check without this software.

**Nothing leaves the local network.** No cloud service, no outside lookups, no names for anyone who has not been
enrolled with consent, and no automatic action against a person: every match is a suggestion for a human reviewer.

> **Status: research project, not a product.** Features are measured on small or synthetic test sets, and the
> [Feature and Capability Guide](FEATURE-GUIDE.pdf) states each one's status, measured accuracy and
> limits. Several optional models are licensed for **non-commercial use only** (see [Model licences](#model-licences)).

<p align="center"><img src="docs/public/img/screens/replay-demo.jpg" alt="3D replay of a walking person: path, footprints, fitted body and the camera image projected onto the ground" width="100%"><br>
<sub>3D replay of a stock video sample: path, footprints, fitted body and the camera image projected onto the ground.</sub></p>

## What is in here

| Part | What it does | Where |
|---|---|---|
| Detector | camera stream -> motion gate -> YOLOv8 (Open Images V7, ONNX on CPU) -> N-of-M trigger -> local vision-model check -> deterrent | `squirrel_guard/main.py` |
| Console | install, configure and run any machine role: commands, a terminal menu and a local web page (`./sg`) | `squirrel_guard/console/`, [docs/console.md](docs/console.md) |
| Recorder stack | stream relay, continuous recording with capture-time seals, event database, alerts, single sign-on, review interfaces | `deploy/recorder/` |
| GPU server stack | Frigate NVR, the listener that acts on its events, and the profiler (people and vehicle analysis) | `deploy/gpuserver/` |
| Review interfaces | visitor log with explained matches and human review, evidence packages, audit log, 3D replay | `deploy/recorder/persons-ui/`, `deploy/recorder/replay/` |
| Analysis modules | re-identification, gait measurements, faces of enrolled people, plates, colour, paths, 3D scenes | `squirrel_guard/` |
| Documentation | feature guide (PDF and Markdown), architecture diagrams, per-module notes | `docs/public/`, `docs/` |

```
camera -> stream relay -> NVR (detect, track) -> MQTT -> workers (verify, deter, profile) -> database + rules
       -> push alert to a person -> review on web pages behind single sign-on -> optional signed evidence export
```


<img src="docs/public/diagrams/2-event-pipeline.png" alt="One event from camera frame to alert" width="100%">
<sub>One event from camera frame to alert. Animals take the top path; people and vehicles go through the profiler lane.</sub>

## A look at it

All screenshots come from the [Feature and Capability Guide](FEATURE-GUIDE.pdf). They use test data, invented
names and example addresses; faces and camera views are blurred for publication.

**Detection.** The shipped detector on stock photos: the cat is not called a squirrel.

<img src="docs/public/img/detector-examples.jpg" alt="Detector boxes on three stock photos: squirrel, cat, car" width="100%">

**Review: every match is explained and decided by a person.**

<table>
<tr><td width="50%" valign="top"><img src="docs/public/img/screens/pane-overview.jpg" alt="Single pane of glass: the match band and its reasons, objects seen, estimates labelled as estimates." width="100%"><br><sub>Single pane of glass: the match band and its reasons, objects seen, estimates labelled as estimates.</sub></td><td width="50%" valign="top"><img src="docs/public/img/screens/persons-why.jpg" alt="The "Why?" page: the decision in one sentence, its reason code and the evidence behind it." width="100%"><br><sub>The "Why?" page: the decision in one sentence, its reason code and the evidence behind it.</sub></td></tr>
<tr><td width="50%" valign="top"><img src="docs/public/img/screens/persons-index.jpg" alt="Identity list: unknown people stay numbered; only enrolled people have names." width="100%"><br><sub>Identity list: unknown people stay numbered; only enrolled people have names.</sub></td><td width="50%" valign="top"><img src="docs/public/img/screens/persons-alerts.jpg" alt="Alerts: only unknown or watch-listed subjects and long-parked vehicles; household members never." width="100%"><br><sub>Alerts: only unknown or watch-listed subjects and long-parked vehicles; household members never.</sub></td></tr>
<tr><td width="50%" valign="top"><img src="docs/public/img/screens/pane-timeline.jpg" alt="Timeline of every visit, with one click to the clip, the 3D replay and the evidence page." width="100%"><br><sub>Timeline of every visit, with one click to the clip, the 3D replay and the evidence page.</sub></td><td width="50%" valign="top"><img src="docs/public/img/screens/persons-identity.jpg" alt="An identity: key facts as cards, actions in one bar, enrolled faces with their consent record (blurred)." width="100%"><br><sub>An identity: key facts as cards, actions in one bar, enrolled faces with their consent record (blurred).</sub></td></tr>
</table>

**3D replay and people analysis.**

<table>
<tr><td width="50%" valign="top"><img src="docs/public/img/screens/replay-demo-pane.jpg" alt="Replay pane on the demo sample: duration, distance walked, stops, heading, step rate." width="100%"><br><sub>Replay pane on the demo sample: duration, distance walked, stops, heading, step rate.</sub></td><td width="50%" valign="top"><img src="docs/public/img/screens/gait-walk-comparison.png" alt="A walk compared with the same identity's earlier walks (synthetic walker)." width="100%"><br><sub>A walk compared with the same identity's earlier walks (synthetic walker).</sub></td></tr>
<tr><td width="50%" valign="top"><img src="docs/public/img/screens/face3d.jpg" alt="3D face viewer, redacted for publication: deviation from the same person's average shape." width="100%"><br><sub>3D face viewer, redacted for publication: deviation from the same person's average shape.</sub></td><td width="50%" valign="top"><img src="docs/public/img/screens/pane-related.jpg" alt="Related sightings within 90 days, ranked by the evidence named on each link." width="100%"><br><sub>Related sightings within 90 days, ranked by the evidence named on each link.</sub></td></tr>
</table>

**Evidence handling: tamper-evident by design.**

<table>
<tr><td width="50%" valign="top"><img src="docs/public/img/screens/evidence-review.png" alt="Human review at the top of a sighting: decision, reviewer, time and note." width="100%"><br><sub>Human review at the top of a sighting: decision, reviewer, time and note.</sub></td><td width="50%" valign="top"><img src="docs/public/img/screens/evidence-export-form.png" alt="Export form: unreviewed items are flagged, and the purpose must be stated." width="100%"><br><sub>Export form: unreviewed items are flagged, and the purpose must be stated.</sub></td></tr>
<tr><td width="50%" valign="top"><img src="docs/public/img/screens/evidence-package.png" alt="A package page detecting a recording segment changed after it was sealed." width="100%"><br><sub>A package page detecting a recording segment changed after it was sealed.</sub></td><td width="50%" valign="top"><img src="docs/public/img/screens/evidence-audit.png" alt="The audit log reporting the exact entry where its hash chain breaks." width="100%"><br><sub>The audit log reporting the exact entry where its hash chain breaks.</sub></td></tr>
</table>

**Install and operate.**

<table>
<tr><td width="50%" valign="top"><img src="docs/public/img/screens/console-install.png" alt="Web console install page: each step shows the commands it will run first." width="100%"><br><sub>Web console install page: each step shows the commands it will run first.</sub></td><td width="50%" valign="top"><img src="docs/public/img/screens/console-checks.png" alt="Preflight checks: each failure names the install step that fixes it." width="100%"><br><sub>Preflight checks: each failure names the install step that fixes it.</sub></td></tr>
</table>

**Architecture.**

<table>
<tr><td width="50%" valign="top"><img src="docs/public/diagrams/1-software-stack.png" alt="Software stack by layer, coloured by the machine role that runs each part." width="100%"><br><sub>Software stack by layer, coloured by the machine role that runs each part.</sub></td><td width="50%" valign="top"><img src="docs/public/diagrams/3-deployment-and-trust.png" alt="Machine roles, network zones and permitted connections." width="100%"><br><sub>Machine roles, network zones and permitted connections.</sub></td></tr>
</table>

## Quick start: one detector on one camera

Needs Linux, Python 3.11 or newer, ffmpeg, and an RTSP camera (any camera with an RTSP stream; the project was
developed with Thingino-firmware cameras). A local vision model (llama.cpp with a vision model and its `--mmproj`
file, or any OpenAI-compatible server that accepts images) is optional but recommended.

```
git clone <this repository> && cd squirrel-guard
./sg setup --dry-run      # shows every install step and its commands, changes nothing
./sg setup                # the install menu: Python environment, packages, model export, configuration
./sg env set SG_RTSP_URL  # camera stream address, asked for hidden, stored in a private .env file
./sg endpoints set verifier http://127.0.0.1:8080   # your vision model server; `./sg endpoints test` checks it
./sg check                # preflight checks with what to do about each failure
./sg service start
```

`./sg` alone opens the terminal menu; `./sg gui` opens the web console on this machine only. The deterrent starts
in **dry run** (`actuator.dry_run = true`): events are logged and nothing plays until you switch it on.

The full multi-machine deployment (recorder host + GPU server) is described in `deploy/recorder/` and
`deploy/gpuserver/`; the files there are a working reference deployment with example addresses (192.0.2.x,
198.51.100.x, `cam.example.org`). Replace them with your own.

## Measured, briefly

- Animal detector, 13 stock photos at 640x360: 8 of 8 squirrels found, 0 of 4 cats called squirrel, ~90 ms per frame on a CPU.
- Second-opinion vision model agreed on 13 of 13; ~1.3 s per check on a GPU server, 15-20 s on a laptop CPU.
- Stock photos are close-ups. Real garden squirrels are smaller; measure on your own camera before trusting any figure.

Everything else, including what has **not** been validated, is in the [Feature and Capability Guide](FEATURE-GUIDE.pdf).

## Privacy and safety

The software refuses to: name anyone not enrolled with consent; look anything up outside; infer race or ethnicity;
use age, gender, gait or face shape to decide identity; act automatically on a person. Identity features are
optional and should stay off unless there is a lawful basis and documented consent. Recording people, neighbours,
public areas or audio is regulated differently in different places: follow your local law. See chapter 12 of the guide.

<img src="docs/public/diagrams/4-privacy-and-retention.png" alt="Retention periods and the rules the software enforces" width="100%">
<sub>Retention periods and the rules the software enforces.</sub>

## Model licences

Model weights are **not** in this repository; scripts download or export them on your machine. Their licences
differ and some are restrictive:

- Ultralytics YOLOv8 code and weights: **AGPL-3.0**.
- Face recognition (InspireFace, AdaFace weights), attribute models trained on PA100K / VeRi-776, action models
  trained on NTU RGB+D, the SMPL body model and VGGT-1B: **non-commercial / research only**.
- The learned gait model (OpenGait GaitBase) has no clear licence.

The full table is in chapter 14 of the guide. Check every licence before any use beyond personal research.

## Tests

```
.venv/bin/python -m pytest tests          # some tests need PostgreSQL (psycopg) or Docker and skip without them
```

## Licence

Copyright the squirrel-guard contributors. Licensed under the GNU Affero General Public License v3.0 (see
[LICENSE](LICENSE)). Bundled third-party code keeps its own licence (e.g. `squirrel_guard/third_party/`, Apache-2.0).
Documentation icons: Lucide (ISC); software logos in the guide: Simple Icons (CC0), trademarks of their owners.

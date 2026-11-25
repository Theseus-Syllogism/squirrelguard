# Replay investigation pane ("single pane of glass")

The 3D trajectory replay (`deploy/recorder/replay/`, served by the persons UI at `/replay/`) has a sub-dashboard over
the scene: click or tap the person (stick figure, 3D body, current video cut-out or the position marker), press
**Inspect person**, or focus the 3D view and press **I** / **Enter**. Everything opens in the same page: clips play
inline, related 3D replays load into the same viewer (browser Back returns), only the Why? / identity / Frigate pages
open in a new tab.

The pane is draggable by its header on a desktop (double-click the header to put it back; the position is remembered
per browser), collapses with ▾, and closes with × or Esc. Under 560 px it is a bottom sheet. Keyboard: tabs move with
←/→/Home/End, graph nodes are focusable buttons, Esc closes the open card first.

## Panels

| Tab | Shows | Source |
|---|---|---|
| Overview | headline with band and estimated %, reason codes in words, why-this-match evidence, what they did (stay, walk, carried objects, skeleton actions, the LLM's activity), apparent look (clothing, apparent age/gender, PULC tags, height, step rate, apparent expression as a description), routine surprise against the alert threshold, the replay's own movement facts (path, speed, stops, head direction), alerts and the decision trail, evidence card, links | `sightings.explain`, `attributes`, `pulc`, `objects`, `actions`, `routine`, `height_cm`, `gait`, `behavior`; `alerts`, `decisions`; the replay JSON's `metrics` and `gaze.summary` |
| Face | the visit's face crops, the enrolled face it matched (only for an actual gated match of a household/known identity), the decision with the cosine against the 0.40 threshold, liveness, quality, face size, consistency | `sightings.face_meta`, `face`, `face_templates` (crop only) |
| Related | node-link graph: this visit in the centre; its identity; the "possible" identity; up to 10 earlier visits; vehicles within ±10 min. Closer and thicker = stronger; edge colour = band; edge label = the evidence (re-ID, face, gait, assigned, possible, minutes for vehicles). Click a node for its card: thumbnail, time, every piece of evidence with its value and whether it raised or lowered the estimate, and actions (play clip, open 3D replay, focus the pane on that visit, show an identity's timeline, Why?, Frigate). A plain list of all related items follows the graph. | `/api/visit/<event>` `related` |
| Clips | this visit's clip plus earlier visits that correlate at "possible" (40 %) or better, as Frigate thumbnails; plays inline | Frigate `/api/events/<id>/clip.mp4` and `thumbnail.webp` |
| Timeline | the identity's visits on a time strip and by day (newest 200), its median height and step rate, enrolled-face count, vehicles seen with it; per visit: play, 3D, focus, Why? | `/api/identity/<id>` |

Demo samples (`demo/*.json`) have no stored visit: the pane shows Overview (source, movement, head direction),
**Scene & model** (calibration, replay media, the WHAM body fit) and an empty Related tab saying that related-sighting
analysis appears on real visits. Nothing is invented for them. (The basement sample is a hand-recorded clip, not a
Frigate event, so there is no visit to link.)

## How "related" is ranked (`deploy/recorder/persons-ui/related.py`)

Candidates: sightings in the 90 days before the visit (`PANE_RELATED_DAYS`), newest 600 (`PANE_RELATED_CANDIDATES`).
For each one, the same arithmetic as the bridge's readable scores (`bridge/explain.py`):

1. start from the re-ID appearance probability of the embedding cosine (logistic: 0.70 -> 70 %, 0.55 -> 40 %);
2. face templates (both visits have `face_vec`): cosine >= 0.40 x4 odds, >= 0.30 x1.5, < 0.15 x0.5;
3. gait signature (`gait_vec`, fetched only for the best 40): distance <= 0.05 x1.5, >= 0.15 x0.6;
4. filed under the same identity x2 (x3 if reviewed).

The top 12 (`PANE_RELATED_TOP`) are returned with every factor listed. Vehicles: closer in time and seen near this
identity before (2-minute co-occurrence) = stronger. All percentages are estimates for ranking and say so; they never
change an identity, an alert or a decision (the bridge's matching is untouched).

## API (persons-ui, same sign-in)

* `GET /api/visit/<frigate event id>`: the visit, its explanation, face summary, attributes, objects, actions,
  routine, height/gait, evidence URLs, Frigate links, alerts, decisions and `related` (each with `strength
  {p, pct, band, edges[{kind, value, effect, text}]}` and links).
* `GET /api/identity/<id>`: the identity, its visits (newest 200), height/step-rate medians, enrolled-face count,
  vehicles seen with it.

The viewer finds the event from the replay JSON's `event` field (written by the profiler from now on, with
`camera_name`) or, for older files, from the `<event>_traj.json` file name. `?api=` points the viewer at another API
base (tests use a static stub).

Test: `tests/test_replay_pane_api.py` (throwaway `postgres:16-alpine` via docker; ranking, window, vehicles, auth).

## Privacy

* **SSO only.** `/api/*` sits behind the same Authelia check as every persons-UI page (the app also refuses requests
  without `Remote-User`, so nothing, not even whether an event exists, is visible signed out). Responses are
  `Cache-Control: private, no-store`. No outside lookups: the API reads the local database only; Frigate clips and
  thumbnails are loaded by the signed-in browser itself.
* **Names come only from the user.** Identities show the label the user gave, else "Unknown #n". A face names someone
  only through the bridge's enrolled-template match (household/known, consented, liveness passed); the pane shows the
  enrolled crop only for such a match.
* **Face similarity between two visits** is shown as correlation evidence (a number on an edge) and never produces a
  name. It uses the per-visit templates that already exist and follow the 90-day retention of unknown identities.
  `PANE_FACE_LINKS=0` in the persons-ui environment turns it off (face edges then disappear; nothing else changes).
* **Expression is a description only** ("apparent expression"), never an input to identity, ranking or alerts.
  Apparent age/gender are labelled as estimates. No race/ethnicity anywhere.

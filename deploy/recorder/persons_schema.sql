-- Person sightings and identities (private visitor log). Fingerprints are OSNet re-ID vectors (appearance,
-- mostly clothing). Age/gender are APPARENT guesses from the LLM, stored with their confidence.
CREATE TABLE IF NOT EXISTS identities (
    id          serial PRIMARY KEY,
    label       text,                                   -- NULL -> shown as "Unknown #<id>"
    kind        text NOT NULL DEFAULT 'unknown',        -- unknown | household | known
    notes       text,
    centroid    real[] NOT NULL,                        -- running mean fingerprint (L2-normalised)
    n           integer NOT NULL DEFAULT 1,             -- sightings folded into the centroid
    first_seen  timestamptz NOT NULL,
    last_seen   timestamptz NOT NULL
);
CREATE TABLE IF NOT EXISTS sightings (
    id              bigserial PRIMARY KEY,
    frigate_event   text UNIQUE NOT NULL,
    camera          text NOT NULL,
    ts_start        timestamptz NOT NULL,
    ts_end          timestamptz,
    score           real,
    crops           integer,
    embedding       real[] NOT NULL,
    attributes      jsonb,
    age_low         integer,
    age_high        integer,
    gender          text,
    gender_conf     real,
    crop_file       text,
    identity_id     integer REFERENCES identities(id) ON DELETE SET NULL,
    match_sim       real,                               -- cosine to the identity centroid at assignment
    match_kind      text,                               -- match | new | new_possible (0.55-0.70 to possible_id)
    possible_id     integer,
    possible_sim    real,
    reviewed        boolean NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS sightings_identity_idx ON sightings (identity_id);
CREATE INDEX IF NOT EXISTS sightings_ts_idx ON sightings (ts_start);

-- Vehicles. Plate text comes from Frigate's LPR (stored for matching only; no outside lookups).
CREATE TABLE IF NOT EXISTS vehicles (
    id          serial PRIMARY KEY,
    label       text,
    kind        text NOT NULL DEFAULT 'unknown',        -- unknown | household | known
    notes       text,
    plate       text,                                   -- normalised (A-Z0-9)
    body_type   text,
    color       text,
    make_model  text,
    centroid    real[] NOT NULL,
    n           integer NOT NULL DEFAULT 1,
    first_seen  timestamptz NOT NULL,
    last_seen   timestamptz NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS vehicles_plate_uidx ON vehicles (plate) WHERE plate IS NOT NULL;
CREATE TABLE IF NOT EXISTS vehicle_sightings (
    id              bigserial PRIMARY KEY,
    frigate_event   text UNIQUE NOT NULL,
    camera          text NOT NULL,
    label           text,
    ts_start        timestamptz NOT NULL,
    ts_end          timestamptz,
    score           real,
    crops           integer,
    plate           text,
    plate_score     real,
    embedding       real[] NOT NULL,
    attributes      jsonb,
    crop_file       text,
    vehicle_id      integer REFERENCES vehicles(id) ON DELETE SET NULL,
    match_sim       real,
    match_kind      text,                               -- plate | appearance | new | new_possible | merged
    possible_id     integer,
    possible_sim    real
);
CREATE INDEX IF NOT EXISTS vehicle_sightings_vehicle_idx ON vehicle_sightings (vehicle_id);
CREATE INDEX IF NOT EXISTS vehicle_sightings_ts_idx ON vehicle_sightings (ts_start);

-- Phase 3/4: behaviour summaries, alerts, reports
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS behavior jsonb;
ALTER TABLE vehicle_sightings ADD COLUMN IF NOT EXISTS behavior jsonb;
CREATE TABLE IF NOT EXISTS alerts (
    id          bigserial PRIMARY KEY,
    ts          timestamptz NOT NULL DEFAULT now(),
    kind        text NOT NULL,          -- night_unknown | repeat_unknown | linger_unknown | parked_vehicle | recurring_vehicle | zone_unknown
    subject     text NOT NULL,          -- person:<id> | vehicle:<id>
    dedupe_key  text UNIQUE NOT NULL,
    message     text NOT NULL,
    sighting    text,                   -- frigate_event
    sent        boolean NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS alerts_ts_idx ON alerts (ts);
CREATE TABLE IF NOT EXISTS reports (
    id          bigserial PRIMARY KEY,
    period      text NOT NULL,          -- daily | weekly
    ts          timestamptz NOT NULL DEFAULT now(),
    since       timestamptz NOT NULL,
    until       timestamptz NOT NULL,
    facts       jsonb NOT NULL,
    text        text
);
-- Phase 2/3: height (needs camera calibration) and gait, from the end-of-event keypoint analysis
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS height_cm real;
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS height_iqr_cm real;
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS height_frames int;
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS gait jsonb;
-- Readable scores + visual evidence
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS explain jsonb;          -- explain.explain_person() result
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS evidence_card text;     -- files under snapshots/evidence/
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS evidence_gif text;
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS match_strip text;
CREATE TABLE IF NOT EXISTS decisions (
  id bigserial PRIMARY KEY,
  ts timestamptz NOT NULL DEFAULT now(),
  stage text NOT NULL,              -- identity_match | motion | alert | action
  frigate_event text,
  subject text,                     -- person:<id> / vehicle:<id>
  band text,                        -- strong | likely | possible | unlikely
  pct int,                          -- estimated probability the decision is right
  reasons text[] NOT NULL DEFAULT '{}',
  scores jsonb,
  message text,                     -- the same sentence emails and ntfy show
  versions jsonb
);
CREATE INDEX IF NOT EXISTS decisions_event ON decisions (frigate_event);
CREATE INDEX IF NOT EXISTS decisions_ts ON decisions (ts);
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS actions jsonb;   -- skeleton action tags (fall, stagger, throw, ...)
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS gait_vec real[];    -- GaitBase (GREW) signature, 4096-d, L2-normalised
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS trajectory_file text;   -- evidence/<event>_traj.json for the 3D replay
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS fused_trajectory_file text;   -- evidence/<primary event>_fused_traj.json: one
                                   -- rig over every camera that saw the visit (squirrel_guard.fusion, docs/fusion.md)
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS objects jsonb;   -- open-vocabulary objects (YOLO-World): label, score, frames, carried
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS routine jsonb;   -- routine.Routine score: bits, parts, reasons, unusual
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS face jsonb;      -- Frigate face recognition: {name, score} (enrolled people only)
-- Face identification (squirrel_guard/faceid.py: InspireFace + FASR-inspired per-visit template; docs/faceid.md).
-- face_vec: the visit's 512-d face template (L2-normalised); kept with the sighting, so it follows the 90-day
-- retention of unknown identities and is deleted with them. face_meta: counts, quality, liveness, apparent
-- expression (description only), the match decision; never race/ethnicity.
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS face_vec real[];
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS face_meta jsonb;
-- Enrolled face templates: ONLY for household/known identities, created from the persons UI with the person's
-- consent; deleted with the identity (CASCADE) or when it stops being household/known.
CREATE TABLE IF NOT EXISTS face_templates (
    id            serial PRIMARY KEY,
    identity_id   integer NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    vec           real[] NOT NULL,
    model         text,                 -- InspireFace pack (templates of different packs are not comparable)
    aggregator    text,
    n_faces       integer,
    quality       real,
    liveness      real,
    source_event  text,                 -- sighting it was taken from
    crop_jpeg     bytea,                -- small face crop (evidence of what was enrolled); deleted with the row
    note          text,
    created       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS face_templates_identity_idx ON face_templates (identity_id);
-- Face geometry (squirrel_guard/facegeom.py, docs/facegeom.md): EXPLANATORY evidence, never an identity decision.
-- face_geom: per-visit proportions (/ inter-pupil distance) and relative depths with spread + n, face px/pose used,
-- files (snapshots/evidence/<event>_facegeom_{overlay,depth}.jpg, _mesh.gltf) and the comparison with the matched
-- identity's enrolled geometry. Never race/ethnicity. face_templates.geom: the geometry of the visit a template was
-- enrolled from (copied by the bridge; deleted with the template).
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS face_geom jsonb;
ALTER TABLE face_templates ADD COLUMN IF NOT EXISTS geom jsonb;
-- PaddleClas PULC attributes (squirrel_guard/pulc.py): visit summary {attributes: {label: mean p}, orientation, views,
-- apparent: {gender_female, age}, tags, spread, crops} or {crops: 0, skipped}; vehicles {color, type, colors, types}.
-- APPARENT estimates; used only as weak evidence (explain.py PULC_* reasons) and for /search filters.
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS pulc jsonb;
ALTER TABLE vehicle_sightings ADD COLUMN IF NOT EXISTS pulc jsonb;
-- Fitted face (squirrel_guard/facefit.py, docs/facefit.md): one face shape over the whole visit (multi-frame, lens-aware,
-- metric only with two calibrated cameras), features with bootstrap sd, detail levels LOD0-3 in
-- snapshots/evidence/<event>_facefit.glb. EXPLANATORY evidence, never an identity decision; never race/ethnicity.
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS face_fit jsonb;
-- Face recognition v2 (squirrel_guard/faceid_v2.py, docs/faceid-v2.md): the visit's AdaFace IR-101 template (512-d).
-- Enrolled AdaFace templates are ordinary face_templates rows with model = the AdaFace model name.
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS face_vec_v2 real[];
-- Vehicle occupant faces (squirrel_guard/occupants.py, docs/faceid-v2.md "Vehicle occupants") and unnamed-face
-- auto-tracking (facematch.decide_unknown, FACE_AUTO_TRACK). Only the persons UI (the user) ever labels or enrols an
-- identity; auto-tracked identities stay kind 'unknown' with no label. Never race/ethnicity.
-- identities.origin: NULL = founded by a person sighting; 'vehicle_occupant' = founded by a face seen inside a vehicle
-- (no body fingerprint: centroid all zeros, so body re-ID never matches it).
ALTER TABLE identities ADD COLUMN IF NOT EXISTS origin text;
-- vehicle_sightings: occupants = faces groups found in the cabin (a count); occupant_face = per-occupant decisions
-- (status, sim, reasons; identity only for an enrolled match or a tracked unnamed identity); the driver link.
ALTER TABLE vehicle_sightings ADD COLUMN IF NOT EXISTS occupants integer;
ALTER TABLE vehicle_sightings ADD COLUMN IF NOT EXISTS occupant_face jsonb;
ALTER TABLE vehicle_sightings ADD COLUMN IF NOT EXISTS driver_identity_id integer REFERENCES identities(id) ON DELETE SET NULL;
ALTER TABLE vehicle_sightings ADD COLUMN IF NOT EXISTS driver_sim real;
ALTER TABLE vehicle_sightings ADD COLUMN IF NOT EXISTS driver_kind text;      -- face_template | face_unknown | face_new
CREATE INDEX IF NOT EXISTS vehicle_sightings_driver_idx ON vehicle_sightings (driver_identity_id);
-- One row per occupant face LINKED to an identity (unusable faces are never stored). vec/vec_v2 only for unnamed
-- tracked identities (enrolled people match through face_templates). Deleted with the vehicle sighting or the identity,
-- and after the 90-day retention for anyone not household/known (bridge occupant_retention). crop_file: one crop in
-- snapshots/faces/<event>_occ<slot>_face0.jpg (removed by the bridge sweep once its row is gone).
CREATE TABLE IF NOT EXISTS occupant_faces (
    id                  bigserial PRIMARY KEY,
    vehicle_sighting_id bigint NOT NULL REFERENCES vehicle_sightings(id) ON DELETE CASCADE,
    frigate_event       text NOT NULL,
    ts                  timestamptz NOT NULL,
    slot                integer,
    identity_id         integer NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    match_kind          text,                 -- face_template | face_unknown | face_new
    sim                 real,
    vec                 real[],
    vec_v2              real[],
    model               text,
    v2_model            text,
    meta                jsonb,                -- n_faces, quality, size_px, consistency, liveness, rel_x
    crop_file           text
);
CREATE INDEX IF NOT EXISTS occupant_faces_identity_idx ON occupant_faces (identity_id);
CREATE INDEX IF NOT EXISTS occupant_faces_ts_idx ON occupant_faces (ts);

-- Vehicle readings from the profiler (squirrel_guard/vehicle_read.py) and the explained match (vehmatch.py)
ALTER TABLE vehicle_sightings ADD COLUMN IF NOT EXISTS plate_read jsonb;   -- own plate reading: text, per-character confidence
ALTER TABLE vehicle_sightings ADD COLUMN IF NOT EXISTS colour jsonb;       -- measured colour; night = infrared, no colour
ALTER TABLE vehicle_sightings ADD COLUMN IF NOT EXISTS motion jsonb;       -- direction of travel, time standing still
ALTER TABLE vehicle_sightings ADD COLUMN IF NOT EXISTS explain jsonb;      -- why it was filed under its vehicle (reasons, band)

-- Evidence packages, chain of custody, audit log and human review (persons-ui/evidence_export.py, access.py,
-- docs/evidence-export.md). Every table here is append-only: triggers refuse UPDATE, DELETE and TRUNCATE. The two logs
-- are hash chains (entry_hash covers prev_hash), so an edit made by someone who bypasses the triggers still shows.
CREATE TABLE IF NOT EXISTS evidence_exports (
  id bigserial PRIMARY KEY,
  created timestamptz NOT NULL,
  sighting_id bigint,                       -- first item; no foreign key: the record outlives merged or deleted sightings
  frigate_event text NOT NULL,
  camera text,
  items jsonb NOT NULL,                     -- [{n, sighting, event, camera, start}]
  export_ref text NOT NULL UNIQUE,          -- EX-YYYYMMDD-NNNNN, printed on the report
  case_ref text NOT NULL,
  purpose text NOT NULL,
  exported_by text NOT NULL,
  exported_role text NOT NULL,
  package_file text NOT NULL,
  package_sha256 text NOT NULL,
  package_bytes bigint NOT NULL,
  manifest_sha256 text NOT NULL,
  manifest jsonb NOT NULL,
  has_original boolean NOT NULL,
  signing_fingerprint text NOT NULL,
  tsa jsonb,                                -- outside timestamp of manifest.json: url, gen_time, serial, offset
  clock jsonb
);
CREATE TABLE IF NOT EXISTS custody_log (
  id bigserial PRIMARY KEY,
  export_id bigint NOT NULL REFERENCES evidence_exports(id),
  ts text NOT NULL,                         -- UTC, exactly the text that was hashed
  actor text NOT NULL,
  action text NOT NULL,                     -- created | downloaded | verified | verify_failed | release_*
  detail text NOT NULL DEFAULT '',
  remote text NOT NULL DEFAULT '',
  prev_hash text NOT NULL,
  entry_hash text NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS custody_log_export ON custody_log (export_id, id);
CREATE TABLE IF NOT EXISTS audit_log (     -- who looked at what: every page/API request and every refusal
  id bigserial PRIMARY KEY,
  ts text NOT NULL,
  actor text NOT NULL,
  role text NOT NULL,
  action text NOT NULL,                     -- view:<page> | change:<page> | denied | review | export ...
  target text NOT NULL,
  detail text NOT NULL DEFAULT '',
  remote text NOT NULL DEFAULT '',
  prev_hash text NOT NULL,
  entry_hash text NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS audit_log_actor ON audit_log (actor, id);
CREATE TABLE IF NOT EXISTS match_reviews ( -- a person's decision on a sighting's match; the latest one counts
  id bigserial PRIMARY KEY,
  sighting_id bigint NOT NULL,
  frigate_event text NOT NULL,
  ts timestamptz NOT NULL DEFAULT now(),
  reviewer text NOT NULL,
  role text NOT NULL,
  outcome text NOT NULL CHECK (outcome IN ('confirmed', 'rejected', 'unsure')),
  note text NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS match_reviews_sighting ON match_reviews (sighting_id, id);
CREATE TABLE IF NOT EXISTS release_copies (-- redacted copies for disclosure, linked to the package they came from
  id bigserial PRIMARY KEY,
  export_id bigint NOT NULL REFERENCES evidence_exports(id),
  created timestamptz NOT NULL,
  created_by text NOT NULL,
  package_file text NOT NULL,
  package_sha256 text NOT NULL,
  package_bytes bigint NOT NULL,
  manifest_sha256 text NOT NULL,
  summary jsonb NOT NULL
);
CREATE OR REPLACE FUNCTION evidence_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION '% is append-only: % refused', TG_TABLE_NAME, TG_OP; END $$;
DO $$ DECLARE t text; BEGIN
  FOREACH t IN ARRAY ARRAY['evidence_exports', 'custody_log', 'audit_log', 'match_reviews', 'release_copies'] LOOP
    EXECUTE format('DROP TRIGGER IF EXISTS %I ON %I', t || '_append_only', t);
    EXECUTE format('CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION evidence_append_only()', t || '_append_only', t);
    EXECUTE format('DROP TRIGGER IF EXISTS %I ON %I', t || '_no_truncate', t);
    EXECUTE format('CREATE TRIGGER %I BEFORE TRUNCATE ON %I FOR EACH STATEMENT EXECUTE FUNCTION evidence_append_only()', t || '_no_truncate', t);
  END LOOP;
END $$;

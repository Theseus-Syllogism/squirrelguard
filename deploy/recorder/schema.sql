-- squirrel-guard event history (one row per candidate the trigger window let through)
CREATE TABLE IF NOT EXISTS events (
    id         bigserial PRIMARY KEY,
    ts         timestamptz NOT NULL DEFAULT now(),
    camera     text NOT NULL DEFAULT 'frontyard',
    label      text NOT NULL,          -- detector class
    score      real NOT NULL,          -- detector confidence
    box        integer[],              -- x1,y1,x2,y2 in 640x360 frame pixels
    verifier   text,                   -- VLM answer, NULL if the verifier was unreachable
    confirmed  boolean NOT NULL,
    action     text NOT NULL,          -- fired | dry-run | cooldown | rejected | no-command
    detect_ms  integer,
    snapshot   text                    -- file name of the annotated frame
);
CREATE INDEX IF NOT EXISTS events_ts_idx ON events (ts);

#!/usr/bin/env python3
"""Seal recording segments soon after they are written: a hash-chained ledger per camera, anchored daily by an
outside timestamp. Standard library only (runs in the recorder's Alpine image).

  /recordings/<camera>/<YYYY-mm-dd_HH-MM-SS>.mkv         written by record.sh (read-only here)
  /ledger/<camera>.jsonl                                 one line per closed segment
  /ledger/anchors/<camera>.jsonl                         one line per day: ledger head + RFC 3161 token

Ledger line: {"seq", "camera", "file", "bytes", "sha256", "start_local", "mtime_utc", "sealed_utc", "prev", "hash"}
with hash = sha256(prev + "\\n" + canonical JSON of every other field). Changing, removing or reordering a line breaks
every later hash; the daily anchor proves the chain up to its head existed on that day. A segment is sealed when the
next segment of the same camera exists, or when it has not changed for SEGMENT_S + 120 seconds.

What it proves: a segment whose bytes still hash to its ledger entry is unchanged since it was sealed (a few minutes
after it was written). What it does not prove: anything about the minutes between capture and sealing.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tsa  # noqa: E402

REC = Path(os.environ.get("RECORDINGS", "/recordings"))
LEDGER = Path(os.environ.get("LEDGER", "/ledger"))
SEGMENT_S = int(os.environ.get("SEGMENT_S", "600"))
TSA_URL = os.environ.get("TSA_URL", "").strip()
EVERY = int(os.environ.get("SEAL_EVERY_S", "60"))
NAME = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.mkv$")
GENESIS = "0" * 64


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def chain_hash(prev: str, fields: dict) -> str:
    return hashlib.sha256(prev.encode() + b"\n" + canonical(fields)).hexdigest()


def utc(t: float | None = None) -> str:
    return dt.datetime.fromtimestamp(time.time() if t is None else t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_file(p: Path) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def verify_lines(lines: list[dict], prev: str = GENESIS) -> int | None:
    """Index of the first broken line, or None when the chain holds from `prev`."""
    for i, d in enumerate(lines):
        fields = {k: v for k, v in d.items() if k not in ("prev", "hash")}
        if d.get("prev") != prev or d.get("hash") != chain_hash(prev, fields):
            return i
        prev = d["hash"]
    return None


class Camera:
    def __init__(self, name: str):
        self.name = name
        self.path = LEDGER / f"{name}.jsonl"
        lines = read_ledger(self.path)
        bad = verify_lines(lines)
        if bad is not None:
            # never rewrite history: stop sealing this camera and say so loudly every round
            self.broken = f"ledger {self.path} is broken at line {bad + 1}; sealing for {name} stopped"
        else:
            self.broken = None
        self.sealed = {d["file"] for d in lines}
        self.seq = lines[-1]["seq"] if lines else 0
        self.head = lines[-1]["hash"] if lines else GENESIS

    def closed(self, files: list[Path]) -> list[Path]:
        out, now = [], time.time()
        for i, f in enumerate(files):
            rel = f"{self.name}/{f.name}"
            if rel in self.sealed:
                continue
            later = i + 1 < len(files)
            try:
                idle = now - f.stat().st_mtime
            except FileNotFoundError:
                continue                                        # retention removed it meanwhile
            if later or idle > SEGMENT_S + 120:
                out.append(f)
        return out

    def seal(self, f: Path) -> dict:
        digest, size = sha256_file(f)
        st = f.stat()
        self.seq += 1
        fields = {"seq": self.seq, "camera": self.name, "file": f"{self.name}/{f.name}", "bytes": size, "sha256": digest,
                  "start_local": f.stem, "mtime_utc": utc(st.st_mtime), "sealed_utc": utc()}
        h = chain_hash(self.head, fields)
        line = dict(fields, prev=self.head, hash=h)
        with open(self.path, "a") as out:
            out.write(json.dumps(line, sort_keys=True) + "\n")
            out.flush()
            os.fsync(out.fileno())
        self.head = h
        self.sealed.add(fields["file"])
        return line

    def anchor(self) -> str | None:
        """Timestamp the ledger head once per UTC day (and whenever the day's last anchor is behind the head)."""
        if not TSA_URL or self.seq == 0:
            return None
        apath = LEDGER / "anchors" / f"{self.name}.jsonl"
        anchors = read_ledger(apath)
        today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        if anchors and anchors[-1]["date"] == today:
            return None
        if anchors and anchors[-1]["head"] == self.head:
            return None
        try:
            s = tsa.stamp(self.head, TSA_URL)
        except tsa.TSAError as e:
            return f"{self.name}: anchor not made ({e}); retrying next round"
        import base64
        apath.parent.mkdir(exist_ok=True)
        rec = {"date": today, "camera": self.name, "seq": self.seq, "head": self.head, "tsa_url": TSA_URL,
               "gen_time": s.gen_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "clock_offset_s": s.offset_s,
               "round_trip_s": s.round_trip_s, "tsr_b64": base64.b64encode(s.tsr).decode()}
        with open(apath, "a") as out:
            out.write(json.dumps(rec, sort_keys=True) + "\n")
        return f"{self.name}: anchored head {self.head[:16]} (seq {self.seq}) at {rec['gen_time']}, clock offset {s.offset_s:+.2f} s"


def once(cams: dict) -> None:
    for d in sorted(p for p in REC.iterdir() if p.is_dir() and not p.name.startswith(".")):
        cam = cams.get(d.name) or cams.setdefault(d.name, Camera(d.name))
        if cam.broken:
            print(cam.broken, file=sys.stderr, flush=True)
            continue
        files = sorted(f for f in d.iterdir() if NAME.match(f.name))
        for f in cam.closed(files):
            try:
                line = cam.seal(f)
                print(f"sealed {line['file']} {line['bytes']} bytes sha256 {line['sha256'][:16]}", flush=True)
            except FileNotFoundError:
                pass
        msg = cam.anchor()
        if msg:
            print(msg, flush=True)


def main() -> None:
    LEDGER.mkdir(exist_ok=True)
    cams: dict = {}
    print(f"sealer: {REC} -> {LEDGER}, every {EVERY} s, timestamps {'from ' + TSA_URL if TSA_URL else 'OFF'}", flush=True)
    while True:
        try:
            once(cams)
        except OSError as e:                                    # NTFS mount gone etc.: report, keep trying
            print(f"sealer: {e}", file=sys.stderr, flush=True)
        if os.environ.get("SEAL_ONCE"):
            return
        time.sleep(EVERY)


if __name__ == "__main__":
    main()

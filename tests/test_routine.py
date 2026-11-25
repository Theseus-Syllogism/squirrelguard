"""Routine model (deploy/recorder/bridge/routine.py) on a synthetic camera history.

Run:  .venv/bin/python tests/test_routine.py      (or pytest)
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy/recorder/bridge"))
import routine as rt  # noqa: E402

TZ = ZoneInfo("America/Denver")
DAY0 = dt.datetime(2026, 9, 1, tzinfo=TZ)


def path(a, b, bend=0.0, n=16):
    t = np.linspace(0, 1, n)[:, None]
    p = (1 - t) * np.asarray(a) + t * np.asarray(b)
    p[:, 1] += bend * np.sin(np.pi * t[:, 0])
    return np.round(p, 3).tolist()


def visit(day, hour, entry, exit_, dur, still=0, bend=0.0):
    return {"ts": DAY0 + dt.timedelta(days=day, hours=hour),
            "behavior": {"entry": entry, "exit": exit_, "duration_s": dur, "motionless_frames": still,
                         "path": path(entry, exit_, bend)}}


def history(rng):
    h = []
    for d in range(60):
        h.append(visit(d, 11 + rng.normal(0, 0.4), [0.02, 0.7], [0.98, 0.72], 25 + rng.normal(0, 5)))   # mail: left->right
        if rng.random() < 0.8:
            h.append(visit(d, 17.5 + rng.normal(0, 0.6), [0.5, 0.99], [0.5, 0.4], 12 + rng.normal(0, 3)))  # household: bottom->middle
        if rng.random() < 0.3:
            h.append(visit(d, 19 + rng.normal(0, 1.5), [0.98, 0.7], [0.02, 0.7], 30 + rng.normal(0, 8)))   # evening walker
    return h


def test_unusual_night_visit_is_flagged():
    rng = np.random.default_rng(0)
    m = rt.Routine(history(rng), TZ)
    s = m.score(DAY0 + dt.timedelta(days=61, hours=3.2),
                {"entry": [0.5, 0.02], "exit": [0.5, 0.02], "duration_s": 400, "motionless_frames": 300,
                 "path": path([0.5, 0.02], [0.5, 0.3], 0.0)})
    assert s["ready"] and s["unusual"], s
    codes = {p["code"] for p in s["parts"] if p["bits"] >= 4}
    assert {"hour", "stay"} <= codes, s["parts"]
    assert s["reasons"] and "03:" in s["reasons"][0] or any("03:" in r for r in s["reasons"])


def test_ordinary_visits_rarely_flagged():
    rng = np.random.default_rng(1)
    m = rt.Routine(history(rng), TZ)
    fresh = history(np.random.default_rng(2))[:120]            # new visits from the same routine
    flagged = sum(m.score(v["ts"], v["behavior"])["unusual"] for v in fresh)
    assert flagged <= 0.05 * len(fresh), flagged


def test_cold_start_never_alerts():
    rng = np.random.default_rng(3)
    m = rt.Routine(history(rng)[:10], TZ)
    s = m.score(DAY0 + dt.timedelta(hours=3), {"entry": [0.5, 0.02], "exit": [0.5, 0.02], "duration_s": 900})
    assert not s["ready"] and not s["unusual"] and s["note"]


def test_unusual_route_at_a_normal_time():
    rng = np.random.default_rng(4)
    m = rt.Routine(history(rng), TZ)
    s = m.score(DAY0 + dt.timedelta(days=61, hours=11), {"entry": [0.5, 0.02], "exit": [0.02, 0.98], "duration_s": 25,
                                                         "motionless_frames": 0,
                                                         "path": path([0.5, 0.02], [0.02, 0.98], 0.3)})
    route = next(p for p in s["parts"] if p["code"] == "route")
    hour = next(p for p in s["parts"] if p["code"] == "hour")
    assert route["bits"] >= 4 and hour["bits"] < 1, s["parts"]


if __name__ == "__main__":
    for f in [v for n, v in list(globals().items()) if n.startswith("test_")]:
        f()
        print("ok", f.__name__)
    rng = np.random.default_rng(0)
    m = rt.Routine(history(rng), TZ)
    s = m.score(DAY0 + dt.timedelta(days=61, hours=3.2), {"entry": [0.5, 0.02], "exit": [0.5, 0.02], "duration_s": 400,
                                                          "motionless_frames": 300, "path": path([0.5, 0.02], [0.5, 0.3])})
    print(f"night visit: {s['bits']} bits (threshold {s['threshold']}), unusual={s['unusual']}")
    for r in s["reasons"]:
        print("  -", r)

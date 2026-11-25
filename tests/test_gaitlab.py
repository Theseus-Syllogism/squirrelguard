"""Gait analysis (squirrel_guard/gaitlab.py) on synthetic walkers with a known gait: realistic knee-flexion cycle,
hip bob, arm swing, keypoint noise and dropped frames; a limp lowers one knee's swing and makes steps unequal."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard import gaitlab  # noqa: E402
from squirrel_guard.pose import Pose  # noqa: E402

THIGH = SHANK = 0.45


def knee_curve(phi, swing_peak=57.0):
    """Knee flexion (deg) over one stride from heel strike: loading bump ~20 deg at 15 %, swing peak ~60 deg at 72 %."""
    return 4 + 15 * math.exp(-((phi - 0.15) / 0.07) ** 2) + swing_peak * math.exp(-((phi - 0.72) / 0.11) ** 2)


def hip_curve(phi):
    return 8 + 17 * math.cos(2 * math.pi * phi)                # flexed 25 deg at heel strike, extended -9 at 50 %


def walker(cadence=110.0, seconds=8.0, fps=10.0, limp=0.0, view="side", seed=0, noise=2.0, drop=0.08, arm=0.22,
           scale=220.0, speed=None):
    rng = np.random.default_rng(seed)
    stride_s = 120.0 / cadence
    speed = speed or 0.75 * stride_s * cadence / 120 * 1.0 / stride_s * 1.4    # ~1.3 m/s at 110 spm
    trk, n = [], int(seconds * fps)
    for i in range(n):
        t = i / fps
        phase = (t / stride_s) % 1.0
        warp = phase + limp * 0.08 * math.sin(2 * math.pi * phase) / (2 * math.pi)  # unequal steps
        x = speed * t
        bob = 0.025 * math.cos(4 * math.pi * phase)
        pts3 = {}
        hip_z = THIGH + SHANK + bob - 0.02
        for side, ph, peak in (("l", warp, 57.0), ("r", (warp + 0.5) % 1.0, 57.0 * (1 - 0.45 * limp))):
            th, kn = math.radians(hip_curve(ph)), math.radians(knee_curve(ph, peak))
            knee = (x + THIGH * math.sin(th), hip_z - THIGH * math.cos(th))
            ank = (knee[0] + SHANK * math.sin(th - kn), knee[1] - SHANK * math.cos(th - kn))
            pts3[side] = ((x, hip_z), knee, ank)
        sh_z = hip_z + 0.5
        swing = arm * math.sin(2 * math.pi * warp)
        world = {gaitlab.NOSE: (x + 0.08, sh_z + 0.2), gaitlab.L_SH: (x, sh_z), gaitlab.R_SH: (x, sh_z),
                 gaitlab.L_EL: (x - swing / 2, sh_z - 0.28), gaitlab.R_EL: (x + swing / 2, sh_z - 0.28),
                 gaitlab.L_WR: (x - swing, sh_z - 0.55), gaitlab.R_WR: (x + swing, sh_z - 0.55),
                 gaitlab.L_HIP: pts3["l"][0], gaitlab.R_HIP: pts3["r"][0], gaitlab.L_KNEE: pts3["l"][1], gaitlab.R_KNEE: pts3["r"][1],
                 gaitlab.L_ANK: pts3["l"][2], gaitlab.R_ANK: pts3["r"][2]}
        kp = np.zeros((17, 3))
        for j in range(17):
            fw, z = world.get(j, (x, sh_z + 0.2))
            if view == "side":
                u, v, s = 100 + fw * scale, 600 - z * scale, 1.0
            else:                                                 # walking toward the camera: depth shrinks, size grows
                depth = 12.0 - fw
                s = 6.0 / depth
                lat = {gaitlab.L_SH: -0.19, gaitlab.R_SH: 0.19, gaitlab.L_HIP: -0.1, gaitlab.R_HIP: 0.1, gaitlab.L_KNEE: -0.1,
                       gaitlab.R_KNEE: 0.1, gaitlab.L_ANK: -0.1, gaitlab.R_ANK: 0.1, gaitlab.L_WR: -0.24, gaitlab.R_WR: 0.24}.get(j, 0)
                u, v = 640 + lat * scale * s, 300 + (1.0 - z) * scale * s
            kp[j] = (u + rng.normal(0, noise), v + rng.normal(0, noise), rng.uniform(0.6, 0.95))
        if rng.random() < drop:
            trk.append(None)
            continue
        x1, y1 = kp[:, :2].min(0) - 4
        x2, y2 = kp[:, :2].max(0) + 4
        trk.append(Pose((float(x1), float(y1), float(x2), float(y2)), 0.9, kp))
    return trk


@pytest.mark.parametrize("cadence,fps", [(100, 10), (115, 10), (130, 25)])
def test_cadence_is_accurate(cadence, fps):
    w = gaitlab.analyse(walker(cadence, fps=fps), fps)
    assert w["usable"] and w["view"] == "side"
    assert abs(w["cadence_spm"] - cadence) <= 4, w["cadence_spm"]


def test_knee_and_arm_measurements_match_the_model():
    w = gaitlab.analyse(walker(110, fps=25, noise=1.0), 25)
    assert 50 <= w["knee_rom_l"] <= 70 and 50 <= w["knee_rom_r"] <= 70, (w["knee_rom_l"], w["knee_rom_r"])
    assert 50 <= w["knee_peak_swing"] <= 72 and len(w["knee_cycle"]["l"]) == gaitlab.CYCLE_POINTS
    assert w["knee_asym"] < 0.12 and w["symmetry"] > 0.8
    assert 0.3 <= w["arm_swing"] <= 0.7 and abs(w["lean_deg"]) < 6


def test_a_limp_shows_up():
    normal = gaitlab.analyse(walker(110, fps=25, seed=1), 25)
    limp = gaitlab.analyse(walker(110, fps=25, seed=1, limp=1.0), 25)
    assert limp["knee_asym"] > 0.25 > normal["knee_asym"]
    assert limp["knee_rom_r"] < normal["knee_rom_r"] - 12
    strong = gaitlab.analyse(walker(110, fps=25, seed=1, limp=3.0), 25)               # unequal steps as well
    assert strong["step_time_asym"] > normal["step_time_asym"] + 0.05, (strong["step_time_asym"], normal["step_time_asym"])
    import json
    json.dumps(strong)                                                                # stored as JSON: plain types only


def test_profile_and_compare():
    prof = gaitlab.profile([gaitlab.analyse(walker(110, fps=10, seed=s), 10) for s in range(4)])
    assert prof["walks"] == 4 and prof["features"]["cadence_spm"]["n"] == 4
    same = gaitlab.compare(gaitlab.analyse(walker(112, fps=10, seed=9), 10), prof)
    assert same["verdict"] == "consistent", same
    limp = gaitlab.compare(gaitlab.analyse(walker(110, fps=10, seed=9, limp=1.0), 10), prof)
    assert limp["verdict"] != "consistent"
    assert any(r["key"].startswith("knee") and r["verdict"] != "within the usual range" for r in limp["rows"])
    fast = gaitlab.compare(gaitlab.analyse(walker(140, fps=10, seed=9), 10), prof)
    assert fast["verdict"] != "consistent" and next(r for r in fast["rows"] if r["key"] == "cadence_spm")["verdict"] == "different"
    assert "does not identify" in same["text"]


def test_front_view_measures_rhythm_only():
    w = gaitlab.analyse(walker(110, fps=10, view="front", seed=3), 10)
    assert w["view"] == "front/back" and abs(w["cadence_spm"] - 110) <= 5
    assert "knee_rom_l" not in w and "arm_swing" not in w and "cadence_spm" in w["measured"]
    prof = gaitlab.profile([gaitlab.analyse(walker(110, fps=10, seed=s), 10) for s in range(3)])
    c = gaitlab.compare(w, prof)
    assert all(r["key"] in gaitlab.RHYTHM for r in c["rows"])


def test_too_short_or_no_legs():
    assert gaitlab.analyse(walker(110, seconds=2.0), 10)["walking"] is False
    assert gaitlab.analyse([None] * 40, 10) is None
    assert gaitlab.compare({"usable": False}, {})["verdict"] == "not comparable"


def test_svg_chart():
    w = gaitlab.analyse(walker(110, fps=25), 25)
    prof = gaitlab.profile([gaitlab.analyse(walker(110, fps=25, seed=s), 25) for s in range(3)])
    svg = gaitlab.svg_cycle(w, prof)
    assert svg.startswith("<svg") and svg.count("<polyline") == 2 and "<polygon" in svg and "usual range" in svg
    assert gaitlab.svg_cycle({"walking": True}) == ""


def test_kinematics_gait_includes_the_new_measurements():
    from squirrel_guard import kinematics
    g = kinematics.gait(walker(110, fps=10), 10.0, None, (1280, 720))
    assert g["usable"] and g["version"] == gaitlab.VERSION and "symmetry" in g and "knee_rom_l" in g
    assert abs(g["cadence_spm"] - 110) <= 4                     # the old keys keep their meaning


def test_persons_ui_copy_is_identical():
    assert (ROOT / "squirrel_guard/gaitlab.py").read_bytes() == (ROOT / "deploy/recorder/persons-ui/gaitlab.py").read_bytes()

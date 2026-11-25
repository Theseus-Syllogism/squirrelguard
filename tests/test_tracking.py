"""kinematics.track guards: a weak-torso object after the person leaves must not continue the track, a short
occlusion must not lose a walking person, and a differently-dressed person must not be adopted after a loss.

Run:  .venv/bin/python tests/test_tracking.py      (or pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squirrel_guard import kinematics as k  # noqa: E402
from squirrel_guard.pose import Pose  # noqa: E402

W, H = 640, 360


def person(x, torso=0.9, w=60, h=200, y=100):
    kp = np.zeros((17, 3))
    kp[:, 0], kp[:, 1], kp[:, 2] = x + w / 2, np.linspace(y + 10, y + h - 5, 17), 0.9
    kp[[5, 6, 11, 12], 2] = torso
    kp[[5, 6], 1], kp[[11, 12], 1] = y + 0.3 * h, y + 0.55 * h
    return Pose((x, y, x + w, y + h), 0.9, kp)


class FakeEstimator:
    """Returns scripted detections per frame index (frames are coloured arrays carrying their index)."""
    def __init__(self, script):
        self.script = script

    def _dets(self, fr):
        return self.script.get(int(fr[0, 0, 2]), [])

    def __call__(self, fr):
        return self._dets(fr)

    def in_region(self, fr, box):
        return self._dets(fr)


def frames(n, colours=None):
    for i in range(n):
        fr = np.zeros((H, W, 3), np.uint8)
        for (x1, x2, col) in (colours or {}).get(i, []):
            fr[100:300, x1:x2] = col
        fr[0, 0, 2] = i
        yield fr


def test_weak_torso_object_ends_the_track():
    script = {i: [person(100 + 10 * i)] for i in range(10)}                  # person walks right, then leaves
    script.update({i: [person(400, torso=0.05, w=150)] for i in range(12, 25)})  # a coat: "person" w/o torso
    trk, _ = k.track(frames(25), FakeEstimator(script))
    assert all(trk[i] is not None for i in range(10))
    assert all(p is None for p in trk[12:]), [i for i, p in enumerate(trk) if p is not None and i >= 12]


def test_weak_arrival_is_kept():
    """Someone entering close to the lens has no clear torso at first: the track must not be cut off."""
    script = {i: [person(20 + 10 * i, torso=0.05 if i < 6 else 0.9, w=200)] for i in range(20)}
    trk, _ = k.track(frames(20), FakeEstimator(script))
    assert sum(p is not None for p in trk) >= 19, [i for i, p in enumerate(trk) if p is None]


def test_suspended_track_resumes_for_the_same_person():
    red = (0, 0, 220)
    script, cols = {}, {}
    for i in list(range(8)) + list(range(14, 22)):             # same red-shirted person, briefly turned/hidden
        x = 100 + 10 * i
        script[i] = [person(x)]
        cols[i] = [(x, x + 60, red)]
    for i in range(8, 14):
        script[i] = [person(100 + 10 * i, torso=0.05)]          # torso unreadable for 6 frames
    trk, _ = k.track(frames(22, cols), FakeEstimator(script))
    assert all(trk[i] is not None for i in range(14, 22))


def test_short_occlusion_is_bridged_by_motion_prediction():
    script = {i: [person(50 + 25 * i)] for i in range(14) if i not in (6, 7)}  # hidden for 2 frames
    trk, _ = k.track(frames(14), FakeEstimator(script))
    assert trk[5] is not None and trk[8] is not None and trk[13] is not None


def test_different_clothes_not_adopted_after_loss():
    red, blue = (0, 0, 220), (220, 0, 0)
    script, cols = {}, {}
    for i in range(8):                                        # red-shirted person walking
        x = 100 + 10 * i
        script[i] = [person(x)]
        cols[i] = [(x, x + 60, red)]
    for i in range(12, 20):                                   # after a gap, a blue-shirted person far away
        x = 450
        script[i] = [person(x)]
        cols[i] = [(x, x + 60, blue)]
    trk, _ = k.track(frames(20, cols), FakeEstimator(script))
    assert all(trk[i] is not None for i in range(8))
    assert all(p is None for p in trk[12:])


if __name__ == "__main__":
    for f in [v for n, v in list(globals().items()) if n.startswith("test_")]:
        f()
        print("ok", f.__name__)

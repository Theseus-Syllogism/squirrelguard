"""Vehicle reading and matching: plate comparison, explained match decisions, colour (night images refused), motion,
and the plate reader on stock photos when its models are available (SG_VEHICLE_MODELS)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard import vehmatch  # noqa: E402


def test_plate_similarity():
    assert vehmatch.plate_similarity("ABC 123", "abc-123")["score"] == 1.0
    conf = vehmatch.plate_similarity("8KX4O1", "BKX401")                       # two easily confused characters
    one = vehmatch.plate_similarity("ABC123", "ABC128")
    clipped = vehmatch.plate_similarity("ABC123", "BC123")                     # first character cut off at the edge
    other = vehmatch.plate_similarity("ABC123", "XYZ789")
    assert conf["score"] > one["score"] and "easily misread" in conf["text"]
    assert clipped["score"] >= 0.9 and other["score"] < 0.3
    low = vehmatch.plate_similarity("ABC123", "ABC128", None, [1, 1, 1, 1, 1, 0.3])  # the reader was unsure of the 8
    assert low["score"] > one["score"]
    assert vehmatch.plate_similarity(None, "ABC") is None


def test_explained_decisions():
    same = vehmatch.explain({"plate": "STAT04", "colour": "green", "body": "hatchback"},
                            {"plate": "STATO4", "colour": "mint green", "body": "hatchback"}, 0.71)
    assert same["decision"] == "same" and same["band"] in ("strong", "likely")
    assert [r["code"] for r in same["reasons"]][:2] == ["PLATE", "APPEARANCE"]
    diff = vehmatch.explain({"plate": "ABC123"}, {"plate": "XYZ789"}, 0.92)       # look-alike cars, different plates
    assert diff["decision"] == "different"
    night = vehmatch.explain({"colour": None, "colour_night": True, "body": "sedan"}, {"colour": "red", "body": "sedan"}, 0.86)
    assert any(r["code"] == "COLOUR_UNKNOWN" for r in night["reasons"]) and night["decision"] == "same"
    recolour = vehmatch.explain({"colour": "blue", "body": "sedan"}, {"colour": "red", "body": "sedan"}, 0.83)
    assert recolour["decision"] != "same" and any(r["code"] == "COLOUR_DIFFERENT" for r in recolour["reasons"])
    weak = vehmatch.explain({"colour": "grey"}, {"colour": "white"}, 0.66)
    assert weak["decision"] in ("possible", "different") and any(r["code"] == "COLOUR_NEAR" for r in weak["reasons"])


def test_copies_are_identical():
    src = (ROOT / "squirrel_guard/vehmatch.py").read_bytes()
    for d in ("deploy/recorder/bridge", "deploy/recorder/persons-ui"):
        assert (ROOT / d / "vehmatch.py").read_bytes() == src, d


cv2 = pytest.importorskip("cv2")


def test_night_images_have_no_colour():
    from squirrel_guard import vehicle_read as vr
    rng = np.random.default_rng(0)
    grey = np.repeat(rng.integers(0, 255, (180, 320, 1), dtype=np.uint8), 3, axis=2)
    assert vr.body_colour(grey[40:140, 60:260], grey)["night"] is True
    day = np.full((180, 320, 3), 128, np.uint8)
    day[40:140, 60:260] = (40, 40, 200)                                        # a red car (BGR) on a grey street
    c = vr.body_colour(day[40:140, 60:260], day, (60, 40, 260, 140))
    assert c["name"] == "red" and c["night"] is False


def test_white_balance_does_not_turn_a_red_car_blue():
    from squirrel_guard import vehicle_read as vr
    frame = np.full((200, 300, 3), (90, 110, 150), np.uint8)                   # warm evening light on a grey street
    frame[20:190, 10:290] = (30, 30, 190)                                      # a red car fills the frame
    c = vr.body_colour(frame[20:190, 10:290], frame, (10, 20, 290, 190))
    assert c["name"] == "red"


def test_motion():
    from squirrel_guard import vehicle_read as vr
    path = [[[0.1, 0.6], 0.0], [[0.3, 0.6], 1.0], [[0.31, 0.6], 4.0], [[0.8, 0.62], 6.0]]
    m = vr.motion(path)
    assert m["direction"] == "left to right" and m["stopped_s"] == 3.0
    assert vr.motion([[[0.5, 0.2], 0], [[0.5, 0.8], 3]])["direction"] == "toward the camera"
    assert vr.motion(None) is None


MODELS = Path(os.environ.get("SG_VEHICLE_MODELS", "/models/vehicle"))


@pytest.mark.skipif(not (MODELS / "cct_s_v2_global.onnx").exists(), reason="plate models not here (SG_VEHICLE_MODELS)")
def test_plate_reader_on_stock_photos():
    from squirrel_guard import vehicle_read as vr
    r = vr.PlateReader(MODELS)
    img = cv2.imread(str(ROOT / "data/testimg/cars/3_nissan_car_wallpapers_figaro_automobile_antique_car_youngtim.jpg"))
    res = r.read([img, cv2.resize(img, None, fx=0.8, fy=0.8)])
    assert res["readable"] and res["text"] == "STATO4" and res["reads"] == 2 and res["agree"] == 1.0
    junk = r.read([cv2.imread(str(ROOT / "data/testimg/cars/11_A_red_car_parked_on_the_side_of_the_road.jpg"))])
    assert junk is None or not junk["readable"]                                # a garbled reading is not passed on as a plate

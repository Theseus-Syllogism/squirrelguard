"""Person profiler: one sighting per finished Frigate person event -> fingerprint + attributes -> MQTT.

frigate/events (type end, label person) -> event clip from Frigate -> up to N frames -> person crops
-> OSNet fingerprint (averaged) + Qwen attributes on the best crop -> squirrelguard/<camera>/sighting.
The recorder host bridge stores it and matches it against known identities.

Run: python -m squirrel_guard.profiler -c config.toml   (needs [profiler] in the config)
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import logging
import os
import uuid
import queue
import tempfile
import threading
import time
import tomllib
from pathlib import Path

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import requests

from .attributes import describe
from .detector import Detector
from .reid import Embedder
from . import reid_v2
from .attributes import describe_vehicle
from .vehicle import VehicleEmbedder, vehicle_label
from .geometry import Camera
from .kinematics import analyse
from .pose import PoseEstimator
from . import evidence, gaze
from .action_skeleton import EVENTS_OF_INTEREST, SkeletonActionRecognizer
from .gait_embed import GaitEmbedder
from .openvocab import ObjectAggregator, OpenVocabDetector, describe as describe_objects
from . import trajectory
from . import replay_assets
from . import pulc
from . import kinematics, rigpipe, smplfit

log = logging.getLogger("squirrel_guard.profiler")
PEOPLE = {"person", "man", "woman", "boy", "girl"}
VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle"}  # Frigate labels (after our labelmap remap)


def frames_from_clip(mp4: bytes, n: int) -> list:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(mp4)
        f.flush()
        cap = cv2.VideoCapture(f.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        out = []
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / n) if total else i * 10)
            ok, fr = cap.read()
            if ok:
                out.append(fr)
        cap.release()
        return out


def face_of(a: dict) -> dict | None:
    """Frigate face recognition result for a PERSON event: the enrolled name (sub label) and its score, or None.
    Frigate only names people enrolled in its Face Library; everyone else stays unnamed."""
    if a.get("label") != "person":
        return None
    sl = a.get("sub_label")
    name, score = (sl[0], sl[1]) if isinstance(sl, (list, tuple)) and sl else (sl, None)
    if not name or not isinstance(name, str):
        return None
    score = score if score is not None else (a.get("data") or {}).get("sub_label_score")
    return {"name": name, "score": round(float(score), 3) if score is not None else None}


def behavior_of(a: dict) -> dict:
    """Summarise a finished Frigate event: dwell, path through the frame, stillness, zones."""
    pts = [p[0] for p in (a.get("path_data") or []) if p and p[0]]
    length = sum(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 for (x1, y1), (x2, y2) in zip(pts, pts[1:]))
    start, end = a.get("start_time"), a.get("end_time")
    return {"duration_s": round(end - start, 1) if start and end else None,
            "path_points": len(pts), "path_length": round(length, 3),  # in frame widths/heights (0-1 units)
            "entry": [round(v, 3) for v in pts[0]] if pts else None, "exit": [round(v, 3) for v in pts[-1]] if pts else None,
            "net_displacement": round(((pts[-1][0] - pts[0][0]) ** 2 + (pts[-1][1] - pts[0][1]) ** 2) ** 0.5, 3) if len(pts) > 1 else 0,
            "motionless_frames": a.get("motionless_count"), "position_changes": a.get("position_changes"),
            "entered_zones": a.get("entered_zones") or [], "avg_speed": a.get("average_estimated_speed"),
            "max_area": a.get("area"), "path": resample_path(pts, 16)}


def load_pulc(pc: dict) -> tuple:
    """PULC attribute classifiers ([profiler] pulc_person_model / pulc_vehicle_model; "" or missing = off). A missing or
    stale model file only switches the feature off (logged), never the profiler."""
    out = []
    for key, cls in (("pulc_person_model", pulc.PersonAttributes), ("pulc_vehicle_model", pulc.VehicleAttributes)):
        m = None
        if pc.get(key):
            try:
                m = cls(pc[key], threads=int(pc.get("pulc_threads", 2)))
                log.info("PULC %s: %s (%d labels)", key, pc[key], len(m.labels))
            except Exception:
                log.exception("PULC model %s unusable; %s off", pc[key], key)
        out.append(m)
    return tuple(out)


def full_body(box, frame_shape, min_aspect: float = 1.5, margin: int = 2) -> bool:
    """PA100K-like view: upright (height/width >= min_aspect) and not cut off by the top or bottom of the frame.
    Close-ups (head and shoulders at a desk, a hand with a drill) make PULC hallucinate bags, coats and gender."""
    x1, y1, x2, y2 = box
    h = frame_shape[0]
    return (y2 - y1) >= min_aspect * max(1, x2 - x1) and y1 > margin and y2 < h - margin


def pulc_person_visit(model, cands: list, max_crops: int = 6, min_aspect: float = 1.5) -> dict | None:
    """cands: (weight, crop, box, frame_shape) per frame -> PULC visit summary over the best full-body crops, or a
    {"crops": 0, "skipped": ...} note when no crop qualifies (so the UI can say why there are no attributes)."""
    if model is None or not cands:
        return None
    ok = [(w, c) for w, c, b, shp in cands if full_body(b, shp, min_aspect)]
    if not ok:
        return {"model": model.kind, "crops": 0, "skipped": "no full-body view", "considered": len(cands)}
    try:
        v = model.visit([c for _, c in ok], [w for w, _ in ok], max_crops)
    except Exception:
        log.exception("PULC person attributes failed")
        return None
    if v is not None:
        v["considered"] = len(cands)
    return v


def pulc_vehicle_visit(model, crops: list, max_crops: int = 6) -> dict | None:
    """crops: (weight, crop) -> PULC vehicle colour/type summary."""
    if model is None or not crops:
        return None
    try:
        return model.visit([c for _, c in crops], [w for w, _ in crops], max_crops)
    except Exception:
        log.exception("PULC vehicle attributes failed")
        return None


def resample_path(pts: list, n: int = 16) -> list | None:
    """Frigate path (image 0-1 units) -> n points evenly spaced along its length (the routine model's route shape)."""
    if len(pts) < 2:
        return None
    P = np.asarray(pts, float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] <= 1e-6:
        return np.round(np.repeat(P[:1], n, 0), 3).tolist()
    t = np.linspace(0, d[-1], n)
    return np.round(np.c_[np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])], 3).tolist()


# -- face identification (faceid.py), behind [profiler] face_enabled; kept apart from the rest for easy merging ------
def face_setup(pc: dict):
    """[profiler] face_* keys -> (engine, settings) or None. A missing InspireFace/model pack only disables faces."""
    if not pc.get("face_enabled"):
        return None
    try:
        from . import faceid
        Engine = faceid.FaceEngine
        if pc.get("face_v2_adaface"):           # opt-in v2 (docs/faceid-v2.md): same engine + 5-pt aligned crops
            from . import faceid_v2
            Engine = faceid_v2.FaceEngineV2
        eng = Engine(pc.get("face_model_pack", "/models/inspireface/Megatron"),
                                detect_level=int(pc.get("face_detect_level", 320)),
                                expression=bool(pc.get("face_expression", True)),
                                attributes=bool(pc.get("face_attributes", False)))   # age/gender; race always discarded
    except Exception:
        log.exception("face identification unavailable (InspireFace or model pack %s); faces off", pc.get("face_model_pack"))
        return None
    log.info("face identification on: %s", eng.pack)
    eng.ada = None
    if pc.get("face_v2_adaface"):
        try:
            from . import faceid_v2
            eng.ada = faceid_v2.AdaFace(pc["face_v2_adaface"], threads=int(pc.get("face_v2_threads", 2)))
            log.info("face v2 (AdaFace) on: %s", pc["face_v2_adaface"])
        except Exception:
            log.exception("AdaFace unavailable (%s); v2 off", pc.get("face_v2_adaface"))
    facegeom_setup(eng, pc)
    facefit_setup(eng, pc)
    return eng, pc


# -- face geometry (facegeom.py), behind [profiler] facegeom_enabled; explanatory evidence only -----------------------
def facegeom_setup(eng, pc: dict) -> None:
    """[profiler] facegeom_* keys -> a MediaPipe face mesher on the face engine (eng.geom_mesher) or nothing. A
    missing mediapipe/model only disables the geometry, never the face template."""
    eng.geom_mesher = None
    if not pc.get("facegeom_enabled"):
        return
    try:
        from . import facegeom
        eng.geom_mesher = facegeom.FaceMesher(pc.get("facegeom_model", "/opt/mediapipe/face_landmarker.task"))
        eng.dense_landmarks = True          # InspireFace 106 points for the overlay
    except Exception:
        log.exception("face geometry unavailable (mediapipe or %s); geometry off", pc.get("facegeom_model"))
        return
    log.info("face geometry on: %s", pc.get("facegeom_model"))


def facegeom_attach(fs, face) -> None:
    """Give a FaceSampler a per-visit geometry collector when the engine has a mesher."""
    eng, pc = face
    if getattr(eng, "geom_mesher", None) is None:
        return
    from . import facegeom
    fs.geom = facegeom.GeomCollector(eng.geom_mesher, facegeom.GeomParams(
        keep=int(pc.get("facegeom_faces", 8)), max_yaw=float(pc.get("facegeom_max_yaw", 25)),
        min_px=float(pc.get("facegeom_min_px", 32))))


# -- face fit (facefit.py, docs/facefit.md), behind [profiler] facefit_enabled; explanatory evidence only -------------
def facefit_setup(eng, pc: dict) -> None:
    """[profiler] facefit_* keys -> a MediaPipe Tasks landmarker + the model's canonical face (shape prior) on the face
    engine (eng.fit_lm, eng.fit_prior), or nothing."""
    eng.fit_lm = eng.fit_prior = None
    if not pc.get("facefit_enabled"):
        return
    model = pc.get("facefit_model", "/opt/mediapipe/face_landmarker.task")
    try:
        from . import facefit
        eng.fit_lm = facefit.make_landmarker(model)
        eng.fit_prior = facefit.canonical_from_task(model)
    except Exception:
        eng.fit_lm = None
        log.exception("face fit unavailable (mediapipe or %s); face fit off", model)
        return
    log.info("face fit on: %s (canonical prior %s)", model, "yes" if eng.fit_prior is not None else "no")


def facefit_attach(fs, face, cam: str = "", calib=None) -> None:
    eng, pc = face
    if getattr(eng, "fit_lm", None) is None:
        return
    from . import facefit
    fs.fit = facefit.FitCollector(eng.fit_lm, cam=cam, calib=(calib, None) if calib is not None else None,
                                  max_frames=int(pc.get("facefit_max_frames", 96)),
                                  max_ms=float(pc.get("facefit_max_ms", 4000)), prior=eng.fit_prior)


def face_sampler(face, cam: str = "", calib=None):
    """A fresh per-event FaceSampler (bounded CPU/memory) or None."""
    if face is None:
        return None
    from . import faceid
    eng, pc = face
    params = faceid.AggParams(min_px=float(pc.get("face_min_px", 24)), min_quality=float(pc.get("face_min_quality", 0.0)),
                              max_faces=int(pc.get("face_max_frames", 96)))
    kw = dict(every=int(pc.get("face_every", 1)), max_attempts=int(pc.get("face_max_attempts", 150)),
              max_cpu_s=float(pc.get("face_max_cpu_s", 8)))
    if getattr(eng, "ada", None) is not None:
        from . import faceid_v2
        fs = faceid_v2.FaceSamplerV2(eng, params, ada=eng.ada, **kw)
    else:
        fs = faceid.FaceSampler(eng, params, **kw)
    facegeom_attach(fs, face)
    facefit_attach(fs, face, cam, calib)
    return fs


# -- vehicle occupant faces (occupants.py), behind [profiler] vehicle_occupant_faces; needs face_enabled --------------
def occupant_setup(pc: dict, face):
    """A second InspireFace session with a larger detector input for the cabin tiles (same pack, same process), the
    v2 AdaFace model of the person path, and the sampling settings. None when off or faces are unavailable."""
    if not pc.get("vehicle_occupant_faces"):
        return None
    if face is None:
        log.error("vehicle_occupant_faces needs face_enabled (InspireFace); occupant faces off")
        return None
    eng0, _ = face
    try:
        from . import faceid, faceid_v2, occupants
        Engine = faceid_v2.FaceEngineV2 if getattr(eng0, "ada", None) is not None else faceid.FaceEngine
        lvl = int(pc.get("vehicle_occupant_detect_level", 640))
        eng = Engine(pc.get("face_model_pack", "/models/inspireface/Megatron"), detect_level=lvl,
                     expression=False, attributes=False)        # occupants: no expression/age/gender at all
    except Exception:
        log.exception("occupant face engine unavailable; occupant faces off")
        return None
    occ = occupants.OccParams(det_px=lvl, tile_px=int(pc.get("vehicle_occupant_tile_px", 320)),
                              max_tiles=int(pc.get("vehicle_occupant_max_tiles", 12)),
                              max_cpu_s=float(pc.get("vehicle_occupant_max_cpu_s", 12)),
                              region_bottom=float(pc.get("vehicle_occupant_region_bottom", 0.70)))
    agg = faceid.AggParams(min_px=float(pc.get("face_min_px", 24)), min_quality=float(pc.get("face_min_quality", 0.0)),
                           max_faces=int(pc.get("face_max_frames", 96)))
    log.info("vehicle occupant faces on: detector %d px, %d px tiles, <= %.0f s CPU per visit", lvl, occ.tile_px,
             occ.max_cpu_s)
    return {"engine": eng, "ada": getattr(eng0, "ada", None), "occ": occ, "agg": agg,
            "fps": float(pc.get("vehicle_occupant_fps", 2)), "max_s": float(pc.get("vehicle_occupant_max_s", 20)),
            "max_frames": int(pc.get("vehicle_occupant_frames", 8)),
            "max_templates": int(pc.get("vehicle_occupant_max_templates", 3))}


def occupant_windows(a: dict, max_s: float, now: float, lag_s: float = 12.0) -> list[tuple[float, float]]:
    """Recording windows to search for occupants: the first max_s seconds of the visit (arrival) and, for a long
    finished visit, its last max_s seconds (departure). A visit still in progress ends lag_s before now (the
    newest recording segment is not on disk yet)."""
    t0 = float(a.get("start_time") or now)
    t1 = float(a["end_time"]) if a.get("end_time") else now - lag_s
    if t1 <= t0:
        return []
    w = [(t0 - 1.0, min(t1, t0 + max_s))]
    if a.get("end_time") and t1 - t0 > 2 * max_s:
        w.append((t1 - max_s, t1 + 1.0))
    elif a.get("end_time") and t1 - t0 > max_s:
        w[0] = (t0 - 1.0, t1 + 1.0)
    return w


def faceid_of_boxes(face, frame_boxes: list) -> dict | None:
    """Face template from the sighting's few sampled frames + detector person boxes (no keypoints)."""
    fs = face_sampler(face)
    if fs is None:
        return None
    for i, (fr, box) in enumerate(frame_boxes):
        fs.on_box(fr, box, None, t=i)
    return fs.result()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.toml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = tomllib.loads(Path(args.config).read_text())
    pc, fc = cfg["profiler"], cfg.get("frigate", {})
    api = fc.get("api", "http://frigate:5000").rstrip("/")
    cameras = set(fc.get("cameras", []))
    det = Detector(pc["detector_model"], conf=float(pc.get("conf", 0.3)), threads=int(pc.get("threads", 4)))
    # re-ID: reid_backend "osnet" (default) = Embedder(reid_model); "v2" / reid_shadow_model: docs/reid-gait.md
    emb = reid_v2.load_embedder(pc, threads=int(pc.get("threads", 4)))
    emb_shadow = reid_v2.load_shadow(pc, threads=int(pc.get("threads", 4)))
    vemb = VehicleEmbedder(pc["vehicle_model"], threads=int(pc.get("threads", 4))) if pc.get("vehicle_model") else None
    plates = None
    if vemb is not None and pc.get("plate_models"):
        try:
            from . import vehicle_read
            plates = vehicle_read.PlateReader(pc["plate_models"], threads=2)
            log.info("own plate reading on (%s)", pc["plate_models"])
        except Exception as e:                                   # missing models or onnxruntime: Frigate's plates only
            log.error("plate reading off: %s", e)
    plate_sent: collections.OrderedDict = collections.OrderedDict()  # vehicle event -> plate already published
    pulc_person, pulc_vehicle = load_pulc(pc)
    pulc_max = int(pc.get("pulc_max_crops", 6))
    llm = pc.get("llm_url") or cfg.get("verifier", {}).get("url")
    llm_model = pc.get("llm_model", "") if pc.get("llm_url") else cfg.get("verifier", {}).get("model", "")
    n_frames = int(pc.get("frames", 6))
    pose = PoseEstimator(pc["pose_model"], threads=int(pc.get("threads", 4))) if pc.get("pose_model") else None
    actions_model = SkeletonActionRecognizer(pc["action_model"], threads=int(pc.get("threads", 4))) \
        if pc.get("action_model") and pose is not None else None
    gait_emb = GaitEmbedder(pc["gait_model"], pc["gait_seg_model"], threads=int(pc.get("threads", 4))) \
        if pc.get("gait_model") and pc.get("gait_seg_model") and pose is not None else None
    ov_classes = pc.get("openvocab_classes") or []
    if isinstance(ov_classes, str):
        ov_classes = [c for c in ov_classes.split(",") if c.strip()]
    ov = None
    if pc.get("openvocab_model") and pose is not None:
        try:   # optional: a missing/stale model file must not take the whole profiler down
            ov = OpenVocabDetector(pc["openvocab_model"], classes=ov_classes,
                                   min_score=float(pc.get("openvocab_min_score", 0.25)), threads=int(pc.get("threads", 4)))
        except Exception:
            log.exception("open-vocabulary model %s unusable; object detection off", pc["openvocab_model"])
    if ov is not None:
        log.info("open-vocabulary objects: %s", ", ".join(ov.vocabulary))
    face = face_setup(pc)
    occ_cfg = occupant_setup(pc, face)
    calib_dir = Path(pc.get("calib_dir", "/calib"))
    calib_cache: dict = {}  # camera -> (mtime, Camera | None)
    # IK rig (squirrel_guard.rig, docs/rigfit.md): whole-body keypoints + procedural skeleton fit for every replay of a
    # calibrated camera; RTMW is loaded on first use (needs torch for the fit)
    rig_cfg = {"enabled": bool(pc.get("rig_enabled", False)) and pose is not None,
               "model": pc.get("rig_model", "/models/rtmw/rtmw-dw-l-m_simcc-cocktail14_270e-256x192.onnx"),
               "fps": float(pc.get("rig_fps", 12)), "max_s": float(pc.get("rig_max_s", 20)),
               "threads": int(pc.get("rig_threads", 2)),
               "iters_root": int(pc.get("rig_iters_root", 300)), "iters_full": int(pc.get("rig_iters_full", 900))}
    rig_state: dict = {}
    # 3D body (squirrel_guard.smplfit): SMPL dressed onto the rig of each replay; the SMPL model file is mounted at run
    # time (non-commercial licence: never baked into an image, never published)
    body_cfg = {"enabled": bool(pc.get("body_enabled", False)) and rig_cfg["enabled"],
                "model": pc.get("body_smpl_model", "/models/smpl/SMPL_NEUTRAL.npz"),
                "mesh_step": int(pc.get("body_mesh_step", 2)), "iters": int(pc.get("body_iters", 300))}
    if body_cfg["enabled"] and not Path(body_cfg["model"]).is_file():
        body_cfg["enabled"] = False
        log.error("body_enabled but %s is missing; 3D body off", body_cfg["model"])

    def smpl_model():
        if "smpl" not in rig_state:
            rig_state["smpl"] = smplfit.SMPLModel(body_cfg["model"])
        return rig_state["smpl"]

    def rig_wb():
        if "wb" not in rig_state:
            from .wholebody import WholeBody
            rig_state["wb"] = WholeBody(rig_cfg["model"], threads=rig_cfg["threads"])
            log.info("rig: whole-body keypoints %s", rig_cfg["model"])
        return rig_state["wb"]
    if rig_cfg["enabled"]:
        try:
            import torch  # noqa: F401
            torch.set_num_threads(rig_cfg["threads"])
            rigpipe.PARTS = tuple(pc.get("rig_parts", ()))     # ("face", "hands"): head pose/jaw + hands
            log.info("rig on: %s fps, <= %s s per event, parts %s", rig_cfg["fps"], rig_cfg["max_s"],
                     list(rigpipe.PARTS) or "body only")
        except ImportError:
            rig_cfg["enabled"] = False
            log.error("rig_enabled but torch is not installed; rig off")
    # multi-camera fusion (squirrel_guard.fusion, docs/fusion.md): when overlapping person events of 2+ cameras that
    # share a calibration frame have all ended, one rig over every view -> one fused replay linked from each sighting
    fusion_cfg = {"enabled": bool(pc.get("fusion_enabled", False)) and rig_cfg["enabled"],
                  "fps": float(pc.get("fusion_fps", rig_cfg["fps"])), "max_s": float(pc.get("fusion_max_s", 20)),
                  "max_cameras": int(pc.get("fusion_max_cameras", 3)),
                  "max_offset_s": float(pc.get("fusion_max_offset_s", 4)),
                  "max_gap_m": float(pc.get("fusion_max_gap_m", 0.15)), "pad_s": float(pc.get("fusion_pad_s", 3))}

    def camera_calib(cam: str):
        f = calib_dir / f"{cam}.json"
        mt = f.stat().st_mtime if f.is_file() else None
        if calib_cache.get(cam, (0,))[0] != mt:  # reload when the calibration file changes
            calib_cache[cam] = (mt, Camera.load(f) if mt else None)
            log.info("camera %s: %s", cam, f"calibrated ({calib_cache[cam][1].meta.get('method')})"
                     if calib_cache[cam][1] else "no height calibration (gait only)")
        return calib_cache[cam][1]
    min_h = int(pc.get("min_crop_height", 64))

    work: queue.Queue = queue.Queue(maxsize=100)
    seen: collections.OrderedDict = collections.OrderedDict()

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="squirrel-guard-profiler")
    c.username_pw_set(os.environ.get("SG_MQTT_USER"), os.environ.get("SG_MQTT_PASS"))
    c.reconnect_delay_set(1, 60)

    pub_lock = threading.Lock()

    def publish(topic: str, payload: str) -> None:
        """QoS-1 publish over a FRESH, short-lived MQTT connection (connect, publish, wait for PUBACK, disconnect).
        The recorder host's inline Suricata reassembles only the first stream.reassembly.depth bytes of a TCP flow; on the
        long-lived subscriber connection the broker's PUBACKs then trip rule 2229001 ("PUBLISH not seen before
        PUBACK"), the IPS drops them, the flow dies and the visit is lost (seen daily since evidence made messages
        ~1-2 MB). A new flow per message keeps every PUBLISH/PUBACK inside the inspected window; `c` only subscribes."""
        with pub_lock:
            for attempt in range(3):
                p = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sg-profiler-pub-{os.getpid()}-{uuid.uuid4().hex[:8]}")  # unique: a lingering previous session must not be taken over
                p.username_pw_set(os.environ.get("SG_MQTT_USER"), os.environ.get("SG_MQTT_PASS"))
                try:
                    p.connect(os.environ["SG_MQTT_HOST"], int(os.environ.get("SG_MQTT_PORT", "1883")), keepalive=30)
                    p.loop_start()
                    info = p.publish(topic, payload, qos=1)
                    info.wait_for_publish(timeout=30)
                    if info.is_published():
                        return
                    log.warning("publish to %s not acknowledged (attempt %d, %d bytes)", topic, attempt + 1, len(payload))
                except Exception as e:  # noqa: BLE001
                    log.warning("publish to %s failed (attempt %d): %s", topic, attempt + 1, e)
                finally:
                    try:
                        p.disconnect(); p.loop_stop()
                    except Exception:  # noqa: BLE001
                        pass
                time.sleep(2 * (attempt + 1))
            log.error("giving up publishing to %s (%d bytes)", topic, len(payload))

    def on_connect(cl, ud, fl, rc, pr):
        log.info("MQTT connected (%s); profiling person%s events on %s", rc, " + vehicle" if vemb else "",
                 sorted(cameras) or "all cameras")
        cl.subscribe("frigate/events", qos=0)

    def on_message(cl, ud, msg):
        try:
            ev = json.loads(msg.payload)
        except ValueError:
            return
        a = ev.get("after") or {}
        label = (a.get("label") or "").lower()
        is_vehicle = label in VEHICLE_LABELS and vemb is not None
        if (label != "person" and not is_vehicle) or a.get("false_positive") or ev.get("type") not in ("update", "end"):
            return
        if cameras and a.get("camera") not in cameras:
            return
        # Frigate's LPR often reads the plate after we profiled: send it when the event ends
        if is_vehicle and ev.get("type") == "end" and a["id"] in seen and a.get("recognized_license_plate") \
                and plate_sent.get(a["id"]) != a["recognized_license_plate"]:
            plate_sent[a["id"]] = a["recognized_license_plate"]
            publish(f"squirrelguard/{a['camera']}/vehicle", json.dumps({
                "plate_update": True, "camera": a["camera"], "frigate_event": a["id"],
                "plate": a["recognized_license_plate"], "plate_score": a.get("recognized_license_plate_score")}))
            publish(f"squirrelguard/{a['camera']}/behavior", json.dumps({
                "kind": "vehicle", "camera": a["camera"], "frigate_event": a["id"], "end": a.get("end_time"),
                "behavior": behavior_of(a)}))
            return
        kind = "vehicle" if is_vehicle else "person"
        if ev.get("type") == "end" and a["id"] in seen:  # already profiled while live: send behaviour now
            if kind == "person" and pose is not None:  # ... after the clip-based height/gait analysis (worker)
                try:
                    work.put_nowait({**a, "_end": True})
                except queue.Full:
                    log.warning("profiler queue full; skipping analysis of %s", a["id"])
                return
            publish(f"squirrelguard/{a['camera']}/behavior", json.dumps({
                "kind": kind, "camera": a["camera"], "frigate_event": a["id"], "end": a.get("end_time"),
                "behavior": behavior_of(a)}))
            return
        # profile once per event: when it ends, or once tracked long enough (people/parked cars that stay
        # in view keep the event open indefinitely, so waiting for 'end' misses them)
        age = time.time() - float(a.get("start_time") or time.time())
        wait = float(pc.get("vehicle_profile_after_s", 20) if is_vehicle else pc.get("profile_after_s", 10))
        if ev.get("type") != "end" and age < wait:
            return
        if a["id"] in seen:
            return
        seen[a["id"]] = 1
        while len(seen) > 1000:
            seen.popitem(last=False)
        try:
            work.put_nowait(a)
        except queue.Full:
            log.warning("profiler queue full; skipping %s", a["id"])

    def live_frames(cam: str, n: int) -> list:
        out = []
        for _ in range(n):  # one per second from Frigate's latest frame (detect resolution)
            try:
                r = requests.get(f"{api}/api/{cam}/latest.jpg", timeout=10)
                if r.status_code == 200 and r.content:
                    out.append(cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR))
            except requests.RequestException:
                pass
            time.sleep(1)
        return [f for f in out if f is not None]

    def profile(a: dict) -> None:
        eid, cam = a["id"], a["camera"]
        frames = []
        if a.get("end_time"):  # finished: use the event clip
            time.sleep(float(pc.get("settle_s", 15)))  # let the last segment reach storage
            try:
                r = requests.get(f"{api}/api/events/{eid}/clip.mp4", timeout=60)
                if r.status_code == 200 and r.content:
                    frames = frames_from_clip(r.content, n_frames)
            except requests.RequestException:
                pass
        else:  # still in view: sample live frames now
            frames = live_frames(cam, n_frames)
        if not frames:  # fall back to the snapshot
            r = requests.get(f"{api}/api/events/{eid}/snapshot.jpg", params={"bbox": 0}, timeout=15)
            if r.status_code == 200:
                frames = [cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)]
        label = (a.get("label") or "").lower()
        if label in VEHICLE_LABELS:
            return profile_vehicle(a, frames)
        crops, person_boxes, pulc_cands = [], [], []
        for fr in frames:
            ds = [d for d in det(fr) if d.label.lower() in PEOPLE]
            if not ds:
                continue
            d = max(ds, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))  # largest person
            x1, y1, x2, y2 = d.box
            if y2 - y1 >= min_h:
                crops.append((d.score * (y2 - y1), fr[y1:y2, x1:x2]))
                person_boxes.append((fr, d.box))
                pulc_cands.append((crops[-1][0], crops[-1][1], d.box, fr.shape))
        if not crops and pose is not None and pc.get("zoom_fallback", True):
            # small, distant person (or a sweeper event): the whole-frame detector sees too few pixels. Scan zoomed
            # tiles along the band where the event's box was, with the COCO person model (docs/sweep.md)
            rel = (a.get("data") or {}).get("box")
            for fr in frames:
                H, W = fr.shape[:2]
                cy, bh = ((rel[1] + rel[3] / 2) * H, rel[3] * H) if rel else (H / 2, H / 8)
                tile = int(min(H, max(320, 3 * bh)))
                ty = int(np.clip(cy - 0.55 * tile, 0, H - tile))
                xs = list(range(0, max(1, W - tile + 1), max(1, tile * 2 // 3)))
                if xs[-1] != W - tile and W > tile:
                    xs.append(W - tile)
                best_p = None
                for tx in xs:
                    for p in pose(fr[ty:ty + tile, tx:tx + tile]):
                        if p.score >= float(pc.get("zoom_min_score", 0.5)) and (best_p is None or p.score > best_p[1]):
                            best_p = ((int(p.box[0]) + tx, int(p.box[1]) + ty, int(p.box[2]) + tx, int(p.box[3]) + ty),
                                      p.score)
                if best_p is None:
                    continue
                (x1, y1, x2, y2), sc = best_p
                if y2 - y1 >= min_h:
                    crops.append((sc * (y2 - y1), fr[y1:y2, x1:x2]))
                    person_boxes.append((fr, (x1, y1, x2, y2)))
                    pulc_cands.append((crops[-1][0], crops[-1][1], (x1, y1, x2, y2), fr.shape))
            if crops:
                log.info("%s: %d person crops found by the zoomed fallback (small/distant person)", eid, len(crops))
        if not crops:
            log.info("no usable person crop for %s", eid)
            return
        vec = emb.one([cr for _, cr in crops])
        best = max(crops, key=lambda t: t[0])[1]
        attrs = describe(llm, best, model=llm_model) if llm else None
        ok, jpg = cv2.imencode(".jpg", best, [cv2.IMWRITE_JPEG_QUALITY, 88])
        msg = {"camera": cam, "frigate_event": eid, "start": a.get("start_time"), "end": a.get("end_time"),
               "score": round(float(a.get("top_score") or a.get("score") or 0), 3), "crops": len(crops),
               "embedding": [round(float(v), 5) for v in vec], "attributes": attrs,
               "crop_jpeg_b64": base64.b64encode(jpg.tobytes()).decode() if ok else None, "face": face_of(a)}
        if emb_shadow is not None:   # dual-embedding period: a second re-ID vector the bridge does not match on (yet)
            try:
                v2 = emb_shadow.one([cr for _, cr in crops])
                msg["embedding_v2"] = {"model": emb_shadow.name, "vec": [round(float(v), 5) for v in v2]}
            except Exception:
                log.exception("shadow re-ID embedding failed for %s", eid)
        if face is not None:
            try:
                msg["faceid"] = faceid_of_boxes(face, person_boxes)
            except Exception:
                log.exception("face identification failed for %s", eid)
        pv = pulc_person_visit(pulc_person, pulc_cands, pulc_max, float(pc.get("pulc_min_aspect", 1.5)))
        if pv is not None:
            msg["pulc"] = pv
        publish(f"squirrelguard/{cam}/sighting", json.dumps(msg))
        log.info("SIGHTING %s %s crops=%d attrs=%s%s", cam, eid, len(crops), "yes" if attrs else "no",
                 f" pulc={pulc.describe_person(pv) or pv.get('skipped')}" if pv else "")

    def iou(a, b) -> float:
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def parked_boxes(per: list, frac: float = 0.8) -> list:
        """Boxes present at the same place (IoU > 0.7) in >= frac of the frames with detections: parked vehicles."""
        n = sum(1 for ds in per if ds)
        if n < 3:
            return []
        out = []
        for ds in per:
            for d in ds:
                if any(iou(d.box, b) > 0.7 for b in out):
                    continue
                hits = sum(1 for o in per if any(iou(d.box, e.box) > 0.7 for e in o))
                if hits >= frac * n:
                    out.append(d.box)
        return out

    def profile_vehicle(a: dict, frames: list) -> None:
        eid, cam, label = a["id"], a["camera"], a["label"].lower()
        crops, where = [], []                   # where: (frame, box) of each crop, for colour balance and night detection
        per = []
        for fr in frames:
            per.append([d for d in det(fr) if vehicle_label(d.label) == label or (label == "car" and vehicle_label(d.label))])
        parked = parked_boxes(per)            # the event is about the vehicle that MOVED, not the biggest parked one
        for fr, ds in zip(frames, per):
            moving = [d for d in ds if not any(iou(d.box, b) > 0.7 for b in parked)] or (ds if not parked else [])
            if not moving:
                continue
            d = max(moving, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
            x1, y1, x2, y2 = d.box
            if min(x2 - x1, y2 - y1) >= min_h:
                crops.append(((x2 - x1) * (y2 - y1), fr[y1:y2, x1:x2]))
                where.append((fr, d.box))
        if not crops and parked:              # nothing moved (e.g. a car that parked): fall back to the largest
            for fr, ds in zip(frames, per):
                if ds:
                    d = max(ds, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
                    x1, y1, x2, y2 = d.box
                    if min(x2 - x1, y2 - y1) >= min_h:
                        crops.append(((x2 - x1) * (y2 - y1), fr[y1:y2, x1:x2]))
                        where.append((fr, d.box))
        if not crops:
            log.info("no usable vehicle crop for %s", eid)
            return
        vec = vemb.one([cr for _, cr in crops])
        best = max(crops, key=lambda t: t[0])[1]
        attrs = describe_vehicle(llm, best, model=llm_model) if llm else None
        ok, jpg = cv2.imencode(".jpg", best, [cv2.IMWRITE_JPEG_QUALITY, 88])
        plate = a.get("recognized_license_plate")
        if plate:
            plate_sent[eid] = plate
        while len(plate_sent) > 1000:
            plate_sent.popitem(last=False)
        msg = {"camera": cam, "frigate_event": eid, "label": label, "start": a.get("start_time"), "end": a.get("end_time"),
               "score": round(float(a.get("top_score") or a.get("score") or 0), 3), "crops": len(crops),
               "plate": plate, "plate_score": a.get("recognized_license_plate_score"),
               "embedding": [round(float(v), 5) for v in vec], "attributes": attrs,
               "crop_jpeg_b64": base64.b64encode(jpg.tobytes()).decode() if ok else None}
        pvv = pulc_vehicle_visit(pulc_vehicle, crops, pulc_max)
        if pvv is not None:
            msg["pulc"] = pvv
        msg.update(read_vehicle(a, crops, where))
        if occ_cfg is not None:
            try:
                msg.update(occupants_of(a, label))
            except Exception:
                log.exception("occupant faces failed for %s", eid)
        publish(f"squirrelguard/{cam}/vehicle", json.dumps(msg))
        log.info("VEHICLE %s %s %s crops=%d plate=%s attrs=%s occupants=%s", cam, eid, label, len(crops), plate or "-",
                 "yes" if attrs else "no", msg.get("occupants_detected", "-"))

    def read_vehicle(a: dict, crops: list, where: list) -> dict:
        """Own plate reading (all crops voted), lighting-corrected colour of the largest crop, and how it moved."""
        from . import vehicle_read
        out = {}
        try:
            if plates is not None:
                pr = plates.read([cr for _, cr in crops])
                if pr is not None:
                    out["plate_read"] = pr
            k = max(range(len(crops)), key=lambda i: crops[i][0])
            fr, box = where[k] if k < len(where) else (None, None)
            out["colour"] = vehicle_read.body_colour(crops[k][1], fr, box)
            mv = vehicle_read.motion(a.get("path_data") or (a.get("data") or {}).get("path_data"))
            if mv:
                out["motion"] = mv
        except Exception:
            log.exception("vehicle reading failed for %s", a.get("id"))
        return out

    def occupants_of(a: dict, label: str) -> dict:
        """Occupant faces of a vehicle visit (occupants.py): full-resolution recording windows (arrival, departure),
        two passes per window, faces inside the cabin of the visit's vehicle (largest box of its class), standing
        pedestrians excluded. -> {"occupants_detected", "occupant_faces": [templates], "occupant_stats"}."""
        from . import occupants
        smp = occupants.OccupantSampler(occ_cfg["engine"], occ_cfg["agg"], occ_cfg["occ"], ada=occ_cfg["ada"])

        def boxes(fr):
            dd = det(fr)
            ds = [d for d in dd if vehicle_label(d.label) == label or (label == "car" and vehicle_label(d.label))]
            if not ds:
                return None
            v = max(ds, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])).box
            return v, [d.box for d in dd if d.label.lower() in PEOPLE]
        cam, wins = a["camera"], occupant_windows(a, occ_cfg["max_s"], time.time())
        for w0, w1 in wins:
            if smp.done():
                break
            try:
                r = requests.get(f"{api}/api/{cam}/start/{w0:.1f}/end/{w1:.1f}/clip.mp4", timeout=120)
            except requests.RequestException as e:
                log.info("occupant clip unavailable (%s): %s", a["id"], e)
                continue
            if r.status_code != 200 or len(r.content) < 1000:
                continue
            occupants.sample_clip(r.content, boxes, smp, fps=occ_cfg["fps"], max_s=w1 - w0 + 1.0,
                                  max_frames=max(1, occ_cfg["max_frames"] // len(wins)), t0=w0 - float(a["start_time"]))
        res = smp.result(max_templates=occ_cfg["max_templates"])
        tpl = res.pop("occupant_faces")
        log.info("OCCUPANTS %s %d group(s), %d faces seen (max %.0f px, %d pedestrian), %d frames, %.1fs cpu", a["id"],
                 res["occupants_detected"], res["faces_seen"], res["max_size_px"], res["pedestrian_faces"],
                 res["frames_with_vehicle"], res["cpu_s"])
        return {"occupants_detected": res.pop("occupants_detected"), "occupant_faces": tpl, "occupant_stats": res}

    def motion_of(a: dict, settle: bool) -> dict | None:
        """Height (if calibrated) + gait from the event's recording. Long events: first and last stretch only."""
        if pose is None or not a.get("end_time"):
            return None
        if settle:
            time.sleep(float(pc.get("settle_s", 15)))
        cam, eid, t0, t1 = a["camera"], a["id"], float(a["start_time"]), float(a["end_time"])
        max_s = float(pc.get("analysis_max_s", 40))
        urls = [f"{api}/api/events/{eid}/clip.mp4"] if t1 - t0 <= max_s else [
            f"{api}/api/{cam}/start/{t0:.1f}/end/{t0 + max_s / 2:.1f}/clip.mp4",
            f"{api}/api/{cam}/start/{t1 - max_s / 2:.1f}/end/{t1:.1f}/clip.mp4"]
        parts = []
        for u in urls:
            try:
                r = requests.get(u, timeout=120)
                if r.status_code == 200 and r.content:
                    parts.append(r.content)
            except requests.RequestException as e:
                log.info("clip for analysis unavailable (%s): %s", eid, e)
        if not parts:
            return None
        t = time.time()
        sils: list = []

        def cut_silhouette(fr, p):   # gait signature input, cut while the frame is still in memory
            if gait_emb is not None and p is not None and len(sils) < 720:
                s = gait_emb.silhouettes([fr], [p.box])
                if len(s):
                    sils.append(s[0])

        calib = camera_calib(cam)
        # 3D replay cut-outs + projected background need a calibrated camera; collecting is nearly free otherwise
        coll = replay_assets.Collector(every=2, max_cutouts=120) if calib is not None else None

        # open-vocabulary objects: every Nth analysed frame with the tracked person, at most max_frames per event
        objs = ObjectAggregator(min_frames=int(pc.get("openvocab_min_frames", 2))) if ov is not None else None
        ov_every, ov_max = max(1, int(pc.get("openvocab_every", 2))), int(pc.get("openvocab_max_frames", 60))
        ov_state = {"i": 0, "s": 0.0}
        a_fps = float(pc.get("analysis_fps", 8))

        def find_objects(fr, p):
            i = ov_state["i"]
            ov_state["i"] += 1
            if objs is None or p is None or i % ov_every or objs.frames >= ov_max:
                return
            t = time.time()
            try:
                objs.add(ov.detect_around(fr, p.box), p, t=round(i / a_fps, 1), size=(fr.shape[1], fr.shape[0]))
            except Exception:
                log.exception("open-vocabulary detection failed")
            ov_state["s"] += time.time() - t

        faces = face_sampler(face, cam, calib)

        def on_frame(fr, p):
            if faces is not None:
                faces.on_frame(fr, p)
            cut_silhouette(fr, p)
            find_objects(fr, p)
            if coll is not None:
                coll.on_frame(fr, p)

        res = analyse(parts, pose, calib, fps=float(pc.get("analysis_fps", 8)), max_s=max_s,
                      keep_every=int(pc.get("evidence_keep_every", 3)), on_frame=on_frame)
        res["_replay_collector"] = coll
        res["_clips"] = parts                                   # raw clip bytes for the rig (popped below)
        if faces is not None:
            res["faceid"] = faces.result()
            fr_ = res["faceid"]
            log.info("FACES %s %d faces (%d seen, %d rejected) q=%s live=%s (%.1fs cpu)", eid, fr_.get("n_faces", 0),
                     fr_.get("faces_seen", 0), fr_.get("n_rejected", 0), fr_.get("quality"),
                     (fr_.get("liveness") or {}).get("ok"), fr_.get("cpu_s", 0))
        res["gait_vec"] = None
        if gait_emb is not None and len(sils) >= 10:
            try:
                v = gait_emb.part_features(np.stack(sils)).reshape(-1).astype(np.float32)
                v /= np.linalg.norm(v) + 1e-12
                res["gait_vec"] = [round(float(x), 5) for x in v]
            except Exception:
                log.exception("gait signature failed for %s", eid)
        res["gait_frames"] = len(sils)
        res["actions"] = actions_of(res)
        if objs is not None:
            res["objects"] = objs.summary()
            log.info("OBJECTS %s %s (%d frames, %.1fs)", eid, describe_objects(res["objects"]) or "-", objs.frames,
                     ov_state["s"])
        res["trajectory"] = trajectory_of(res, calib, cam, eid)
        res.pop("_replay_collector", None)
        res.pop("_clips", None)
        g = res.get("gait") or {}
        log.info("MOTION %s height=%s gait=%s (%.0fs)", eid, (res.get("height") or {}).get("cm"),
                 f"{g.get('cadence_spm')}spm usable={g.get('usable')}" if g.get("walking") else "-", time.time() - t)
        return res

    def actions_of(res: dict) -> list:
        """Skeleton actions (ST-GCN++, NTU-120) in 4 s windows; a tag counts only at >= 0.5 in two consecutive
        windows (NTU is indoor/lab data and always picks some class, so single hits are mostly noise)."""
        parts = (res.get("evidence") or {}).get("parts") or []
        if actions_model is None or not parts:
            return []
        out: dict = {}
        for part in parts:
            trk = part["track"]
            kps = np.stack([p.kp if p is not None else np.zeros((17, 3), np.float32) for p in trk]).astype(np.float32)
            if (kps[:, :, 2].max(1) > 0.3).sum() < part["fps"] * 3:
                continue
            try:
                wins = actions_model.windows(kps, part["size"], part["fps"])
            except Exception:
                log.exception("action recognition failed")
                continue
            prev: dict = {}
            for t0, top in wins:
                tags: dict = {}                       # tag -> (summed probability, best NTU label)
                for label, p in top:
                    tag = EVENTS_OF_INTEREST.get(label)
                    if tag:
                        sp, bl = tags.get(tag, (0.0, label))
                        tags[tag] = (sp + p, bl)
                tags = {t: v for t, v in tags.items() if v[0] >= 0.5}
                for tag, (p, label) in tags.items():
                    if tag in prev and (tag not in out or p > out[tag]["p"]):
                        out[tag] = {"tag": tag, "p": round(float(max(p, prev[tag][0])), 2), "t": round(t0, 1),
                                    "label": label}
                prev = tags
        return sorted(out.values(), key=lambda a: -a["p"])

    def finish_replay(tr: dict, coll, calib, prefix: str, fps: float, frame_offset: int = 0) -> dict:
        """3D body (SMPL on the rig) + replay assets (cut-outs, background) for a replay, shipped base64 in
        tr["replay_files"] as <prefix>_replay_* and <prefix>_body*.bin (per-camera replays: prefix = the event id;
        fused replays: <event id>_fused)."""
        body_files: dict = {}
        if body_cfg["enabled"] and tr.get("rig"):
            try:                          # SMPL dressed onto the rig: a 3D body for this replay
                t_b = time.time()
                r3 = smplfit.body_for_replay(smpl_model(), tr, prefix, mesh_step=body_cfg["mesh_step"],
                                             iters=body_cfg["iters"])
                if r3:
                    blk3, vb3, fb3 = r3
                    tr["body3d"] = blk3
                    body_files[blk3["mesh"]["vertices"]] = vb3
                    body_files[blk3["mesh"]["faces"]] = fb3
                    log.info("BODY %s %d frames, marker error %.1f cm (%.0fs, %.1f MB)", prefix,
                             blk3["stats"]["replay_frames_with_body"], blk3["stats"]["marker_err_cm_median"],
                             time.time() - t_b, len(vb3) / 2 ** 20)
            except Exception:
                log.exception("rig body failed for %s", prefix)
        if coll is not None:
            try:   # build the files in a temp dir, ship them base64 in the behaviour message
                masker = gait_emb or replay_assets.PersonMasker(pc.get("gait_seg_model", "/models/yolov8s-seg-320.onnx"))
                rp, atlas, bg = replay_assets.build(coll, calib, tr, masker, fps=fps, frame_offset=frame_offset)
                with tempfile.TemporaryDirectory() as td:
                    replay_assets.attach(tr, rp, atlas, bg, td, prefix=prefix)
                    tr["replay_files"] = {n: base64.b64encode((Path(td) / n).read_bytes()).decode()
                                          for n in tr.get("assets", {}).values()}
            except Exception:
                log.exception("replay assets failed for %s", prefix)
        if body_files:
            tr.setdefault("replay_files", {}).update({n: base64.b64encode(b).decode() for n, b in body_files.items()})
        return tr

    def trajectory_of(res: dict, calib, cam_name: str, eid: str) -> dict | None:
        """Metric ground path with Gaussian uncertainty (Kalman/RTS + GP prediction + estimated footprints), a rough
        3D stick figure, and the replay assets (perspective-fitted person cut-outs + background projected onto the
        ground) for the 3D replay. Needs a calibrated camera; None otherwise."""
        parts = (res.get("evidence") or {}).get("parts") or []
        if calib is None or not parts:
            return None
        k = max(range(len(parts)), key=lambda j: sum(t is not None for t in parts[j]["track"]))
        part = parts[k]
        try:
            tr = trajectory.estimate(part["track"], part["fps"], part["size"], calib, gait=res.get("gait"),
                                     heights_px=True)
            if tr:
                sk = trajectory.skeleton3d(part["track"], calib, part["size"])
                tr["skeleton"] = sk[tr["frame0"]:tr["frame0"] + len(tr["t"])]
                try:                                                  # head direction -> view cone in the replay
                    gz = gaze.estimate(part["track"], tr, calib, part["size"])
                    if gz:
                        tr["gaze"] = gz
                except Exception:
                    log.exception("gaze estimate failed")
                geo = (getattr(calib, "meta", None) or {}).get("geo") or {}
                if geo.get("aerial"):                                 # map-calibrated camera: aerial photo as the ground
                    tr["aerial"] = geo["aerial"]
                tr["source"] = f"{cam_name} event {time.strftime('%Y-%m-%d %H:%M')}"
                tr["event"], tr["camera_name"] = eid, cam_name   # replay pane: links the scene to its stored visit
                if (getattr(calib, "meta", None) or {}).get("frame"):    # persons-ui attaches that frame's room scene
                    tr["calibration_frame"] = calib.meta["frame"]
                clips = res.get("_clips") or []
                if rig_cfg["enabled"] and k < len(clips):
                    try:                          # IK rig + gait at a higher frame rate, sampled onto the replay
                        t_r = time.time()
                        frames, rfps = kinematics.decode(clips[k], rig_cfg["fps"], rig_cfg["max_s"])
                        kpd = rigpipe.extract(frames, pose, rig_wb())
                        rc = {cam_name: (kpd, calib, 0)}
                        its = (rig_cfg["iters_root"], rig_cfg["iters_full"])
                        rres, rviews = rigpipe.fit_for_replay(tr, rc, rfps, iters=its)
                        blk = None
                        if rres is not None:          # the rig drives the path (path, stick figure, body agree)
                            keep = {k2: tr[k2] for k2 in ("aerial", "source", "event", "camera_name", "calibration_frame") if k2 in tr}
                            tr2 = trajectory.estimate(part["track"], part["fps"], part["size"], calib,
                                                      gait=res.get("gait"), heights_px=True,
                                                      ground_override=rigpipe.rig_ground(rres, len(part["track"]),
                                                                                         part["fps"], rfps))
                            if tr2:
                                tr = tr2
                                tr.update(keep)
                                try:
                                    gz = gaze.estimate(part["track"], tr, calib, part["size"])
                                    if gz:
                                        tr["gaze"] = gz
                                except Exception:
                                    log.exception("gaze estimate failed")
                            blk = rigpipe.attach_rig(tr, rc, rfps, res=rres, views=rviews)
                        if blk:
                            log.info("RIG %s stature=%.2f frames=%d err=%s gait=%s (%.0fs: keypoints %.0fs)", eid,
                                     blk["stature_m"], blk["frames_fitted"],
                                     (blk["error"].get(cam_name) or {}).get("median_err_px"),
                                     {k2: blk["gait"].get(k2) for k2 in ("cadence_spm", "knee_symmetry")},
                                     time.time() - t_r, kpd["seconds"])
                    except Exception:
                        log.exception("rig fit failed for %s", eid)
                finish_replay(tr, res.get("_replay_collector"), calib, eid, part["fps"],
                              frame_offset=sum(len(p["track"]) for p in parts[:k]))
            return tr
        except Exception:
            log.exception("trajectory failed")
            return None

    zone_cache: dict = {}  # camera -> (fetched, zones, detect size)

    def camera_zones(cam: str):
        z = zone_cache.get(cam)
        if not z or time.time() - z[0] > 3600:
            try:
                cc = requests.get(f"{api}/api/config", timeout=10).json()["cameras"][cam]
                z = (time.time(), cc.get("zones") or {}, (cc["detect"]["width"], cc["detect"]["height"]))
            except Exception:
                z = (time.time(), {}, (1280, 720))
            zone_cache[cam] = z
        return z[1], z[2]

    def render_evidence(a: dict, b: dict, mot: dict) -> dict:
        """Evidence card (JPEG) + flip-book GIF for the recorder host's emails/UI. Visual facts only: the identity match and
        its explanation are added on the recorder host."""
        ev = mot.pop("evidence", None)
        if not ev:
            return {}
        h, g = mot.get("height") or {}, mot.get("gait") or {}
        dur = b.get("duration_s") or 0
        facts = [f"{a['camera']}  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(a['start_time'])))}",
                 f"stayed {int(dur // 60)} min {int(dur % 60)} s" if dur >= 60 else f"stayed {dur:.0f} s",
                 f"height ~{h['cm']:.0f} cm (+/-{max(2, h.get('iqr_cm', 0) / 2):.0f})" if h.get("cm") else
                 "height: " + ("not measurable (no full upright view)" if mot.get("calibrated") else "camera not calibrated"),
                 (f"walk {g['cadence_spm']:.0f} steps/min" + (f", {g['speed_mps']} m/s" if g.get("speed_mps") else "")
                  if g.get("usable") else "walk: no clear side-on walk"),
                 f"zones: {', '.join(b.get('entered_zones') or []) or 'none'}"]
        for act in (mot.get("actions") or [])[:2]:
            facts.append(f"action: {act['tag']} ({act['label']}) {act['p'] * 100:.0f}% at +{act['t']:.0f}s")
        if mot.get("objects"):
            facts.append("objects: " + describe_objects(mot["objects"][:3]))
        zones, dsize = camera_zones(a["camera"])
        t = time.time()
        out = {}
        try:
            cj = evidence.card(ev, facts, f"Person at {a['camera']} - visit summary", zones, dsize)
            gf = evidence.gif(ev)
            if cj:
                out["card_jpeg_b64"] = base64.b64encode(cj).decode()
            if gf and len(gf) < 3_000_000:
                out["gif_b64"] = base64.b64encode(gf).decode()
            log.info("EVIDENCE %s card=%dKB gif=%dKB (%.1fs)", a["id"], len(cj or b"") // 1024, len(gf or b"") // 1024,
                     time.time() - t)
        except Exception:
            log.exception("evidence rendering failed for %s", a["id"])
        return out

    def send_behavior(a: dict, mot: dict | None) -> None:
        lab = (a.get("label") or "").lower()
        msg = {"kind": "vehicle" if lab in VEHICLE_LABELS else "person", "camera": a["camera"],
               "frigate_event": a["id"], "end": a.get("end_time"), "behavior": behavior_of(a), "face": face_of(a)}
        if mot:
            ev = render_evidence(a, msg["behavior"], mot)
            msg.update({"height": mot.get("height"), "gait": mot.get("gait"), "actions": mot.get("actions") or [],
                        "gait_vec": mot.get("gait_vec"), "gait_frames": mot.get("gait_frames"),
                        "trajectory": mot.get("trajectory")})
            if mot.get("objects") is not None:
                msg["objects"] = mot["objects"]
            if mot.get("faceid") is not None:   # whole-visit face template (better than the sighting's few frames)
                msg["faceid"] = mot["faceid"]
            if ev:
                msg["evidence"] = ev
        publish(f"squirrelguard/{a['camera']}/behavior", json.dumps(msg))

    # -- multi-camera fusion: its own thread + small queue, so per-camera sightings never wait for it ---------------
    fusion_q: queue.Queue = queue.Queue(maxsize=2)
    live_fusion = None
    if fusion_cfg["enabled"]:
        from . import fusion

        def fr_json(path: str):
            r = requests.get(f"{api}{path}", timeout=30)
            return r.json() if r.status_code == 200 else None

        def fr_bytes(path: str):
            r = requests.get(f"{api}{path}", timeout=120)
            return r.content if r.status_code == 200 and r.content else None

        fusion_cams = sorted(set(cameras) | {f.stem for f in calib_dir.glob("*.json")})
        live_fusion = fusion.LiveFusion(fr_json, fr_bytes, camera_calib, fusion_cams, fps=fusion_cfg["fps"],
                                        max_s=fusion_cfg["max_s"], max_cameras=fusion_cfg["max_cameras"],
                                        pad_s=fusion_cfg["pad_s"], max_offset_s=fusion_cfg["max_offset_s"],
                                        max_gap_m=fusion_cfg["max_gap_m"],
                                        iters=(rig_cfg["iters_root"], rig_cfg["iters_full"]))
        log.info("fusion on: %s fps, <= %s s, <= %d cameras per visit", fusion_cfg["fps"], fusion_cfg["max_s"],
                 fusion_cfg["max_cameras"])

    def fusion_submit(a: dict) -> None:
        """After a person event's own analysis: queue a fusion if overlapping events of other cameras in the same
        calibration frame have all ended (the LAST one to end triggers it; earlier ones see it still running)."""
        if live_fusion is None or (a.get("label") or "").lower() != "person" or not a.get("end_time"):
            return
        try:
            p = live_fusion.candidates(a)
        except Exception:
            log.exception("fusion planning failed for %s", a.get("id"))
            return
        if p.get("status") != "fuse":
            if p.get("status") == "wait":
                log.info("fusion: %s waits for %s to end", a["id"], p.get("waiting_for"))
            return
        try:
            fusion_q.put_nowait(p)
            live_fusion.mark(p["members"], "queued")
            log.info("fusion queued: %s", ", ".join(f"{e['camera']}:{e['id']}" for e in p["members"]))
        except queue.Full:
            log.warning("fusion queue full; %s stays per-camera", a["id"])

    def fusion_worker():
        from . import fusion
        while True:
            p = fusion_q.get()
            try:
                end = max(float(e["end_time"]) for e in p["members"])
                wait = end + float(pc.get("settle_s", 15)) - time.time()   # last segments must reach storage
                if wait > 0:
                    time.sleep(wait)
                t0 = time.time()
                fz, info = live_fusion.run(p, pose, rig_wb(),
                                           make_collector=lambda: replay_assets.Collector(every=2, max_cutouts=120))
                if fz is None or not fz.get("ok"):
                    log.info("FUSION none %s: %s %s", [e["id"] for e in p["members"]],
                             (fz or {}).get("reason") or info.get("reason"),
                             json.dumps({k: v.get("reason") for k, v in ((fz or {}).get("report") or {}).items()
                                         if v.get("role") == "dropped"}))
                    continue
                prim = fz["primary"]
                pm = [e for e in p["members"] if e["camera"] == prim]
                if not pm:
                    log.info("FUSION none: primary %s has no member event", prim)
                    continue
                peid = max(pm, key=lambda e: float(e["end_time"]) - float(e["start_time"]))["id"]
                fid = f"{peid}_fused"
                members = [{"camera": e["camera"], "frigate_event": e["id"], "start": e["start_time"],
                            "end": e["end_time"]} for e in p["members"]]
                byname = {v["name"]: v for v in fz["views"]}
                tr = fusion.build_replay(fz, fz["views"], live_fusion.fps, fz["step"], members=members,
                                         window=tuple(p["window"]))
                if tr is None:
                    log.info("FUSION none %s: no trajectory", fid)
                    continue
                calib = byname[prim]["cam"]
                tr["source"] = (f"{' + '.join(v['name'] for v in fz['aligned'])} fused, "
                                f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(p['window'][0]))}")
                tr["event"], tr["camera_name"] = peid, prim
                tr["events_by_camera"] = {e["camera"]: e["id"] for e in p["members"]}
                tr["fusion"]["seconds"] = round(time.time() - t0, 1)
                tr["fusion"]["keypoint_s"] = info.get("keypoint_s")
                finish_replay(tr, byname[prim]["collector"], calib, fid, live_fusion.fps / fz["step"])
                for v in fz["views"]:
                    v["collector"] = None
                live_fusion.mark(p["members"], fid)
                publish(f"squirrelguard/{prim}/fusion", json.dumps({
                    "camera": prim, "fusion_id": fid, "frigate_event": peid, "members": members, "trajectory": tr}))
                rg = tr.get("rig") or {}
                log.info("FUSION %s %s stature=%s frames=%s err=%s gait=%s (%.0fs, keypoints %.0fs)", fid,
                         "+".join(tr["fusion"]["cameras"]), rg.get("stature_m"), rg.get("frames_fitted"),
                         {k: (v or {}).get("median_err_px") for k, v in (rg.get("error") or {}).items()},
                         (rg.get("gait") or {}).get("cadence_spm"), time.time() - t0, info.get("keypoint_s") or 0)
            except Exception:
                log.exception("fusion failed for %s", [e.get("id") for e in p.get("members", [])])

    def worker():
        while True:
            a = work.get()
            try:
                if a.get("_end"):  # profiled earlier while live; the event has now ended
                    send_behavior(a, motion_of(a, settle=True))
                else:
                    profile(a)
                    if a.get("end_time"):  # profiled at its end (clip already settled): behaviour goes after the sighting
                        lab = (a.get("label") or "").lower()
                        send_behavior(a, motion_of(a, settle=False) if lab not in VEHICLE_LABELS else None)
                if a.get("end_time"):
                    fusion_submit(a)
            except Exception:
                log.exception("profiling failed for %s", a.get("id"))

    threading.Thread(target=worker, daemon=True).start()
    if pc.get("sweep_cameras") and pc.get("pose_model"):   # distant walkers Frigate's detector misses (docs/sweep.md)
        from squirrel_guard.sweep import Sweeper

        def sweep_enqueue(ev: dict):
            if ev.get("id") in seen:
                return
            seen[ev["id"]] = 1
            try:
                work.put_nowait(ev)
            except queue.Full:
                log.warning("profiler queue full; skipping swept person %s", ev.get("id"))
        Sweeper(api, list(pc["sweep_cameras"]), pc["pose_model"], sweep_enqueue, pc).start()
    if live_fusion is not None:
        threading.Thread(target=fusion_worker, daemon=True).start()
    c.on_connect, c.on_message = on_connect, on_message
    c.connect(os.environ["SG_MQTT_HOST"], int(os.environ.get("SG_MQTT_PORT", "1883")), keepalive=60)
    c.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-clip room scenes: every replay on the recorder host gets a Gaussian splat + 3D objects of the room AS IT WAS DURING
THAT VISIT, built on the laptop's GPU with no manual step (docs/scenes.md).

  python3 scripts/scene_worker.py                  # loop: poll, queue, build, publish, sweep
  python3 scripts/scene_worker.py --once           # one poll + queue drain, then exit
  python3 scripts/scene_worker.py --event 1794140173.664091-js55kc [--frame FRAME] [--force] [--no-publish]
  deploy/laptop/scene-worker.sh                    # transient systemd --user unit (Restart=always)

Per replay (recorder host /srv/camstack/snapshots/evidence/<stem>_traj.json without replay/scenes/clip/<stem>.json):
  1. frame: the replay's fusion.frame / calibration.frame / calibration_frame, else the installed calibration whose
     camera centre matches the replay's camera (replays published before the tag existed); none -> skipped.
     Installed calibrations are read from gpuserver (/mnt/media-extra/frigate/config/calib) and cached per frame.
  2. cache: a scene of the same frame built for a visit within --cache-h hours is reused (manifest only) unless a
     quick look at the cameras now (one frame each, median grey difference) says the view changed.
  3. frames: every calibrated camera's Frigate recording around the visit (gpuserver `docker exec frigate curl
     .../clip.mp4`, padded by --pad-s), one person-free frame each (scripts/scene_frames.py: a person-free sample, or
     the person boxes filled from matching samples).
  4. solve: scripts/multicam_rig.py --align-to <installed calibrations>: VGGT on those frames, then a similarity
     transform onto the installed cameras (rotations + centres). Residual above --max-centre-m / --max-rot-deg ->
     status "camera_moved": no scene is attached, an ntfy notification goes out (once per frame per --cache-h).
  5. splat (make_splat.py train, anti-spike flags) -> ply_to_splat.py -> rig_objects.py (VLM naming) ->
     publish_scene.py -> rsync to recorder host replay/scenes/clip/ -> manifest replay/scenes/clip/<stem>.json.
GPU: every GPU step waits until nvidia-smi shows --min-free-mib free (the llama services share the card); a busy
GPU skips the cycle instead of evicting anything. Retention: --sweep-h hourly, clip manifests whose replay file is
gone (the bridge's retention janitor deletes <event>_* with the sighting) are removed, then builds no manifest uses.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()
SPLAT_PY = str(HOME / "src/splat/.venv/bin/python")
PIX_PY = str(HOME / "src/pixarmesh/.venv/bin/python")
recorder host = "db-host"
PVE = "gpuserver"
EVID = "/srv/camstack/snapshots/evidence"
SCENES = "/srv/camstack/replay/scenes"
CALIB_PVE = "/mnt/media-extra/frigate/config/calib"
FRIGATE = "http://127.0.0.1:5000"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def slug(s: str) -> str:            # same rule as publish_scene.py / persons-ui scene_slug
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-")[:120]


def run(cmd, timeout=1800, **kw):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    if r.returncode:
        raise RuntimeError(f"{Path(str(cmd[0])).name} {' '.join(map(str, cmd[1:3]))} failed ({r.returncode}): "
                           f"{(r.stderr or r.stdout)[-800:]}")
    return r.stdout, round(time.time() - t, 1)


def remote_py(host: str, code: str, arg: str = "", timeout=120) -> str:
    """Run a Python snippet on host (stdin), argument as JSON in argv[1]."""
    r = subprocess.run(["ssh", host, "python3 - " + shlex.quote(json.dumps(arg) if arg else "{}")], input=code,
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"ssh {host}: {r.stderr[-500:]}")
    return r.stdout


# ---------------------------------------------------------------- recorder host side snippets (stdlib python3 there)
DISCOVER = r'''
import json, os, sys, time
a = json.loads(sys.argv[1]); EV, SC = a["evid"], a["scenes"] + "/clip"
out = []
cut = time.time() - a["lookback_h"] * 3600
for f in os.listdir(EV):
    if not f.endswith("_traj.json"):
        continue
    p = os.path.join(EV, f)
    st = os.stat(p)
    if st.st_mtime < cut:
        continue
    stem = f[:-len("_traj.json")]
    if os.path.exists(os.path.join(SC, stem + ".json")) and not a.get("all"):
        continue
    try:
        t = json.load(open(p))
    except ValueError:
        continue
    fz = t.get("fusion") or {}
    out.append({"stem": stem, "mtime": st.st_mtime, "event": t.get("event"), "camera_name": t.get("camera_name"),
                "frame": fz.get("frame") or (t.get("calibration") or {}).get("frame") or t.get("calibration_frame"),
                "camera_C": (t.get("camera") or {}).get("C"), "members": fz.get("members"),
                "window": fz.get("window"), "events_by_camera": t.get("events_by_camera")})
print(json.dumps(out))
'''

PUT_MANIFEST = r'''
import json, os, sys
a = json.loads(sys.argv[1]); d = a["scenes"] + "/clip"
os.makedirs(d, exist_ok=True)
p = os.path.join(d, a["stem"] + ".json"); tmp = p + ".tmp"
open(tmp, "w").write(json.dumps(a["manifest"], indent=1)); os.replace(tmp, p)
print(p)
'''

SWEEP = r'''
import json, os, sys, time
a = json.loads(sys.argv[1]); EV, d = a["evid"], a["scenes"] + "/clip"
gone, used = [], set()
if not os.path.isdir(d):
    print(json.dumps({"manifests_removed": 0, "builds_removed": 0})); sys.exit()
for f in os.listdir(d):
    if f.endswith(".json") and not f.startswith("b-"):
        stem = f[:-5]
        if not os.path.exists(os.path.join(EV, stem + "_traj.json")):
            os.remove(os.path.join(d, f)); gone.append(stem); continue
        try:
            b = json.load(open(os.path.join(d, f))).get("build")
        except ValueError:
            b = None
        if b:
            used.add(b)
removed = 0
for f in os.listdir(d):
    if not f.startswith("b-"):
        continue
    b = f.split("_objects")[0].split(".")[0]
    if b in used or time.time() - os.stat(os.path.join(d, f)).st_mtime < a["min_age_h"] * 3600:
        continue
    p = os.path.join(d, f)
    if os.path.isdir(p):
        for g in os.listdir(p):
            os.remove(os.path.join(p, g))
        os.rmdir(p)
    else:
        os.remove(p)
    removed += 1
print(json.dumps({"manifests_removed": len(gone), "builds_removed": removed, "stems": gone[:20]}))
'''

NOTIFY = r'''
import json, sys, urllib.request, base64
a = json.loads(sys.argv[1]); E = {}
for line in open("/srv/camstack/.env"):
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.rstrip("\n").split("=", 1); E[k.strip()] = v.strip().strip('"').strip("'")
h = {"Title": a["title"], "Tags": "warning,camera"}
if E.get("NTFY_PUB_TOKEN"):
    h["Authorization"] = "Bearer " + E["NTFY_PUB_TOKEN"]
elif E.get("NTFY_PUB_USER"):
    h["Authorization"] = "Basic " + base64.b64encode((E["NTFY_PUB_USER"] + ":" + E["NTFY_PUB_PASS"]).encode()).decode()
url = E.get("NTFY_URL", "http://127.0.0.1:8090/squirrels")
r = urllib.request.urlopen(urllib.request.Request(url, data=a["msg"].encode(), headers=h), timeout=15)
print(r.status)
'''


class Worker:
    def __init__(self, a):
        self.a = a
        self.state_dir = Path(a.state)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.sf = self.state_dir / "state.json"
        self.st = json.loads(self.sf.read_text()) if self.sf.is_file() else {}
        for k in ("builds", "skip", "notified"):
            self.st.setdefault(k, {})

    def save(self):
        tmp = self.sf.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.st, indent=1))
        tmp.replace(self.sf)

    # ------------------------------------------------------------ calibrations
    def sync_calibs(self) -> dict:
        """gpuserver installed calibrations -> STATE/calib/<frame slug>/<cam>.json (history kept). {frame: dir}."""
        code = r'''
import json, os, sys
d = "%s"
out = {}
for f in sorted(os.listdir(d)):
    if f.endswith(".json"):
        try: out[f[:-5]] = json.load(open(os.path.join(d, f)))
        except ValueError: pass
print(json.dumps(out))
''' % CALIB_PVE
        try:
            cur = json.loads(remote_py(PVE, code))
        except Exception as e:                     # gpuserver down: work from the cache
            log("calib sync failed:", e)
            cur = {}
        for cam, m in cur.items():
            if m.get("frame"):
                d = self.state_dir / "calib" / slug(m["frame"])
                d.mkdir(parents=True, exist_ok=True)
                p = d / f"{cam}.json"
                if not p.is_file() or json.loads(p.read_text()) != m:
                    p.write_text(json.dumps(m, indent=1))
        self.installed = {cam: m.get("frame") for cam, m in cur.items()}
        frames = {}
        for d in (self.state_dir / "calib").glob("*"):
            fr = {json.loads(p.read_text()).get("frame") for p in d.glob("*.json")}
            if len(fr) == 1:
                frames[fr.pop()] = d
        return frames

    def resolve_frame(self, item, frames):
        if item.get("frame"):
            return item["frame"] if item["frame"] in frames else None
        C, cam = item.get("camera_C"), item.get("camera_name")
        if not C or not cam:
            return None
        for fr, d in frames.items():               # replays made before the frame tag: match the camera centre
            p = d / f"{cam}.json"
            if p.is_file():
                Ci = json.loads(p.read_text())["C"]
                if sum((x - y) ** 2 for x, y in zip(C, Ci)) ** 0.5 < 0.005:
                    return fr
        return None

    # ------------------------------------------------------------ GPU
    def gpu_ok(self, wait_s=None) -> bool:
        wait_s = self.a.gpu_wait_s if wait_s is None else wait_s
        t0 = time.time()
        while True:
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=20).stdout
                free = int(o.split()[0])
            except Exception:
                free = 0
            if free >= self.a.min_free_mib:
                return True
            if time.time() - t0 > wait_s:
                log(f"GPU busy: {free} MiB free < {self.a.min_free_mib}; deferring")
                return False
            time.sleep(15)

    # ------------------------------------------------------------ frigate
    def frigate_get(self, path: str, out: Path | None = None, timeout=180):
        cmd = ["ssh", PVE, f"docker exec frigate curl -sf '{FRIGATE}{path}'"]
        if out:
            with open(out, "wb") as f:
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL, timeout=timeout)
            if r.returncode or out.stat().st_size < 10000:
                raise RuntimeError(f"frigate {path}: no clip ({out.stat().st_size} bytes)")
            return out
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode:
            raise RuntimeError(f"frigate {path} failed")
        return json.loads(r.stdout)

    def visit_window(self, item):
        if item.get("members"):
            s = min(m["start"] for m in item["members"])
            e = max(m["end"] or m["start"] for m in item["members"])
        elif item.get("window"):
            s, e = item["window"]
        else:
            ev = self.frigate_get(f"/api/events/{item['event']}")
            s, e = ev["start_time"], ev.get("end_time") or ev["start_time"] + 10
        return float(s), float(e)

    # ------------------------------------------------------------ cache
    def grab(self, cam: str, t0: float, t1: float, work: Path, live_ok: bool):
        """Recording of cam over [t0, t1] (Frigate keeps motion/object segments only), else -- for a visit that just
        ended -- the camera's current frame (/api/<cam>/latest.jpg, detect resolution). Path + how, or None."""
        try:
            return self.frigate_get(f"/api/{cam}/start/{int(t0)}/end/{int(t1) + 1}/clip.mp4", work / f"{cam}.mp4"), "recording"
        except Exception:
            pass
        if live_ok:
            try:
                return self.frigate_get(f"/api/{cam}/latest.jpg?h=720", work / f"{cam}.jpg"), "live frame (no recording)"
            except Exception:
                pass
        return None, None

    def view_changed(self, build: dict, cams: list, t0: float, t1: float, work: Path):
        """Cheap camera-moved check before reusing a cached scene: per camera, a frame from the visit's pre-roll vs the
        build's frame, compared by the correlation of their edge maps (same view >= 0.8 even with a person in it;
        a turned camera ~0.2). Returns (changed, cameras checked)."""
        code = r'''
import sys, cv2, numpy as np
a, b = sys.argv[1], sys.argv[2]
if a.endswith(".jpg"):
    f = cv2.imread(a); ok = f is not None
else:
    cap = cv2.VideoCapture(a); ok, f = cap.read()
ok = ok and float(f.std()) >= 8                  # blank frame: camera not checkable
def e(x):
    x = cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (320, 180)).astype(np.float32)
    m = cv2.GaussianBlur(np.hypot(cv2.Sobel(x, cv2.CV_32F, 1, 0), cv2.Sobel(x, cv2.CV_32F, 0, 1)), (7, 7), 0)
    return (m - m.mean()) / (m.std() + 1e-6)
print(float((e(f) * e(cv2.imread(b))).mean()) if ok else -9)
'''
        checked = 0
        live_ok = time.time() - t1 < self.a.live_max_s
        for cam in cams:
            ref = Path(build["raw"]) / f"{cam}.jpg"
            if not ref.is_file():
                continue
            p, _ = self.grab(cam, t0, t0 + 3, work, live_ok)
            if p is None:
                continue
            try:
                o, _ = run([SPLAT_PY, "-I", "-c", code, str(p), str(ref)], timeout=120)
            except Exception as e:
                log("  view check failed:", e)
                continue
            ncc = float(o.strip())
            if ncc < -1:
                log(f"  view check {cam}: no usable frame")
                continue
            checked += 1
            log(f"  view check {cam}: edge correlation {ncc:.2f}")
            if ncc < self.a.view_ncc:
                return True, checked
        return False, checked

    def cached_build(self, frame, t):
        best = None
        for bid, b in self.st["builds"].items():
            if (b["frame"] == frame and b["status"] == "ok" and abs(b["visit_t"] - t) < self.a.cache_h * 3600
                    and (b.get("published") or self.a.no_publish)):
                if best is None or abs(b["visit_t"] - t) < abs(best[1]["visit_t"] - t):
                    best = (bid, b)
        return best

    # ------------------------------------------------------------ build
    def build(self, item, frame, frames, force=False) -> dict:
        a = self.a
        cdir = frames[frame]
        calibs = {p.stem: json.loads(p.read_text()) for p in sorted(cdir.glob("*.json"))}
        cams = [c for c in frame.split(":")[-1].split("+") if c in calibs] or sorted(calibs)
        s, e = self.visit_window(item)
        t0, t1 = s - a.pad_s, min(e + a.pad_s, s - a.pad_s + a.max_clip_s)
        tmid = (s + e) / 2
        work = Path(a.work) / item["stem"]
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        timings, man = {}, {"event": item["stem"], "frame": frame, "visit": [round(s, 2), round(e, 2)],
                            "cameras": cams}
        if not force:
            hit = self.cached_build(frame, tmid)
            if hit:
                bid, b = hit
                tc = time.time()
                changed, checked = self.view_changed(b, cams, t0, e, work)
                if not changed:
                    timings["view_check_s"] = round(time.time() - tc, 1)
                    man.update({k: b[k] for k in ("build", "splat", "objects", "residual", "built")})
                    man.update({"status": "ok", "reused": True, "timings": timings,
                                "note": f"room scene of a visit {abs(b['visit_t'] - tmid) / 60:.0f} min away "
                                        + (f"(cameras unchanged: {checked} checked)" if checked else "(cameras NOT re-checked: no recordings)")})
                    return man
                log("  view changed since the cached build: rebuilding")
        if not self.gpu_ok():
            raise TimeoutError("gpu busy")
        tt = time.time()
        clips, how = [], {}
        live_ok = time.time() - e < a.live_max_s
        for cam in cams:
            p, h = self.grab(cam, t0, t1, work, live_ok)
            if p is None:
                log(f"  {cam}: no recording of the visit")
            else:
                clips.append((cam, p))
                how[cam] = h
        timings["clips_s"] = round(time.time() - tt, 1)
        have = [c for c, _ in clips]
        if len(have) < 2:
            raise RuntimeError(f"recordings for {have} only: a joint solve needs 2+ cameras")
        cmd = [PIX_PY, str(ROOT / "scripts/scene_frames.py"), str(work / "raw"), "--report", str(work / "frames.json")]
        for cam, p in clips:
            W, H = calibs[cam]["image_size"]
            cmd += ["--clip", f"{cam}={p}", "--size", f"{cam}={W}x{H}"]
        _, timings["frames_s"] = run(cmd, timeout=900)
        fr = json.loads((work / "frames.json").read_text())
        have = [c for c in have if (work / "raw" / f"{c}.jpg").is_file()]
        if len(have) < 2:
            raise RuntimeError(f"usable frames for {have} only: a joint solve needs 2+ cameras")
        man["frames"] = {c: {"source": how[c], **{k: fr[c].get(k) for k in ("method", "person_px", "first_last_diff")}}
                         for c in have}
        if not self.gpu_ok():
            raise TimeoutError("gpu busy")
        cmd = [SPLAT_PY, str(ROOT / "scripts/multicam_rig.py"), str(work / "rig"), "--raw", str(work / "raw"),
               "--align-to", str(cdir), "--splat-work", str(work / "splat"), "--vggt-python", SPLAT_PY, "--cams"]
        cmd += [f"{c}={calibs[c]['image_size'][0]}x{calibs[c]['image_size'][1]}" for c in have]
        cmd += ["--lens"] + [f"{c}={calibs[c].get('lens_division_lambda', 0.0)}" for c in have]
        if (cdir / "ref" / "vggt.npz").is_file():          # metric scale from the calibration's own depth maps
            cmd += ["--align-ref", str(cdir / "ref")]
        _, timings["solve_s"] = run(cmd, timeout=900)
        rig = json.loads((work / "rig/calib/rig.json").read_text())
        al = rig["alignment"]
        man["residual"] = {k: al.get(k) for k in ("max_centre_m", "max_rot_deg", "cameras", "scale", "scale_from",
                                                   "scale_centres", "scale_depth", "scale_depth_per_camera")}
        log(f"  alignment: centre <= {al['max_centre_m']} m, rotation <= {al['max_rot_deg']} deg, scale {al['scale']} "
            f"({al.get('scale_from')}; centres {al.get('scale_centres')}, depth {al.get('scale_depth')})")
        if al["max_centre_m"] > a.max_centre_m or al["max_rot_deg"] > a.max_rot_deg:
            moved = [c for c, v in al["cameras"].items()
                     if v["centre_m"] > a.max_centre_m or v["rot_deg"] > a.max_rot_deg]
            man.update({"status": "camera_moved", "moved": moved, "timings": timings,
                        "note": f"cameras moved since the calibration ({', '.join(moved)}): scene not attached; "
                                f"recalibrate (scripts/multicam_rig.py) and reinstall"})
            return man
        if not self.gpu_ok():
            raise TimeoutError("gpu busy")
        _, timings["splat_s"] = run([SPLAT_PY, str(ROOT / "splat/make_splat.py"), "train", str(work / "splat"),
                                     "--iters", str(a.iters), "--sh-degree", "1", "--aniso-weight", "0.1",
                                     "--max-ratio", "4", "--max-scale", "0.3"], timeout=1800)
        C = [calibs[c]["C"] for c in have]
        lo = [min(c[i] for c in C) - a.room_m for i in range(2)]
        hi = [max(c[i] for c in C) + a.room_m for i in range(2)]
        bbox = f"{lo[0]},{lo[1]},-0.3,{hi[0]},{hi[1]},3.0"
        ply = next((work / "splat").rglob("splat.ply"), None) or next((work / "splat").rglob("*.ply"))
        run([SPLAT_PY, str(ROOT / "splat/ply_to_splat.py"), str(ply), str(work / "scene.splat"),
             "--max-ratio", "6", "--max-scale", "0.3", f"--bbox={bbox}"], timeout=300)
        objects = None
        if not a.no_objects and self.gpu_ok():
            try:
                _, timings["objects_s"] = run([PIX_PY, str(ROOT / "scripts/rig_objects.py"), str(work / "rig"),
                                               "--out", str(work / "obj"), "--name", "scene", "--vlm-url", a.vlm_url],
                                              timeout=1500)
                objects = work / "obj/scene_objects.json"
            except Exception as ex:
                log("  objects failed (splat only):", ex)
        bid = f"b-{time.strftime('%Y%m%dT%H%M%S')}-{hashlib.sha1(frame.encode()).hexdigest()[:8]}"
        built = time.strftime("%Y-%m-%d %H:%M")
        cmd = [sys.executable, str(ROOT / "scripts/publish_scene.py"), "--frame", frame, "--name", bid,
               "--url-base", "/replay/scenes/clip/", "--splat", str(work / "scene.splat"), "--out", str(work / "pub"),
               "--note", f"per-clip scene ({item['stem']})",
               "--extra", json.dumps({"visit": man["visit"], "residual": man["residual"]})]
        if objects and objects.is_file():
            cmd += ["--objects", str(objects)]
        run(cmd, timeout=120)
        base = "/replay/scenes/clip/"
        man.update({"status": "ok", "reused": False, "build": bid, "built": built, "splat": f"{base}{bid}.splat",
                    "objects": f"{base}{bid}_objects.json" if objects and objects.is_file() else None,
                    "timings": timings, "note": f"room during this visit ({', '.join(have)})"})
        # keep the person-free frames for later cache checks
        keep = self.state_dir / "builds" / bid
        keep.mkdir(parents=True, exist_ok=True)
        for p in (work / "raw").glob("*.jpg"):
            shutil.copy(p, keep / p.name)
        shutil.copytree(work / "pub", keep / "pub", dirs_exist_ok=True)   # work/<stem> is rebuilt per replay
        self.st["builds"][bid] = {"frame": frame, "visit_t": tmid, "status": "ok", "raw": str(keep),
                                  "build": bid, "splat": man["splat"], "objects": man["objects"],
                                  "residual": man["residual"], "built": built, "pub": str(keep / "pub")}
        return man

    def publish(self, item, man):
        b = self.st["builds"].get(man.get("build") or "")
        if man.get("status") == "ok" and b and Path(b["pub"]).is_dir():     # no-op for files already there
            pub = Path(b["pub"])
            subprocess.run(["ssh", recorder host, f"mkdir -p {SCENES}/clip"], check=True, timeout=60)
            r = subprocess.run(["rsync", "-a", f"{pub}/", f"{recorder host}:{SCENES}/clip/"], capture_output=True, text=True,
                               timeout=600)
            if r.returncode:
                raise RuntimeError("rsync: " + r.stderr[-300:])
            b["published"] = True
        remote_py(recorder host, PUT_MANIFEST, {"scenes": SCENES, "stem": item["stem"], "manifest": man})

    def notify_moved(self, frame, man):
        last = self.st["notified"].get(frame, 0)
        if time.time() - last < self.a.cache_h * 3600:
            return
        res = man.get("residual") or {}
        msg = (f"Room scene for visit {man['event']} not attached: camera(s) {', '.join(man.get('moved', []))} moved "
               f"since calibration {frame} (centre residual {res.get('max_centre_m')} m, rotation "
               f"{res.get('max_rot_deg')} deg). Replays use the old calibration until the rig is re-solved.")
        try:
            remote_py(recorder host, NOTIFY, {"title": "Camstack: camera moved", "msg": msg})
            self.st["notified"][frame] = time.time()
            log("  notified: camera moved")
        except Exception as e:
            log("  notify failed:", e)

    # ------------------------------------------------------------ loop
    def process(self, item, frames, force=False):
        stem = item["stem"]
        frame = self.a.frame or self.resolve_frame(item, frames)
        if not frame:
            self.st["skip"][stem] = {"t": time.time(), "why": "no installed calibration frame for this replay"}
            log(f"{stem}: skipped (replay is not in a known calibration frame)")
            return None
        log(f"{stem}: building ({frame})")
        t = time.time()
        man = self.build(item, frame, frames, force=force)
        man["timings"]["total_s"] = round(time.time() - t, 1)
        if not self.a.no_publish:
            self.publish(item, man)
        if man["status"] == "camera_moved" and not self.a.no_publish:
            self.notify_moved(frame, man)
        self.st["skip"].pop(stem, None)
        self.save()
        log(f"{stem}: {man['status']} in {man['timings']['total_s']} s {json.dumps(man['timings'])}")
        (self.state_dir / "last").mkdir(exist_ok=True)
        (self.state_dir / "last" / f"{stem}.json").write_text(json.dumps(man, indent=1))
        return man

    def sweep(self):
        o = remote_py(recorder host, SWEEP, {"evid": EVID, "scenes": SCENES, "min_age_h": self.a.cache_h})
        log("sweep:", o.strip())
        for bid, b in list(self.st["builds"].items()):          # local: forget builds too old to be reused
            if time.time() - b["visit_t"] > max(self.a.cache_h * 3600, 86400):
                shutil.rmtree(b.get("raw", ""), ignore_errors=True)
                shutil.rmtree(b.get("pub", ""), ignore_errors=True)
                del self.st["builds"][bid]
        for d in Path(self.a.work).glob("*"):                    # work dirs of finished builds
            if time.time() - d.stat().st_mtime > 3600:
                shutil.rmtree(d, ignore_errors=True)
        self.st["last_sweep"] = time.time()
        self.save()

    def cycle(self):
        frames = self.sync_calibs()
        items = json.loads(remote_py(recorder host, DISCOVER, {"evid": EVID, "scenes": SCENES,
                                                      "lookback_h": self.a.lookback_h}))
        now = time.time()
        q = [i for i in items if now - self.st["skip"].get(i["stem"], {}).get("t", 0) > self.a.retry_h * 3600]
        q.sort(key=lambda i: -i["mtime"])                          # newest first
        if q:
            log(f"queue: {len(q)} replay(s) without a scene")
        for item in q:
            try:
                self.process(item, frames)
            except TimeoutError:
                return                                             # GPU busy: try again next cycle
            except Exception as e:
                log(f"{item['stem']}: failed: {e}")
                self.st["skip"][item["stem"]] = {"t": time.time(), "why": str(e)[:300]}
                self.save()
        if now - self.st.get("last_sweep", 0) > self.a.sweep_h * 3600:
            try:
                self.sweep()
            except Exception as e:
                log("sweep failed:", e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--event", help="build for this replay stem (<frigate event>[_fused]) now")
    ap.add_argument("--frame", help="with --event: calibration frame to use")
    ap.add_argument("--force", action="store_true", help="with --event: no cache reuse")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--no-objects", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="run the retention sweep now and exit")
    ap.add_argument("--add-ref", metavar="RIG_WORK", help="register a calibration's multicam_rig.py work dir (vggt.npz + "
                                                           "calib/rig.json) as its scale reference, then exit")
    ap.add_argument("--state", default=str(HOME / ".cache/scene-worker"))
    ap.add_argument("--work", default=str(HOME / ".cache/scene-worker/work"))
    ap.add_argument("--interval", type=float, default=120)
    ap.add_argument("--lookback-h", type=float, default=48, help="only replays written within this many hours")
    ap.add_argument("--retry-h", type=float, default=6, help="retry skipped/failed replays after this many hours")
    ap.add_argument("--cache-h", type=float, default=6, help="reuse a scene of the same frame within this many hours")
    ap.add_argument("--view-ncc", type=float, default=0.6, help="edge correlation below which a camera 'changed'")
    ap.add_argument("--live-max-s", type=float, default=1800,
                    help="a visit that ended less than this ago may use a camera's live frame when it has no recording")
    ap.add_argument("--pad-s", type=float, default=8)
    ap.add_argument("--max-clip-s", type=float, default=60)
    ap.add_argument("--max-centre-m", type=float, default=0.10)
    ap.add_argument("--max-rot-deg", type=float, default=3.0)
    ap.add_argument("--min-free-mib", type=int, default=2500)
    ap.add_argument("--gpu-wait-s", type=float, default=300)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--room-m", type=float, default=6.0, help="splat box: camera centres +- this (m), Z -0.3..3")
    ap.add_argument("--vlm-url", default="http://203.0.113.20:5920")
    ap.add_argument("--sweep-h", type=float, default=1)
    a = ap.parse_args()
    w = Worker(a)
    lock = open(w.state_dir / "lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another scene_worker is running (lock held)")
    if a.add_ref:                             # the installed calibration's own solve -> metric scale reference
        src = Path(a.add_ref)
        rig = json.loads(((src / "calib/rig.json") if (src / "calib/rig.json").is_file() else src / "rig.json").read_text())
        d = w.state_dir / "calib" / slug(rig["frame"]) / "ref"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "vggt.npz", d / "vggt.npz")
        (d / "rig.json").write_text(json.dumps(rig, indent=1))
        print(f"reference for {rig['frame']} -> {d}")
        return
    if a.sweep:
        w.sweep()
        return
    if a.event:
        frames = w.sync_calibs()
        items = json.loads(remote_py(recorder host, DISCOVER, {"evid": EVID, "scenes": SCENES, "lookback_h": 1e6, "all": True}))
        it = next((i for i in items if i["stem"] == a.event), None)
        if it is None:                            # no replay file (yet): build from the Frigate event alone
            it = {"stem": a.event, "event": a.event.split("_")[0], "mtime": time.time()}
        man = w.process(it, frames, force=a.force)
        print(json.dumps(man, indent=1))
        return
    while True:
        try:
            w.cycle()
        except Exception as e:
            log("cycle failed:", e)
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()

"""Download PaddleClas PULC attribute models (and optionally PP-ShiTuV2) and convert them to ONNX for squirrel_guard.pulc.

Usage:
  # one-off converter environment (paddle2onnx 1.0.x converts the old .pdmodel format WITHOUT PaddlePaddle; it has no
  # Python 3.13 wheels, so it lives in its own 3.10 venv):
  uv venv -p 3.10 ~/.cache/sg-paddle2onnx && VIRTUAL_ENV=~/.cache/sg-paddle2onnx uv pip install paddle2onnx==1.0.6 six
  # then, from the repo root, with the project venv (needs onnx + onnxruntime + opencv):
  .venv/bin/python scripts/export_pulc.py [--models person,vehicle,shitu] [--out models/] [--check]

Outputs (untracked, models/):
  pulc_person_attribute.onnx   PPLCNet_x1_0, input "x" float32 (N,3,256,192) RGB, ImageNet mean/std; output (N,26) sigmoid
  pulc_vehicle_attribute.onnx  PPLCNet_x1_0, input "x" float32 (N,3,192,256);                       output (N,19) sigmoid
  ppshituv2_general_rec.onnx   PPLCNetV2_base general recognition, (N,3,224,224) -> (N,512) embedding (offline eval only)
Each file carries ONNX metadata: sg_kind, sg_labels (JSON list, one per output column), sg_input_wh, sg_mean, sg_std,
sg_output ("sigmoid" / "embedding"), sg_thresholds (PaddleClas defaults), sg_source (URL + sha256 + date), sg_dataset,
sg_licence. squirrel_guard/pulc.py reads the labels from there, so a model and its label order cannot drift apart.

Sources (PaddleClas release/2.6, Apache-2.0; the weights are PaddlePaddle's released inference models):
  person_attribute_infer.tar   trained on PA100K   (academic / non-commercial research use)
  vehicle_attribute_infer.tar  trained on VeRi-776 (academic / non-commercial research use)
  general_PPLCNetV2_base_pretrained_v1.0_infer.tar  PP-ShiTuV2 recognition, trained on a mix of public sets
       (Aliproduct, GLDv2, VeRi-Wild, LogoDet-3K, SOP, Inshop, iNat, ...; see PaddleClas docs/en/PPShiTu)
The label order and thresholds below are copied from ppcls/data/postprocess/attr_rec.py (PersonAttribute,
VehicleAttribute) and the preprocessing from deploy/configs/PULC/*/inference_*.yaml.

--check runs the converted models on PaddleClas' own demo images (pulc_demo_imgs.zip) and compares with the outputs
printed in the PaddleClas docs (e.g. 0002_c002_00030670_0.jpg -> yellow 0.9893, hatchback 0.9734).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "squirrel-guard-pulc"
P2O_DEFAULT = Path.home() / ".cache" / "sg-paddle2onnx" / "bin" / "paddle2onnx"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# PaddleClas PersonAttribute (attr_rec.py) output columns, in order. PA100K's attributes, regrouped by PaddleClas.
PERSON_LABELS = ["Hat", "Glasses", "ShortSleeve", "LongSleeve", "UpperStride", "UpperLogo", "UpperPlaid",
                 "UpperSplice", "LowerStripe", "LowerPattern", "LongCoat", "Trousers", "Shorts", "Skirt&Dress",
                 "Boots", "HandBag", "ShoulderBag", "Backpack", "HoldObjectsInFront", "AgeLess18", "Age18-60",
                 "AgeOver60", "Female", "Front", "Side", "Back"]
PERSON_THRESHOLDS = {"default": 0.5, "Glasses": 0.3, "HoldObjectsInFront": 0.6}
VEHICLE_LABELS = ["yellow", "orange", "green", "gray", "red", "blue", "white", "golden", "brown", "black",
                  "sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus", "truck", "estate"]
VEHICLE_THRESHOLDS = {"default": 0.5}

MODELS = {
    "person": {
        "url": "https://paddleclas.bj.bcebos.com/models/PULC/inference/person_attribute_infer.tar",
        "sha256": "576cc739749021298418e61dfa44362acf427a99e055f302c3f895d638a2bde4",
        "dir": "person_attribute_infer", "out": "pulc_person_attribute.onnx",
        "meta": {"sg_kind": "pulc_person_attribute", "sg_labels": PERSON_LABELS, "sg_input_wh": [192, 256],
                 "sg_output": "sigmoid", "sg_thresholds": PERSON_THRESHOLDS,
                 "sg_dataset": "PA100K (Liu et al. 2017; academic/non-commercial research use)",
                 "sg_licence": "PaddleClas code+weights Apache-2.0; training data PA100K research-only"}},
    "vehicle": {
        "url": "https://paddleclas.bj.bcebos.com/models/PULC/inference/vehicle_attribute_infer.tar",
        "sha256": "bba326342be959292ec628473d82a0dd9cea43d115e3b4364b4244a0fd104c8b",
        "dir": "vehicle_attribute_infer", "out": "pulc_vehicle_attribute.onnx",
        "meta": {"sg_kind": "pulc_vehicle_attribute", "sg_labels": VEHICLE_LABELS, "sg_input_wh": [256, 192],
                 "sg_output": "sigmoid", "sg_thresholds": VEHICLE_THRESHOLDS,
                 "sg_dataset": "VeRi-776 (Liu et al. 2016; academic/non-commercial research use)",
                 "sg_licence": "PaddleClas code+weights Apache-2.0; training data VeRi-776 research-only"}},
    "shitu": {
        "url": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/PP-ShiTuV2/"
               "general_PPLCNetV2_base_pretrained_v1.0_infer.tar",
        "sha256": "b55f63f2f93eb42b0ff232154180f5e54d99da0891a200a2439d7a0a0ef8c026",
        "dir": "general_PPLCNetV2_base_pretrained_v1.0_infer", "out": "ppshituv2_general_rec.onnx",
        "meta": {"sg_kind": "ppshituv2_general_rec", "sg_labels": [], "sg_input_wh": [224, 224],
                 "sg_output": "embedding",
                 "sg_dataset": "PP-ShiTuV2 training mix (Aliproduct, GLDv2, VeRi-Wild, LogoDet-3K, SOP, Inshop, iNat2021, "
                               "...); several are research-only",
                 "sg_licence": "PaddleClas code+weights Apache-2.0; training data mixed, research-only parts"}},
}
DEMO_URL = "https://paddleclas.bj.bcebos.com/data/PULC/pulc_demo_imgs.zip"
# expected outputs as printed in PaddleClas docs/en/PULC/PULC_{person,vehicle}_attribute_en.md
DEMO_EXPECT = {
    "vehicle_attribute/0002_c002_00030670_0.jpg": {"yellow": 0.9893, "hatchback": 0.9734},
    "vehicle_attribute/0014_c012_00040750_0.jpg": {"red": 0.9999, "sedan": 1.0},
    "person_attribute/090004.jpg": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
}


def fetch(url: str, dest: Path, sha256: str | None, tries: int = 8) -> Path:
    """Download with resume (the Baidu CDN drops long transfers mid-way) and verify the pinned sha256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and sha256 and hashlib.sha256(dest.read_bytes()).hexdigest() == sha256:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(tries):
        have = part.stat().st_size if part.is_file() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have and r.status == 206 else "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            break
        except Exception as e:  # noqa: BLE001 - retry anything network-shaped
            print(f"  {url}: {type(e).__name__} {e}; retry {attempt + 1}/{tries}")
            time.sleep(2 + attempt)
    else:
        raise SystemExit(f"download failed: {url}")
    got = hashlib.sha256(part.read_bytes()).hexdigest()
    if sha256 and got != sha256:
        part.unlink()
        raise SystemExit(f"sha256 mismatch for {url}: {got} (expected {sha256})")
    part.rename(dest)
    return dest


def safe_extract(tar: Path, into: Path) -> None:
    with tarfile.open(tar) as t:
        for m in t.getmembers():
            if not (m.isfile() or m.isdir()) or m.name.startswith("/") or ".." in Path(m.name).parts:
                raise SystemExit(f"refusing tar member {m.name!r}")
        t.extractall(into, filter="data")


def convert(key: str, out_dir: Path, p2o: Path, opset: int) -> Path:
    spec = MODELS[key]
    tar = fetch(spec["url"], CACHE / Path(spec["url"]).name, spec["sha256"])
    out = out_dir / spec["out"]
    with tempfile.TemporaryDirectory() as td:
        safe_extract(tar, Path(td))
        mdir = Path(td) / spec["dir"]
        raw = Path(td) / "raw.onnx"
        cmd = [str(p2o), "--model_dir", str(mdir), "--model_filename", "inference.pdmodel",
               "--params_filename", "inference.pdiparams", "--save_file", str(raw), "--opset_version", str(opset),
               "--enable_onnx_checker", "True"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=td)
        if r.returncode or not raw.is_file():
            raise SystemExit(f"paddle2onnx failed for {key}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        import onnx
        m = onnx.load(str(raw))
        meta = dict(spec["meta"], sg_mean=MEAN, sg_std=STD,
                    sg_source=f"{spec['url']} sha256={spec['sha256']} converted {dt.date.today()} "
                              f"paddle2onnx opset {opset}")
        for k, v in meta.items():
            p = m.metadata_props.add()
            p.key, p.value = k, v if isinstance(v, str) else json.dumps(v)
        out_dir.mkdir(parents=True, exist_ok=True)
        onnx.save(m, str(out))
    print(f"{key}: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def check(out_dir: Path) -> bool:
    """Converted models vs the PaddleClas documentation outputs on its demo images."""
    import cv2
    import sys
    sys.path.insert(0, str(ROOT))
    from squirrel_guard.pulc import PersonAttributes, VehicleAttributes
    z = fetch(DEMO_URL, CACHE / "pulc_demo_imgs.zip", None)
    ok = True
    with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(z) as zf:
        for name, want in DEMO_EXPECT.items():
            member = f"pulc_demo_imgs/{name}"
            img = cv2.imdecode(__import__("numpy").frombuffer(zf.read(member), "uint8"), cv2.IMREAD_COLOR)
            if name.startswith("vehicle"):
                p = out_dir / MODELS["vehicle"]["out"]
                if not p.is_file():
                    continue
                probs = VehicleAttributes(str(p), threads=1).probs([img])[0]
                good = all(abs(probs[k] - v) < 2e-3 for k, v in want.items())
                print(f"  {name}: " + ", ".join(f"{k} {probs[k]:.4f} (doc {v})" for k, v in want.items()),
                      "OK" if good else "MISMATCH")
            else:
                p = out_dir / MODELS["person"]["out"]
                if not p.is_file():
                    continue
                pa = PersonAttributes(str(p), threads=1)
                probs = pa.probs([img])[0]
                pred = [int(probs[lab] > pa.thresholds.get(lab, pa.thresholds["default"])) for lab in pa.labels]
                good = pred == want
                print(f"  {name}: binarised {pred} {'OK' if good else 'MISMATCH (doc ' + str(want) + ')'}")
            ok &= good
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default="person,vehicle", help="comma list of: person, vehicle, shitu")
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--paddle2onnx", default=str(P2O_DEFAULT if P2O_DEFAULT.is_file() else "paddle2onnx"))
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--check", action="store_true", help="compare with the PaddleClas doc outputs on its demo images")
    a = ap.parse_args()
    out_dir = Path(a.out)
    p2o = Path(a.paddle2onnx) if os.sep in a.paddle2onnx else Path(shutil.which(a.paddle2onnx) or a.paddle2onnx)
    for key in [k.strip() for k in a.models.split(",") if k.strip()]:
        if key not in MODELS:
            raise SystemExit(f"unknown model {key!r}")
        convert(key, out_dir, p2o, a.opset)
    if a.check and not check(out_dir):
        raise SystemExit("converted models do not reproduce the PaddleClas demo outputs")


if __name__ == "__main__":
    main()

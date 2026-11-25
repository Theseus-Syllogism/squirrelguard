"""Download the Ultralytics YOLOv8 Open Images V7 weights and export them to ONNX.

Usage: .venv/bin/python scripts/export_model.py [n|s|m]   (default s)
Note: Ultralytics code and weights are AGPL-3.0. Fine for personal use; commercial use needs their licence
or a different model.
"""
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

size = sys.argv[1] if len(sys.argv) > 1 else "s"
models = Path(__file__).resolve().parent.parent / "models"
models.mkdir(exist_ok=True)
name = f"yolov8{size}-oiv7"
model = YOLO(f"{name}.pt")
onnx = Path(model.export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False))
dst = models / f"{name}.onnx"
shutil.move(onnx, dst)
pt = Path(f"{name}.pt")
if pt.exists():
    shutil.move(pt, models / pt.name)
names = model.names
sq = [i for i, n in names.items() if n.lower() == "squirrel"]
print(f"exported {dst} ({len(names)} classes; Squirrel id = {sq})")

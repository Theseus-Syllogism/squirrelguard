# squirrel-guard detector. Two targets:
#   runtime (default): the detector itself, about 900 MB. Config, secrets, model and events are mounted, not baked in.
#   export: PyTorch + Ultralytics, used once to write models/yolov8s-oiv7.onnx (`docker compose run --rm export`).
FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

FROM base AS export
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install ultralytics onnx onnxruntime onnxslim
COPY scripts/export_model.py scripts/export_model.py
ENV HOME=/tmp YOLO_CONFIG_DIR=/tmp
WORKDIR /tmp
CMD ["python", "/app/scripts/export_model.py", "s"]

FROM base AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg pulseaudio-utils && rm -rf /var/lib/apt/lists/*
RUN pip install onnxruntime opencv-python-headless requests numpy paho-mqtt
COPY squirrel_guard/ squirrel_guard/
ENV HOME=/tmp
CMD ["python", "-m", "squirrel_guard.main", "-c", "config.toml"]

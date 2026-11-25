# squirrel-guard in Frigate mode. onnxruntime + ffmpeg are only for the boxed event clips.
FROM python:3.12.8-slim
# libegl1/libgles2: mediapipe (facegeom) loads libEGL/GLES at import even when running on CPU
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libegl1 libgles2 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "paho-mqtt==2.1.0" "requests==2.32.3" "numpy==2.1.3" \
    "opencv-python-headless==4.10.0.84" "onnxruntime==1.20.1"
# face identification (profiler, face_enabled): the SDK only; the model pack is mounted at run time (academic licence)
RUN pip install --no-cache-dir "inspireface==1.2.4.post3"
# face geometry (profiler, facegeom_enabled; docs/facegeom.md): MediaPipe Tasks, Apache-2.0. --no-deps: its declared
# opencv-contrib-python would replace opencv-python-headless and sounddevice is unused; matplotlib is imported at
# load time by its drawing utils. The Face Landmarker model (Apache-2.0, 3.6 MB) is pinned by checksum.
RUN pip install --no-cache-dir "matplotlib==3.9.4" && \
    pip install --no-cache-dir --no-deps "mediapipe==1.2.2" "absl-py==2.3.1" "flatbuffers==25.9.23"
ADD --checksum=sha256:64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff --chmod=644 \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
    /opt/mediapipe/face_landmarker.task
# IK rig (profiler, rig_enabled; docs/rigfit.md): the fit uses torch autograd; CPU-only wheel (no CUDA libraries)
RUN pip install --no-cache-dir "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu
# vehicle reading (profiler, plate_models; squirrel_guard/vehicle_read.py): plate detector + plate reader, MIT-licensed code
# (ankandrew/open-image-models, ankandrew/fast-plate-ocr), pinned by checksum
ADD --checksum=sha256:c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076 --chmod=644 \
    https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-640-license-plates-end2end.onnx \
    /opt/vehicle/yolo-v9-t-640-license-plates-end2end.onnx
ADD --checksum=sha256:384bbbd2cea3ef54761d3df70822ef3a349ee1a112aeafddbe0e3ba06bc6e47b --chmod=644 \
    https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_s_v2_global.onnx /opt/vehicle/cct_s_v2_global.onnx
WORKDIR /app
COPY squirrel_guard/ /app/squirrel_guard/
COPY config.toml /app/config.toml
USER 1000:1000
CMD ["python", "-u", "-m", "squirrel_guard.frigate_listener", "-c", "/app/config.toml"]

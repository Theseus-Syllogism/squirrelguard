FROM python:3.12.8-slim
# openssl: checks outside timestamps (openssl ts -verify); ca-certificates: the timestamp services' root certificates
RUN apt-get update && apt-get install -y --no-install-recommends openssl ca-certificates curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "flask==3.1.0" "waitress==3.0.2" "psycopg[binary]==3.2.3" "numpy==2.1.3" \
    "cryptography==43.0.3" "opencv-python-headless==4.10.0.84" "onnxruntime==1.20.1"
# release-copy detectors (redact.py), pinned by SHA-256. Both MIT-licensed code:
#   YuNet face detector, OpenCV Zoo;  YOLOv9-t licence-plate detector, ankandrew/open-image-models
RUN mkdir -p /app/models && cd /app/models \
 && curl -fsSL -o face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
 && curl -fsSL -o yolo-v9-t-640-license-plates-end2end.onnx https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-640-license-plates-end2end.onnx \
 && echo "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  face_detection_yunet_2023mar.onnx" > SHA256SUMS \
 && echo "c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076  yolo-v9-t-640-license-plates-end2end.onnx" >> SHA256SUMS \
 && sha256sum -c SHA256SUMS
# facegeom.py: a copy of squirrel_guard/facegeom.py (render_html for the face geometry panel; numpy only)
COPY app.py related.py facegeom.py facemodels.py evidence_export.py tsa.py access.py redact.py gaitlab.py vehmatch.py /app/
# shared design system (ui.css, ui.js), served at /ui/
COPY static /app/static
USER 1000:1000
CMD ["python", "-u", "/app/app.py"]

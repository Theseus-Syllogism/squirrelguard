# easelscan: easel notes -> straightened page -> local vision LLM -> PDF/Markdown/JSON/.ics + ntfy (docs/easelscan.md)
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir "opencv-python-headless==4.10.0.84" "numpy<2.3" "reportlab==4.2.5"
WORKDIR /app
COPY easelscan/ /app/easelscan/
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "easelscan", "watch"]

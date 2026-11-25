FROM python:3.12.8-slim
# ssh client for the camera deterrent; ssh needs a passwd entry for the uid it runs as
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client && rm -rf /var/lib/apt/lists/* \
    && useradd -u 1000 -M -d /tmp camstack
RUN pip install --no-cache-dir "paho-mqtt==2.1.0" "psycopg[binary]==3.2.3" "requests==2.32.3" "numpy==2.1.3" "tzdata==2024.2" "pillow==11.0.0"
COPY app.py intel.py explain.py strip.py routine.py facematch.py facegeomcmp.py vehmatch.py /app/
USER camstack
WORKDIR /app
CMD ["python", "-u", "/app/app.py"]

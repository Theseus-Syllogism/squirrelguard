FROM alpine:3.21
RUN apk add --no-cache ffmpeg tzdata findutils coreutils python3
COPY record.sh janitor.sh snapshot.sh seal.py tsa.py /usr/local/bin/
USER 1000:1000

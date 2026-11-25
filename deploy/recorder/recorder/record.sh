#!/bin/sh
# Continuous recording from go2rtc in 10-minute MKV segments, stream copy only (the recorder host cannot transcode).
# MKV survives a crash mid-segment, unlike MP4.
set -u
CAM=${CAMERA:-frontyard}
DIR=/recordings/$CAM
mkdir -p "$DIR"
while true; do
  ffmpeg -nostdin -hide_banner -loglevel warning -fflags +genpts -rtsp_transport tcp -timeout 15000000 \
    -i "rtsp://camstack:${GO2RTC_RTSP_PASS}@go2rtc:8554/${CAM}" \
    -map 0 -c copy -f segment -segment_time "${SEGMENT_S:-600}" -segment_atclocktime 1 \
    -reset_timestamps 1 -strftime 1 -segment_format matroska "$DIR/%Y-%m-%d_%H-%M-%S.mkv"
  echo "$(date) ffmpeg exited ($?), retrying in 10s" >&2
  sleep 10
done

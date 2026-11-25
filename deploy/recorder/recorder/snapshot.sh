#!/bin/sh
# Latest camera frame for the portal (live.jpg), from go2rtc's 640x360 sub stream over RTSP inside the Docker
# network. Keyframes only (-skip_frame nokey): no full decode, negligible CPU. Written atomically for nginx.
set -u
CAM=${CAMERA:-frontyard}
OUT=${SNAP_FILE:-live.jpg}   # live_v4.jpg etc. for further cameras
mkdir -p /snap
while true; do
  ffmpeg -nostdin -hide_banner -loglevel error -rtsp_transport tcp -timeout 15000000 -skip_frame nokey \
    -i "rtsp://camstack:${GO2RTC_RTSP_PASS}@go2rtc:8554/${CAM}_sub" \
    -an -fps_mode passthrough -q:v 5 -f image2 -update 1 -atomic_writing 1 "/snap/$OUT"
  echo "$(date) snapshot ffmpeg exited ($?), retrying in 10s" >&2
  sleep 10
done

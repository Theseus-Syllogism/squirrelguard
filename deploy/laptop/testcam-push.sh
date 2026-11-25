#!/bin/sh
# Push the laptop's USB webcam (Sonix "USB 2.0 Camera") to the recorder host go2rtc as stream `testcam` (second camera for
# multi-camera testing). Password: ~/.config/testcam/env (PASS=..., mode 600; the go2rtc RTSP password).
#   systemd-run --user --unit=testcam-push ~/src/squirrel-guard/deploy/laptop/testcam-push.sh
set -eu
. "$HOME/.config/testcam/env"
DEV=${TESTCAM_DEV:-}
if [ -z "$DEV" ]; then                      # find the webcam by name (video numbers change with plug order)
    for d in /sys/class/video4linux/video*; do
        if grep -q "USB 2.0 Camera" "$d/name" 2>/dev/null; then DEV=/dev/$(basename "$d"); break; fi
    done
fi
[ -n "$DEV" ] || { echo "USB 2.0 Camera not found" >&2; exit 1; }
exec ffmpeg -hide_banner -loglevel warning -nostdin \
    -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 15 -i "$DEV" \
    -c:v libx264 -preset veryfast -tune zerolatency -g 30 -b:v 2500k -maxrate 3000k -bufsize 3000k -pix_fmt yuv420p \
    -an -f rtsp -rtsp_transport tcp "rtsp://camstack:${PASS}@192.0.2.20:8554/testcam"

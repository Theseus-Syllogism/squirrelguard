#!/bin/bash
# Synthetic-squirrel test stream. Runs in the squirrel-test container and PUBLISHES to Frigate's go2rtc:
#   $1 = rtsp://frigate:8554/frontyard_squirrel_test   (loops forever, reconnecting)
# Live main stream from the recorder host go2rtc, scaled to 1280x720, with a background-removed squirrel cutout and a
# solid silhouette walking across it. 60 s cycle, 15 s per phase:
#   0-15 cutout 120 px | 15-30 cutout 200 px | 30-45 cutout 300 px | 45-60 silhouette 300 px  (widths of 1280)
FF=ffmpeg
D=/test
SRC="rtsp://camstack:${FRIGATE_RTSP_PASSWORD}@192.0.2.20:8554/frontyard"
# walk left->right across the frame within each 15 s phase; small bob so it reads as moving
X='(mod(t,15)/15)*(W+w)-w'
Y='H-h-30+6*sin(t*7)'
while true; do
"$FF" -hide_banner -loglevel error -rtsp_transport tcp -i "$SRC" \
  -loop 1 -i "$D/squirrel_cutout.png" -loop 1 -i "$D/squirrel_silhouette.png" \
  -filter_complex "
    [0:v]scale=1280:720,fps=15[bg];
    [1:v]split=3[c1][c2][c3];
    [c1]scale=120:-1,format=rgba[s120];
    [c2]scale=200:-1,format=rgba[s200];
    [c3]scale=300:-1,format=rgba[s300];
    [2:v]scale=300:-1,format=rgba[sil];
    [bg][s120]overlay=x='$X':y='$Y':enable='lt(mod(t,60),15)':shortest=1[a];
    [a][s200]overlay=x='$X':y='$Y':enable='between(mod(t,60),15,30)':shortest=1[b];
    [b][s300]overlay=x='$X':y='$Y':enable='between(mod(t,60),30,45)':shortest=1[c];
    [c][sil]overlay=x='$X':y='$Y':enable='gte(mod(t,60),45)':shortest=1[out]" \
  -map "[out]" -an -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -pix_fmt yuv420p \
  -rtsp_transport tcp -f rtsp "$1"
echo "$(date) compositor exited ($?), restarting in 5s" >&2
sleep 5
done

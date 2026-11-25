# Frigate on gpuserver (203.0.113.20 / 192.0.2.30)

- Deployed at `/mnt/media-extra/frigate` (compose.yaml, config/, media/, .env with secrets, admin-password.txt).
- UI: https://192.0.2.30:8971 (self-signed cert), user `admin`, password in `/mnt/media-extra/frigate/admin-password.txt`.
- Detector: OpenVINO on the Intel UHD 770 (`renderD128`), YOLOv8s Open Images V7 ONNX (`yolo-generic`, 640),
  ~43 ms/inference. The AMD GPUs are left to the llama.cpp verifier.
- Input: go2rtc on the recorder host (`frontyard_sub` for detect, `frontyard` for event clips). gpuserver cannot reach the
  camera's 198.51.100.0/24 directly: its `wg1` WireGuard interface owns that range.
- Recording: event clips only (alerts = squirrel, 14 d; detections = other animals/people, 7 d). The recorder host records 24/7.
- MQTT: recorder host Mosquitto, user `frigate`, prefix `frigate/`.

## AppArmor (required)
`/etc/apparmor/parser.conf` pins `policy-features` to an old feature set, so profiles without an `abi` line
(including Docker 26's generated `docker-default`) are compiled without unix-socket rules while kernel 6.17 enforces
af_unix mediation. s6-overlay then dies with `s6-ipcserver-socketbinder: unable to create socket: Permission denied`.
Fix: `docker-frigate.apparmor` (= docker-default + `unix,`, pinned to `abi <abi/4.0>`), installed as
`/etc/apparmor.d/docker-frigate` and loaded with `apparmor_parser -r -W`; compose uses `apparmor=docker-frigate`.

## Docker data-root
Moved 2026-11-05 from /mnt/jellyfin/docker (99% full) to /mnt/media-extra/docker; drop-in
`docker.service.d/zz-data-root-mount.conf` has `RequiresMountsFor=/mnt/media-extra`.

## squirrel-guard (Frigate mode)
Service `squirrel-guard` in the same compose project: listens on `frigate/events` and `frigate/reviews`
(manual/API events in 0.18 only appear as reviews, with objects like `squirrel: <sub_label>`), fetches the cropped
snapshot from `http://frigate:5000`, asks Qwen on `host.docker.internal:5922`, runs the deterrent (dry run), and
publishes to `squirrelguard/<camera>/event` for the recorder host bridge (Postgres, ntfy, email). Secrets in `sg.env`.
Deploy: copy `squirrel_guard/{__init__,frigate_listener,verifier,actuator,publish}.py` into `squirrel-guard/squirrel_guard/`.
Test without animals: `docker exec frigate curl -X POST -H 'Content-Type: application/json'
-d '{"duration":6,"include_recording":false,"score":0.9,"sub_label":"pipeline-test"}'
http://127.0.0.1:5000/api/events/frontyard/squirrel/create` (then delete the event and review item).

## Notifications (Wyze-style)
- `squirrel` events: Qwen check -> ntfy topic `squirrels` (boxed snapshot), at most one per 2 min.
- `person` events: no Qwen check -> ntfy topic `people`, at most one per 5 min.
- Every notification has a Click link to `https://192.0.2.30:8971/explore?event_id=<id>`.
- After the event ends, `clips.py` fetches Frigate's clip, re-runs the OIV7 detector (every 2nd frame at 12 fps),
  burns in boxes, encodes 960x540 H.264 (<9.5 MB) and the bridge posts it as a second notification ("Clip: ...").
- Frigate event retention is `mode: all` so clips cover the whole event window. With `active_objects`, segments
  without a moving tracked object are dropped, which also leaves manual test events with "No recordings found".

## Synthetic squirrel test rig
- `test/squirrel_cutout.png` (background-removed stock squirrel) and `test/squirrel_silhouette.png` (same alpha, solid).
- `squirrel-test` container runs `test/squirrel-test.sh`: live main stream from the recorder host go2rtc, scaled to 1280x720,
  with the overlays walking across; 60 s cycle: cutout 120 / 200 / 300 px, then silhouette 300 px (widths of 1280).
  It publishes to `mediamtx` (compose network only); Frigate camera `frontyard_test` (detect 1280x720) reads it.
- Not via Frigate's go2rtc: `exec:` sources need GO2RTC_ALLOW_ARBITRARY_EXEC=true, `{output}` collides with Frigate's
  `{VAR}` substitution, and an empty publish-only stream list is dropped and breaks the generated go2rtc.yaml.
- Stop: `docker compose stop squirrel-test mediamtx` and set `frontyard_test` `enabled: false` in config.yml.
- Grafana panels exclude cameras matching `%_test`.

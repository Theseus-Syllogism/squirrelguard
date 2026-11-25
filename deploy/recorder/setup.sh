#!/bin/bash
# Run ON db-host from /srv/camstack after copying deploy/recorder/* there.
# Idempotent: generates secrets only if missing, never prints them.
# Needs /srv/camstack/pg.url (from `sudo recorder-newdb squirrelguard`).
set -euo pipefail
cd "$(dirname "$0")"
umask 077

gen() { head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 24; }

[ -s pg.url ] || { echo "missing pg.url (run: sudo recorder-newdb squirrelguard)" >&2; exit 1; }
if [ ! -s .env ]; then
  url=$(cat pg.url)
  rest=${url#*://}; creds=${rest%%@*}; db=${rest#*/}; db=${db%%\?*}
  {
    echo "PG_USER=${creds%%:*}"
    echo "PG_PASS=${creds#*:}"
    echo "PG_DB=$db"
    echo "GF_SECURITY_ADMIN_PASSWORD=$(gen)"
    echo "MQTT_USER=squirrelguard"
    echo "MQTT_PASS=$(gen)"
    echo "NTFY_ADMIN_USER=operator"
    echo "NTFY_ADMIN_PASS=$(gen)"
    echo "NTFY_PUB_USER=squirrelguard"
    echo "NTFY_PUB_PASS=$(gen)"
  } > .env
  echo "generated .env"
fi
# keys added after the first release: append if missing, never overwrite
grep -q '^PG_URL=' .env || echo "PG_URL=$(cat pg.url)" >> .env
grep -q '^GO2RTC_PASS=' .env || echo "GO2RTC_PASS=$(gen)" >> .env
grep -q '^GO2RTC_RTSP_PASS=' .env || echo "GO2RTC_RTSP_PASS=$(gen)" >> .env
grep -q '^FRIGATE_MQTT_PASS=' .env || echo "FRIGATE_MQTT_PASS=$(gen)" >> .env
grep -q '^PERSONS_UI_USER=' .env || { echo "PERSONS_UI_USER=admin"; echo "PERSONS_UI_PASS=$(gen)"; } >> .env
# evidence packages, the signing key and the capture-time ledger: written by uid 1000 (persons-ui, sealer)
install -d -m 750 evidence-store ledger && install -d -m 700 evidence-keys
[ "$(id -u)" = 0 ] && chown 1000:1000 evidence-store evidence-keys ledger
grep -q '^FILEBROWSER_PASS=' .env || echo "FILEBROWSER_PASS=$(gen)" >> .env
grep -q '^KUMA_USER=' .env || { echo "KUMA_USER=admin"; echo "KUMA_PASS=$(gen)"; } >> .env
grep -q '^NTFY_ALERT_USER=' .env || { echo "NTFY_ALERT_USER=kuma"; echo "NTFY_ALERT_PASS=$(gen)"; } >> .env
grep -q '^CAM_RTSP_PASS=.' .env || echo "WARNING: CAM_RTSP_PASS (camera RTSP password) missing from .env; go2rtc cannot reach the camera" >&2
grep -q '^SMTP_HOST=' .env || cat >> .env <<'EOF'
# Email via Proton Mail Bridge (off while SMTP_HOST is empty). Bridge SMTP: 127.0.0.1:1025, STARTTLS,
# username = your Proton address, password = the Bridge-generated one (not your Proton password).
SMTP_HOST=
SMTP_PORT=1025
SMTP_USER=
SMTP_PASS=
SMTP_FROM=
SMTP_TO=
SMTP_TLS_VERIFY=0
SMTP_MIN_INTERVAL_S=1800
NTFY_MIN_INTERVAL_S=120
EOF
chmod 600 .env pg.url
set -a; . ./.env; set +a
mkdir -p snapshots

# MQTT password file (hashed); readable by the container's mosquitto user
if [ ! -s mosquitto/passwd ]; then
  touch mosquitto/passwd
  docker run --rm -v "$PWD/mosquitto:/m" eclipse-mosquitto:2.0.20 \
    mosquitto_passwd -b /m/passwd "$MQTT_USER" "$MQTT_PASS"
  echo "created mosquitto/passwd"
fi
if ! grep -q "^frigate:" mosquitto/passwd; then
  docker run --rm -v "$PWD/mosquitto:/m" eclipse-mosquitto:2.0.20 mosquitto_passwd -b /m/passwd frigate "$FRIGATE_MQTT_PASS"
  echo "added frigate MQTT user"
fi
chmod 644 mosquitto/passwd mosquitto/mosquitto.conf ntfy/server.yml
chmod -R a+rX grafana mosquitto ntfy
chmod 755 .

# event table
psql "$(cat pg.url)" -v ON_ERROR_STOP=1 -q -f schema.sql && psql "$(cat pg.url)" -v ON_ERROR_STOP=1 -q -f persons_schema.sql && echo "schema applied"


# File Browser: initialise its database once with a generated admin password, read-only permissions
if ! docker run --rm -v camstack_filebrowser-db:/database filebrowser/filebrowser:v2.31.2 \
     users ls --database /database/filebrowser.db >/dev/null 2>&1; then
  docker compose create filebrowser >/dev/null 2>&1   # lets compose own the volume (no 'external' warning)
  docker run --rm -v camstack_filebrowser-db:/database alpine:3.21 chown 1000:1000 /database
  docker run --rm --user 1000:1000 -v camstack_filebrowser-db:/database filebrowser/filebrowser:v2.31.2 \
    config init --database /database/filebrowser.db --auth.method=json >/dev/null
  docker run --rm --user 1000:1000 -v camstack_filebrowser-db:/database filebrowser/filebrowser:v2.31.2 \
    users add admin "$FILEBROWSER_PASS" --database /database/filebrowser.db \
    --perm.admin=false --perm.create=false --perm.delete=false --perm.modify=false --perm.rename=false \
    --perm.share=false --perm.execute=false --perm.download=true >/dev/null
  echo "filebrowser initialised (read-only admin)"
fi
docker compose up -d --build
sleep 5

# ntfy users: admin for the phone app, write-only publisher for squirrel-guard.
# (Capture the list first: `cmd | grep -q` under pipefail fails when grep exits early, which made
#  this re-add an existing user and abort the script.)
users=$(docker compose exec -T ntfy ntfy user list 2>&1 || true)   # the ntfy CLI prints lists on stderr
if ! grep -q "user $NTFY_ADMIN_USER" <<<"$users"; then
  docker compose exec -T -e NTFY_PASSWORD="$NTFY_ADMIN_PASS" ntfy ntfy user add --role=admin "$NTFY_ADMIN_USER" >/dev/null
  echo "ntfy admin user created"
fi
if ! grep -q "user $NTFY_PUB_USER" <<<"$users"; then
  docker compose exec -T -e NTFY_PASSWORD="$NTFY_PUB_PASS" ntfy ntfy user add "$NTFY_PUB_USER" >/dev/null
  docker compose exec -T ntfy ntfy access "$NTFY_PUB_USER" squirrels write-only >/dev/null
  echo "ntfy publisher user created (write-only on topic 'squirrels')"
fi
docker compose exec -T ntfy ntfy access "$NTFY_PUB_USER" people write-only >/dev/null   # person alerts (idempotent)
docker compose exec -T ntfy ntfy access "$NTFY_PUB_USER" visitors write-only >/dev/null  # visitor alerts + reports
if ! grep -q "user $NTFY_ALERT_USER" <<<"$users"; then
  docker compose exec -T -e NTFY_PASSWORD="$NTFY_ALERT_PASS" ntfy ntfy user add "$NTFY_ALERT_USER" >/dev/null
  docker compose exec -T ntfy ntfy access "$NTFY_ALERT_USER" camstack-alerts write-only >/dev/null
  echo "ntfy alert user created (write-only on topic 'camstack-alerts')"
fi
# Access token for the bridge (Bearer auth; Suricata drops plaintext Basic auth on this host)
if ! grep -q '^NTFY_PUB_TOKEN=tk_' .env; then
  tok=$(docker compose exec -T ntfy ntfy token add "$NTFY_PUB_USER" 2>&1 | grep -o -E 'tk_[A-Za-z0-9]+' | head -1)
  [ -n "$tok" ] && { sed -i '/^NTFY_PUB_TOKEN=/d' .env; echo "NTFY_PUB_TOKEN=$tok" >> .env; echo "ntfy token created"; \
    docker compose up -d bridge >/dev/null 2>&1; } || echo "WARNING: could not create ntfy token" >&2
fi
docker compose ps --format 'table {{.Name}}\t{{.Status}}'

#!/bin/sh
# Per-clip room scenes (scripts/scene_worker.py) as a TRANSIENT systemd --user unit on the laptop (it needs the
# RTX 4070; the recorder host has no GPU and gpuserver is AMD). Logs: journalctl --user -u scene-worker -f
#   deploy/laptop/scene-worker.sh            start (extra args go to scene_worker.py, e.g. --cache-h 12)
#   systemctl --user stop scene-worker       stop (Restart=always only restarts after a crash/exit, not a stop)
#   systemctl --user status scene-worker
# Gone after a reboot/logout by design: see docs/scenes.md for a persistent install.
set -eu
REPO=$(cd "$(dirname "$0")/../.." && pwd)
if systemctl --user is-active --quiet scene-worker; then
    echo "scene-worker already running (systemctl --user stop scene-worker first)" >&2; exit 1
fi
systemctl --user reset-failed scene-worker 2>/dev/null || true
exec systemd-run --user --unit=scene-worker --description="squirrel-guard per-clip room scenes" \
    -p Restart=always -p RestartSec=60 -p Nice=10 -p IOSchedulingClass=idle \
    --setenv=PYTHONUNBUFFERED=1 --setenv=PATH="$PATH" --setenv=SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-}" \
    --working-directory="$REPO" \
    /usr/bin/python3 "$REPO/scripts/scene_worker.py" "$@"

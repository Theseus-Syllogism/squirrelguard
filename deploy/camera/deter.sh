#!/bin/sh
# Squirrel deterrent on the Thingino camera. Run ONLY as the forced command of the recorder host's deter SSH key:
#   command="/etc/deter/deter.sh",no-port-forwarding,no-agent-forwarding,no-X11-forwarding,no-pty ssh-ed25519 ...
# Request ($SSH_ORIGINAL_COMMAND): "<sound|random> [volume 0-100]". Anything else is refused.
# random picks only from sounds/ (squirrel deterrents); announce/ files (e.g. person_announce) play by name only.
D=/etc/deter/sounds
A=/etc/deter/announce
set -- $SSH_ORIGINAL_COMMAND
name=${1:-random}
vol=${2:-80}
case "$vol" in ''|*[!0-9]*) echo "bad volume" >&2; exit 2 ;; esac
[ "$vol" -gt 100 ] && vol=100
if [ "$name" = "random" ]; then
  set -- $D/*.opus
  n=$(awk -v n=$# 'BEGIN { srand(); print int(rand() * n) + 1 }')   # BusyBox od has no -A
  eval "file=\${$n}"
else
  case "$name" in *[!a-z0-9_]*) echo "bad sound name" >&2; exit 2 ;; esac
  file="$D/$name.opus"
  [ -f "$file" ] || file="$A/$name.opus"
fi
[ -f "$file" ] || { echo "no such sound: $name" >&2; exit 2; }
play stop >/dev/null 2>&1
play -v "$vol" "$file"
echo "$(date "+%F %T") played $(basename "$file") vol=$vol" >> /tmp/deter.log
echo "played $(basename "$file" .opus) vol=$vol"

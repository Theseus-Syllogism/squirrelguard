#!/bin/sh
# Every 5 min: report recording freshness to Uptime Kuma (push monitor). Every hour: retention
# (delete segments older than RETAIN_DAYS, then oldest-first until under RETAIN_MAX_GB).
set -u
DAYS=${RETAIN_DAYS:-7}
MAX=$(( ${RETAIN_MAX_GB:-350} * 1024 * 1024 * 1024 ))
STALE=${STALE_S:-900}
n=0
while true; do
  # one Uptime Kuma push monitor PER CAMERA folder: KUMA_PUSH_URL = frontyard (original), KUMA_PUSH_URL_<CAM> others
  # (e.g. KUMA_PUSH_URL_V4), so a working camera can never hide a dead one
  for d in /recordings/*/; do
    cam=$(basename "$d")
    var=KUMA_PUSH_URL_$(echo "$cam" | tr 'a-z-' 'A-Z_')
    url=$(eval echo "\${$var:-}")
    [ "$cam" = frontyard ] && url=${url:-${KUMA_PUSH_URL:-}}
    [ -n "$url" ] || continue
    newest=$(find "$d" -type f -name '*.mkv' -printf '%T@\n' | sort -n | tail -1 | cut -d. -f1)
    cage=$(( $(date +%s) - ${newest:-0} ))
    if [ "$cage" -le "$STALE" ]; then
      wget -q -O /dev/null "${url}?status=up&msg=${cam}+newest+segment+${cage}s+old&ping=" || echo "kuma push failed ($cam)" >&2
    else
      wget -q -O /dev/null "${url}?status=down&msg=${cam}+no+new+segment+for+${cage}s&ping=" || echo "kuma push failed ($cam)" >&2
    fi
  done
  newest=$(find /recordings -type f -name '*.mkv' -printf '%T@\n' | sort -n | tail -1 | cut -d. -f1)
  age=$(( $(date +%s) - ${newest:-0} ))
  if [ $((n % 12)) -eq 0 ]; then
    find /recordings -type f -name '*.mkv' -mtime +"$DAYS" -print -delete | sed 's/^/expired: /'
    used=$(find /recordings -type f -name '*.mkv' -printf '%s\n' | awk '{s+=$1} END {print s+0}')
    if [ "$used" -gt "$MAX" ]; then
      find /recordings -type f -name '*.mkv' -printf '%T@ %s %p\n' | sort -n | while read -r _ size path; do
        [ "$used" -le "$MAX" ] && break
        rm -f "$path" && used=$((used - size)) && echo "over cap, removed: $path"
      done
    fi
    echo "$(date) recordings: $(( used / 1024 / 1024 / 1024 )) GiB (cap ${RETAIN_MAX_GB:-350} GiB, ${DAYS} days); newest segment ${age}s old"
  fi
  n=$((n + 1))
  sleep 300
done

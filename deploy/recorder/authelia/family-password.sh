#!/bin/sh
# Create the 'family' camera login, or set a new password for it. Run on the recorder host in a real terminal:
#   ssh -t db-host 'sudo /srv/camstack/authelia/family-password.sh'
# 'family' can open only https://frigate.cam.example.org (live cameras, recordings, events; Frigate's read-only
# viewer role), with a password and no one-time code.
exec "$(dirname "$0")/user-password.sh" family family "${1:-admin@example.org}"

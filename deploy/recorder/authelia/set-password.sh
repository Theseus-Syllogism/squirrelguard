#!/bin/sh
# Reset the admin camstack SSO account (default operator). Run on the recorder host in a terminal:
#   sudo /srv/camstack/authelia/set-password.sh [username] [email]
# Now a wrapper around user-password.sh, which edits only that account. (The old version rewrote users.yml
# with a single user, which would have deleted every other account, e.g. 'family'.)
exec "$(dirname "$0")/user-password.sh" "${1:-operator}" admins "${2:-admin@example.org}"

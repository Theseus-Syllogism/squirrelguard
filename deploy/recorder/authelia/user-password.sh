#!/bin/sh
# Create or update ONE camstack SSO account, leaving every other account untouched. Run on the recorder host in a real
# terminal (it prompts for the password twice, nothing echoed):
#   sudo /srv/camstack/authelia/user-password.sh USER GROUP [EMAIL]
# GROUP decides access (configuration.yml access_control):
#   admins  every camstack site, password + 2FA
#   family  frigate.cam.example.org only, password only (Frigate gives non-admins its read-only "viewer" role)
#   sg-viewer / sg-investigator / sg-supervisor
#           persons.cam.example.org only, password + 2FA; the role inside it (docs/evidence-export.md)
# Authelia watches users.yml, so the change works at once; existing passkeys/TOTP stay registered.
set -eu
U=${1:?usage: user-password.sh USER GROUP [EMAIL]}
G=${2:?usage: user-password.sh USER GROUP [EMAIL]}
MAIL=${3:-admin@example.org}      # password-reset codes go here
case "$U" in *[!a-z0-9_.-]*|"") echo "user name: lowercase letters, digits, . _ - only" >&2; exit 2 ;; esac
case "$G" in admins|family|sg-viewer|sg-investigator|sg-supervisor) ;;
  *) echo "group must be admins, family, sg-viewer, sg-investigator or sg-supervisor" >&2; exit 2 ;; esac
DIR=/srv/camstack/authelia/users
F=$DIR/users.yml
IMG=authelia/authelia:4.39.28
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
[ -t 0 ] || { echo "needs a terminal (run it yourself, or ssh -t)" >&2; exit 1; }
install -d -m 700 "$DIR"
echo "Password for $U (group $G). Authelia asks twice:"
OUT=$(docker run --rm -it "$IMG" authelia crypto hash generate argon2 | tr -d '\r')
HASH=$(printf '%s\n' "$OUT" | sed -n 's/^Digest: //p')
case "$HASH" in '$argon2id$'*) ;; *) echo "no hash produced; nothing changed" >&2; exit 1 ;; esac
[ -f "$F" ] && cp -p "$F" "$F.bak-$(date +%Y%m%d-%H%M%S)"
umask 077
python3 - "$F" "$U" "$G" "$MAIL" "$HASH" <<'EOF'
import os, re, sys
f, user, group, mail, h = sys.argv[1:]
s = open(f).read() if os.path.exists(f) else "users:\n"
if not re.search(r"(?m)^users:\s*$", s):
    s = s.rstrip("\n") + "\nusers:\n"
block = (f"  {user}:\n    disabled: false\n    displayname: \"{user}\"\n    email: \"{mail}\"\n"
         f"    groups: [{group}]\n    password: '{h}'\n")
pat = re.compile(r"(?ms)^  %s:\n(?:    .*\n?)*" % re.escape(user))
s = pat.sub(lambda m: block, s, count=1) if pat.search(s) else s.rstrip("\n") + "\n" + block
s = s.replace("# Camstack SSO users (written by set-password.sh). Group 'admins' may open every camstack site.",
              "# Camstack SSO users (written by user-password.sh). Group 'admins': every site + 2FA; 'family': Frigate only, password only.")
tmp = f + ".new"
open(tmp, "w").write(s)
os.replace(tmp, f)
EOF
chmod 600 "$F"
echo "password set for $U (group $G); accounts now: $(grep -E '^  [a-z0-9_.-]+:$' "$F" | tr -d ' :' | tr '\n' ' ')"

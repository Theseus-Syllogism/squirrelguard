#!/bin/sh
# Let's Encrypt wildcard certificate for cam.example.org + *.cam.example.org via DNS-01 at Njalla (lego).
# Token: /srv/camstack/certs/lego.env (NJALLA_TOKEN=..., root 600, never in git). Run by camstack-cert.timer (daily);
# lego only renews within 30 days of expiry. nginx is reloaded when the certificate changed.
set -eu
IMG=goacme/lego:v4.35.2
DATA=/srv/camstack/certs/lego
OUT=/etc/nginx/camstack/sso
D=cam.example.org
install -d -m 700 "$DATA"; install -d -m 750 "$OUT"
set -- --accept-tos --path /data --dns njalla --dns.resolvers 1.1.1.1:53 --domains "$D" --domains "*.$D"
if [ -f "$DATA/certificates/$D.crt" ]; then ACTION="renew --days 30"; else ACTION=run; fi
# shellcheck disable=SC2086
docker run --rm --env-file /srv/camstack/certs/lego.env -v "$DATA:/data" "$IMG" "$@" $ACTION
NEW="$DATA/certificates/$D.crt"
if ! cmp -s "$NEW" "$OUT/fullchain.pem" 2>/dev/null; then
    install -m 644 "$NEW" "$OUT/fullchain.pem"
    install -m 600 "$DATA/certificates/$D.key" "$OUT/privkey.pem"
    nginx -t -q && systemctl reload nginx
    echo "certificate installed: $(openssl x509 -in "$OUT/fullchain.pem" -noout -enddate)"
fi

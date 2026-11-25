#!/bin/sh
# Install/refresh the camstack portal + single sign-on on the recorder host (run with sudo from /srv/camstack/portal).
# Sign-in = Authelia (users: /srv/camstack/authelia/set-password.sh). TLS = Let's Encrypt wildcard for
# *.cam.example.org from renew-cert.sh (needs /srv/camstack/certs/lego.env with NJALLA_TOKEN).
set -eu
cd "$(dirname "$0")"
# nginx must start even when the Wi-Fi adapter (198.51.100.20) is not up yet, or every site on this host fails
printf "net.ipv4.ip_nonlocal_bind = 1\n" > /etc/sysctl.d/60-camstack-portal.conf
sysctl -q -p /etc/sysctl.d/60-camstack-portal.conf
install -d -m 755 /srv/camstack/portal/www
install -m 644 index.html /srv/camstack/portal/www/index.html
install -d -m 750 -o root -g www-data /etc/nginx/camstack
if [ ! -f /etc/nginx/camstack/portal.crt ]; then   # only for the redirect on the bare IP
    openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -subj "/CN=camstack 198.51.100.20" \
        -addext "subjectAltName=IP:198.51.100.20" \
        -keyout /etc/nginx/camstack/portal.key -out /etc/nginx/camstack/portal.crt 2>/dev/null
    chmod 640 /etc/nginx/camstack/portal.key; chgrp www-data /etc/nginx/camstack/portal.key
fi
rm -f /etc/nginx/camstack/htpasswd
install -d -m 755 -o 1000 -g 1000 /srv/camstack/portal/snap   # written by the snapshot container (uid 1000)
# Frigate trusts the user headers only with this shared secret (same value in gpuserver's Frigate config: proxy.auth_secret)
if [ ! -f /etc/nginx/camstack/frigate-secret.conf ]; then
    S=$(openssl rand -hex 32)
    printf 'proxy_set_header X-Proxy-Secret "%s";\n' "$S" > /etc/nginx/camstack/frigate-secret.conf
    chmod 640 /etc/nginx/camstack/frigate-secret.conf; chgrp www-data /etc/nginx/camstack/frigate-secret.conf
fi
install -m 644 nginx/sso-tls.conf nginx/sso-authz.conf nginx/sso-check.conf nginx/sso-proxy.conf /etc/nginx/camstack/
install -m 644 camstack-portal.conf.in /etc/nginx/sites-available/camstack-portal
ln -sf /etc/nginx/sites-available/camstack-portal /etc/nginx/sites-enabled/camstack-portal
install -m 755 renew-cert.sh /srv/camstack/portal/renew-cert.sh 2>/dev/null || true
install -m 644 camstack-cert.service camstack-cert.timer portal-status.service portal-status.timer /etc/systemd/system/
systemctl daemon-reload
[ -f /etc/nginx/camstack/sso/fullchain.pem ] || /srv/camstack/portal/renew-cert.sh
install -m 644 camstack-sso.conf /etc/nginx/sites-available/camstack-sso
ln -sf /etc/nginx/sites-available/camstack-sso /etc/nginx/sites-enabled/camstack-sso
systemctl enable --now camstack-cert.timer portal-status.timer
install -m 755 portal-status.py /usr/local/sbin/portal-status
/usr/local/sbin/portal-status
nginx -t && systemctl reload nginx
echo "portal: https://cam.example.org/"

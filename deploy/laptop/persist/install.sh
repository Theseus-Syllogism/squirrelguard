#!/bin/sh
# Run as yourself (not root): it uses sudo only for the two network files.
#   sh ~/src/squirrel-guard/deploy/laptop/persist/install.sh
set -e
D=$(cd "$(dirname "$0")" && pwd)
echo "== 1. Network names (takes effect at next boot / replug)"
sudo mv /etc/udev/rules.d/70-lan-gw-eth1.rules /etc/udev/rules.d/70-lan-gw-eth1.rules.disabled 2>/dev/null || true
sudo install -m 644 "$D/10-lan-gw-usb-eth1.link" "$D/10-onboard-lan0.link" /etc/systemd/network/
sudo update-initramfs -u >/dev/null 2>&1 || true      # .link files are also applied in the initramfs
echo "== 2. Laptop services (survive logout and reboot)"
systemctl --user stop testcam-push scene-worker 2>/dev/null || true
systemctl --user reset-failed testcam-push scene-worker 2>/dev/null || true
mkdir -p ~/.config/systemd/user
install -m 644 "$D/testcam-push.service" "$D/scene-worker.service" ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now testcam-push scene-worker
sudo loginctl enable-linger "$USER"
systemctl --user is-active testcam-push scene-worker
echo "done. Undo: sudo rm /etc/systemd/network/10-*.link; sudo mv /etc/udev/rules.d/70-lan-gw-eth1.rules.disabled /etc/udev/rules.d/70-lan-gw-eth1.rules; systemctl --user disable --now testcam-push scene-worker; sudo loginctl disable-linger $USER"

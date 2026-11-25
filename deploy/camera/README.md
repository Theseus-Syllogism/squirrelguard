# Camera speaker deterrent (Thingino on the Wyze v3, 198.51.100.103)

- `/etc/deter/deter.sh` + `/etc/deter/sounds/*.opus` (sweeps_a, sweeps_b, screech, clatter; 1.5-8 kHz, made by
  `scripts/make_deter_sounds.py`). Request = `<sound|random> [volume]`; anything else is refused.
- Flow: listener (gpuserver) -> MQTT `squirrelguard/<cam>/deter` -> recorder host bridge -> `ssh` with `/srv/camstack/deter/id_ed25519`
  (pinned camera host key in `/srv/camstack/deter/known_hosts`) -> forced command `deter.sh` -> Thingino `play`.
- The key must be authorized on the camera with a forced command (done by the owner, not automatically):
  `command="/etc/deter/deter.sh",no-port-forwarding,no-agent-forwarding,no-X11-forwarding,no-pty ssh-ed25519 ... camstack-deter`
- Measured: screech at volume 50 raised the camera mic RMS from ~280 to ~570-690 (~+7 dB).
- Audible to people and pets (camera speaker cannot do ultrasound). Quiet hours and volume: gpuserver squirrel-guard config.
- Person announcement: `announce/person_announce.opus` (Piper TTS, voice en_US-ryan-high on gpuserver, two-tone chime +
  "Person detected. Video surveillance is recording.", 300 Hz high-pass, loudness-normalised) in `/etc/deter/announce/`.
  Played by name only (never by `random`); listener `[announce]`: once per 5 min, no quiet hours.

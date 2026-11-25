"""Deterrent trigger. Runs a configurable command (play a sound, hit an ESP32 over HTTP, ...) with a cooldown."""
from __future__ import annotations

import logging
import shlex
import subprocess
import time

log = logging.getLogger(__name__)


class Actuator:
    def __init__(self, command: str, cooldown_s: float = 60.0, dry_run: bool = True):
        self.argv = shlex.split(command) if command else []
        self.cooldown_s = cooldown_s
        self.dry_run = dry_run
        self._last = 0.0

    def ready(self) -> bool:
        return time.monotonic() - self._last >= self.cooldown_s

    def fire(self) -> str:
        """Returns what happened: 'fired', 'dry-run', 'cooldown' or 'no-command'."""
        if not self.ready():
            return "cooldown"
        self._last = time.monotonic()
        if not self.argv:
            return "no-command"
        if self.dry_run:
            log.info("DRY RUN: would run %s", shlex.join(self.argv))
            return "dry-run"
        log.info("firing deterrent: %s", shlex.join(self.argv))
        subprocess.Popen(self.argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        return "fired"

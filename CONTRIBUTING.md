# Contributing

Thank you for helping. A few rules keep this project safe to publish and honest about what it can do.

1. **No personal data.** No real camera footage, faces, names, addresses, hostnames, credentials or site imagery in
   code, tests, fixtures, screenshots or issues. Use stock material with a recorded licence, synthetic data, invented
   names, and the documentation address ranges (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24) and `example.org`.
   `python3 scripts/public_check.py .` must report the tree clean.
2. **No model weights in git.** Add a download or export step and record the weight licence in chapter 14 of
   `docs/public/FEATURE-GUIDE.md`. Non-commercial weights must stay optional.
3. **Measure, then claim.** A new or changed feature states what was measured, on what data, and its limits, in the
   guide's format (status, Measured, Limits, Safeguards). Small or synthetic test sets are named as such.
4. **Keep a human in the loop.** Nothing may act on a person automatically, name a person who was not enrolled with
   consent, or look anything up outside the local network.
5. **Tests.** `.venv/bin/python -m pytest tests` passes; add tests with the change.
6. **Licence.** Contributions are under the AGPL-3.0, like the rest of the project.

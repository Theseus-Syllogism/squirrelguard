#!/usr/bin/env python3
"""Fail if a folder contains anything that must not be published.

    python3 scripts/public_check.py DIR [--map .public-map.toml]

Always checked (no configuration needed):
  - private IPv4 addresses (10/8, 172.16/12, 192.168/16, 100.64/10). Loopback and the documentation ranges
    192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24 are allowed;
  - e-mail addresses outside example.org/.com/.net and noreply addresses;
  - private keys, and common token formats (GitHub, OpenAI-style, AWS, Slack, ntfy);
  - files over 10 MB, and model weights (.onnx .pt .pth .safetensors .engine .ckpt .npz).
With --map: also every word listed under [check] markers in that (private, untracked) file.

A line containing the words `public-check: ignore` is skipped (for lines that list ranges on purpose).
Text files are read as UTF-8 with errors ignored; binary files are checked for the markers as bytes too.
Exit code 0 = clean, 1 = something found (each hit is printed as path:line: rule), 2 = usage error.
"""
from __future__ import annotations

import argparse
import ipaddress
import re
import sys
import tomllib
from pathlib import Path

ALLOWED_NETS = [ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24", "127.0.0.0/8")]
PRIVATE_NETS = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10")]  # public-check: ignore
IPV4 = re.compile(r"(?<![\d.])(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?![\d])")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+\.)+[A-Za-z]{2,}")
OK_EMAIL = re.compile(r"@(example\.(org|com|net)|users\.noreply\.github\.com|noreply\.)|^noreply@|^git@", re.I)
SECRETS = [("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
           ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
           ("API key", re.compile(r"\bsk-[A-Za-z0-9_-]{24,}")),
           ("AWS key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
           ("Slack token", re.compile(r"\bxox[abpr]-[A-Za-z0-9-]{10,}")),
           ("ntfy token", re.compile(r"\btk_[a-z0-9]{29}\b"))]
WEIGHTS = {".onnx", ".pt", ".pth", ".safetensors", ".engine", ".ckpt", ".npz"}
MAX_BYTES = 10 * 1024 * 1024
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules"}


def scan(root: Path, markers: list[str]) -> list[str]:
    hits: list[str] = []
    marks = [(m, re.compile(re.escape(m), re.I)) for m in markers]
    for p in sorted(root.rglob("*")):
        if not p.is_file() or any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        rel = p.relative_to(root).as_posix()
        for m, rx in marks:
            if rx.search(rel):
                hits.append(f"{rel}: file name contains a private marker")
        size = p.stat().st_size
        if size > MAX_BYTES:
            hits.append(f"{rel}: {size / 1e6:.1f} MB file")
        if p.suffix.lower() in WEIGHTS:
            hits.append(f"{rel}: model weights or arrays are not published")
        data = p.read_bytes()
        if b"\0" in data[:8192]:                                   # binary: markers only
            low = data.lower()
            for m, _ in marks:
                if m.lower().encode() in low:
                    hits.append(f"{rel}: binary file contains a private marker")
            continue
        for n, line in enumerate(data.decode("utf-8", "ignore").splitlines(), 1):
            if "public-check: ignore" in line:                     # a line that names the ranges on purpose
                continue
            for ip in IPV4.findall(line):
                try:
                    a = ipaddress.ip_address(ip)
                except ValueError:
                    continue
                if any(a in net for net in PRIVATE_NETS) and not any(a in net for net in ALLOWED_NETS):
                    hits.append(f"{rel}:{n}: private address {ip}")
            for mo in EMAIL.finditer(line):
                if p.name.upper().startswith(("LICENSE", "COPYING")):   # upstream licence texts name their authors
                    break
                if not OK_EMAIL.search(mo.group(0)) and not re.search(r"\.(png|jpg|svg|webp)$", mo.group(0)):
                    hits.append(f"{rel}:{n}: e-mail address {mo.group(0)}")
            for name, rx in SECRETS:
                if rx.search(line):
                    hits.append(f"{rel}:{n}: {name}")
            for m, rx in marks:
                if rx.search(line):
                    hits.append(f"{rel}:{n}: private marker")      # the marker itself is not printed
    return hits


def markers_from(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    return list(tomllib.loads(path.read_text()).get("check", {}).get("markers", []))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--map", default=None, help="private mapping file with [check] markers")
    a = ap.parse_args()
    root = Path(a.dir)
    if not root.is_dir():
        print(f"not a folder: {root}", file=sys.stderr)
        return 2
    hits = scan(root, markers_from(Path(a.map) if a.map else None))
    for h in hits:
        print(h)
    print(f"public_check: {'clean' if not hits else f'{len(hits)} problem(s)'} in {root}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())

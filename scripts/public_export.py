#!/usr/bin/env python3
"""Make the public copy of this repository: tracked files only, private values replaced, private files left out.

    python3 scripts/public_export.py OUT [--map .public-map.toml] [--git] [--force]

The private repository (with its history) stays as it is. The mapping file is private and untracked; it lists:
  [[replace]]     regex + with: applied in order to every text file's content and to every path
  [[drop_lines]]  files + regex: lines removed from one file (after nothing else matched it)
  [exclude]       paths: glob patterns (of the private paths) that are not copied
  [check]         markers: words that must not survive (scripts/public_check.py)
  [export]        author: the author of the single commit made with --git
OUT must not exist (or pass --force to replace it). The copy is checked with public_check.py; on any problem the
command fails, and --git does not run. --git makes OUT a new repository with one commit and no history.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import public_check  # noqa: E402


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p and (ROOT / p).is_file()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--map", default=str(ROOT / ".public-map.toml"))
    ap.add_argument("--git", action="store_true", help="make OUT a new git repository with one commit")
    ap.add_argument("--force", action="store_true", help="replace OUT if it exists")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.map).read_text())
    rules = [(re.compile(r["regex"]), r["with"]) for r in cfg.get("replace", [])]
    drops = [(d["files"], re.compile(d["regex"])) for d in cfg.get("drop_lines", [])]
    excl = cfg.get("exclude", {}).get("paths", [])
    out = Path(a.out).resolve()
    if out == ROOT or ROOT in out.parents:
        sys.exit("OUT must be outside the repository")
    if out.exists():
        if not a.force:
            sys.exit(f"{out} exists (use --force to replace it)")
        shutil.rmtree(out)

    def sub(s: str) -> str:
        for rx, w in rules:
            s = rx.sub(w, s)
        return s

    copied = skipped = changed = 0
    for rel in tracked():
        if any(fnmatch.fnmatch(rel, g) for g in excl):
            skipped += 1
            continue
        dst = out / sub(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = (ROOT / rel).read_bytes()
        if b"\0" not in data[:8192]:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = None
            if text is not None:
                for pattern, rx in drops:
                    if fnmatch.fnmatch(rel, pattern):
                        text = "".join(line for line in text.splitlines(keepends=True) if not rx.search(line))
                new = sub(text)
                changed += new != data.decode("utf-8")
                data = new.encode("utf-8")
        dst.write_bytes(data)
        shutil.copymode(ROOT / rel, dst)
        copied += 1
    print(f"public_export: {copied} files copied ({changed} changed), {skipped} left out -> {out}")

    hits = public_check.scan(out, public_check.markers_from(Path(a.map)))
    for h in hits:
        print(h)
    if hits:
        print(f"public_export: {len(hits)} problem(s) left; not ready to publish")
        return 1
    print("public_export: check clean")
    if a.git:
        author = cfg.get("export", {}).get("author", "squirrel-guard contributors <noreply@example.org>")
        name, email = re.match(r"(.*?)\s*<(.*)>", author).groups()
        env = {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email, "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email}
        run = lambda *c: subprocess.run(["git", *c], cwd=out, check=True, capture_output=True, env={**env, "PATH": "/usr/bin:/bin"})  # noqa: E731
        run("init", "-q", "-b", "main")
        run("add", "-A")
        run("commit", "-q", "-m", "squirrel-guard: first public release")
        print(f"public_export: new repository with one commit by {author}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

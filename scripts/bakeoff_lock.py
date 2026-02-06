#!/usr/bin/env python3
"""Per-target lock helper for bakeoff runs.

KISS: no state machine. Just a lock file + stale detection.

Lock location (in bakeoff repo):
  locks/<slug>.lock

Lock contents are informational JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOCKS = ROOT / "locks"


def _set_locks_dir(p: str | None) -> None:
    global LOCKS
    if p:
        LOCKS = Path(p).expanduser().resolve()


def slug_for_repo(repo: str) -> str:
    p = Path(repo).expanduser().resolve()
    h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:10]
    # keep some human readability
    name = p.name.replace(" ", "-")
    return f"{name}-{h}"


def lock_path(repo: str) -> Path:
    return LOCKS / f"{slug_for_repo(repo)}.lock"


def cmd_acquire(args: argparse.Namespace) -> int:
    _set_locks_dir(getattr(args, "locks_dir", None))
    LOCKS.mkdir(parents=True, exist_ok=True)
    lp = lock_path(args.repo)
    now = time.time()

    if lp.exists():
        age = now - lp.stat().st_mtime
        if age < args.stale_seconds:
            print(f"LOCKED {lp} age={int(age)}s")
            return 2
        # stale
        if args.force:
            lp.unlink()
        else:
            print(f"STALE_LOCK {lp} age={int(age)}s (use --force to remove)")
            return 3

    payload = {
        "repo": str(Path(args.repo).expanduser().resolve()),
        "createdAt": int(now),
        "pid": os.getpid(),
        "note": args.note or "",
    }
    lp.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"ACQUIRED {lp}")
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    _set_locks_dir(getattr(args, "locks_dir", None))
    lp = lock_path(args.repo)
    if lp.exists():
        lp.unlink()
        print(f"RELEASED {lp}")
        return 0
    print(f"NO_LOCK {lp}")
    return 1


def cmd_status(args: argparse.Namespace) -> int:
    _set_locks_dir(getattr(args, "locks_dir", None))
    lp = lock_path(args.repo)
    if not lp.exists():
        print("UNLOCKED")
        return 0
    try:
        data = json.loads(lp.read_text())
    except Exception:
        data = None
    age = int(time.time() - lp.stat().st_mtime)
    print(json.dumps({"lock": str(lp), "ageSeconds": age, "data": data}, indent=2))
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("acquire")
    a.add_argument("--repo", required=True)
    a.add_argument("--locks-dir")
    a.add_argument("--stale-seconds", type=int, default=45 * 60)
    a.add_argument("--note")
    a.add_argument("--force", action="store_true")
    a.set_defaults(fn=cmd_acquire)

    r = sub.add_parser("release")
    r.add_argument("--repo", required=True)
    r.add_argument("--locks-dir")
    r.set_defaults(fn=cmd_release)

    s = sub.add_parser("status")
    s.add_argument("--repo", required=True)
    s.add_argument("--locks-dir")
    s.set_defaults(fn=cmd_status)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

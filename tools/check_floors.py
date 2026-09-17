"""Fail if requirements.txt admits a torch or transformers version with known advisories.

The floors are the lowest versions pip-audit reported as fixing every advisory
against these packages on 2026-09-16 (torch.load RCE class, transformers
deserialization and ReDoS). Raise them when new advisories land.
"""

import re
import sys
from pathlib import Path

MIN_SAFE = {"torch": (2, 13, 0), "transformers": (5, 10, 0)}


def parse(version):
    return tuple(int(p) for p in version.split("."))


def main():
    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    found = {}
    for line in req.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Za-z0-9_.-]+)\s*>=\s*([0-9.]+)", line)
        if m:
            found[m.group(1).lower()] = parse(m.group(2))
    errors = []
    for name, floor in MIN_SAFE.items():
        have = found.get(name)
        if have is None:
            errors.append(f"{name}: no '>=' floor in requirements.txt")
        elif have < floor:
            errors.append(f"{name}: floor {'.'.join(map(str, have))} is below {'.'.join(map(str, floor))}")
    for e in errors:
        print("FAIL", e)
    if errors:
        return 1
    print("OK: requirement floors are at or above the known-safe versions")
    return 0


if __name__ == "__main__":
    sys.exit(main())

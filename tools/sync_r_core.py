"""Copy the C++ core (src/*.cpp, src/*.h) into the R package (R-package/src/).

CRAN and R-universe build the R package from R-package/ alone, so the core must
be committed there as well.  Run this after any change to src/:

    python tools/sync_r_core.py           # copy
    python tools/sync_r_core.py --check   # exit 1 if the copy is out of date (CI, tests)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src"
DEST = ROOT / "R-package" / "src"
OWN = {"bindings.cpp", "RcppExports.cpp", "communityhierarchy_types.h"}  # the R package's own


def core_files() -> list[Path]:
    return sorted(p for p in CORE.iterdir() if p.suffix in (".cpp", ".h"))


def main() -> int:
    check = "--check" in sys.argv[1:]
    want = {p.name: p.read_bytes() for p in core_files()}
    have = {
        p.name: p.read_bytes()
        for p in DEST.iterdir()
        if p.suffix in (".cpp", ".h") and p.name not in OWN
    }
    stale = sorted(n for n in want if have.get(n) != want[n])
    extra = sorted(set(have) - set(want))
    if check:
        for n in stale:
            print(f"out of date: R-package/src/{n}")
        for n in extra:
            print(f"not in src/: R-package/src/{n}")
        if stale or extra:
            print("run: python tools/sync_r_core.py")
            return 1
        print("R-package/src is in sync with src/")
        return 0
    for n in stale:
        (DEST / n).write_bytes(want[n])
    for n in extra:
        (DEST / n).unlink()
    print(f"copied {len(stale)}, removed {len(extra)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

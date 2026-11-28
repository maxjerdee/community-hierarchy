"""The R package (R-package/) carries a copy of the C++ core, which must match src/."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_r_package_core_matches_src():
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "sync_r_core.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr

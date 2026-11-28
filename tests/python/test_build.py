"""The build: the installed version, imports, and that every compiled copy of the core is current."""

from __future__ import annotations

import importlib.metadata
import pathlib
import subprocess
import sys
from pathlib import Path

import community_hierarchy as ch
import community_hierarchy.MCMC_core as native
import pytest


def test_version():
    assert importlib.metadata.version("community-hierarchy") == ch.__version__


# asbm and community_hierarchy each bind a bare (un-namespaced) MCMC and
# FitResultNative.  Their RTTI names collide, and pybind11's type registry is
# process-wide, so importing both used to raise
#   ImportError: generic_type: type "MCMC" is already registered!
# py::module_local() on both classes keeps each registration inside its own
# extension module.  Either package can be imported first, so test both orders.
@pytest.mark.parametrize("first", ["asbm", "community_hierarchy"])
def test_imports_alongside_community_hierarchy(first):
    pytest.importorskip("asbm")
    second = "community_hierarchy" if first == "asbm" else "asbm"
    code = (
        f"import {first}, {second}; "
        "import asbm.core as ac, community_hierarchy.fitting as dc; "
        "assert ac.fit_backend is not None, 'asbm native backend failed to load'; "
        "assert dc._fit_backend is not None, 'community_hierarchy native backend failed to load'; "
        "import asbm._asbm, community_hierarchy.MCMC_core as dm; "
        "assert asbm._asbm.MCMC is not dm.MCMC"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr


REPO = pathlib.Path(__file__).resolve().parents[2]


# Each artifact is compared against the sources IT is built from: the Python
# extension does not rebuild from bindings/wasm, and the wasm bundle does not
# rebuild from bindings/python.  Comparing against everything gives false
# positives the moment you touch a binding you are not testing.
EXTENSION_SOURCES = [REPO / "src", REPO / "bindings" / "python"]


WASM_SOURCES = [REPO / "src", REPO / "bindings" / "wasm" / "src"]


SOURCE_DIRS = EXTENSION_SOURCES


WASM_DIST = REPO / "bindings" / "wasm" / "js" / "dist" / "ds_core.mjs"


REBUILD = "python -m pip install -e . --no-build-isolation --no-deps"


def _newest_source(dirs=None) -> tuple[pathlib.Path, float]:
    newest, newest_mtime = None, 0.0
    for d in dirs or EXTENSION_SOURCES:
        if not d.is_dir():
            continue
        for path in d.rglob("*"):
            if path.suffix.lower() not in {".cpp", ".h", ".hpp"}:
                continue
            mtime = path.stat().st_mtime
            if mtime > newest_mtime:
                newest, newest_mtime = path, mtime
    return newest, newest_mtime


def _skip_unless_source_checkout() -> None:
    if not all(d.is_dir() for d in SOURCE_DIRS):
        pytest.skip("not a source checkout: nothing to compare the build against")


def _fmt(t: float) -> str:
    import datetime

    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def test_extension_is_not_older_than_the_sources():
    """The loaded extension must post-date every .cpp/.h it is built from."""
    _skip_unless_source_checkout()

    newest, newest_mtime = _newest_source()
    if newest is None:
        pytest.skip("no C++ sources found")
    built = pathlib.Path(native.__file__)
    built_mtime = built.stat().st_mtime

    # One minute of slack: copying an editable install's .so into site-packages
    # is not atomic with the compile, and filesystems differ in timestamp
    # granularity (drvfs under WSL is coarse).
    assert built_mtime + 60 >= newest_mtime, (
        f"the extension you are importing is OLDER than the sources.\n"
        f"  extension: {built}\n             built {_fmt(built_mtime)}\n"
        f"  newest source: {newest.relative_to(REPO)}\n"
        f"             changed {_fmt(newest_mtime)}\n"
        f"Results will not reflect your edits. Rebuild and reinstall:\n"
        f"  {REBUILD}"
    )


@pytest.mark.skipif(not WASM_DIST.exists(), reason="wasm bundle not built")
def test_wasm_bundle_is_not_older_than_the_sources():
    """A stale wasm bundle silently passes any test that only checks DLs.

    The consensus partition, for instance, can change with no change to the
    description length, so a DL-only comparison will not notice.
    """
    _skip_unless_source_checkout()
    newest, newest_mtime = _newest_source(WASM_SOURCES)
    if newest is None:
        pytest.skip("no C++ sources found")
    built_mtime = WASM_DIST.stat().st_mtime
    assert built_mtime + 60 >= newest_mtime, (
        f"the wasm bundle is OLDER than the sources.\n"
        f"  bundle: {WASM_DIST.relative_to(REPO)}\n          built {_fmt(built_mtime)}\n"
        f"  newest source: {newest.relative_to(REPO)}\n"
        f"             changed {_fmt(newest_mtime)}\n"
        f"Anything comparing it with the native build is comparing two different\n"
        f"versions of the code. Rebuild:\n"
        f"  emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release\n"
        f"  cmake --build build-wasm -j"
    )


ROOT = Path(__file__).resolve().parents[2]


def test_r_package_core_matches_src():
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "sync_r_core.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr

"""Fail loudly when a built artifact is older than the sources it was built from.

A stale build is indistinguishable from a real numerical disagreement: the
editable install serves the extension from a COPY under site-packages, so
`cmake --build` updates the source tree and `import directedstructure` keeps loading the old
binary. That cost two sessions an afternoon of chasing a "parity failure" that
was only a stale .so, so it is now a test.

These skip outside a source checkout (an installed wheel has no src/ to compare
against), and they never fail for a reason other than staleness.
"""

from __future__ import annotations

import os
import pathlib

import pytest

import directedstructure

REPO = pathlib.Path(__file__).resolve().parents[2]
SOURCE_DIRS = [REPO / "src", REPO / "bindings" / "python"]
WASM_DIST = REPO / "bindings" / "wasm" / "js" / "dist" / "ds_core.mjs"
REBUILD = "python -m pip install -e . --no-build-isolation --no-deps"


def _newest_source() -> tuple[pathlib.Path, float]:
    newest, newest_mtime = None, 0.0
    for d in SOURCE_DIRS:
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
    import directedstructure.MCMC_core as native

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
    newest, newest_mtime = _newest_source()
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

from __future__ import annotations

import importlib.metadata

import directedstrcture as m


def test_version():
    assert importlib.metadata.version("directedstrcture") == m.__version__

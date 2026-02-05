from __future__ import annotations

import importlib.metadata

import directedstructure as m


def test_version():
    assert importlib.metadata.version("directedstructure") == m.__version__

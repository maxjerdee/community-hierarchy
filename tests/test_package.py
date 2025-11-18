from __future__ import annotations

import importlib.metadata

import community_hierarchy as m


def test_version():
    assert importlib.metadata.version("community_hierarchy") == m.__version__

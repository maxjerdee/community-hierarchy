"""
Copyright (c) 2025 Max Jerdee. All rights reserved.

directedstructure: Infer communities, hierarchies, and their connection in directed graphs
"""

from __future__ import annotations

from ._version import version as __version__

# first party imports
from directedstructure.core import samples, node_properties, network_properties

from ._version import version as __version__

__all__ = [
    "__version__",
    "node_properties",
    "network_properties",
    "samples",
]

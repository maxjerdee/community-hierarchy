"""
Copyright (c) 2025 Max Jerdee. All rights reserved.

directedstructure: Infer communities, hierarchies, and their connection in directed graphs
"""

from __future__ import annotations

from ._version import version as __version__

# first party imports
from directedstructure.core import samples, node_properties, network_properties
from directedstructure.input_output import plot_node_properties, plot_MCMC_entropy, write_gml_with_inferences

from ._version import version as __version__

__all__ = [
    "__version__",
    "node_properties",
    "network_properties",
    "plot_node_properties",
    "plot_MCMC_entropy",
    "write_gml_with_inferences",
    "samples",
]

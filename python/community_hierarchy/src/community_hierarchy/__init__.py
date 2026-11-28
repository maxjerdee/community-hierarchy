"""
Copyright (c) 2025 Maximilian Jerdee. Released under the MIT License (see LICENSE).

community_hierarchy: Infer communities, hierarchies, and their connection in directed graphs
"""

from __future__ import annotations

from ._version import version as __version__
from .config import Config
from .fitting import fit
from .gml import write_gml_with_inferences
from .plots import plot_adjacency_matrix, plot_MCMC_entropy, plot_node_properties
from .result import Result

__all__ = [
    "__version__",
    "Config",
    "Result",
    "fit",
    "plot_node_properties",
    "plot_adjacency_matrix",
    "plot_MCMC_entropy",
    "write_gml_with_inferences",
]

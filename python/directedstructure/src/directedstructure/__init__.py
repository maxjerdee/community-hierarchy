"""
Copyright (c) 2025 Max Jerdee. All rights reserved.

directedstructure: Infer communities, hierarchies, and their connection in directed graphs
"""

from __future__ import annotations

from ._version import version as __version__
from .core import Config, Result, fit
from .input_output import plot_node_properties, plot_MCMC_entropy, write_gml_with_inferences

__all__ = [
    "__version__",
    "Config",
    "Result",
    "fit",
    "plot_node_properties",
    "plot_MCMC_entropy",
    "write_gml_with_inferences",
]

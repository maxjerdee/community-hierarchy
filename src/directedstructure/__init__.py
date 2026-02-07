"""
Copyright (c) 2025 Max Jerdee. All rights reserved.

directedstructure: Infer communities, hierarchies, and their connection in directed graphs
"""

from __future__ import annotations

from ._version import version as __version__

# first party imports
from directedstructure.core import samples, node_properties, network_properties
from directedstructure.input_output import plot_groups, write_gml, get_neighbors
from directedstructure.clusterings import consensus_clustering
from directedstructure.prediction import expected_edges, anomalies, link_predictions
from .MCMC_core import MCMC

from ._version import version as __version__

__all__ = [
    "__version__",
    "node_groups",
    "node_properties",
    "network_properties",
    "consensus_clustering",
    "plot_groups",
    "write_gml",
    "mixing_matrix",
    "parameters",
    "samples",
    "expected_edges",
    "anomalies",
    "link_predictions",
    "MCMC",
    "get_neighbors",
]

"""
# Example API from the README:

# Find network communities (uses the default canonical model with generalized degree correction)
node_groups_df = ng.node_groups(G)

# Draw the groups on the network
ng.plot_groups(G, node_groups_df)

# Write the GML with a new "group" attribute so external tools (e.g., Gephi) can color by group
ng.write_gml(G, node_groups_df)

# Compute the probabilities of connecting between each pair of inferred groups
mixing_matrix = ng.mixing_matrix(G, node_groups_df)

# We can also measure other SBM parameters
parameters_df = ng.parameters(G)

# Get the full posterior samples of the node group assignments and other SBM parameters
samples_df = ng.samples(G)

posterior_predictives_df = ng.posterior_predictives(G)
```

Which can be used in order to rank the edges of the network according to how unusual they are (anomaly detection)
```python 
anomalies_df = ng.anomalies(G)
```

Or to predict which absences of edges are most suprising (link prediction)
```python
link_predicitions_df = ng.link_predictions(G)

"""

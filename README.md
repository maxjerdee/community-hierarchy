# directedstructure

[![Documentation Status][rtd-badge]][rtd-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/maxjerdee/directedstructure/workflows/CI/badge.svg
[actions-link]:             https://github.com/maxjerdee/directedstructure/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/directedstructure
[conda-link]:               https://github.com/conda-forge/directedstructure-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/maxjerdee/directedstructure/discussions
[pypi-link]:                https://pypi.org/project/directedstructure/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/directedstructure
[pypi-version]:             https://img.shields.io/pypi/v/directedstructure
[rtd-badge]:                https://readthedocs.org/projects/directedstructure/badge/?version=latest
[rtd-link]:                 https://directedstructure.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

### Infer communities, hierarchies, and their connection in directed graphs

##### Maximilian Jerdee, Elizabeth Bruch, Mark Newman

This python package uses Bayesian inference to identify communities and hierarchies of nodes in a directed network, as well as measure the interaction between those structures. 

We model community structure using a stochastic block model, hierarchy structure with a Bradley-Terry model, and their interaction according to our work here [link paper].

## Installation

`directedstructure` can be built locally by cloning this repository and running

```bash
pip install .
```

in the base directory (requires a C++ compiler).

## Typical usage
Once installed, the package can be used to identify the network structures.

### Load a network
We recommend using [NetworkX](https://networkx.org/) to load the network and then using the `directedstructure` package to infer the node grouping and hierarchy.


```python
import directedstructure as ds
import networkx as nx
import pandas as pd

# Load a network using NetworkX (this can also be read from an edgelist or other format)
G = nx.read_gml("examples/data/friends.gml")
```

### Infer node properties
```python
node_properties_df = ds.node_properties(G) # pandas DataFrame of inferred community identity and hierarchical position of each node in the network

```

### Infer network properties
```python
network_properties_df = ds.network_properties(G) # pandas DataFrame of network properties (for example depth of hierarchy within and between communities) inferred by the model, as well as their uncertainities

```



### Full samples
To get a more complete picture of the inference, we can consider the full posterior distribution of Monte Carlo samples
```python
samples_df = ds.samples(G)
```

With these samples we can ask more detailed questions like what is the posterior distribution of possible numbers of groups?


### Customization
As this package focuses on the potential link between community and hierarchy, we can swap out the models considered of either community or hierarchy in isolation. 

Neutral interactions can also be considered within the model if further interaction type information is provided (either type = dominant or type = neutral). If no type information is provided all interactions will be assumed to be dominant. 


Further usage examples can be found in the `examples` directory of the
repository and the [package documentation][rtd-link].

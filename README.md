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

This Python package uses Bayesian inference to identify communities and hierarchies of nodes in a directed network, as well as measure the interaction between those structures.

We model community structure using a stochastic block model, hierarchy structure with a Bradley-Terry model, and their interaction according to our work (link forthcoming).

## Installation

```bash
pip install directedstructure
```

Or build locally by cloning this repository and running

```bash
pip install .
```

in the base directory (requires a C++ compiler).

## Usage

The public API has three objects: `Config`, `fit`, and `Result`.

### Load a network

```python
import directedstructure as ds
import networkx as nx

G = nx.read_gml("examples/data/networks/friends.gml")
```

Edges may carry a `type` attribute (`'dominant'` or `'neutral'`) to distinguish directed pairwise comparisons from symmetric ties. If no type information is provided all edges are treated as dominant.

### Configure the model

```python
config = ds.Config(
    groups_model="general_canonical",   # community structure preset
    hierarchy_model="bradley_terry_ties",  # auto-selected if omitted
    interaction="coupled",              # coupling between communities and hierarchy
)
```

`Config` is a frozen dataclass — see the API docs for the full list of parameters (mixing variation, degree correction, individual/group depth, etc.).

### Run inference

```python
result = ds.fit(
    config, G,
    num_samples=1000,       # posterior samples to collect
    sweeps_per_sample=10,   # MCMC sweeps between samples
    timeout=60.0,           # wall-clock limit in seconds
    seed=42,
)
```

### Inspect results

```python
# A concise summary is printed by the result itself
print(result)
# directedstructure.Result(n=62, samples=1000, mean_groups=3.2, mdl=412.100)

# Per-node hierarchy scores: posterior mean and uncertainty
scores = result.score_means    # np.ndarray, shape [n_nodes]
uncertainty = result.score_stds  # np.ndarray, shape [n_nodes]

# Network-level parameter summary: mean and std of each inferred parameter
net_df = result.network_properties()

# Integer community labels for the consensus sample
partition = result.consensus_partition()

# n×n co-assignment frequency matrix
comatrix = result.coincidence_matrix()

# Full posterior samples DataFrame
samples_df = result.samples_df()
```

### Visualization

```python
import matplotlib.pyplot as plt

fig, ax = ds.plot_node_properties(result)
plt.savefig("node_scores.png")

fig, ax = ds.plot_MCMC_entropy(result)   # convergence diagnostic
```

Export the graph with inferred attributes for use in Gephi or other tools:

```python
ds.write_gml_with_inferences(G, "output.gml", result)
```

### Fixing model components

Pass fixed values in `Config` to test submodels. For example, to fix group structure from node attributes and infer hierarchy only:

```python
config = ds.Config(group_attribute="department", interaction="independent")
result = ds.fit(config, G)
```

Or fix the number of groups:

```python
config = ds.Config(num_groups=4)
result = ds.fit(config, G)
```

### Parallel tempering

For complex posteriors, enable parallel tempering:

```python
result = ds.fit(config, G, num_tempering_chains=4, beta=1.0)
```

Further usage examples can be found in the `examples/python/` directory of the
repository and the [package documentation][rtd-link].

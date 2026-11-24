# community-hierarchy

[![Documentation Status][rtd-badge]][rtd-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/maxjerdee/community-hierarchy/workflows/CI/badge.svg
[actions-link]:             https://github.com/maxjerdee/community-hierarchy/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/community-hierarchy
[conda-link]:               https://github.com/conda-forge/community-hierarchy-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/maxjerdee/community-hierarchy/discussions
[pypi-link]:                https://pypi.org/project/community-hierarchy/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/community-hierarchy
[pypi-version]:             https://img.shields.io/pypi/v/community-hierarchy
[rtd-badge]:                https://readthedocs.org/projects/community-hierarchy/badge/?version=latest
[rtd-link]:                 https://community-hierarchy.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

### Infer communities, hierarchies, and their connection in directed graphs

##### Maximilian Jerdee, Elizabeth Bruch, Mark Newman

This Python package uses Bayesian inference to identify communities and hierarchies of nodes in a directed network, as well as measure the interaction between those structures.

We model community structure using a stochastic block model, hierarchy structure with a Bradley-Terry model, and their interaction according to our work (link forthcoming).

## Installation

```bash
pip install community-hierarchy
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
import community_hierarchy as ch
import networkx as nx

G = nx.read_gml("examples/data/networks/friends.gml")
```

Edges may carry a `type` attribute (`'dominant'` or `'neutral'`) to distinguish directed pairwise comparisons from symmetric ties. If no type information is provided all edges are treated as dominant.

### Configure the model

```python
config = ch.Config(
    groups_model="general_ASBM",        # community structure preset
    hierarchy_model="bradley_terry_ties",  # auto-selected if omitted
    interaction="coupled",              # coupling between communities and hierarchy
)
```

`Config` is a frozen dataclass — see the API docs for the full list of parameters (mixing variation, degree correction, individual/group depth, etc.).

### Run inference

```python
result = ch.fit(
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
# community_hierarchy.Result(n=62, samples=1000, mean_groups=3.2, mdl=412.100)

# Network-level posterior summary: columns parameter, mean, std, hdi_lo, hdi_hi
# Includes ICC (intraclass correlation) and sigma_T.
# Computed in C++ without materialising the full samples DataFrame.
npdf = result.network_properties_df()          # default 90% HDI
npdf90 = result.network_properties_df(prob=0.90)

icc_row = npdf[npdf["parameter"] == "ICC"].iloc[0]
print(f"ICC = {icc_row['mean']:.3f}  90% HDI [{icc_row['hdi_lo']:.3f}, {icc_row['hdi_hi']:.3f}]")

# Per-node posterior summary: columns node, score_mean, score_std, group, group_score_mean
# For n ≤ 3000 uses the consensus partition (posterior summary);
# for larger networks falls back to the MDL partition with a warning.
node_df = result.node_properties_df()                    # auto
node_df = result.node_properties_df(partition="consensus")  # always consensus
node_df = result.node_properties_df(partition="mdl")        # always MDL (free)

print(node_df.sort_values("score_mean", ascending=False).head())

# Minimum description length value (model comparison)
print(result.mdl_value)

# n×n co-assignment frequency matrix (posterior)
comatrix = result.coincidence_matrix()

# Full posterior samples DataFrame (all parameters + per-node groups/scores)
samples_df = result.samples_df()

# Posterior-predictive probability of each directed outcome between two nodes.
# Returns {"a_to_b": ..., "tie": ..., "b_to_a": ...}, values sum to 1.
p = result.direction_probabilities("Spain", "Morocco")

# Set ties_allowed=False for standard Bradley-Terry (no tie term).
p = result.direction_probabilities("Spain", "Morocco", ties_allowed=False)
# {"a_to_b": ..., "b_to_a": ...}
```

### Visualization

```python
import matplotlib.pyplot as plt

# Node scores ranked within each inferred community (±1 std error bars,
# group mean line, communities labeled G1/G2/…)
fig, ax = ch.plot_node_properties(result)
plt.savefig("node_scores.png")

# Adjacency matrix of G reordered by inferred community, with colored
# community strips on the top and left edges.
# G can be any graph — it does not have to be the graph used in fit().
fig, ax = ch.plot_adjacency_matrix(result, G, cmap="Greens", title="Network")
plt.savefig("adjacency.png")

# Compose both panels side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ch.plot_node_properties(result, ax=axes[0])
ch.plot_adjacency_matrix(result, G, ax=axes[1])
fig.tight_layout()
plt.savefig("combined.png")

fig, ax = ch.plot_MCMC_entropy(result)   # convergence diagnostic
```

Export the graph with inferred attributes for use in Gephi or other tools:

```python
ch.write_gml_with_inferences(G, "output.gml", result)
```

### Fixing model components

Pass fixed values in `Config` to test submodels. For example, to fix group structure from node attributes and infer hierarchy only:

```python
config = ch.Config(group_attribute="department", interaction="independent")
result = ch.fit(config, G)
```

Or fix the number of groups:

```python
config = ch.Config(num_groups=4)
result = ch.fit(config, G)
```

### Parallel tempering

For complex posteriors, enable parallel tempering:

```python
result = ch.fit(config, G, num_tempering_chains=4, beta=1.0)
```

Further usage examples can be found in the `examples/python/` directory of the
repository and the [package documentation][rtd-link].

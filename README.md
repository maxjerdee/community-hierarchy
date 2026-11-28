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

**Infer communities, hierarchies, and their connection in directed graphs**

*Maximilian Jerdee, Elizabeth Bruch, Mark Newman*

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

The public API is three objects: `Config` (the model), `fit` (the sampler) and
`Result` (samples and summaries).  The [documentation][rtd-link] describes the model,
its conventions and every parameter.

### Load a network

```python
import community_hierarchy as ch
import networkx as nx

# International football 2023-2026, in this repository: one edge per match,
# winner -> loser (type="dominant") or a draw (type="neutral").
G = nx.read_gml("examples/paper/international_football/data/football_group_stage.gml",
                label="label")
```

- An edge `u -> v` means **u beat v**; pass `source_high_rank=False` to `fit` when the
  target is the winner (a citation, a friendship nomination).
- `type="neutral"` marks a draw (direction ignored).  If any edge has a `type`, all must.
- An edge's `weight` is its number of interactions (a positive integer), or a positive
  real strength with `Config(weighted_SBM=True)`.  Self-loops are not allowed.

### Fit

```python
result = ch.fit(
    ch.Config(),            # coupled model, degree-corrected SBM, ties modelled
    G,
    num_chains=4,           # independent chains, run in parallel
    samples_per_chain=1000,
    burn_sweeps=1000,
    seed=1,
    timeout=600,            # seconds; returns the samples collected so far
)
print(result)
```

### Summaries

```python
npdf = result.network_properties_df()        # mean, std, 90% HDI of every parameter
print(npdf.set_index("parameter").loc[["ICC", "sigma_T"]])

nodes = result.node_properties_df()          # per node: score_mean, score_std, group
print(nodes.sort_values("score_mean", ascending=False).head())

p = result.direction_probabilities("Spain", "Morocco")
# {"a_to_b": P(Spain beats Morocco), "tie": ..., "b_to_a": ...}

samples = result.samples_df()                # every sample, with chain_id
```

`ICC` is the fraction of the variance in scores explained by the groups, and
`sigma_T` the total spread of scores.  All arrays and tables follow the order of
`G.nodes()`; the groups are the maximum-overlap consensus partition.

### Plots and export

```python
fig, ax = ch.plot_node_properties(result)          # scores within each group
fig, ax = ch.plot_adjacency_matrix(result, G)      # adjacency matrix ordered by group
fig, ax = ch.plot_MCMC_entropy(result)             # description length, one line per chain
ch.write_gml_with_inferences(G, "football_inferred.gml", result)
```

### Model options

```python
ch.Config(interaction="independent")       # no group term in the scores
ch.Config(groups_model="traditional_SBM")  # simpler mixing structure (see the docs)
ch.Config(degree_correction=False)
ch.Config(group_attribute="confederation") # groups fixed to a node attribute
ch.Config(num_groups=4)                    # fixed number of groups
ch.Config(weighted_SBM=True)               # real-valued edge weights

ch.fit(config, G, group_graph=H)           # communities from a second, undirected graph
ch.fit(config, G, num_tempering_chains=4)  # parallel tempering
```

Setting a value that the chosen model fixes (for example `group_depth` with
`interaction="independent"`) raises a `ValueError`.

A JavaScript/WebAssembly build of the sampler is described in
[`bindings/wasm/README.md`](https://github.com/maxjerdee/community-hierarchy/blob/main/bindings/wasm/README.md).  The scripts that reproduce the
paper's football and journal examples are in [`examples/paper/`](https://github.com/maxjerdee/community-hierarchy/tree/main/examples/paper).

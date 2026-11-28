# The model

`community_hierarchy` fits a joint model of community structure and ranking in a
directed network, and measures how far the ranking is explained by the communities.

## What is inferred

Every node $i$ belongs to a group $b_i$ and has a score $s_i$ (its rank or strength).

- **Communities.**
  - The undirected network (who interacts with whom, ignoring direction) follows a stochastic block model.
  - By default the SBM is degree-corrected, and the densities of edges within and between groups are inferred separately (`assortative=True`).
  - The number of groups is inferred.
- **Hierarchy.**
  - Each interaction between $i$ and $j$ is decided in favour of $i$ with the Bradley–Terry probability $e^{s_i} / (e^{s_i} + e^{s_j})$.
  - With ties allowed (`hierarchy_model="bradley_terry_ties"`, chosen automatically when the network has neutral edges), the interaction is drawn with probability proportional to $\nu\, e^{(s_i + s_j)/2}$ (Davidson 1970).
- **Coupling.** In the coupled model (`interaction="coupled"`, the default), a node's score is its group's score plus an individual deviation,

  $$s_i = c_{b_i} + \epsilon_i, \qquad c_r \sim N(0, \sigma_G^2/2), \quad \epsilon_i \sim N(0, \sigma_I^2/2).$$

  The independent model (`interaction="independent"`) has no group term ($\sigma_G = 0$).

The summaries reported for a fit are:

| Quantity | Name in output | Meaning |
|---|---|---|
| $\sigma_I$ | `individual_depth` | spread of scores within a group |
| $\sigma_G$ | `group_depth` | spread of group scores |
| $\sigma_T = \sqrt{\sigma_I^2 + \sigma_G^2}$ | `sigma_T` | total depth of the hierarchy |
| $\text{ICC} = \sigma_G^2 / \sigma_T^2$ | `ICC` | fraction of the variance in scores attributable to the groups |
| $\nu$ | `ties_parameter` | propensity for ties |
| $q$ | `num_groups` | number of groups |

Because the score difference of two randomly chosen nodes has variance $\sigma_T^2$, a
typical pair differs by $\sigma_T$. The better of the two then wins with odds of about
$e^{\sigma_T}$.

All parameters are inferred by default with these priors:
- $\sigma_I$, $\sigma_G$ and $\nu$: half-Cauchy with scale 4.
- SBM densities: exponential.
- Mixing variations: uniform.

Any of them can be fixed through {class}`~community_hierarchy.Config`.

## Input conventions

- **Graph type.** `fit` takes a `networkx.DiGraph` or `MultiDiGraph`.
- **Interactions and weights.** Each edge is one interaction, or `weight` interactions; parallel edges are summed.
- **Decided edges.** By default an edge `u → v` means **u beat v**.
  - Pass `source_high_rank=False` when the target is the winner: for example a citation (being cited) or a friendship nomination (being named).
  - With that flag the scores need no flipping afterwards.
- **Neutral edges.** An edge with `type="neutral"` is a draw (a tie, or a reciprocated nomination); its direction is ignored.
  - `hierarchy_model="bradley_terry"` has no draws, so it raises a ValueError on a graph with neutral edges.
  - If any edge has a `type`, every edge must have one, either `"dominant"` or `"neutral"`.
  - Draws count towards the community structure as well as towards the tie rate.
- **Self-loops** are not allowed: a node cannot interact with itself.
- **Node order.** Every array in a {class}`~community_hierarchy.Result`, and every node table, follows the order of `G.nodes()`.

## Choosing the community model

`Config(groups_model=...)` selects a preset for the SBM's mixing structure:

| `groups_model` | Mixing variation (in / out) | `assortative` |
|---|---|---|
| `"traditional_SBM"` (default) | 0.5 / 0.5 | `False` |
| `"general_ASBM"` | both inferred | `True` |
| `"hybrid_ASBM"` | 0.5 / 0 | `True` |
| `"simple_ASBM"` | 0.5 / 0.5 | `True` |
| `"planted_partition"` | 0 / 0 | `True` |

The default is the standard degree-corrected SBM: one density shared by all group pairs. The assortative presets give edges within and between groups separate densities (0.5.0 and earlier defaulted to `"general_ASBM"`).

- **Mixing variation.** It controls how unequal the edge densities may be across group pairs, within groups ("in") and between them ("out").
  - 0 means all equal.
  - Values are inferred unless fixed.
- **`assortative`.** With `assortative=True`, edges within groups and between groups get separate density parameters. With `False`, a single density is shared.
- **Explicit values override the preset.** `Config(variation_in=..., variation_out=..., assortative=...)` overrides it.

Degree correction is set separately: `degree_correction=True` is the default. Keep it on
whenever degrees vary for reasons other than group membership.

This matters most when status itself drives degree, as in friendship nominations. A
model without degree correction can only explain degree differences through the groups.
It then sorts nodes into groups by degree, and so by status, which inflates the ICC.

### Weighted networks

- **`weighted_SBM=True`.** The weights are treated as real-valued strengths, not counts:
  - The binary network (which pairs are joined at all) follows the SBM.
  - The weights between two groups are uniform on the simplex of their total (Peixoto 2018).
  - The totals of the group pairs joined by an edge are uniform on the simplex of the overall weight.
  - As a result, the inferred groups do not depend on the unit in which the weights are measured.
- **Use it** for weights such as citation counts, or association indices such as the simple ratio index, when you want communities from the pattern of strong and weak ties.
- **Weights** must be finite and positive.

### Two-layer networks

- **`fit(config, G, group_graph=H)`** infers the communities from a separate undirected graph `H` on the same nodes, while the hierarchy is still inferred from `G`.
  - Example: communities from grooming, ranks from aggression.
- **Weighted `H`.** With `weighted_SBM=True`, the weights of `H` are modelled as above.

### Known groups or scores

- **`Config(group_attribute="team")`** fixes the partition to a node attribute, and only the hierarchy is inferred.
  - The ICC then measures how much of the variation in rank the known groups explain.
- **`Config(hierarchy_attribute=...)`** fixes the scores.
- **`Config(num_groups=k)`** fixes the number of groups.

## Sampling

- **The sampler.** `fit` runs Markov chain Monte Carlo with:
  - single-node and merge-split moves for the partition (Peixoto 2020);
  - moves in the cumulative-distribution space of each prior for the scores and parameters;
  - optional parallel tempering (`num_tempering_chains`).
- **Running chains.** `num_chains` independent chains run in parallel threads.
- **Burn-in.**
  - `num_warmup` sweeps (each one proposal per node) are discarded.
  - During burn-in the step sizes adapt towards an acceptance rate of 0.44; they are then frozen.
  - After burn-in, each chain records `num_samples` samples, `sweeps_per_sample` sweeps apart.
- **Checking convergence.**
  - Use several chains.
  - Compare their summaries: `samples_df()` has a `chain_id` column.
  - Watch `plot_MCMC_entropy` for a stationary trace.
- **Typical settings** for networks of a few hundred nodes are 4 chains, 1,000 burn-in sweeps, and 1,000–5,000 samples per chain.

## Summaries of a fit

- **Network-level parameters.** {meth}`~community_hierarchy.Result.network_properties_df` gives the posterior mean, standard deviation and highest-density interval of each one, including `ICC` and `sigma_T`.
- **Node-level summaries.** {meth}`~community_hierarchy.Result.node_properties_df` gives each node's posterior mean score, its standard deviation and its consensus group.
- **Consensus partition.** The point estimate of the partition is the maximum-overlap consensus (Peixoto 2021): the partition that agrees with the most labels across samples after optimal relabelling.
- **Predictions.** {meth}`~community_hierarchy.Result.direction_probabilities` gives the posterior predictive probability that one node beats another, or that they tie.

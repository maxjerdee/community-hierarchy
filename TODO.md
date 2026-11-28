# Open items

Resolved items were pruned (2026-11); see the git history of this file for the
development notes they came from.  Known sampler issues are listed under
"Known open issues" in docs/ARCHITECTURE.md.

## Model / core

- Decide whether a graph with no edges should raise instead of fitting the prior.
- Investigate the over-fragmentation on `snr150_sigC5_rep09.gml` (too many
  communities even in independent mode); possibly the degree-correction
  singleton trap described in ARCHITECTURE.md.

## API

- Let the summary and plotting functions (e.g. `plot_node_properties`) take a
  `samples_df` as well as a `Result`.
- A plot of the status decomposition (group mean vs individual score).
- Naming: consider `type` -> `interaction_type` for the edge attribute.

## Packaging

- R package (`R-package/`): register on R-universe, then submit to CRAN.

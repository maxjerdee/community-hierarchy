# Tests

## Native (C++) tests

Configure with tests on and run them through CTest. Run from the repository
root:

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

| test | what it checks |
|---|---|
| `mcmc_core_test` | Construction-time validation (self-loops, weights, symmetric ties, `num_groups`), the description length against an exact enumeration of stub matchings, fixed scores staying fixed under the σ moves, the accuracy of the Gaussian quantile and the Hastings ratio of the score proposal, plain Bradley–Terry with neutral edges, `value_check`, and n = 50,000 without overflow (about 2 s). |
| `weighted_bookkeeping_test` | Incremental weighted-SBM bookkeeping (H_W, weight counts) against a recount after 40,000 single-node flips and 4,000 sweeps, also in two-layer mode (under 1 s). |
| `flip_nodes_multi_test` | The multi-node path of `flip_nodes`, which merge and split use and single-site moves never touch. It is checked against two oracles: a from-scratch rebuild, and the same move made as single-node flips. |
| `merge_split_test` | Four checks (about 90 s). §1 bookkeeping after every collective move. §2 the β = 0 uniform-q prior. §2b the exact partition target `q! exp(-H_b - H_s)` over all 52 partitions of n = 5, with the score prior untempered. §3 single-site and merge/split agree at β = 1. |

All four have a plain `main()` and exit non-zero on failure; none needs data files. They can also be built by hand:

```bash
g++ -std=c++17 -O2 -Isrc tests/merge_split_test.cpp \
  src/MCMC_core.cpp src/MCMC_proposals.cpp src/MCMC_bookkeeping.cpp \
  src/MCMC_merge_split.cpp src/helpers.cpp src/globals.cpp -o build/merge_split_test
```

## Mathematica reference values

`tests/data/outputs/` holds values computed in Mathematica from the model's
definition, in the notebook `mathematica/sbm_mcmc.nb` of the paper's reproduction
repository (`community-hierarchy_reproduction`, private):
- the entropy components of a 3-node network (`wins_ties_3node_point_eval_components.csv`);
- the exact partition posterior of a 6-node network, unweighted and weighted SBM, over all
  203 partitions (`exact6_partition_posterior.csv`);
- the coincidence matrices of small fits (`*_coincidence.csv`).

The first two come from the notebook's section "community-hierarchy test fixtures" (cells
tagged `ch-fixtures`). Regenerate them with

```bash
wolframscript -file tests/mathematica/generate_fixtures.wls [path/to/sbm_mcmc.nb]
```

Without an argument it looks for `../../package-testing/community-hierarchy_reproduction/mathematica/sbm_mcmc.nb`.
The script reads the notebook
and does not change it.

## Python tests

```bash
pytest tests/python -q
```

`tests/python/test_regressions.py` has one test per bug fixed in the 2026-09
review.

See `docs/ARCHITECTURE.md` for how each test maps to a property of the sampler.

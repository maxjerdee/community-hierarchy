# community_hierarchy: architecture

How the package is put together, what each part of the model is, why each MCMC
move is valid, and where each property is tested. Read this before changing
the sampler.

## Layers

| layer | files | role |
|---|---|---|
| Python API | `python/community_hierarchy/src/community_hierarchy/core.py` | `Config`, `fit()`, `Result` (summaries, plots) |
| input handling | `.../input_output.py` | input validation, GML output, plotting |
| R API | `R-package/` | `ch_config()`, `ch_fit()`, summaries; a copy of `src/` (`tools/sync_r_core.py`) |
| binding | `bindings/python/bindings.cpp` | pybind11: `fit_backend`, `FitResult`, and the raw `MCMC` class |
| model presets | `src/model_presets.{h,cpp}` | named presets + explicit values → sampler parameters, and their checks (every binding) |
| starting state | `src/initial_state.{h,cpp}`, `src/modularity.{h,cpp}` | default partition (greedy modularity, `modularity`) and scores (win fractions), used by `fit()` when none are given |
| driver | `src/fit.{h,cpp}` | chains (threads), parallel tempering, burn-in, sampling, timeout/abort, progress callbacks |
| model + moves | `src/MCMC_core.{h,cpp}`, `src/MCMC_proposals.cpp`, `src/MCMC_merge_split.cpp` | the state, every move, `change_parameter` |
| energies | `src/MCMC_bookkeeping.cpp` | from-scratch computation of every term; `check_state_bookkeeping` |
| helpers | `src/helpers.{h,cpp}`, `src/globals.{h,cpp}` | special functions, prior-CDF proposals, per-thread RNG, merge/split tuning |
| results | `src/fit_result.{h,cpp}` | samples, consensus, coincidence matrix, summaries |
| cross-validation | `src/loo.{h,cpp}` | PSIS (a port of ArviZ's) and leave-one-out of the edge directions (`direction_loo_df()` in Python, `direction_loo()` in R) |

## The model (the energy the sampler targets)

The state is a partition `b` (q groups), a score `s_i` per node, and
continuous parameters. The description length (negative log joint, nats) is a
sum of terms, each cached on the `MCMC` object and recomputed from scratch at
the end of every sweep (`check_state_bookkeeping`, which throws on mismatch):

| term | meaning | depends on |
|---|---|---|
| `H_b` | partition prior: uniform q, uniform group sizes, multinomial assignment (labelled) | b |
| `H_M` | block edge counts, negative binomial with means `density_in n_r²/2`, `density_out n_r n_s` | b, densities, variations |
| `H_k` | degrees within groups, Dirichlet-multinomial (multinomial when no degree correction) | b, degree_correction |
| `H_A_G_k_M_b` | edge placement given block counts and degrees (microcanonical configuration model, Peixoto 2017 PRE 95:012317): multi-edges contribute Σ_{i<j} log A_ij! once per pair; the term is a constant in `weighted_SBM` and two-layer mode and is skipped there | b |
| `H_W` | weighted SBM only: edge weights uniform on the simplex of each block pair's total, and the totals of the Q block pairs with edges uniform on the simplex of the overall weight, (Q−1)!/w^(Q−1); counting only pairs with edges keeps the partition posterior independent of the weight unit | b |
| `H_A_G_b` | `H_M + H_k + H_A_G_k_M_b (+ H_W)`: the undirected network given b | |
| `H_Adir_G_A_s_nu` | edge directions given scores: Bradley–Terry with ties (a tie ∝ `ties_parameter·e^{(s_i+s_j)/2}`) | s, ties_parameter |
| `H_s_G_b_depths` | score prior given the partition: `s_i = c_r + ε_i`, group means `c_r` integrated out | s, b, individual_depth, group_depth |

The undirected network is `wins + ties`, or, in **two-layer mode**
(`group_graph=`), a separate undirected graph drives the communities while
`wins`/`ties` still drive the directions. Edge weights are integer
multiplicities except in `weighted_SBM` mode. `fit()` rejects
`hierarchy_model="bradley_terry"` (ties fixed at zero) on a graph with neutral edges,
because a draw has probability 0 under it. The core itself, if given ties with
`ties_parameter` fixed at 0, keeps them in the SBM and out of `H_Adir`.

Input handling (`input_output.py`) rejects, before the core is built: self-loops in
either layer; weights that are not finite and positive (and non-integer weights unless
`weighted_SBM`); explicit values that the chosen model fixes (e.g. `group_depth` with
`interaction="independent"`). The initial partition is built in `G.nodes()` order in
both modes. The core checks again and throws `std::invalid_argument` on self-loops in
any layer, node ids out of range, ties not listed in both directions, weights that are
not finite (or not positive in `weighted_SBM` and two-layer mode), and a fixed
`num_groups` that disagrees with the initial partition (q is always read from the
partition).

The state holds the scores `s_i` themselves (and per-group sums of them), and
`H_s_G_b_depths` is the density of `s`:
`s ~ N(0, individual_depth²/2 · I + group_depth²/2 · B)`, `B` the same-group
indicator. Every move that rescales scores carries its Jacobian in `s`.

## Tempering convention

Every move, the merge/split target and the parallel-tempering swaps agree on

    pi_beta  ∝  P(b) · P(s | b, depths) · P(parameters) · exp(-beta · (H_A_G_b + H_Adir_G_A_s_nu))

Only the data likelihood is tempered. The partition prior, the score prior and
the parameter priors are not. This is forced by the score move (its proposal
is reversible with respect to `exp(-H_s_G_b_depths)`). It is documented at
`MCMC::beta`; the regression test is `merge_split_test` section 2b.

## Moves (all in one sweep of n proposals; weights are state-independent)

| move | where | why it is valid |
|---|---|---|
| single-site | `propose_single_site_move` | block-informed proposal of Peixoto (2020) Eq. 14/15, c = 0.5, new group with d = 0.01; Hastings ratio of the proposals; accept on `ΔH_b + beta·ΔH_A_G_b + ΔH_s` (with the label factor when q changes) |
| merge / split / joint merge-split | `MCMC_merge_split.cpp` | Peixoto arXiv:2003.07070 with staged splits as auxiliary variables; label-multiplicity factor for a label-array representation; pitfalls P1–P5 documented in the file |
| score | `propose_score_move` | CDF-space random walk on the exact conditional of `s_i` under the score prior; accept on `beta·ΔH_Adir`. The prior cancels only if the Gaussian CDF and quantile are exact, so they work in the nearer tail (Acklam's inverse plus one Halley step on `erfc`, relative error < 4e-13 down to tail mass 1e-300) |
| group score shift | `propose_group_score_move` | symmetric random walk shifting all of one group's scores; accept on `beta·ΔH_Adir + ΔH_s` |
| parameters | `propose_parameter_move` → `change_parameter` | random walks in each prior's CDF space (reflected), so the prior cancels |
| `individual_depth` (σ_I), two kernels by a fixed ½ coin | `propose_individual_depth_centred`, `propose_individual_depth_residual` | (1) scores fixed: accept on `ΔH_s`, no Jacobian; (2) deviations from group means scaled by λ = σ_I'/σ_I, group means fixed: accept on `ΔH_s + beta·ΔH_Adir`, Jacobian λ^(n−q) |
| `group_depth` (σ_G), two kernels by a fixed ½ coin | `change_parameter`, `propose_group_depth_rescale` | (1) scores fixed: accept on `ΔH_s`; (2) group means scaled by λ = σ_G'/σ_G, deviations fixed: accept on `ΔH_s + beta·ΔH_Adir`, Jacobian λ^q |

With `scores_fixed` only kernel (1) of each depth is used, so no move changes a score.

Each kernel has its own step size. Steps adapt during burn-in only (Robbins–Monro
on the log step towards acceptance 0.44) and are frozen for sampling.

Priors on parameters: variations and degree correction Uniform(0,1); densities
Exponential(1); `individual_depth`, `group_depth`, `ties_parameter`
half-Cauchy(4).

## The driver (`src/fit.cpp`)

- `num_chains` independent chains run on `std::async` threads. The random
  generator is `thread_local` (`globals.h`); each chain builds its `MCMC`
  objects on its own thread, so chains are independent and runs are
  reproducible from `seed`.
- Parallel tempering (`num_tempering_chains = K > 1`): K replicas per chain on
  a geometric β ladder from `beta/(K-1)` to `beta`, plus β = 0; adjacent swaps
  after every sample; the β = `beta` replica is recorded. The chain's thread
  generator is seeded once with `seed + chain`; replicas are built with seed −1
  (no reseed) and share that stream.
- Starting partition: unless `initial_partition` is given, every chain starts
  from a greedy-modularity partition of the undirected graph (small groups
  merged into their best-connected neighbour; `src/modularity.cpp`, the same
  in every binding), which avoids starting in the slow-to-coarsen many-groups
  regime. Since 0.6.0 it is computed in C++ with exact (integer) modularity
  gains; 0.5.0 and earlier used networkx, whose order among equal gains and
  equal-sized groups differs, so seeded fits from the default start differ.
- `num_param_warmup`: sweeps with partition and scores frozen, letting the
  continuous parameters settle first.
- `timeout` (seconds, default 60) stops a fit early, **with fewer samples than
  requested**. Raise it for long runs.

## Validation map

| property | test |
|---|---|
| incremental energies = from-scratch energies | every sweep (`check_state_bookkeeping(true)`) |
| multi-node `flip_nodes` bookkeeping | `tests/flip_nodes_multi_test.cpp` (two oracles) |
| partition moves sample the prior at β = 0 | `tests/merge_split_test.cpp` §2 |
| partition moves sample `q! exp(-H_b - H_s)` exactly (score prior untempered) | `tests/merge_split_test.cpp` §2b (n = 5, all 52 partitions) |
| single-site and merge/split agree at β = 1 | `tests/merge_split_test.cpp` §3 |
| constructor validation, DL vs exact stub-matching enumeration, `scores_fixed`, quantile accuracy and the score proposal's Hastings ratio, plain Bradley–Terry with neutral edges | `tests/mcmc_core_test.cpp` |
| weighted-SBM bookkeeping (H_W, weight counts) after many flips and sweeps, also two-layer | `tests/weighted_bookkeeping_test.cpp` |
| core energies, and the partition kernels, against Mathematica's exact posterior of a 6-node network (203 partitions), unweighted and weighted | `tests/python/test_sampler_exactness.py` |
| entropy components against Mathematica at one point | `tests/python/test_package.py::test_point_eval_entropy_components` |
| Python API, presets, reference values | `tests/python/` |
| fixed bugs | `tests/python/test_regressions.py` |

## Known open issues

- **Depth parameters mix slowly.** The prior-CDF step is 0.01; per-chain
  medians of `group_depth` on a 6–8 node graph still scatter by ±1 after
  ~10⁵ sweeps. A joint (σ_I, σ_C) block move is the natural fix.
- **Fragmentation with degree correction.** As in asbm, degree-corrected fits
  of weakly structured graphs can sit near all-singletons (e.g. q ≈ 11 of 12);
  see the asbm singleton-trap notes.

## After changing anything in `src/`

Every binding keeps its own copy of the compiled core, and none of them notices
that the others are stale. A stale artifact looks exactly like a numerical
disagreement, so rebuild all of them, or expect to lose an afternoon:

```bash
cmake --build build -j                                   # native + the Python extension
python -m pip install -e . --no-build-isolation --no-deps # editable installs serve a COPY under site-packages
emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release && cmake --build build-wasm -j
```

`tests/python/test_build_freshness.py` fails loudly when the extension or the wasm
bundle is older than `src/`, with the command to fix it. Run
it first when a comparison between two bindings disagrees.

Two traps worth knowing:

- **The editable install serves the extension from a copy** under
  `site-packages`, not from the source tree. `cmake --build` alone updates the
  tree and leaves `import community_hierarchy` on the old binary.
- **A stale wasm bundle passes any test that only compares description
  lengths.** The consensus partition can change with no change to any sample's
  DL, which is exactly how the refinement in 0245bd5 slipped past a green
  parity suite. Compare partitions, not just numbers.

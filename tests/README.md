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
| `mcmc_core_test` (Catch2) | Construction, bookkeeping, moves and the `fit` driver on the football network. It reads `examples/data/...`, so CTest runs it from the repository root. |
| `flip_nodes_multi_test` | The multi-node path of `flip_nodes`, which merge and split use and single-site moves never touch. It is checked against two oracles: a from-scratch rebuild, and the same move made as single-node flips. |
| `merge_split_test` | Four checks (about 40 s). §1 bookkeeping after every collective move. §2 the β = 0 uniform-q prior. §2b the exact partition target `q! exp(-H_b - H_s)` over all 52 partitions of n = 5, with the score prior untempered. §3 single-site and merge/split agree at β = 1. |

`flip_nodes_multi_test` and `merge_split_test` have a plain `main()` and exit
non-zero on failure. They can also be built by hand:

```bash
g++ -std=c++17 -O2 -Isrc tests/merge_split_test.cpp \
  src/MCMC_core.cpp src/MCMC_proposals.cpp src/MCMC_bookkeeping.cpp \
  src/MCMC_merge_split.cpp src/helpers.cpp src/globals.cpp -o build/merge_split_test
```

## Python tests

```bash
pytest tests/python -q
```

`tests/python/test_regressions.py` has one test per bug fixed in the 2026-09
review.

See `docs/ARCHITECTURE.md` for how each test maps to a property of the sampler.

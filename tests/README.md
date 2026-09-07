# C++-Only Test Build (No pybind11)

## Merge/split tests (added with src/MCMC_merge_split.cpp)

```bash
g++ -std=c++17 -O2 -Isrc tests/flip_nodes_multi_test.cpp \
  src/MCMC_core.cpp src/MCMC_proposals.cpp src/MCMC_bookkeeping.cpp \
  src/MCMC_merge_split.cpp src/helpers.cpp src/globals.cpp -o build/flip_nodes_multi_test
g++ -std=c++17 -O2 -Isrc tests/merge_split_test.cpp \
  src/MCMC_core.cpp src/MCMC_proposals.cpp src/MCMC_bookkeeping.cpp \
  src/MCMC_merge_split.cpp src/helpers.cpp src/globals.cpp -o build/merge_split_test
```

`flip_nodes_multi_test` validates the MULTI-NODE path of flip_nodes (never
exercised by single-site moves) against two oracles.  `merge_split_test` checks
bookkeeping, the beta=0 uniform-q prior, and beta=1 coincidence agreement
between single-site and merge/split.  The beta=1 check is the important one: the
H_s_G_b_depths term is multiplied by beta, so a beta=0 test alone cannot catch
its omission from the merge/split energy.

NOTE: `tests/mcmc_core_test.cpp` does not currently compile -- it predates the
`group_adj` constructor argument and was already broken at HEAD.


This folder contains a standalone C++ test harness: `mcmc_core_test.cpp`.

You can compile and run it **without pybind11/Python** using plain `g++` commands.

## 1) Build (release)

Run from the repository root (`package-development/directedstructure`):

```bash
g++ -std=c++17 -O3 -march=native -Wall -Wextra -Isrc \
  tests/mcmc_core_test.cpp \
  src/MCMC_core.cpp \
  src/MCMC_proposals.cpp \
  src/MCMC_bookkeeping.cpp \
  src/MCMC_merge_split.cpp \
  src/globals.cpp \
  src/helpers.cpp \
  -o build/mcmc_core_test_cpp_only
```

Then run:

```bash
./build/mcmc_core_test_cpp_only
```

## 2) Build (debug)

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Isrc \
  tests/mcmc_core_test.cpp \
  src/MCMC_core.cpp \
  src/MCMC_proposals.cpp \
  src/MCMC_bookkeeping.cpp \
  src/MCMC_merge_split.cpp \
  src/globals.cpp \
  src/helpers.cpp \
  -o build/mcmc_core_test_cpp_only_debug
```

Run:

```bash
./build/mcmc_core_test_cpp_only_debug
```

## 3) Optional: run under valgrind

```bash
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
  ./build/mcmc_core_test_cpp_only_debug
```

---

# run_sampler_to_csv (C++ sampler → CSV)

`run_sampler_to_csv.cpp` reads a GML file, runs the MCMC sampler, and writes
samples to a CSV file in the same column format as `examples/testing_samples.csv`.

## 1) Build (release)

Run from the repository root (`package-development/directedstructure`):

```bash
g++ -std=c++17 -O3 -march=native -Wall -Wextra -Isrc \
  tests/run_sampler_to_csv.cpp \
  src/MCMC_core.cpp \
  src/MCMC_proposals.cpp \
  src/MCMC_bookkeeping.cpp \
  src/MCMC_merge_split.cpp \
  src/globals.cpp \
  src/helpers.cpp \
  -o build/run_sampler_to_csv
```

## 2) Run

```bash
./build/run_sampler_to_csv \
  --input-gml  examples/data/networks/english-football-restricted.gml \
  --output-csv examples/data/samples/football_test.csv \
  --num-samples 1000 \
  --sweeps-per-sample 10 \
  --seed 0
```

Only `--input-gml` and `--output-csv` are required; the rest have the defaults shown above.

## 3) Build (debug)

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Isrc \
  tests/run_sampler_to_csv.cpp \
  src/MCMC_core.cpp \
  src/MCMC_proposals.cpp \
  src/MCMC_bookkeeping.cpp \
  src/MCMC_merge_split.cpp \
  src/globals.cpp \
  src/helpers.cpp \
  -o build/run_sampler_to_csv_debug
```
---

# run_sampler_to_csv

`run_sampler_to_csv.cpp` has moved to `AddHealth_analysis/`.
See [AddHealth_analysis/README.md](../AddHealth_analysis/README.md) for
compilation and usage instructions.

---

## Notes

- These commands intentionally **do not** use:
  - `-DDIRECTEDSTRUCTURE_PYBIND`
  - `python -m pybind11 --includes`
  - Python linker flags
- Make sure the `build/` directory exists first:

```bash
mkdir -p build
```

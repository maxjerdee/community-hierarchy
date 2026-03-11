# AddHealth Analysis

This folder now uses the `mcmc_core_test` harness as the sampler source,
renamed to `run_sampler_to_csv.cpp`.

## Directory layout

```
AddHealth_analysis/
  run_sampler_to_csv.cpp   # test-harness-based sampler source
  input_networks/          # optional place for .gml files
  samples/                 # optional output folder
```

---

## 1) Compile the sampler (release)

Run from inside `AddHealth_analysis/`:

```bash
g++ -std=c++17 -O3 -march=native -Wall -Wextra -I../src \
  run_sampler_to_csv.cpp \
  ../src/MCMC_core.cpp \
  ../src/MCMC_proposals.cpp \
  ../src/MCMC_bookkeeping.cpp \
  ../src/MCMC_merge_split.cpp \
  ../src/consensus.cpp \
  ../src/helpers.cpp \
  -o run_sampler_to_csv
```

You can also compile via Makefile:

```bash
make csv
```

## 2) Run

```bash
cd AddHealth_analysis
./run_sampler_to_csv dominance_networks/english-football-restricted.gml
```

Optional explicit output path:

```bash
./run_sampler_to_csv dominance_networks/english-football-restricted.gml samples/english-football-restricted.csv
```

If `output.csv` is not provided, output defaults to `samples/<input_stem>_samples.csv`.

This harness currently uses hard-coded paths/settings inside `main()`.
Edit settings in `run_sampler_to_csv.cpp` before running if needed.

---

## Notes

- Compilation does not require pybind11 or Python (`-DDIRECTEDSTRUCTURE_PYBIND` is omitted).
- If needed, create `build/` first when using Makefile helpers: `mkdir -p build`.

# community-hierarchy WASM binding

WebAssembly build of the `community_hierarchy` C++ core, following the same
session-step pattern as the `asbm` WASM binding. A fit runs in a Web Worker
(or a Node `worker_threads` worker) and is stepped in chunks, so it never
blocks the page and can be cancelled.

## Build

Requires [Emscripten](https://emscripten.org/) (`emcmake`).

```bash
cd community-hierarchy
emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build-wasm -j
```

The output is `bindings/wasm/js/dist/ds_core.mjs`, a self-contained WASM
module (SINGLE_FILE=1, the binary is base64-inlined). Rebuild it after any
change to `src/` or `bindings/wasm/src/`: `tests/python/test_build_freshness.py`
fails when it is older than its sources. Tests: `cd bindings/wasm/js && npm test`
and `python -m pytest tests/test_wasm_parity.py`.

## Usage

```js
import { load } from "community-hierarchy";

const m = await load();

// wins: [[u, v], ...] -- a directed edge u -> v means u beats v (u ranks above v)
// ties: [[u, v], ...] -- undirected draws (neutral / reciprocated edges)
const run = m.fit(
  wins,
  ties,
  { num_nodes: N, num_warmup: 500, num_samples: 500, sweeps_per_sample: 10, seed: 42 },
  { onProgress(p) { console.log(p.fraction.toFixed(2)); } }
);

const result = await run.result;
// result.numNodes            -- number
// result.numSamples          -- samples recorded (fewer if cancelled)
// result.cancelled           -- boolean
// result.mdlPartition        -- Int32Array of length N (group 0 = lowest mean score)
// result.consensusPartition  -- Int32Array of length N (maximum-overlap consensus)
// result.scoreMeans          -- Float64Array of length N (higher = higher rank)
// result.scoreStds           -- Float64Array of length N
// result.coincidence         -- Float64Array of length N*N (row-major)
// result.mdlValue            -- smallest description length among the samples (nats)
// result.samples             -- per-sample snapshots: chain, numGroups,
//                               descriptionLength, hAdirGAsNu, hSGbDepths,
//                               individualDepth, groupDepth, variationIn,
//                               variationOut, degreeCorrection, tiesParameter

// Cancel early (run.result then resolves with the samples so far):
run.cancel();
```

`onProgress` receives `{burnDone, burnTotal, samplesDone, samplesTotal,
fraction}`; about every 200 ms also the current state (`currentPartition`,
`currentScores`, `currentDescriptionLength`, `currentNumGroups`) and the best
sample so far (`bestPartition`, `bestScores`, `bestDescriptionLength`,
`bestNumGroups`, `samplesSoFar`); and a last tick with `summarizing: true`
before the consensus is computed.

### Edge format

Both `wins` and `ties` accept:
- an array of `[u, v]` or `[u, v, weight]`;
- an object `{from, to, weight?}` of equal-length arrays (typed or plain);
- `[]` or `null` for no edges.

Node ids are integers in `[0, num_nodes)`. Weights must be finite and
positive. Self-loops are rejected. Bad ids or weights throw from `fit()`;
errors found by the core (a self-loop, a bad `initialPartition`) reject
`run.result`.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `num_nodes` | max id + 1 | Number of nodes |
| `num_groups` | unset | With `num_groups_fixed`: the fixed K (the default start then has K groups); otherwise unused |
| `num_groups_fixed` | false | Hold K fixed (at `num_groups`, or at the K of `initialPartition`) |
| `assortative` | false | Separate densities for edges within and between groups |
| `degree_correction_mode` | 1 | 1: degree correction (fixed at 0.5, a uniform prior over degree sequences); 0: none |
| `weighted_sbm` | 0 | 1: real-valued weights, uniform simplex prior (the H_W term) |
| `independent_coupling` | 0 | 1: `group_depth` fixed at 0 (scores independent of the groups) |
| `variation_in`, `variation_out` | 0.5 | Fixed mixing variation in [0, 1]; negative = inferred. With `assortative: false`, the default is the degree-corrected SBM (`traditional_SBM`) |
| `merge_split_enabled` | true | Collective merge/split moves |
| `num_warmup` | 500 | Warm-up iterations (step sizes adapt); one iteration is `sweeps_per_sample` sweeps |
| `num_samples` | 1000 | Sampling iterations, each recorded |
| `sweeps_per_sample` | 10 | MCMC sweeps per iteration (0: record the starting state) |
| `seed` | random | RNG seed |
| `initialPartition` | greedy modularity | Starting labels: length `num_nodes`, using every label 0..K-1 |
| `initialScores` | win rates | Starting scores: length `num_nodes`, used as given; their RMS is the starting `individual_depth` |

The draw rate (`ties_parameter`) is inferred when there is a tie edge and
fixed at 0 (plain Bradley-Terry) otherwise.

### Differences from the Python `fit()`

The model, the C++ core and the defaults are the same; the wrapper is smaller:

| | Python `fit(Config(), G, ...)` | wasm `m.fit(wins, ties, options)` |
|---|---|---|
| input | NetworkX graph; direction read via `source_high_rank` | edge lists; a win `[u, v]` always means u beats v |
| number of groups | `Config(num_groups=k)` fixes K | `num_groups` with `num_groups_fixed: true` |
| degree correction | `Config(degree_correction=True/False)` | `degree_correction_mode: 1/0` |
| weighted SBM | `Config(weighted_SBM=True)` | `weighted_sbm: 1` |
| independent model | `Config(interaction="independent")` | `independent_coupling: 1` |
| mixing presets | `Config(groups_model=..., mixing_variation=...)` | `variation_in` / `variation_out` values |
| hierarchy model | `Config(hierarchy_model=...)` | automatic: draws are modelled iff there are tie edges |
| chains | `num_chains`, `num_tempering_chains`, `timeout` | one chain, no tempering, no timeout (use `cancel()`) |
| not available | | `group_graph` (two-layer mode), `group_attribute` / `hierarchy_attribute`, fixed depths or ties parameter, `weighted_proposals`, `num_param_warmup`, `num_thin`, `beta` |
| summaries | `Result` methods (HDIs, ICC, `node_properties_df`, ...) | the arrays above; summarise `samples` yourself |

Given the same seed, options and explicit starting state, a single-chain
Python fit and a wasm fit run the same chain (`tests/test_wasm_parity.py`).

## Symlink for edgewise

The edgewise project resolves `community-hierarchy` via a local symlink (same
pattern as `asbm`):

```bash
cd edgewise/web
ln -s ../../../package-development/community-hierarchy/bindings/wasm/js \
      node_modules/community-hierarchy
```

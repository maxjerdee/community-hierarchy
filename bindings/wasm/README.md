# directedstructure WASM binding

WebAssembly build of the `directedstructure` C++ core, following the same
session-step pattern as the `asbm` WASM binding.

## Build

Requires [Emscripten](https://emscripten.org/) (`emcmake`, `emmake`).

```bash
cd package-development/directedstructure
mkdir build-wasm && cd build-wasm
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release
emmake make -j4
```

The output is `bindings/wasm/js/dist/ds_core.mjs` — a self-contained WASM
module (SINGLE_FILE=1, the binary is base64-inlined).

## Usage

```js
import { load } from "directedstructure";

const m = await load();

// wins: [[loser, winner], ...] — directed edges where winner beats loser
// ties: [[u, v], ...] — undirected neutral/reciprocated edges
const run = m.fit(
  wins,
  ties,
  { num_nodes: N, num_groups: 5, burn_sweeps: 1000, samples: 500, seed: 42 },
  { onProgress(p) { console.log(p.fraction.toFixed(2)); } }
);

const result = await run.result;
// result.mdlPartition        — Int32Array of length N
// result.scoreMeans          — Float64Array of length N
// result.scoreStds           — Float64Array of length N
// result.consensusPartition  — Int32Array of length N
// result.coincidence         — Float64Array of length N*N (row-major)
// result.mdlValue            — number
// result.samples             — array of per-sample parameter snapshots

// Cancel early:
run.cancel();
```

### Edge format

Both `wins` and `ties` accept:
- An array of `[u, v]` or `[u, v, weight]` pairs
- An object `{from: Int32Array, to: Int32Array, weight?: Float64Array}`
- `[]` or `null` for no edges

Node ids are 0-based integers.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `num_nodes` | max id + 1 | Number of nodes |
| `num_groups` | 5 | Initial (and fixed, if `num_groups_fixed`) number of groups |
| `num_groups_fixed` | false | Prevent merge/split from changing K |
| `assortative` | false | Whether in-group density differs from out-group density |
| `merge_split_enabled` | true | Enable merge/split moves |
| `burn_sweeps` | 1000 | Number of burn-in iterations |
| `samples` | 1000 | Number of posterior samples to collect |
| `sweeps_per_sample` | 1 | MCMC sweeps between recorded samples |
| `seed` | 0 | RNG seed |
| `initialPartition` | round-robin | Starting community assignment (Int32Array) |
| `initialScores` | zeros | Starting score values (Float64Array) |

## Symlink for edgewise

The edgewise project resolves `directedstructure` via a local symlink (same
pattern as `asbm`):

```bash
cd edgewise/web
ln -s ../../../package-development/directedstructure/bindings/wasm/js \
      node_modules/directedstructure
```

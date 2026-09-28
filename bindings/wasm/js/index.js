// directedstructure in the browser (and Node): the C++ core compiled to WebAssembly,
// run inside a worker so fits never block the page.
//
//   import { load } from "directedstructure";
//   const m = await load();
//   const run = m.fit(wins, ties, { num_nodes: N, num_groups: 5 }, { onProgress });
//   const result = await run.result;   // run.cancel() stops early with partial results
//
// wins and ties are edge lists: [[u, v], ...], [[u, v, w], ...] or {from, to, weight?}.

const isNode = typeof process !== "undefined" && !!(process.versions && process.versions.node)
  && typeof window === "undefined";

async function createWorker(workerURL) {
  const url = workerURL || new URL("./worker.js", import.meta.url);
  if (!isNode) {
    const w = new Worker(url, { type: "module" });
    return {
      post: (m, t) => w.postMessage(m, t || []),
      onMessage: (fn) => { w.onmessage = (e) => fn(e.data); },
      onError: (fn) => { w.onerror = (e) => fn(new Error(e.message || "ds worker failed")); },
      terminate: () => w.terminate(),
    };
  }
  const nodeModule = "node:worker_threads";
  const { Worker: NodeWorker } = await import(/* @vite-ignore */ nodeModule);
  const w = new NodeWorker(url);
  return {
    post: (m, t) => w.postMessage(m, t || []),
    onMessage: (fn) => w.on("message", fn),
    onError: (fn) => w.on("error", fn),
    terminate: () => w.terminate(),
  };
}

/** Accepts [[u, v], ...], [[u, v, w], ...] or {from, to, weight?}. */
function normalizeEdges(edges, name) {
  if (!edges || (Array.isArray(edges) && edges.length === 0) ||
      (edges.from && edges.from.length === 0)) {
    return { from: new Int32Array(0), to: new Int32Array(0), weight: null };
  }
  let from, to, weight = null;
  if (Array.isArray(edges)) {
    from = new Int32Array(edges.length);
    to   = new Int32Array(edges.length);
    const weighted = edges.some((e) => e.length > 2);
    if (weighted) weight = new Float64Array(edges.length);
    edges.forEach((e, i) => {
      from[i] = e[0]; to[i] = e[1];
      if (weighted) weight[i] = e.length > 2 ? e[2] : 1;
      if (!Number.isInteger(e[0]) || !Number.isInteger(e[1]) || e[0] < 0 || e[1] < 0)
        throw new TypeError(`${name} edge ${i}: node ids must be non-negative integers`);
    });
  } else if (edges && edges.from && edges.to) {
    from = Int32Array.from(edges.from);
    to   = Int32Array.from(edges.to);
    if (edges.weight) weight = Float64Array.from(edges.weight);
  } else {
    throw new TypeError(`${name} must be an array of [u, v(, w)] or {from, to, weight?}`);
  }
  return { from, to, weight };
}

/**
 * Start the worker and load the wasm core.
 * @param {{workerURL?: string|URL}} [config]
 */
export async function load(config = {}) {
  const worker = await createWorker(config.workerURL);
  const pending = new Map();
  let nextId = 1;

  const ready = new Promise((resolve, reject) => {
    worker.onError((err) => {
      reject(err);
      for (const p of pending.values()) p.reject(err);
      pending.clear();
    });
    worker.onMessage((msg) => {
      if (msg.type === "ready") { resolve(); return; }
      const p = pending.get(msg.id);
      if (!p) return;
      if (msg.type === "progress") {
        if (p.onProgress) {
          const g = msg.progress;
          const done = g.burn_done + g.samples_done, total = g.burn_total + g.samples_total;
          p.onProgress({
            burnDone: g.burn_done, burnTotal: g.burn_total,
            samplesDone: g.samples_done, samplesTotal: g.samples_total,
            fraction: total > 0 ? done / total : 0,
            // Set on the one extra tick runner.js sends right before the
            // (synchronous, potentially slow-for-large-n) consensus/MDL/
            // coincidence-matrix computation -- lets a UI show "Summarizing..."
            // instead of appearing stuck at the last "Samples N / N" tick.
            ...(g.summarizing ? { summarizing: true } : {}),
            ...(g.currentPartition ? {
              currentPartition: g.currentPartition,
              currentScores: g.currentScores,
              currentDescriptionLength: g.currentDescriptionLength,
              currentNumGroups: g.currentNumGroups,
            } : {}),
            ...(g.bestPartition ? {
              bestPartition: g.bestPartition,
              bestScores: g.bestScores,
              bestDescriptionLength: g.bestDescriptionLength,
              bestNumGroups: g.bestNumGroups,
              samplesSoFar: g.samplesSoFar,
            } : {}),
          });
        }
      } else if (msg.type === "result") {
        pending.delete(msg.id);
        p.resolve(msg.result);
      } else if (msg.type === "error") {
        pending.delete(msg.id);
        p.reject(new Error(msg.message));
      }
    });
  });
  await ready;

  return {
    /**
     * Fit the directedstructure model to a directed network.
     * @param wins    directed win edges: [[u, v], ...] or {from, to, weight?}
     *                u beats v means u→v is a win for u
     * @param ties    undirected tie edges: [[u, v], ...] or {from, to, weight?}
     *                pass [] or null for no ties
     * @param options { num_nodes?, num_groups?, num_groups_fixed?, assortative?,
     *                  merge_split_enabled?, burn_sweeps?, samples?,
     *                  sweeps_per_sample?, seed?,
     *                  initialPartition?: Int32Array, initialScores?: Float64Array }
     * @param hooks   { onProgress({burnDone, burnTotal, samplesDone, samplesTotal,
     *                              fraction, currentPartition?, currentScores?,
     *                              bestPartition?, bestScores?, ...}) }
     * @returns {{id: number, result: Promise<object>, cancel: () => void}}
     */
    fit(wins, ties, options = {}, hooks = {}) {
      const { num_nodes, initialPartition, initialScores, ...rest } = options;
      const { from: winsFrom, to: winsTo, weight: winsWeight } = normalizeEdges(wins, "wins");
      const { from: tiesFrom, to: tiesTo, weight: tiesWeight } = normalizeEdges(ties || [], "ties");

      let n = num_nodes;
      if (n === undefined) {
        n = 0;
        for (let i = 0; i < winsFrom.length; i++) n = Math.max(n, winsFrom[i] + 1, winsTo[i] + 1);
        for (let i = 0; i < tiesFrom.length; i++) n = Math.max(n, tiesFrom[i] + 1, tiesTo[i] + 1);
      }

      const job = {
        numNodes: n,
        winsFrom, winsTo, winsWeight,
        tiesFrom, tiesTo, tiesWeight,
        initialPartition: initialPartition ? Int32Array.from(initialPartition) : null,
        initialScores:    initialScores    ? Float64Array.from(initialScores)  : null,
        options: rest,
      };

      const id = nextId++;
      const result = new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject, onProgress: hooks.onProgress });
      });

      const transfer = [winsFrom.buffer, winsTo.buffer];
      if (winsWeight) transfer.push(winsWeight.buffer);
      if (tiesFrom.length) { transfer.push(tiesFrom.buffer, tiesTo.buffer); }
      if (tiesWeight) transfer.push(tiesWeight.buffer);

      worker.post({ type: "fit", id, job }, transfer);
      return { id, result, cancel: () => worker.post({ type: "cancel", id }) };
    },

    /** Stop the worker.  Pending fits are rejected. */
    terminate() {
      worker.terminate();
      for (const p of pending.values()) p.reject(new Error("directedstructure worker terminated"));
      pending.clear();
    },
  };
}

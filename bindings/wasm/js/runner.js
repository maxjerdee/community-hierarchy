// Drives one fit inside a worker: a DsSession stepped in chunks, yielding to
// the event loop between chunks so a "cancel" message can be seen.  No pthreads,
// SharedArrayBuffer or ASYNCIFY required.  Used by worker.js; not public API.

const TARGET_CHUNK_MS = 50;
// How often to attach the current and best-so-far states to a progress report.
const PEEK_INTERVAL_MS = 200;

const yieldToEventLoop = () => new Promise((resolve) => setTimeout(resolve, 0));

function readError(core) {
  return core.UTF8ToString(core._ds_last_error()) || "community-hierarchy: unknown error";
}

function readJSON(core, ptr) {
  if (!ptr) throw new Error(readError(core));
  return JSON.parse(core.UTF8ToString(ptr));
}

// Best (lowest description-length) sample among those recorded so far.
function peekBest(core, handle, samplesSoFar) {
  const ptr = core._ds_session_best(handle);
  if (!ptr) return null;
  const r = readJSON(core, ptr);
  if (!r.partition.length) return null;
  return {
    bestPartition: Int32Array.from(r.partition),
    bestScores: Float64Array.from(r.scores),
    bestDescriptionLength: r.description_length,
    bestNumGroups: r.num_groups,
    samplesSoFar,
  };
}

// The chain's most recently visited state.
function peekCurrent(core, handle) {
  const ptr = core._ds_session_current(handle);
  if (!ptr) return null;
  const r = readJSON(core, ptr);
  if (!r.partition.length) return null;
  return {
    currentPartition: Int32Array.from(r.partition),
    currentScores: Float64Array.from(r.scores),
    currentDescriptionLength: r.description_length,
    currentNumGroups: r.num_groups,
  };
}

/**
 * Run a fit.
 * @param core   the instantiated wasm module
 * @param job    { numNodes, winsFrom, winsTo, winsWeight, tiesFrom, tiesTo, tiesWeight,
 *                 options: { assortative, num_groups, num_groups_fixed,
 *                            merge_split_enabled, burn_sweeps, samples,
 *                            sweeps_per_sample, seed },
 *                 initialPartition: Int32Array|null,
 *                 initialScores: Float64Array|null }
 * @param hooks  { onProgress(progress), isCancelled() -> bool }
 * @returns      result object
 */
export async function runFit(core, job, { onProgress, isCancelled } = {}) {
  const o = job.options;
  const alloc = [];
  const copyIn32 = (arr) => {
    if (!arr || arr.length === 0) return 0;
    const ptr = core._malloc(arr.length * 4);
    alloc.push(ptr);
    core.HEAP32.set(arr, ptr / 4);
    return ptr;
  };
  const copyInF64 = (arr) => {
    if (!arr || arr.length === 0) return 0;
    const ptr = core._malloc(arr.length * 8);
    alloc.push(ptr);
    core.HEAPF64.set(arr, ptr / 8);
    return ptr;
  };

  let handle = 0;
  try {
    const winsFromPtr   = copyIn32(job.winsFrom);
    const winsToPtr     = copyIn32(job.winsTo);
    const winsWeightPtr = copyInF64(job.winsWeight);
    const tiesFromPtr   = copyIn32(job.tiesFrom);
    const tiesToPtr     = copyIn32(job.tiesTo);
    const tiesWeightPtr = copyInF64(job.tiesWeight);
    const partPtr       = copyIn32(job.initialPartition);
    const scoresPtr     = copyInF64(job.initialScores);

    const numWins = job.winsFrom ? job.winsFrom.length : 0;
    const numTies = job.tiesFrom ? job.tiesFrom.length : 0;
    const partLen = job.initialPartition ? job.initialPartition.length : 0;
    const scLen   = job.initialScores   ? job.initialScores.length   : 0;

    handle = core._ds_session_create(
      job.numNodes,
      winsFromPtr, winsToPtr, winsWeightPtr, numWins,
      tiesFromPtr, tiesToPtr, tiesWeightPtr, numTies,
      o.assortative    ? 1 : 0,
      o.num_groups     ?? 0,
      o.num_groups_fixed ? 1 : 0,
      o.merge_split_enabled !== false ? 1 : 0,
      o.burn_sweeps    ?? 1000,
      o.samples        ?? 1000,
      o.sweeps_per_sample ?? 1,
      o.seed           ?? 0,
      partPtr, partLen,
      scoresPtr, scLen,
      // degree_correction_mode: 0 = no DC (fixed at 0), 1 = DC fixed at 0.5 (default)
      o.degree_correction_mode ?? 1,
      // weighted_sbm: 1 = model real-valued edge weights via uniform simplex prior (H_W term)
      o.weighted_sbm ?? 0,
      // independent_coupling: 1 = fix group_depth at 0.0 (hierarchy decoupled from communities)
      o.independent_coupling ?? 0,
      // variation_in / variation_out: a fixed mixing variation (the model presets);
      // omitted or negative = inferred (general_ASBM)
      o.variation_in ?? -1,
      o.variation_out ?? -1);
    if (!handle) throw new Error(readError(core));

    let chunk = 1;
    let cancelled = false;
    let lastPeek = -Infinity;
    let lastProgress = null;
    for (;;) {
      const t0 = performance.now();
      const status = core._ds_session_step(handle, chunk);
      if (status < 0) throw new Error(readError(core));
      const elapsed = performance.now() - t0;
      if (onProgress) {
        const progress = readJSON(core, core._ds_session_progress(handle));
        lastProgress = progress;
        // Both native session_current/session_best are already burn-in-aware
        // (session_current returns a live "still exploring" state throughout
        // burn-in, per their own comments; session_best just returns an empty
        // partition until a sample is actually recorded) -- this used to gate
        // on samples_done > 0 too, which suppressed ANY live peek during the
        // whole burn-in phase even though the native side was designed to
        // support exactly that ("meant to be called on every progress tick
        // for a live 'watch it explore' view" -- see asbm_wasm.cpp). Dropped
        // the redundant gate so a UI can render live results while warming
        // up, not just once sampling starts.
        if (status === 1 || t0 - lastPeek >= PEEK_INTERVAL_MS) {
          lastPeek = t0;
          const current = peekCurrent(core, handle);
          if (current) Object.assign(progress, current);
          const best = peekBest(core, handle, progress.samples_done);
          if (best) Object.assign(progress, best);
        }
        onProgress(progress);
      }
      if (status === 1) break;
      chunk = Math.max(1, Math.min(chunk * 4, Math.round(chunk * TARGET_CHUNK_MS / Math.max(elapsed, 0.01))));
      await yieldToEventLoop();
      if (isCancelled && isCancelled()) { cancelled = true; break; }
    }

    // session_finish (consensus + MDL partition) and session_coincidence (an
    // n x n matrix) are both synchronous native calls with no progress
    // callback of their own -- for a large network they can take a
    // noticeable moment, during which the UI would otherwise be stuck on the
    // last "Samples N / N" tick with no indication anything is still
    // happening. One extra progress tick (marked `summarizing`) plus a yield
    // lets the UI repaint to a "Summarizing..." state before that work starts.
    if (onProgress && !cancelled && lastProgress) {
      onProgress({ ...lastProgress, summarizing: true });
      await yieldToEventLoop();
    }

    const result = readJSON(core, core._ds_session_finish(handle, cancelled ? 1 : 0));
    const n = result.num_nodes;
    const cPtr = core._malloc(n * n * 8);
    alloc.push(cPtr);
    if (core._ds_session_coincidence(handle, cPtr) !== 0) throw new Error(readError(core));
    const coincidence = new Float64Array(core.HEAPF64.buffer, cPtr, n * n).slice();

    return {
      numNodes: n,
      cancelled: result.cancelled,
      numSamples: result.num_samples,
      mdlValue: result.mdl_value,
      mdlPartition: Int32Array.from(result.mdl_partition),
      consensusPartition: Int32Array.from(result.consensus_partition),
      scoreMeans: Float64Array.from(result.score_means),
      scoreStds: Float64Array.from(result.score_stds),
      coincidence,
      samples: result.samples.map((s) => ({
        chain: s.chain,
        numGroups: s.num_groups,
        descriptionLength: s.description_length,
        hAdirGAsNu: s.H_Adir_G_A_s_nu,
        hSGbDepths: s.H_s_G_b_depths,
        individualDepth: s.individual_depth,
        groupDepth: s.group_depth,
        variationIn: s.variation_in,
        variationOut: s.variation_out,
        degreeCorrection: s.degree_correction,
        tiesParameter: s.ties_parameter,
      })),
    };
  } finally {
    if (handle) core._ds_session_free(handle);
    for (const p of alloc) core._free(p);
  }
}

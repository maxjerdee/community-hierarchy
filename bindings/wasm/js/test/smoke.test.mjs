// End-to-end smoke test of the published API: load() -> worker -> wasm -> result.
// Run after building dist/ (see bindings/wasm/CMakeLists.txt):  node --test test/
import { test, after } from "node:test";
import assert from "node:assert/strict";
import { load } from "../index.js";

const m = await load();
after(() => m.terminate());

// Synthetic directed graph: two communities of 8 nodes each.
//   Within each community: symmetric ties (undirected clique) — every pair tied.
//     This creates group cohesion that the SBM can detect.
//   Between communities: A (0-7) dominates B (8-15) — all directed wins.
// The directional asymmetry between communities (A beats B) combined with dense
// within-community ties makes the K=2 {A, B} partition optimal.

const wins = [];
const ties = [];
// Within-community symmetric ties (both communities)
for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) ties.push([i, j]);
for (let i = 8; i < 16; i++) for (let j = i + 1; j < 16; j++) ties.push([i, j]);
// Between communities: A beats B (directed wins only)
for (let i = 0; i < 8; i++) for (let j = 8; j < 16; j++) wins.push([i, j]);

const NUM_NODES = 16;

test("basic fit returns correct structure", async () => {
  const run = m.fit(wins, [], {
    num_nodes: NUM_NODES,
    num_groups: 4,
    assortative: true,
    num_warmup: 200,
    num_samples: 50,
    sweeps_per_sample: 5,
    seed: 42,
  });
  const r = await run.result;

  assert.equal(r.numNodes, NUM_NODES);
  assert.equal(r.cancelled, false);
  assert.equal(r.numSamples, 50);
  assert.equal(r.mdlPartition.length, NUM_NODES);
  assert.equal(r.consensusPartition.length, NUM_NODES);
  assert.equal(r.scoreMeans.length, NUM_NODES);
  assert.equal(r.scoreStds.length, NUM_NODES);
  assert.equal(r.coincidence.length, NUM_NODES * NUM_NODES);
  assert.ok(Number.isFinite(r.mdlValue), "mdlValue should be finite");
  assert.ok(r.mdlValue < 0 || r.mdlValue > 0, "mdlValue should be non-zero");

  // Coincidence matrix should be symmetric with unit diagonal
  for (let i = 0; i < NUM_NODES; i++) {
    const diag = r.coincidence[i * NUM_NODES + i];
    assert.ok(Math.abs(diag - 1.0) < 1e-9, `diagonal[${i}] should be ~1`);
    for (let j = 0; j < NUM_NODES; j++) {
      assert.ok(
        Math.abs(r.coincidence[i * NUM_NODES + j] - r.coincidence[j * NUM_NODES + i]) < 1e-9,
        "coincidence should be symmetric",
      );
    }
  }

  // All samples should have valid description lengths and at least 1 group
  for (const s of r.samples) {
    assert.ok(Number.isFinite(s.descriptionLength), "sample DL should be finite");
    assert.ok(s.numGroups >= 1, "numGroups >= 1");
  }

  // Scores should be ordered: A dominates B, so score_means[0..7] > score_means[8..15]
  const meanA = r.scoreMeans.slice(0, 8).reduce((a, b) => a + b, 0) / 8;
  const meanB = r.scoreMeans.slice(8, 16).reduce((a, b) => a + b, 0) / 8;
  assert.ok(meanA > meanB, `community A mean score (${meanA.toFixed(3)}) should exceed B (${meanB.toFixed(3)})`);
});

test("recovers two planted groups", async () => {
  // Two symmetric tie-cliques with directed wins between them.
  // K=2 fixed, assortative=true: the model should stabilise at {A vs B}.
  const r = await m.fit(wins, ties, {
    num_nodes: NUM_NODES,
    num_groups: 2,
    num_groups_fixed: true,
    assortative: true,
    num_warmup: 300,
    num_samples: 100,
    sweeps_per_sample: 5,
    seed: 7,
  }).result;

  const c = r.consensusPartition;
  // Nodes 0-7 should be in one group, 8-15 in the other
  for (let i = 1; i < 8; i++) assert.equal(c[i], c[0], `node ${i} should share group with node 0`);
  for (let i = 9; i < 16; i++) assert.equal(c[i], c[8], `node ${i} should share group with node 8`);
  assert.notEqual(c[0], c[8], "the two communities should be distinct");
});

test("same seed produces same result", async () => {
  const opts = { num_nodes: NUM_NODES, num_groups: 3, num_warmup: 50, num_samples: 20, sweeps_per_sample: 2, seed: 99 };
  const a = await m.fit(wins, [], opts).result;
  const b = await m.fit(wins, [], opts).result;
  assert.deepEqual([...a.mdlPartition], [...b.mdlPartition]);
  assert.equal(a.mdlValue, b.mdlValue);
  assert.deepEqual(a.samples.map((s) => s.descriptionLength), b.samples.map((s) => s.descriptionLength));
});

test("cancel returns a partial result", async () => {
  let run;
  let seen = 0;
  run = m.fit(wins, [], {
    num_nodes: NUM_NODES, num_groups: 3, num_warmup: 0, num_samples: 100000, sweeps_per_sample: 1, seed: 1,
  }, {
    onProgress: (p) => { if (p.samplesDone > 0 && seen++ === 0) run.cancel(); },
  });
  const r = await run.result;
  assert.equal(r.cancelled, true);
  assert.ok(r.numSamples > 0 && r.numSamples < 100000);
  assert.equal(r.coincidence.length, NUM_NODES * NUM_NODES);
});

test("progress callbacks have correct structure", async () => {
  const progress = [];
  const run = m.fit(wins, [], {
    num_nodes: NUM_NODES, num_groups: 2, num_warmup: 100, num_samples: 30, sweeps_per_sample: 2, seed: 5,
  }, { onProgress: (p) => progress.push(p) });
  await run.result;
  assert.ok(progress.length > 0);
  const last = progress[progress.length - 1];
  assert.equal(last.burnDone, last.burnTotal);
  assert.equal(last.samplesDone, last.samplesTotal);
  assert.equal(last.fraction, 1.0);

  const withBest = progress.filter((p) => p.bestPartition);
  if (withBest.length > 0) {
    for (const p of withBest) {
      assert.equal(p.bestPartition.length, NUM_NODES);
      assert.equal(p.bestScores.length, NUM_NODES);
      assert.ok(Number.isFinite(p.bestDescriptionLength));
    }
  }
});

test("num_groups_fixed prevents group count from changing", async () => {
  const r = await m.fit(wins, [], {
    num_nodes: NUM_NODES, num_groups: 4, num_groups_fixed: true,
    num_warmup: 100, num_samples: 30, sweeps_per_sample: 2, seed: 11,
  }).result;
  for (const s of r.samples) {
    assert.equal(s.numGroups, 4, "num_groups_fixed=true should keep K constant");
  }
});

test("tie edges are accepted", async () => {
  // Add symmetric tie edges between the two communities
  const tieEdges = [[0, 8], [1, 9], [2, 10]];
  const r = await m.fit(wins, tieEdges, {
    num_nodes: NUM_NODES, num_groups: 2, num_warmup: 100, num_samples: 20, sweeps_per_sample: 2, seed: 3,
  }).result;
  assert.equal(r.numNodes, NUM_NODES);
  assert.ok(Number.isFinite(r.mdlValue));
});

test("wins point from winner to loser", async () => {
  // 0 beats everyone, everyone beats 3.
  const w = [[0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [0, 1], [0, 2], [1, 3], [2, 3]];
  const r = await m.fit(w, [], { num_warmup: 50, num_samples: 50, sweeps_per_sample: 2, seed: 1 }).result;
  assert.ok(r.scoreMeans[0] > r.scoreMeans[3], "the source of a win must score higher");
});

test("out-of-range node ids are rejected", () => {
  assert.throws(() => m.fit([[0, 50], [1, 0]], [], { num_nodes: 2 }), RangeError);
  assert.throws(() => m.fit({ from: [0, 1.5], to: [1, 0] }, [], {}), TypeError);
  assert.throws(() => m.fit({ from: [0, 1], to: [1] }, [], {}), TypeError);
  assert.throws(() => m.fit([[0, 1, -2]], [], {}), TypeError);
});

test("the core rejects bad edges and partitions", async () => {
  // Bypass index.js: the worker hands these straight to ds_session_create.
  await assert.rejects(m.fit([[0, 0], [0, 1]], [], { num_warmup: 1, num_samples: 1 }).result, /self-loop/);
  await assert.rejects(
    m.fit([[0, 1], [1, 2]], [], { num_warmup: 1, num_samples: 1, initialPartition: [0, 5, 5] }).result,
    /every label|labels must be/);
  await assert.rejects(
    m.fit([[0, 1], [1, 2]], [], { num_warmup: 1, num_samples: 1, initialPartition: [0, 1] }).result,
    /length num_nodes/);
});

test("initial scores are used as given", async () => {
  const s = [3, 0, 0, -3];
  const r = await m.fit([[0, 1], [1, 2], [2, 3]], [], {
    num_warmup: 0, num_samples: 1, sweeps_per_sample: 0, seed: 1, initialScores: s,
  }).result;
  // Zero sweeps: the one recorded sample is the starting state (Python does
  // not rescale user scores either; it starts individual_depth at their RMS).
  assert.deepEqual(Array.from(r.scoreMeans), s);
});

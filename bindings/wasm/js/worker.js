// The community_hierarchy worker: loads the wasm core once and runs fits on request.
// Runs as a browser module Web Worker or a Node worker_threads Worker.
//
// Protocol (all messages are plain objects):
//   worker -> host  {type: "ready"}
//   host -> worker  {type: "fit", id, job}           job: see runner.js runFit
//   worker -> host  {type: "progress", id, progress}
//   worker -> host  {type: "result", id, result}     (coincidence buffer transferred)
//   worker -> host  {type: "error", id, message}
//   host -> worker  {type: "cancel", id}             stop early; a partial result follows
import createDsCore from "./dist/ds_core.mjs";
import { runFit } from "./runner.js";

const isBrowserWorker = typeof WorkerGlobalScope !== "undefined" && typeof self !== "undefined";
let port;
if (isBrowserWorker) {
  port = { post: (m, t) => self.postMessage(m, t || []), on: (fn) => { self.onmessage = (e) => fn(e.data); } };
} else {
  const nodeModule = "node:worker_threads";
  const { parentPort } = await import(/* @vite-ignore */ nodeModule);
  port = { post: (m, t) => parentPort.postMessage(m, t || []), on: (fn) => parentPort.on("message", fn) };
}

const core = await createDsCore();
const cancelled = new Set();
const queue = [];
let running = false;

async function drain() {
  if (running) return;
  running = true;
  while (queue.length) {
    const { id, job } = queue.shift();
    try {
      const result = await runFit(core, job, {
        onProgress: (progress) => port.post({ type: "progress", id, progress }),
        isCancelled: () => cancelled.has(id),
      });
      const transfer = [result.coincidence.buffer];
      port.post({ type: "result", id, result }, transfer);
    } catch (err) {
      port.post({ type: "error", id, message: String(err && err.message ? err.message : err) });
    } finally {
      cancelled.delete(id);
    }
  }
  running = false;
}

port.on((msg) => {
  if (msg.type === "fit") {
    queue.push(msg);
    drain();
  } else if (msg.type === "cancel") {
    cancelled.add(msg.id);
  }
});

port.post({ type: "ready" });

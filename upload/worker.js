// One compute unit: CQT frames (plain JS) and BassNet windows (onnxruntime-web).
// The page fetches this file and core.js as text and runs them from a blob: URL
// (the viewer's sandbox has an opaque origin, so module workers cannot load the
// page's own files directly); the import of './core.js' below is replaced by the
// inlined core.  Used either as a worker, or in the page itself through
// makeHandler() when workers are unavailable.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.bundle.min.mjs';
import { makeKernels, cqtFrames, prefix, windowInput, BINS } from './core.js';

export function makeHandler(post) {
  let S = null, cfg = null, ks = null, P = null, zKey = null, zmap = null;

  async function init(m) {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmBinary = m.wasm;
    cfg = m.cfg;
    const opts = { executionProviders: [m.ep], graphOptimizationLevel: 'all' };
    S = {};
    for (const k of ['enc', 'zmap', 'dec']) S[k] = await ort.InferenceSession.create(new Uint8Array(m.models[k]), opts);
  }

  async function run(m) {
    const key = m.z.join(',');
    if (key !== zKey) {
      zmap = (await S.zmap.run({ z: new ort.Tensor('float32', Float32Array.from(m.z), [1, m.z.length]) })).zmap;
      zKey = key;
    }
    const ts = m.ts, B = ts.length, per = cfg.channels * cfg.compressed * BINS;
    const x = new Float32Array(B * per);
    for (let i = 0; i < B; i++) windowInput(P, ts[i], cfg, x, i * per);
    const { ex } = await S.enc.run({ x: new ort.Tensor('float32', x, [B, cfg.channels, cfg.compressed, BINS]) });
    const o = await S.dec.run({ ex, zmap });
    const [pitch, level, onset] = await Promise.all([o.pitch.getData(), o.level.getData(), o.onset.getData()]);
    const T = cfg.compressed, E = m.every, n = B * E;
    const frames = new Int32Array(n), midi = new Uint8Array(n), lv = new Float32Array(n), on = new Float32Array(n);
    const sp = new Float32Array(n * 88);            // semitone posterior (semitone s owns bins 3s-1, 3s, 3s+1)
    let q = 0;
    for (let b = 0; b < B; b++) {
      for (let e = E - 1; e >= 0; e--) {            // the window's newest E frames, oldest first
        const tt = T - 1 - e, f = ts[b] - e;
        const base = (b * T + tt) * BINS;
        let best = 0;
        for (let k = 1; k < BINS; k++) if (pitch[base + k] > pitch[base + best]) best = k;
        const mx = pitch[base + best];
        let z = 0;
        for (let k = 0; k < BINS; k++) z += Math.exp(pitch[base + k] - mx);
        for (let k = 0; k < BINS; k++) sp[q * 88 + Math.min(87, Math.floor((k + 1) / 3))] += Math.exp(pitch[base + k] - mx) / z;
        frames[q] = f;
        midi[q] = Math.round(21 + best / 3);       // 36 bins per octave from A0 (MIDI 21)
        lv[q] = 1 / (1 + Math.exp(-level[b * T + tt]));
        on[q] = 1 / (1 + Math.exp(-onset[b * T + tt]));
        q++;
      }
    }
    return { frames, midi, lv, on, post: sp };
  }

  return async (m) => {
    try {
      if (m.type === 'init') { await init(m); post({ type: 'ready' }); }
      else if (m.type === 'cqt') {
        ks ??= makeKernels();
        const out = cqtFrames(m.audio, ks, m.a, m.b);
        post({ type: 'cqt', a: m.a, b: m.b, out }, [out.buffer]);
      } else if (m.type === 'feat') { P = prefix(m.feat, m.n); post({ type: 'featok' }); }
      else if (m.type === 'run') {
        const r = await run(m);
        post({ type: 'res', gen: m.gen, ...r }, [r.frames.buffer, r.midi.buffer, r.lv.buffer, r.on.buffer, r.post.buffer]);
      }
    } catch (err) {
      post({ type: 'error', where: m.type, msg: String(err && err.message || err) });
    }
  };
}

if (typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) {
  const h = makeHandler((msg, transfer) => postMessage(msg, transfer || []));
  onmessage = e => h(e.data);
}

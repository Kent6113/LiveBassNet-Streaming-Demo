// One compute unit: CQT frames (plain JS) and the multi-rate streaming BassNet
// ("B", bassnet/multirate.py) through onnxruntime-web.  The page fetches this file
// and core.js as text and runs them from a blob: URL (the viewer's sandbox has an
// opaque origin, so module workers cannot load the page's own files directly); the
// import of './core.js' below is replaced by the inlined core.  Used either as a
// worker, or in the page itself through makeHandler() when workers are unavailable.
//
// The model runs in chunks of T = 64 frames whose first frame is a multiple of 64:
// chunk.onnx reads the log1p CQT of the H frames before the chunk and of the chunk
// itself (zeros before the song) and the beat features of the chunk, and returns the
// outputs of all T frames.  Frame t only uses frames <= t, so this equals running
// the model frame by frame (web/export_multirate.py checks it against predict_song).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.bundle.min.mjs';
import { makeKernels, cqtFrames, BINS, decodeHmm, barFilter } from './core.js';

export function makeHandler(post) {
  let S = null, cfg = null, ks = null, F = null, N = 0, BF = null, zKey = null, zmap = null, beatnet = null, hmm = null;

  async function init(m) {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmBinary = m.wasm;
    cfg = m.cfg;
    const opts = { executionProviders: [m.ep], graphOptimizationLevel: 'all' };
    S = {};
    for (const k of ['zmap', 'chunk']) S[k] = await ort.InferenceSession.create(new Uint8Array(m.models[k]), opts);
    if (m.beatnet) { beatnet = await ort.InferenceSession.create(new Uint8Array(m.beatnet), opts); hmm = decodeHmm(m.hmm); }
  }

  // beat features of a whole song: the causal beat network in one pass (equal to
  // frame-by-frame use), then the causal bar-pointer filter
  async function beats(m) {
    const T = m.n;
    const { p } = await beatnet.run({ x: new ort.Tensor('float32', m.feat, [1, T, BINS]) });
    const act = await p.getData();
    return barFilter(act, T, hmm);
  }

  const sig = v => 1 / (1 + Math.exp(-v));

  // m.ts: the last frame of each chunk (t = 64 k + 63)
  async function run(m) {
    const key = m.z.join(',');
    if (key !== zKey) {
      zmap = (await S.zmap.run({ z: new ort.Tensor('float32', Float32Array.from(m.z), [1, m.z.length]) })).zmap;
      zKey = key;
    }
    const H = cfg.history, T = cfg.chunk, B = m.ts.length, L = H + T;
    const x = new Float32Array(B * L * BINS), b = new Float32Array(B * T * 7);
    for (let i = 0; i < B; i++) {
      const s = m.ts[i] - T + 1;
      const a = Math.max(0, s - H), e = Math.min(N, s + T);          // song frames present
      if (e > a) x.set(F.subarray(a * BINS, e * BINS), (i * L + (a - (s - H))) * BINS);
      const be = Math.min(N, s + T);
      if (BF && be > s) b.set(BF.subarray(s * 7, be * 7), i * T * 7);
    }
    const o = await S.chunk.run({ x: new ort.Tensor('float32', x, [B, L, BINS]), b: new ort.Tensor('float32', b, [B, T, 7]), zmap });
    const [pitch, level, onset, voicing] = await Promise.all([o.pitch.getData(), o.level.getData(), o.onset.getData(), o.voicing.getData()]);
    const n = B * T;
    const frames = new Int32Array(n), midi = new Uint8Array(n), lv = new Float32Array(n), on = new Float32Array(n), vo = new Float32Array(n);
    const sp = new Float32Array(n * 88);            // semitone posterior (semitone s owns bins 3s-1, 3s, 3s+1)
    for (let q = 0; q < n; q++) {
      const base = q * BINS;
      let best = 0;
      for (let k = 1; k < BINS; k++) if (pitch[base + k] > pitch[base + best]) best = k;
      const mx = pitch[base + best];
      let z = 0;
      for (let k = 0; k < BINS; k++) z += Math.exp(pitch[base + k] - mx);
      for (let k = 0; k < BINS; k++) sp[q * 88 + Math.min(87, Math.floor((k + 1) / 3))] += Math.exp(pitch[base + k] - mx) / z;
      frames[q] = m.ts[Math.floor(q / T)] - T + 1 + (q % T);
      midi[q] = Math.round(21 + best / 3);         // 36 bins per octave from A0 (MIDI 21)
      lv[q] = sig(level[q]); on[q] = sig(onset[q]); vo[q] = sig(voicing[q]);
    }
    return { frames, midi, lv, on, vo, post: sp };
  }

  return async (m) => {
    try {
      if (m.type === 'init') { await init(m); post({ type: 'ready' }); }
      else if (m.type === 'cqt') {
        ks ??= makeKernels();
        const out = cqtFrames(m.audio, ks, m.a, m.b);
        post({ type: 'cqt', a: m.a, b: m.b, out }, [out.buffer]);
      } else if (m.type === 'feat') { F = m.feat; N = m.n; BF = m.beats || null; post({ type: 'featok' }); }
      else if (m.type === 'beats') { const bf = await beats(m); post({ type: 'beats', bf }, [bf.buffer]); }
      else if (m.type === 'run') {
        const r = await run(m);
        post({ type: 'res', gen: m.gen, ...r }, [r.frames.buffer, r.midi.buffer, r.lv.buffer, r.on.buffer, r.vo.buffer, r.post.buffer]);
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

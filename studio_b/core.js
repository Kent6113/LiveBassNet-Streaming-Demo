// Feature pipeline of LiveBassNet, as in training (bassnet/features.py, bassnet/multires.py).
//
// CQT: nnAudio CQT1992v2 kernels (16 kHz, hop 160, 264 bins from A0 at 36 per
// octave, periodic Hann, L1-normalised, librosa scaling sqrt(length)), causal and
// right-aligned: frame n uses the `length` samples that end at n * hop (exclusive).
// Output log1p|CQT|, frames = floor(samples / hop) + 1.

export const SR = 16000, HOP = 160, BINS = 264, BPO = 36, FMIN = 27.5;

export function makeKernels() {
  const Q = 1 / (Math.pow(2, 1 / BPO) - 1);
  const ks = [];
  for (let k = 0; k < BINS; k++) {
    const f = FMIN * Math.pow(2, k / BPO);
    const l = Math.ceil(Q * SR / f);
    const re = new Float32Array(l), im = new Float32Array(l);
    let norm = 0;
    const w = new Float64Array(l);
    for (let n = 0; n < l; n++) { w[n] = 0.5 - 0.5 * Math.cos(2 * Math.PI * n / l); norm += w[n] / l; }
    const scale = Math.sqrt(l) / (l * norm);
    const r0 = Math.floor(-l / 2);
    for (let n = 0; n < l; n++) {
      const ph = (r0 + n) * 2 * Math.PI * f / SR;
      re[n] = w[n] * Math.cos(ph) * scale;
      im[n] = w[n] * Math.sin(ph) * scale;
    }
    ks.push({ l, re, im });
  }
  return ks;
}

// log1p|CQT| of frames [a, b) of `audio` (Float32Array at 16 kHz) -> Float32Array (b - a) * BINS
export function cqtFrames(audio, ks, a, b) {
  const out = new Float32Array((b - a) * BINS);
  for (let n = a; n < b; n++) {
    const end = n * HOP;
    for (let k = 0; k < BINS; k++) {
      const { l, re, im } = ks[k];
      const s0 = end - l;
      let j0 = 0;
      if (s0 < 0) j0 = -s0;
      let sr = 0, si = 0;
      for (let j = j0; j < l; j++) {
        const x = audio[s0 + j];
        sr += re[j] * x; si += im[j] * x;
      }
      out[(n - a) * BINS + k] = Math.log1p(Math.sqrt(sr * sr + si * si));
    }
  }
  return out;
}

export function nFrames(samples) { return Math.floor(samples / HOP) + 1; }

// prefix sums over time, per bin: P[(t) * BINS + k] = sum of feat[u][k] for u < t
export function prefix(feat, n) {
  const P = new Float64Array((n + 1) * BINS);
  for (let t = 0; t < n; t++)
    for (let k = 0; k < BINS; k++) P[(t + 1) * BINS + k] = P[t * BINS + k] + feat[t * BINS + k];
  return P;
}

// Model input for the window ending at frame t (inclusive): (channels, comp, BINS),
// channel c averages blocks of 2^c frames, all channels end at t; frames before 0
// are zeros.  Then BassNet.prep_x ("log" input scale): log(max(expm1(v), amin)),
// standardised.
export function windowInput(P, t, cfg, out, off) {
  const { compressed: comp, expanded: exp, channels, x_amin, x_mean, x_std } = cfg;
  const first = t - exp + 1;
  let o = off;
  for (let c = 0; c < channels; c++) {
    const fct = 1 << c, start = first + exp - comp * fct;
    for (let j = 0; j < comp; j++) {
      const a = Math.max(0, start + j * fct), b = Math.max(0, start + (j + 1) * fct);
      for (let k = 0; k < BINS; k++) {
        const v = (P[b * BINS + k] - P[a * BINS + k]) / fct;
        out[o++] = cfg.input_scale === 'log'
          ? (Math.log(Math.max(Math.expm1(Math.max(v, 0)), x_amin)) - x_mean) / x_std
          : v;
      }
    }
  }
  return o;
}

// Joint beat / downbeat tracking (bassnet/slakh/beat_track.py, filter_bars): a
// bar-pointer HMM (3/4 + 4/4, 55-215 bpm; madmom's state space) run as a causal
// forward filter over the beat network's activations act (T x 2: P(beat),
// P(downbeat)).  Returns the per-frame beat features of the beat-conditioned
// model (T x 7): sin/cos of the beat phase, sin/cos of the sixteenth phase,
// time to the next beat (s, <= 1), sin/cos of the bar phase.
export function decodeHmm(json) {
  const b64 = s => { const bin = atob(s), a = new Uint8Array(bin.length); for (let i = 0; i < a.length; i++) a[i] = bin.charCodeAt(i); return a.buffer; };
  const f = json.meta.files, get = (k, T) => new T(b64(json[k]));
  return { n: json.meta.states, lambda: json.meta.observation_lambda,
    pointers: get('pointers', Int32Array), states: get('states', Int32Array), probs: get('probs', Float32Array),
    position: get('position', Float32Array), interval: get('interval', Float32Array),
    nbeats: get('nbeats', Float32Array), obs: get('obs', Int32Array) };
}

export function barFilter(act, T, hmm) {
  const { n, pointers, states, probs, position, interval, nbeats, obs, lambda } = hmm;
  const cb = new Float32Array(n), sb = new Float32Array(n), cr = new Float32Array(n), sr = new Float32Array(n);
  for (let s = 0; s < n; s++) {
    const ph = 2 * Math.PI * (position[s] % 1), bar = 2 * Math.PI * position[s] / nbeats[s];
    cb[s] = Math.cos(ph); sb[s] = Math.sin(ph); cr[s] = Math.cos(bar); sr[s] = Math.sin(bar);
  }
  let f = new Float64Array(n).fill(1 / n), g = new Float64Array(n);
  const out = new Float32Array(T * 7), dens = new Float64Array(3);
  for (let t = 0; t < T; t++) {
    let pb = Math.min(Math.max(act[2 * t], 1e-6), 1), pd = Math.min(Math.max(act[2 * t + 1], 1e-6), 1);
    const tot = pb + pd; if (tot > 1 - 1e-6) { pb *= (1 - 1e-6) / tot; pd *= (1 - 1e-6) / tot; }
    dens[0] = (1 - pb - pd) / (lambda - 1); dens[1] = pb; dens[2] = pd;
    let z = 0;
    for (let s = 0; s < n; s++) {
      let acc = 0;
      for (let j = pointers[s]; j < pointers[s + 1]; j++) acc += probs[j] * f[states[j]];
      acc *= dens[obs[s]]; g[s] = acc; z += acc;
    }
    let mc = 0, ms = 0, per = 0, bc = 0, bs = 0;
    for (let s = 0; s < n; s++) { const v = g[s] / z; g[s] = v; mc += v * cb[s]; ms += v * sb[s]; per += v * interval[s]; bc += v * cr[s]; bs += v * sr[s]; }
    [f, g] = [g, f];
    const phase = ((Math.atan2(ms, mc) / (2 * Math.PI)) % 1 + 1) % 1, bar = Math.atan2(bs, bc), o = t * 7;
    out[o] = Math.sin(2 * Math.PI * phase); out[o + 1] = Math.cos(2 * Math.PI * phase);
    out[o + 2] = Math.sin(8 * Math.PI * phase); out[o + 3] = Math.cos(8 * Math.PI * phase);
    out[o + 4] = Math.min((1 - phase) * per / 100, 1);
    out[o + 5] = Math.sin(bar); out[o + 6] = Math.cos(bar);
  }
  return out;
}

// beat features of the full-resolution channel's frames t - comp + 1 .. t (zeros before the song)
export function windowBeats(BF, t, comp, out, off) {
  for (let j = 0; j < comp; j++) {
    const f = t - comp + 1 + j;
    for (let k = 0; k < 7; k++) out[off + j * 7 + k] = f >= 0 ? BF[f * 7 + k] : 0;
  }
}

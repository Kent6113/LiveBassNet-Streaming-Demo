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

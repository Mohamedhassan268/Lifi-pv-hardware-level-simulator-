/**
 * Tiny DSP helpers used by the result panels.
 *
 * - radix2FFT: iterative in-place Cooley-Tukey on a power-of-2 buffer.
 * - powerSpectrumDb: zero-pads to next power of 2, windows, FFTs, returns
 *   a one-sided dB spectrum and matching frequency axis (Hz).
 *
 * We don't pull in fft.js / dsp.js — those add a dependency for ~50 lines
 * of code, and our Avast TLS quirk makes npm installs painful. The naive
 * O(N log N) FFT below handles 4 k–8 k point arrays in ~10 ms.
 */

export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * In-place radix-2 FFT. Mutates `re` and `im`. Both arrays must be the same
 * length and a power of 2.
 */
export function radix2FFT(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  if ((n & (n - 1)) !== 0) {
    throw new Error(`radix2FFT: length ${n} is not a power of 2`);
  }

  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  // Butterflies
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < half; k++) {
        const aRe = re[i + k];
        const aIm = im[i + k];
        const bRe = re[i + k + half] * curRe - im[i + k + half] * curIm;
        const bIm = re[i + k + half] * curIm + im[i + k + half] * curRe;
        re[i + k] = aRe + bRe;
        im[i + k] = aIm + bIm;
        re[i + k + half] = aRe - bRe;
        im[i + k + half] = aIm - bIm;
        const tmpRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = tmpRe;
      }
    }
  }
}

/**
 * Compute the one-sided power spectrum of a real signal (Hann-windowed).
 *
 * @param x    — input samples (any length; will be zero-padded to next pow2)
 * @param fs   — sample rate (Hz)
 * @returns    — { freq: Hz axis (length N/2+1), dB: power in dB }
 */
export function powerSpectrumDb(
  x: number[] | Float64Array,
  fs: number,
): { freq: Float64Array; dB: Float64Array } {
  const N = nextPow2(x.length);
  const re = new Float64Array(N);
  const im = new Float64Array(N);

  // Copy + apply Hann window. Zero-pad anything past x.length.
  const L = x.length;
  for (let i = 0; i < L; i++) {
    const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / Math.max(L - 1, 1)));
    re[i] = x[i] * w;
  }

  radix2FFT(re, im);

  const half = N / 2 + 1;
  const freq = new Float64Array(half);
  const dB = new Float64Array(half);
  const eps = 1e-30;
  for (let k = 0; k < half; k++) {
    freq[k] = (k * fs) / N;
    const mag2 = re[k] * re[k] + im[k] * im[k];
    dB[k] = 10 * Math.log10(mag2 + eps);
  }
  return { freq, dB };
}

/**
 * Estimate sample rate from a time-axis array (seconds). Returns Hz.
 */
export function inferSampleRate(time: number[]): number {
  if (time.length < 2) return 0;
  const dt = time[1] - time[0];
  return dt > 0 ? 1 / dt : 0;
}

// ---------------------------------------------------------------------------
// Analytical BER companion — mirrors cosim.modulation.predict_ber.
// erfc and Q-function via Abramowitz–Stegun 7.1.26 (max error ~1.5e-7).
// ---------------------------------------------------------------------------

/** Complementary error function. */
export function erfc(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t) *
      Math.exp(-ax * ax);
  return 1 - sign * y;
}

/** Q-function = 0.5 · erfc(x / √2). */
export function qFunction(x: number): number {
  return 0.5 * erfc(x / Math.SQRT2);
}

/**
 * Theoretical BER vs linear SNR for the modulation schemes the simulator
 * supports. Mirrors cosim/modulation.py::predict_ber. Returns NaN for
 * schemes we can't predict in closed form here (PWM-ASK needs run-specific
 * params; OFDM depends on QAM order).
 */
export function analyticalBer(modulation: string, snrLinear: number): number {
  if (!Number.isFinite(snrLinear) || snrLinear <= 0) return NaN;
  switch (modulation) {
    case "OOK":
    case "OOK_Manchester":
      // BER_OOK = 0.5 · erfc(√(SNR/2))
      return 0.5 * erfc(Math.sqrt(snrLinear / 2));
    case "BFSK":
      // BER_BFSK_coherent = Q(√SNR); non-coherent = 0.5·exp(-SNR/2)
      return 0.5 * Math.exp(-snrLinear / 2);
    default:
      return NaN;
  }
}

/**
 * Wilson score 95% confidence interval for a binomial proportion
 * (k errors out of n bits). Returns [lo, hi] proportions.
 */
export function wilsonCi(k: number, n: number, z = 1.96): [number, number] {
  if (n <= 0) return [0, 1];
  const p = k / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const centre = (p + z2 / (2 * n)) / denom;
  const radius = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom;
  return [Math.max(0, centre - radius), Math.min(1, centre + radius)];
}

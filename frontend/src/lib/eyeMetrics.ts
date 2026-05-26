/**
 * Eye-diagram quantitative metrics computed on the existing V_rx waveform
 * + bits_tx sequence (no backend changes).
 *
 *   eyeOpeningV     — minimum-1 minus maximum-0 voltage at the bit centre
 *   eyeSnrDb        — 20·log10(opening / σ_noise) where σ_noise is the
 *                     within-class standard deviation of bit-centre samples
 *   jitterSigmaSec  — σ of zero-crossing times around nominal bit edges
 *                     (V_rx − threshold) using linear interpolation
 *   edgeCount       — number of transitions detected
 */

export interface EyeMetrics {
  eyeOpeningV: number;
  eyeSnrDb: number;
  jitterSigmaSec: number;
  edgeCount: number;
}

export function computeEyeMetrics(
  time: number[],
  vRx: number[],
  bitsTx: number[],
  dataRateBps: number,
): EyeMetrics | null {
  if (
    !time.length ||
    !vRx.length ||
    !bitsTx.length ||
    !dataRateBps ||
    time.length !== vRx.length
  ) {
    return null;
  }

  const dt = time[1] - time[0];
  if (!(dt > 0)) return null;
  const fs = 1 / dt;
  const tBit = 1 / dataRateBps;
  const samplesPerBit = Math.max(2, Math.floor(fs * tBit));

  // Collect V_rx at bit centres, split by symbol.
  const v0: number[] = [];
  const v1: number[] = [];
  const usableBits = Math.min(bitsTx.length, Math.floor(vRx.length / samplesPerBit));
  for (let i = 0; i < usableBits; i++) {
    const idx = Math.floor((i + 0.5) * samplesPerBit);
    if (idx >= vRx.length) break;
    (bitsTx[i] === 0 ? v0 : v1).push(vRx[idx]);
  }
  if (v0.length < 2 || v1.length < 2) return null;

  const mean0 = mean(v0);
  const mean1 = mean(v1);
  const sd0 = stddev(v0, mean0);
  const sd1 = stddev(v1, mean1);

  // Eye opening: edge-to-edge between the two clouds.
  const eyeOpeningV = Math.max(0, Math.min(...v1) - Math.max(...v0));
  // Within-class noise σ (pooled).
  const sigmaNoise = (sd0 + sd1) / 2;
  const eyeSnrDb =
    sigmaNoise > 1e-12 && eyeOpeningV > 0
      ? 20 * Math.log10(eyeOpeningV / sigmaNoise)
      : 0;

  // Jitter: zero-crossings of (V_rx − threshold), where threshold is the
  // midpoint of mean0 / mean1. Compare against the nominal edge time
  // i * tBit for each detected transition in bitsTx.
  const threshold = (mean0 + mean1) / 2;
  const jitter: number[] = [];
  for (let i = 1; i < usableBits; i++) {
    if (bitsTx[i] === bitsTx[i - 1]) continue;
    const nominalEdge = i * tBit;
    // Search a half-bit window around the nominal edge for a sign change.
    const winStart = Math.max(0, Math.floor((nominalEdge - 0.5 * tBit) / dt));
    const winEnd = Math.min(vRx.length - 1, Math.ceil((nominalEdge + 0.5 * tBit) / dt));
    const tCross = findCrossing(time, vRx, threshold, winStart, winEnd);
    if (tCross !== null) jitter.push(tCross - nominalEdge);
  }
  const jitterSigmaSec = jitter.length > 1 ? stddev(jitter, mean(jitter)) : 0;

  return {
    eyeOpeningV,
    eyeSnrDb,
    jitterSigmaSec,
    edgeCount: jitter.length,
  };
}

function mean(xs: number[]): number {
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function stddev(xs: number[], m: number): number {
  let s = 0;
  for (const x of xs) {
    const d = x - m;
    s += d * d;
  }
  return Math.sqrt(s / Math.max(1, xs.length - 1));
}

function findCrossing(
  time: number[],
  vRx: number[],
  threshold: number,
  i0: number,
  i1: number,
): number | null {
  for (let i = i0; i < i1; i++) {
    const a = vRx[i] - threshold;
    const b = vRx[i + 1] - threshold;
    if (a === 0) return time[i];
    if (a * b < 0) {
      // Linear interpolation.
      const frac = a / (a - b);
      return time[i] + frac * (time[i + 1] - time[i]);
    }
  }
  return null;
}

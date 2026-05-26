/**
 * TypeScript mirror of cosim/probes.py ProbeSpec.
 *
 * Source of truth lives in Python. To regenerate this list, hit GET /api/probes
 * — the App boot fetches it once at startup and stores it in pipelineExecutionStore
 * so we don't have to keep this hand-maintained file in lockstep with Python.
 *
 * The literal-type union of known IDs below is purely a typing aid for editor
 * autocomplete in places that hardcode an ID; runtime probe IDs are always
 * strings keyed against the registry fetched from the backend.
 */

export type ProbeStage = "TX" | "Channel" | "RX";

export type ProbeKind = "array" | "scalar";

export interface ProbeSpec {
  id: string;
  stage: ProbeStage;
  kind: ProbeKind;
  units: string;
  dtype: "float64" | "float32" | "uint8" | "int32";
  label: string;
  description: string;
  equation_key: string | null;
  citation: string | null;
  edge_id: string | null;
}

/** Common known IDs — autocomplete only, not exhaustive. */
export type KnownProbeId =
  | "tx.bits"
  | "tx.P_tx"
  | "channel.G_los"
  | "channel.G_diffuse"
  | "channel.beer_lambert"
  | "channel.G_total"
  | "channel.P_rx"
  | "rx.I_ph"
  | "rx.I_ph_noisy"
  | "rx.noise.shot"
  | "rx.noise.thermal"
  | "rx.noise.ambient"
  | "rx.noise.amplifier"
  | "rx.noise.adc"
  | "rx.noise.processing"
  | "rx.V_sense"
  | "rx.V_ina"
  | "rx.V_bpf"
  | "rx.V_comp"
  | "rx.bits_rx"
  | "rx.snr_db";

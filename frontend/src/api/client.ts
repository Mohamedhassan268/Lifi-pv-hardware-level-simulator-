/**
 * Thin fetch wrapper. In dev VITE_BACKEND_URL falls back to localhost:8000;
 * in a packaged Tauri build the backend port is injected as window.__BACKEND_PORT
 * by the Rust sidecar lifecycle (see Phase 5).
 */

import { getBackendBase } from "@/api/backend";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${getBackendBase()}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ---- Types matching backend/schemas.py ----

export type ConfigDict = Record<string, unknown>;

export type ValidationIssueLevel = "error" | "warning" | "info";

export interface ValidationIssue {
  level: ValidationIssueLevel;
  field: string;       // dotted path, e.g. "tx.bias_current_A"
  message: string;
  suggestion?: string;
  rule_id?: string;
}

export interface ValidateConfigResponse {
  valid: boolean;
  errors: string[];
  issues: ValidationIssue[];
  normalized?: ConfigDict;
}

export interface KicadExportResult {
  preset: string;
  schematic_path: string;
  bom_path: string;
  component_count: number;
  net_count: number;
  warnings: string[];
}

export interface SchematicDrawing {
  index: number;
  name: string;
}

export interface SchematicPresetEntry {
  key: string;
  label: string;
  drawings: SchematicDrawing[];
}

export interface SchematicIndex {
  presets: SchematicPresetEntry[];
}

export interface LinkBudgetRequest {
  distance_m: number;
  tx_angle_deg?: number;
  rx_tilt_deg?: number;
  fov_half_angle_deg?: number;
  led_half_angle_deg?: number;
  led_radiated_power_mW?: number;
  sc_area_cm2?: number;
  sc_responsivity?: number;
  modulation_depth?: number;
  data_rate_bps?: number;
  r_sense_ohm?: number;
}

export interface LinkBudgetResponse {
  channel_gain: number;
  p_rx_W: number;
  p_rx_uW: number;
  i_ph_A: number;
  i_ph_uA: number;
  lambertian_order: number;
  snr_estimate_dB: number;
}

export interface ComponentSummary {
  part: string;
  display_name?: string;
  category: string;
  class: string;
  config_field: string;
}

export interface SpicePort {
  name: string;
  role: string;
}

export interface ComponentDetail {
  part: string;
  display_name?: string;
  class: string;
  category: string;
  config_field: string;
  parameters: Record<string, unknown>;
  spice_ports?: SpicePort[];
}

export interface StandardsProfileSummary {
  id: string;
  label: string;
  spec_section: string;
  notes: string;
  overrides: Record<string, unknown>;
}

export interface FeatureMeta {
  label: string;
  unit: string;
  group: string;
  desc: string;
}
export type FeatureSchema = Record<string, FeatureMeta>;
export interface FeatureSchemaResponse {
  features: FeatureSchema;
  sweepable: string[];
}
export type FeatureRow = Record<string, number | string | null>;
export interface FeatureDatasetRequest {
  config: ConfigDict;
  param: string;
  values: number[];
  n_bits?: number;
}

export interface CoverageRequest {
  preset?: string;
  luminaires: { x: number; y: number }[];
  total_power_W: number;
  height_m: number;
  half_extent_m: number;
  grid: number;
  half_angle_deg: number;
  r_sense_ohm: number;
  node_budget_uW: number;
  dcdc_eff?: number;
  ber_thresh?: number;
}

export interface CoverageResponse {
  xs: number[];
  BER: number[][];
  harvest_uW: number[][];
  snr_dB: number[][];
  deployable: boolean[][];
  data_cov_pct: number;
  deployable_pct: number;
  harvest_min_uW: number;
  harvest_max_uW: number;
}

export const api = {
  listPresets: () => request<string[]>("/api/presets"),
  coverage: (req: CoverageRequest) =>
    request<CoverageResponse>("/api/coverage", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  getFeatureSchema: () => request<FeatureSchemaResponse>("/api/features/schema"),
  featureDatasetJson: (req: FeatureDatasetRequest) =>
    request<{ param: string; rows: FeatureRow[]; schema: FeatureSchema }>(
      "/api/features/dataset",
      { method: "POST", body: JSON.stringify({ ...req, format: "json" }) },
    ),
  // CSV/JSONL come back as a plain-text body for download (not JSON).
  featureDatasetText: async (
    req: FeatureDatasetRequest,
    format: "csv" | "jsonl",
  ): Promise<string> => {
    const res = await fetch(`${getBackendBase()}/api/features/dataset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...req, format }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
    return res.text();
  },
  getPreset: (name: string) => request<ConfigDict>(`/api/presets/${name}`),
  listStandards: () => request<StandardsProfileSummary[]>("/api/standards"),
  getStandard: (id: string) =>
    request<StandardsProfileSummary & { criteria: string[] }>(
      `/api/standards/${encodeURIComponent(id)}`,
    ),
  validateConfig: (cfg: ConfigDict) =>
    request<ValidateConfigResponse>(
      "/api/config/validate",
      { method: "POST", body: JSON.stringify(cfg) },
    ),
  listKicadPresets: () =>
    request<{ presets: string[] }>("/api/kicad/presets"),
  exportKicad: (preset: string) =>
    request<KicadExportResult>("/api/kicad/export", {
      method: "POST",
      body: JSON.stringify({ preset }),
    }),
  kicadDownloadUrl: (preset: string, kind: "sch" | "bom") =>
    `${getBackendBase()}/api/kicad/download/${encodeURIComponent(preset)}/${kind}`,
  listSchematics: () =>
    request<SchematicIndex>("/api/schematic/index"),
  schematicSvgUrl: (preset: string, drawingIndex: number) =>
    `${getBackendBase()}/api/schematic/${encodeURIComponent(preset)}/${drawingIndex}`,
  listComponents: () => request<ComponentSummary[]>("/api/components"),
  listComponentCategories: () => request<string[]>("/api/components/categories"),
  getComponent: (part: string) =>
    request<ComponentDetail>(`/api/components/${encodeURIComponent(part)}`),
  linkBudget: (req: LinkBudgetRequest) =>
    request<LinkBudgetResponse>("/api/link-budget", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  simulateSchematic: (graph: unknown) =>
    request<SchematicSimResponse>("/api/schematic-sim", {
      method: "POST",
      body: JSON.stringify(graph),
    }),
  parseFirmware: (role: "tx" | "rx", source: string, filename?: string) =>
    request<FirmwareParseResponse>("/api/firmware/parse", {
      method: "POST",
      body: JSON.stringify({ role, source, filename }),
    }),
  compareMeasured: (req: MeasuredCompareRequest) =>
    request<MeasuredCompareResponse>("/api/measured/compare", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  generateFirmware: (preset: string, overrides?: Record<string, number | string>) =>
    request<FirmwareGenResponse>("/api/firmware/generate", {
      method: "POST",
      body: JSON.stringify({ preset, overrides }),
    }),
};

export interface FirmwareGenResponse {
  scheme: string;
  tx: string;
  rx: string;
  notes: string[];
}

export interface MeasuredCompareRequest {
  csv: string;
  preset?: string;
  n_bits?: number;
  sample_rate_hz?: number;
  measured_ber?: number;
}
export interface MeasuredCompareBer {
  simulated: number | null;
  measured: number | null;
  abs_error?: number;
  rel_error?: number | null;
}
export interface MeasuredCompareResponse {
  ok: boolean;
  message: string;
  series: { t: number[]; measured: number[]; simulated: number[] };
  metrics: {
    nrmse: number;
    correlation: number;
    agreement_pct: number;
    lag_frac: number;
    ber: MeasuredCompareBer;
  };
  meta: {
    preset: string;
    n_measured: number;
    sample_rate_hz: number | null;
    header: Record<string, string>;
  };
}

export interface FirmwareFinding {
  config_field: string;
  value: number;
  source: string;
  label: string;
  confidence: string;
}
export interface FirmwareInfo {
  label: string;
  value: string;
  source: string;
}
export interface FirmwareParseResponse {
  role: string;
  params: Record<string, number>;
  findings: FirmwareFinding[];
  info: FirmwareInfo[];
  warnings: string[];
}

export interface ProbeResult {
  id: string;
  kind: "dmm" | "scope";
  net: string;
  found: boolean;
  dc: number | null;
  min: number | null;
  max: number | null;
  trace: number[] | null;
  time: number[] | null;
}

export interface SchematicSimResponse {
  ber: number | null;
  ber_incircuit?: number | null;
  n_bits?: number;
  message?: string;
  warnings?: string[];
  diagnostics?: Record<string, unknown>;
  probes?: ProbeResult[];
}

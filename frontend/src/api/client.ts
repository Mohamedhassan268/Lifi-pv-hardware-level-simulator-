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
  category: string;
  class: string;
  config_field: string;
}

export interface ComponentDetail {
  part: string;
  class: string;
  category: string;
  config_field: string;
  parameters: Record<string, unknown>;
}

export interface StandardsProfileSummary {
  id: string;
  label: string;
  spec_section: string;
  notes: string;
  overrides: Record<string, unknown>;
}

export const api = {
  listPresets: () => request<string[]>("/api/presets"),
  getPreset: (name: string) => request<ConfigDict>(`/api/presets/${name}`),
  listStandards: () => request<StandardsProfileSummary[]>("/api/standards"),
  getStandard: (id: string) =>
    request<StandardsProfileSummary & { criteria: string[] }>(
      `/api/standards/${encodeURIComponent(id)}`,
    ),
  validateConfig: (cfg: ConfigDict) =>
    request<{ valid: boolean; errors: string[]; normalized?: ConfigDict }>(
      "/api/config/validate",
      { method: "POST", body: JSON.stringify(cfg) },
    ),
  listComponents: () => request<ComponentSummary[]>("/api/components"),
  listComponentCategories: () => request<string[]>("/api/components/categories"),
  getComponent: (part: string) =>
    request<ComponentDetail>(`/api/components/${encodeURIComponent(part)}`),
  linkBudget: (req: LinkBudgetRequest) =>
    request<LinkBudgetResponse>("/api/link-budget", {
      method: "POST",
      body: JSON.stringify(req),
    }),
};

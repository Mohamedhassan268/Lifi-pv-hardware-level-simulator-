/**
 * configStore — the single source of truth on the frontend, mirroring the
 * PyQt6 SystemConfig fanout pattern in gui/main_window.py::_on_config_changed.
 *
 * The shape mirrors `SystemConfig.to_dict()` (flat keys). We don't redeclare
 * the 75+ field types here — keeping it `Record<string, unknown>` lets the
 * backend remain the source of truth. Components reach into specific keys
 * via typed selector helpers (see useLinkBudgetInput below).
 *
 * Draft slice:
 *   The Builder route (/build) edits a `draft` copy of the config so that
 *   in-progress edits never visually scrub the live system underneath other
 *   routes. `commitDraft` promotes the draft to live; `discardDraft` throws
 *   it away.
 */

import { create } from "zustand";
import { useShallow } from "zustand/react/shallow";

import type { ConfigDict } from "@/api/client";
import { api } from "@/api/client";

export type ConfigSource = "live" | "draft";

interface ConfigState {
  presetName: string | null;
  config: ConfigDict;
  draft: ConfigDict | null;
  loading: boolean;
  error: string | null;

  loadPreset: (name: string) => Promise<void>;
  patch: (changes: Partial<ConfigDict>) => void;
  reset: () => void;

  // Draft (Builder) actions
  startDraft: (seed?: ConfigDict) => void;
  patchDraft: (changes: Partial<ConfigDict>) => void;
  commitDraft: () => void;
  discardDraft: () => void;
}

export const useConfigStore = create<ConfigState>((set, get) => ({
  presetName: null,
  config: {},
  draft: null,
  loading: false,
  error: null,

  async loadPreset(name: string) {
    set({ loading: true, error: null });
    try {
      const cfg = await api.getPreset(name);
      set({ presetName: name, config: cfg, loading: false });
    } catch (e) {
      set({ loading: false, error: e instanceof Error ? e.message : String(e) });
    }
  },

  patch(changes) {
    set((s) => ({ config: { ...s.config, ...changes } }));
  },

  reset() {
    set({ presetName: null, config: {}, error: null });
  },

  startDraft(seed) {
    // Default seed: clone the current live config so the Builder opens
    // showing the existing system rather than blank fields.
    const base = seed ?? get().config;
    set({ draft: { ...base } });
  },

  patchDraft(changes) {
    set((s) => ({ draft: { ...(s.draft ?? {}), ...changes } }));
  },

  commitDraft() {
    const { draft } = get();
    if (!draft) return;
    set({ config: { ...draft }, draft: null, presetName: null });
  },

  discardDraft() {
    set({ draft: null });
  },
}));

// ---- Selectors ----

/**
 * Pluck the subset of fields the link-budget endpoint needs. Returns a stable
 * reference under `shallow` so Setup-route components don't re-render on
 * unrelated config changes.
 *
 * Pass `source: "draft"` to read from the in-progress Builder draft instead
 * of the committed config — used by BuilderRail.
 */
export function useLinkBudgetInput(source: ConfigSource = "live") {
  return useConfigStore(
    useShallow((s) => {
      const src = source === "draft" ? (s.draft ?? s.config) : s.config;
      return {
        distance_m: (src.distance_m as number) ?? 0.325,
        tx_angle_deg: (src.tx_angle_deg as number) ?? 0,
        rx_tilt_deg: (src.rx_tilt_deg as number) ?? 0,
        fov_half_angle_deg: (src.fov_half_angle_deg as number) ?? 90,
        led_half_angle_deg: (src.led_half_angle_deg as number) ?? 9,
        led_radiated_power_mW: (src.led_radiated_power_mW as number) ?? 9.3,
        sc_area_cm2: (src.sc_area_cm2 as number) ?? 9,
        sc_responsivity: (src.sc_responsivity as number) ?? 0.457,
        modulation_depth: (src.modulation_depth as number) ?? 0.33,
        data_rate_bps: (src.data_rate_bps as number) ?? 5000,
        r_sense_ohm: (src.r_sense_ohm as number) ?? 1,
      };
    }),
  );
}

/** Read a single field from the live config or the draft. */
export function useConfigField<T>(field: string, source: ConfigSource = "live"): T | undefined {
  return useConfigStore((s) => {
    const src = source === "draft" ? (s.draft ?? s.config) : s.config;
    return src[field] as T | undefined;
  });
}

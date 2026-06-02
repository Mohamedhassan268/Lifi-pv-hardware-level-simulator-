/**
 * configStore — the single source of truth on the frontend, mirroring the
 * PyQt6 SystemConfig fanout pattern in gui/main_window.py::_on_config_changed.
 *
 * The shape mirrors `SystemConfig.to_dict()` (flat keys). We don't redeclare
 * the 75+ field types here — keeping it `Record<string, unknown>` lets the
 * backend remain the source of truth. Components reach into specific keys
 * via typed selector helpers (see useLinkBudgetInput below).
 *
 * Two-system model:
 *   The builder can host two complete TX→Channel→RX systems (A and B) for
 *   duplex / multi-node studies. Both live in `systems`; `config` always
 *   mirrors the *active* system so every existing single-system consumer keeps
 *   reading `config` unchanged. `systems.B === null` means single-system mode.
 *
 * Draft slice:
 *   The Builder route edits a `draft` copy of the active config so in-progress
 *   edits never visually scrub the live system. `commitDraft` promotes the
 *   draft to live; `discardDraft` throws it away.
 */

import { create } from "zustand";
import { useShallow } from "zustand/react/shallow";

import type { ConfigDict } from "@/api/client";
import { api } from "@/api/client";

export type ConfigSource = "live" | "draft";
export type SystemId = "A" | "B";

/**
 * How the two systems are coupled:
 *   - "none"   : independent links, side by side.
 *   - "duplex" : linked via cross-links (A.TX→B.RX, B.TX→A.RX) + shared rail.
 *   - "shared" : linked AND both systems use the same optical channel.
 */
export type CouplingMode = "none" | "duplex" | "shared";

// Electrical-tie (option a): the shared supply rail + MCU/ADC fields the two
// systems hold in common when linked. Editing one of these on either system
// propagates to the other.
const SHARED_TIE_KEYS = ["vcc_volts", "adc_vref", "adc_bits"] as const;

// Channel/geometry fields kept identical when the systems share one channel.
const SHARED_CHANNEL_KEYS = [
  "distance_m", "tx_angle_deg", "rx_tilt_deg", "fov_half_angle_deg",
  "led_half_angle_deg", "n_reflections", "ambient_illuminance_lux",
] as const;

function tieKeysFor(coupling: CouplingMode): readonly string[] {
  return coupling === "shared" ? [...SHARED_TIE_KEYS, ...SHARED_CHANNEL_KEYS] : SHARED_TIE_KEYS;
}

function sharedSubset(src: ConfigDict, keys: readonly string[] = SHARED_TIE_KEYS): Partial<ConfigDict> {
  const out: Partial<ConfigDict> = {};
  for (const k of keys) if (k in src) out[k] = src[k];
  return out;
}

interface ConfigState {
  presetName: string | null;
  /** Mirror of systems[activeSystem] — what single-system consumers read. */
  config: ConfigDict;
  systems: { A: ConfigDict; B: ConfigDict | null };
  activeSystem: SystemId;
  /** How the two systems are coupled (none / duplex / shared channel). */
  coupling: CouplingMode;
  draft: ConfigDict | null;
  loading: boolean;
  error: string | null;

  loadPreset: (name: string) => Promise<void>;
  loadDefaults: () => Promise<void>;
  loadBlank: () => void;
  patch: (changes: Partial<ConfigDict>) => void;
  reset: () => void;

  // Two-system actions
  addSystemB: () => void;
  removeSystemB: () => void;
  setActiveSystem: (s: SystemId) => void;
  setCoupling: (c: CouplingMode) => void;

  // Draft (Builder) actions
  startDraft: (seed?: ConfigDict) => void;
  patchDraft: (changes: Partial<ConfigDict>) => void;
  commitDraft: () => void;
  discardDraft: () => void;
}

export const useConfigStore = create<ConfigState>((set, get) => {
  const A0: ConfigDict = {};
  return {
    presetName: null,
    config: A0,
    systems: { A: A0, B: null },
    activeSystem: "A",
    coupling: "none",
    draft: null,
    loading: false,
    error: null,

    async loadPreset(name: string) {
      set({ loading: true, error: null });
      try {
        const cfg = await api.getPreset(name);
        set((s) => ({
          presetName: name,
          config: cfg,
          systems: { ...s.systems, [s.activeSystem]: cfg },
          loading: false,
        }));
      } catch (e) {
        set({ loading: false, error: e instanceof Error ? e.message : String(e) });
      }
    },

    async loadDefaults() {
      // "Build your own" starts from the backend's default SystemConfig so the
      // builder opens with sane values to tweak, not blank fields. The validate
      // endpoint echoes a fully-normalized config in `normalized`.
      set({ loading: true, error: null });
      try {
        const r = await api.validateConfig({});
        const cfg = r.normalized ?? {};
        set((s) => ({
          presetName: null,
          config: cfg,
          systems: { ...s.systems, [s.activeSystem]: cfg },
          loading: false,
        }));
      } catch (e) {
        set({ loading: false, error: e instanceof Error ? e.message : String(e) });
      }
    },

    loadBlank() {
      // New "build your own" systems start empty — no preset numbers in the
      // fields. The backend normalizes any unset fields at run time.
      set((s) => ({
        presetName: null,
        config: {},
        systems: { ...s.systems, [s.activeSystem]: {} },
      }));
    },

    patch(changes) {
      set((s) => {
        const active = s.activeSystem;
        const next = { ...s.config, ...changes };
        const systems = { ...s.systems, [active]: next };
        // When linked, keep the shared fields in lockstep across A↔B (rail/MCU
        // always; channel/geometry too when the systems share one channel).
        if (s.coupling !== "none" && s.systems.B) {
          const other: SystemId = active === "A" ? "B" : "A";
          const shared = sharedSubset(changes as ConfigDict, tieKeysFor(s.coupling));
          if (Object.keys(shared).length > 0) {
            systems[other] = { ...(systems[other] as ConfigDict), ...shared };
          }
        }
        return { config: next, systems };
      });
    },

    reset() {
      const A: ConfigDict = {};
      set({
        presetName: null,
        config: A,
        systems: { A, B: null },
        activeSystem: "A",
        coupling: "none",
        error: null,
      });
    },

    addSystemB() {
      set((s) => {
        if (s.systems.B) return {};
        // System B starts INDEPENDENT (blank), not a clone of A — cloning made
        // A's receiver/channel/TX values appear in B the moment B was added.
        // The backend normalizes unset fields at run time (same as a blank
        // "build your own"); the user configures B from scratch. The optional
        // Link A↔B tie still shares only the 3 rail/ADC fields when enabled.
        return { systems: { ...s.systems, B: {} } };
      });
    },

    removeSystemB() {
      set((s) => ({
        systems: { ...s.systems, B: null },
        coupling: "none",
        // Drop any draft seeded against B so it can't commit into A.
        draft: null,
        ...(s.activeSystem === "B"
          ? { activeSystem: "A" as SystemId, config: s.systems.A }
          : {}),
      }));
    },

    setActiveSystem(target) {
      set((s) => {
        const cfg = s.systems[target];
        if (!cfg) return {};
        // Discard any in-progress draft: a draft belongs to the system that was
        // active when it was seeded. Carrying it across a switch would commit
        // one system's edits into the other (the A→B leak). The Inspector
        // re-seeds a fresh draft from the newly active system's config.
        return { activeSystem: target, config: cfg, draft: null };
      });
    },

    setCoupling(c) {
      set((s) => {
        if (c === "none" || !s.systems.B) return { coupling: c };
        // On linking, push the active system's shared fields to the other so
        // they start consistent (rail/MCU, plus channel when shared).
        const active = s.activeSystem;
        const other: SystemId = active === "A" ? "B" : "A";
        const shared = sharedSubset(s.systems[active] ?? {}, tieKeysFor(c));
        return {
          coupling: c,
          systems: { ...s.systems, [other]: { ...(s.systems[other] as ConfigDict), ...shared } },
        };
      });
    },

    startDraft(seed) {
      const base = seed ?? get().config;
      set({ draft: { ...base } });
    },

    patchDraft(changes) {
      set((s) => ({ draft: { ...(s.draft ?? {}), ...changes } }));
    },

    commitDraft() {
      const { draft, activeSystem } = get();
      if (!draft) return;
      const next = { ...draft };
      set((s) => ({
        config: next,
        systems: { ...s.systems, [activeSystem]: next },
        draft: null,
        presetName: null,
      }));
    },

    discardDraft() {
      set({ draft: null });
    },
  };
});

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

/** True when a second system (B) exists. */
export function useTwoSystem(): boolean {
  return useConfigStore((s) => s.systems.B !== null);
}

/**
 * builderUIStore — UI-only state for the modal-based system builder.
 *
 * Kept separate from configStore so opening/closing a modal doesn't trigger
 * config selectors and config patches don't re-render modal chrome.
 *
 * Two-system model:
 *   `configured` is tracked per system (A/B). Single-system consumers read the
 *   *active* system's map via `useActiveConfigured()`; the active system itself
 *   lives in configStore (the single source of truth for A/B).
 *
 * State surfaces:
 *   - openCategory          : which category modal is open (one at a time)
 *   - schematicRev          : monotonic counter the canvas watches to re-render
 *                             without diffing 80 config keys itself
 *   - configuredBySystem    : per-system map; a category is true once the user
 *                             clicks Apply, or the preset launch path marked all.
 */

import { create } from "zustand";

import { useConfigStore, type SystemId } from "@/store/configStore";

export type BuilderCategory = "transmitter" | "receiver" | "geometry" | "noise" | "mcu";

export type ConfiguredMap = Record<BuilderCategory, boolean>;

const initialConfigured: ConfiguredMap = {
  transmitter: false,
  receiver: false,
  geometry: false,
  noise: false,
  mcu: false,
};

const allConfigured: ConfiguredMap = {
  transmitter: true,
  receiver: true,
  geometry: true,
  noise: true,
  mcu: true,
};

interface BuilderUIState {
  openCategory: BuilderCategory | null;
  schematicRev: number;
  configuredBySystem: Record<SystemId, ConfiguredMap>;
  selectedEntity: BuilderCategory | null;

  openModal: (c: BuilderCategory) => void;
  closeModal: () => void;
  bumpSchematic: () => void;
  markConfigured: (c: BuilderCategory) => void;
  markAllConfigured: () => void;
  selectEntity: (c: BuilderCategory | null) => void;
  resetWorkspace: () => void;
}

export const useBuilderUIStore = create<BuilderUIState>((set) => ({
  openCategory: null,
  schematicRev: 0,
  configuredBySystem: { A: { ...initialConfigured }, B: { ...initialConfigured } },
  selectedEntity: "transmitter",

  openModal: (c) => set({ openCategory: c }),
  closeModal: () => set({ openCategory: null }),
  bumpSchematic: () => set((s) => ({ schematicRev: s.schematicRev + 1 })),

  markConfigured: (c) =>
    set((s) => {
      const a = useConfigStore.getState().activeSystem;
      return {
        configuredBySystem: {
          ...s.configuredBySystem,
          [a]: { ...s.configuredBySystem[a], [c]: true },
        },
      };
    }),

  markAllConfigured: () =>
    set((s) => {
      const a = useConfigStore.getState().activeSystem;
      return {
        configuredBySystem: { ...s.configuredBySystem, [a]: { ...allConfigured } },
        schematicRev: s.schematicRev + 1,
      };
    }),

  selectEntity: (c) => set({ selectedEntity: c }),

  resetWorkspace: () =>
    set((s) => ({
      configuredBySystem: { A: { ...initialConfigured }, B: { ...initialConfigured } },
      openCategory: null,
      selectedEntity: "transmitter",
      schematicRev: s.schematicRev + 1,
    })),
}));

/** The configured-map of the currently active system (A or B). */
export function useActiveConfigured(): ConfiguredMap {
  const active = useConfigStore((s) => s.activeSystem);
  return useBuilderUIStore((s) => s.configuredBySystem[active]);
}

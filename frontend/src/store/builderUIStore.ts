/**
 * builderUIStore — UI-only state for the modal-based system builder.
 *
 * Kept separate from configStore so opening/closing a modal doesn't trigger
 * config selectors and config patches don't re-render modal chrome.
 *
 * The preset-vs-own-design choice happens ONCE at the Landing page (via
 * LaunchChoiceModal). Once inside the workspace, Simulate runs straight
 * through — there is no gate.
 *
 * State surfaces:
 *   - openCategory          : which category modal is open (one at a time)
 *   - schematicRev          : monotonic counter the canvas watches to re-render
 *                             without diffing 80 config keys itself
 *   - configured[category]  : true once the user has clicked Apply for that
 *                             category, OR the LandingChoice preset path
 *                             marked the four categories as fully defined.
 */

import { create } from "zustand";

export type BuilderCategory = "transmitter" | "receiver" | "geometry" | "noise";

type ConfiguredMap = Record<BuilderCategory, boolean>;

const initialConfigured: ConfiguredMap = {
  transmitter: false,
  receiver: false,
  geometry: false,
  noise: false,
};

const allConfigured: ConfiguredMap = {
  transmitter: true,
  receiver: true,
  geometry: true,
  noise: true,
};

interface BuilderUIState {
  openCategory: BuilderCategory | null;
  schematicRev: number;
  configured: ConfiguredMap;
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
  configured: { ...initialConfigured },
  selectedEntity: "transmitter",

  openModal: (c) => set({ openCategory: c }),
  closeModal: () => set({ openCategory: null }),
  bumpSchematic: () => set((s) => ({ schematicRev: s.schematicRev + 1 })),
  markConfigured: (c) =>
    set((s) => ({ configured: { ...s.configured, [c]: true } })),
  markAllConfigured: () =>
    set((s) => ({
      configured: { ...allConfigured },
      schematicRev: s.schematicRev + 1,
    })),
  selectEntity: (c) => set({ selectedEntity: c }),

  resetWorkspace: () =>
    set({
      configured: { ...initialConfigured },
      openCategory: null,
      selectedEntity: "transmitter",
    }),
}));

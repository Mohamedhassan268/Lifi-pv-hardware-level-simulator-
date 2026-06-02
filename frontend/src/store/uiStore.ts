/**
 * uiStore — purely client-side view state: which top-level route we're on.
 *
 * The app boots into "landing"; the user advances into "builder" via the
 * LaunchChoiceModal.
 */

import { create } from "zustand";

export type Route =
  | "landing"
  | "setup"
  | "engine"
  | "sweeps"
  | "builder"
  | "schematic"
  | "ac"
  | "inspector";

/**
 * The form(s) a project is built/viewed in: a schematic (the circuit editor)
 * and/or a block diagram (the TX → Channel → RX builder). The onboarding flow
 * sets these; when both are enabled the workspace shows a toggle between them
 * and the TopBar filters its Builder/Schematic tabs to match.
 */
export interface ProjectForms {
  schematic: boolean;
  block: boolean;
}

/**
 * Analysis/result tabs that appear progressively, as the user produces them.
 * A workspace starts with only its design form(s) visible; these flip true
 * once there's something to show (a run started → engine/inspector; a run
 * completed → sweeps/ac). Once revealed, a tab stays revealed for the session.
 */
export interface RevealedTabs {
  engine: boolean;
  sweeps: boolean;
  ac: boolean;
  inspector: boolean;
}

export type RevealTab = keyof RevealedTabs;

interface UIState {
  route: Route;
  setRoute: (r: Route) => void;
  forms: ProjectForms;
  setForms: (f: ProjectForms) => void;
  revealed: RevealedTabs;
  reveal: (tab: RevealTab) => void;
}

export const useUIStore = create<UIState>((set) => ({
  route: "landing",
  setRoute: (r) => set({ route: r }),
  // Default to both enabled so every form is reachable until a project picks.
  forms: { schematic: true, block: true },
  setForms: (f) => set({ forms: f }),
  revealed: { engine: false, sweeps: false, ac: false, inspector: false },
  reveal: (tab) =>
    set((s) => (s.revealed[tab] ? s : { revealed: { ...s.revealed, [tab]: true } })),
}));

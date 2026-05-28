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
  | "ac"
  | "inspector";

interface UIState {
  route: Route;
  setRoute: (r: Route) => void;
}

export const useUIStore = create<UIState>((set) => ({
  route: "landing",
  setRoute: (r) => set({ route: r }),
}));

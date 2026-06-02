/**
 * workspace — the launcher → dedicated-window bridge.
 *
 * The Landing page is only a launcher. Choosing a task opens a *dedicated*
 * workspace for it. In the Tauri desktop build each choice spawns a real,
 * separate OS window (decoupled from the launcher, EDA-tool style). Outside
 * Tauri (plain browser dev) there are no OS windows, so the caller falls back
 * to applying the spec in the current window.
 *
 * A workspace window is told what to be via the URL hash (e.g. `#/schematic`,
 * `#/both`, `#/preset/kadirvelu2021`). On boot, App.tsx parses the hash and
 * calls applyWorkspaceSpec() so the fresh window builds its own stores —
 * windows do NOT share in-memory Zustand state.
 */

import { useBuilderUIStore } from "@/store/builderUIStore";
import { useConfigStore } from "@/store/configStore";
import { useSchematicStore } from "@/store/schematicStore";
import { useUIStore } from "@/store/uiStore";

export type WorkspaceSpec =
  | { mode: "schematic" | "block" | "both" }
  | { mode: "preset"; name: string };

/** True only inside a live Tauri webview (checked at call time, not load time). */
function inTauri(): boolean {
  return (
    typeof window !== "undefined" &&
    (window as unknown as { __TAURI_IPC__?: unknown }).__TAURI_IPC__ !== undefined
  );
}

export function encodeWorkspaceHash(spec: WorkspaceSpec): string {
  return spec.mode === "preset"
    ? `#/preset/${encodeURIComponent(spec.name)}`
    : `#/${spec.mode}`;
}

export function parseWorkspaceHash(hash: string): WorkspaceSpec | null {
  const h = hash.replace(/^#\/?/, "");
  if (h.startsWith("preset/")) {
    const name = decodeURIComponent(h.slice("preset/".length));
    return name ? { mode: "preset", name } : null;
  }
  if (h === "schematic" || h === "block" || h === "both") return { mode: h };
  return null;
}

function workspaceTitle(spec: WorkspaceSpec): string {
  switch (spec.mode) {
    case "preset":
      return `OptiSim — ${spec.name}`;
    case "schematic":
      return "OptiSim — Schematic";
    case "block":
      return "OptiSim — Block diagram";
    case "both":
      return "OptiSim — Schematic + Block";
  }
}

/**
 * Configure this window's stores for the given task and route to it. Shared by
 * the new-window boot path (App.tsx) and the same-window fallback (modal).
 */
export async function applyWorkspaceSpec(spec: WorkspaceSpec): Promise<void> {
  const ui = useUIStore.getState();
  const cfg = useConfigStore.getState();
  const builder = useBuilderUIStore.getState();

  if (spec.mode === "preset") {
    // A research preset IS the project — it configures the block diagram;
    // open both forms so the user can toggle.
    await cfg.loadPreset(spec.name);
    builder.markAllConfigured();
    ui.setForms({ schematic: true, block: true });
    ui.setRoute("builder");
    return;
  }

  const wantSchematic = spec.mode === "schematic" || spec.mode === "both";
  const wantBlock = spec.mode === "block" || spec.mode === "both";
  ui.setForms({ schematic: wantSchematic, block: wantBlock });
  if (wantBlock) {
    // A new system opens blank — no preset numbers in the fields. The user
    // fills them in; unset fields are normalized by the backend at run time.
    builder.resetWorkspace();
    cfg.loadBlank();
  }
  if (wantSchematic) useSchematicStore.getState().clear();
  ui.setRoute(wantBlock ? "builder" : "schematic");
}

/**
 * Open the task in a dedicated OS window. Returns true if a real window was
 * spawned (Tauri), false otherwise so the caller can fall back to the current
 * window.
 */
export async function openWorkspaceWindow(spec: WorkspaceSpec): Promise<boolean> {
  if (!inTauri()) return false;
  try {
    const { WebviewWindow } = await import("@tauri-apps/api/window");
    const label = `ws-${Date.now().toString(36)}-${Math.floor(Math.random() * 1e4)}`;
    const w = new WebviewWindow(label, {
      url: `index.html${encodeWorkspaceHash(spec)}`,
      title: workspaceTitle(spec),
      width: 1440,
      height: 900,
      minWidth: 1100,
      minHeight: 700,
      resizable: true,
    });
    return await new Promise<boolean>((resolve) => {
      w.once("tauri://created", () => resolve(true));
      w.once("tauri://error", () => resolve(false));
      // Safety net: assume success if neither event arrives promptly.
      setTimeout(() => resolve(true), 1500);
    });
  } catch {
    return false;
  }
}

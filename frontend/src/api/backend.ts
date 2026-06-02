/**
 * Backend URL resolution.
 *
 * Three modes:
 *   1. Tauri prod: `window.__BACKEND_PORT` is injected by src-tauri/src/main.rs
 *      AFTER the webview boots — we poll for up to ~5 s before giving up.
 *   2. Tauri dev (`tauri dev`): same as prod.
 *   3. Plain Vite dev: `import.meta.env.VITE_BACKEND_URL` (default localhost:8000).
 *
 * `getBackendBase()` is synchronous and returns whatever is known *now* —
 * fine for fetch() calls after waitForBackend() has resolved.
 */

declare global {
  interface Window {
    __BACKEND_PORT?: number;
  }
}

const DEV_FALLBACK =
  (import.meta.env.VITE_BACKEND_URL as string | undefined) ?? "http://localhost:8000";

function inTauri(): boolean {
  if (typeof window === "undefined") return false;
  // __TAURI_IPC__ is typed globally by @tauri-apps/api; accessing it directly
  // would require importing the types unconditionally, so read it loosely.
  return (window as unknown as { __TAURI_IPC__?: unknown }).__TAURI_IPC__ !== undefined;
}

export function getBackendBase(): string {
  if (typeof window !== "undefined" && window.__BACKEND_PORT) {
    return `http://127.0.0.1:${window.__BACKEND_PORT}`;
  }
  return DEV_FALLBACK;
}

export function getWsBase(): string {
  return getBackendBase().replace(/^http/, "ws");
}

/**
 * Resolves once the backend is reachable. Outside Tauri it returns immediately
 * (Vite dev assumes uvicorn is already up). Inside Tauri we wait for the Rust
 * shell to inject __BACKEND_PORT, then ping /health.
 */
async function ensurePortFromCommand(): Promise<void> {
  if (window.__BACKEND_PORT) return;
  try {
    const { invoke } = await import("@tauri-apps/api/tauri");
    const port = await invoke<number>("backend_port");
    if (typeof port === "number" && port > 0) window.__BACKEND_PORT = port;
  } catch {
    // backend_port command unavailable — fall back to injection polling below.
  }
}

export async function waitForBackend(timeoutMs = 8000): Promise<void> {
  const t0 = Date.now();
  if (!inTauri()) return;

  // Step 1: resolve the port. The Rust shell injects window.__BACKEND_PORT
  // into the main window only; spawned workspace windows must ask for it via
  // the backend_port command. Try the command first (fast, works in every
  // window), then poll for injection as a fallback.
  await ensurePortFromCommand();
  while (Date.now() - t0 < timeoutMs && !window.__BACKEND_PORT) {
    await new Promise((r) => setTimeout(r, 75));
  }

  // Step 2: ping /health
  const base = getBackendBase();
  while (Date.now() - t0 < timeoutMs) {
    try {
      const res = await fetch(`${base}/health`);
      if (res.ok) return;
    } catch {
      // server not up yet — keep retrying
    }
    await new Promise((r) => setTimeout(r, 150));
  }
}

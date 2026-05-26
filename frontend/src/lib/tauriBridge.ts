/**
 * tauriBridge — thin wrapper around the @tauri-apps/api v1 dialog + fs
 * modules. Importing those modules in a non-Tauri context is harmless
 * (they fall back to no-ops at runtime), but we still gate calls on
 * `isTauri` so the React side can disable buttons cleanly outside the
 * desktop build.
 *
 * Detection: Tauri injects `__TAURI__` on window (v1) and also
 * `__BACKEND_PORT` via our Rust sidecar bootstrap (see src-tauri/src/main.rs).
 * Either presence is sufficient.
 */

import { dialog, fs } from "@tauri-apps/api";

export const isTauri: boolean =
  typeof window !== "undefined" &&
  (Boolean((window as unknown as { __TAURI__?: unknown }).__TAURI__) ||
    Boolean((window as unknown as { __BACKEND_PORT?: unknown }).__BACKEND_PORT));

/** Open a native folder picker. Returns the selected path or null on cancel. */
export async function pickFolder(): Promise<string | null> {
  if (!isTauri) return null;
  const result = await dialog.open({ directory: true, multiple: false });
  if (typeof result === "string") return result;
  return null;
}

/** Write a UTF-8 text file at the given path. */
export async function writeText(path: string, contents: string): Promise<void> {
  if (!isTauri) throw new Error("writeText: Tauri APIs unavailable");
  await fs.writeTextFile(path, contents);
}

/** Write a binary file at the given path. */
export async function writeBinary(path: string, bytes: Uint8Array): Promise<void> {
  if (!isTauri) throw new Error("writeBinary: Tauri APIs unavailable");
  await fs.writeBinaryFile(path, bytes);
}

/**
 * Join a folder with a filename using the platform's separator.
 * Tauri exposes `path.join` but we keep this dependency-free since the
 * separator on the picked folder is already in `folder`.
 */
export function joinPath(folder: string, filename: string): string {
  const sep = folder.includes("\\") && !folder.includes("/") ? "\\" : "/";
  return folder.endsWith(sep) ? `${folder}${filename}` : `${folder}${sep}${filename}`;
}

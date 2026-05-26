/**
 * WebSocket URL helper. Single source of truth for the backend host is
 * src/api/backend.ts — this just appends the path.
 */

import { getWsBase } from "@/api/backend";

export function wsUrl(path: string): string {
  return `${getWsBase()}${path}`;
}

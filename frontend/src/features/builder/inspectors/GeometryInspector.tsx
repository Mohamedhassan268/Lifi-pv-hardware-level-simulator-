/**
 * GeometryInspector — reuses the existing ChannelTab form body (which holds
 * the geometry parameters: distance, angles, FoV, reflections).
 */

import { ChannelTab } from "@/features/build/ChannelTab";

export function GeometryInspector() {
  return <ChannelTab />;
}

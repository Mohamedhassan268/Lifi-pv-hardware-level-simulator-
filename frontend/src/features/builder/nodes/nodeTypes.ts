/**
 * Shared types for the schematic block nodes. Each category gets its own
 * SVG-rich component, but they all share this data shape so SchematicCanvas
 * can build them uniformly.
 */

import type { StepName } from "@/store/pipelineStore";

export interface BlockData extends Record<string, unknown> {
  title: string;
  subtitle: string;
  chips: string[];
  step?: StepName;
  configured: boolean;
  /** Channel node only: whether to render the noise overlay (per system). */
  showNoise?: boolean;
}

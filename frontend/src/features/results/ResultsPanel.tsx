/**
 * ResultsPanel — host for the Plotly result panels.
 *
 * Layout: waveforms span full width, then a 2-column grid of derived views
 * (eye diagram + decision scatter), then spectrum and bit comparison below.
 */

import { BitsComparePanel } from "@/features/results/BitsComparePanel";
import { ConstellationPanel } from "@/features/results/ConstellationPanel";
import { EyeDiagramPanel } from "@/features/results/EyeDiagramPanel";
import { NoiseBreakdownPanel } from "@/features/results/NoiseBreakdownPanel";
import { SpectrumPanel } from "@/features/results/SpectrumPanel";
import { WaveformsPanel } from "@/features/results/WaveformsPanel";

export function ResultsPanel() {
  return (
    <div className="grid grid-cols-1 gap-6">
      <WaveformsPanel />
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <EyeDiagramPanel />
        <ConstellationPanel />
      </div>
      <SpectrumPanel />
      <NoiseBreakdownPanel />
      <BitsComparePanel />
    </div>
  );
}

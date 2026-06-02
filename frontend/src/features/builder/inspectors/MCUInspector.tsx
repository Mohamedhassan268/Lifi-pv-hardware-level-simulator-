/**
 * MCUInspector — the node controller (ESP32 / Arduino / …). Board selection
 * auto-fills realistic clock / ADC resolution / Vref / sample rate. ADC
 * resolution feeds the quantization-noise model and the sample rate drives the
 * demod (V_rx is band-limited to it, so a too-slow ADC degrades BER). Board and
 * clock inform the validator. Firmware (.ino) import is not executed here.
 */

import { NumberField } from "@/primitives/NumberField";
import { Select } from "@/primitives/Select";

import { Section } from "@/features/build/Section";
import { useDraftField } from "@/features/build/useDraftField";

// Realistic per-board defaults. maxSps caps the ADC sample-rate field (the
// controller's real sampling ceiling); the chosen rate is what actually drives
// the demod DSP (V_rx is band-limited to it). "Custom" leaves fields as-is.
const BOARD_PROFILES: Record<
  string,
  { clockMHz: number; adcBits: number; adcVref: number; maxSps: number; sampleRate: number }
> = {
  ESP32: { clockMHz: 240, adcBits: 12, adcVref: 3.3, maxSps: 2_000_000, sampleRate: 200_000 },
  "ESP32-S3": { clockMHz: 240, adcBits: 12, adcVref: 3.3, maxSps: 2_000_000, sampleRate: 200_000 },
  "Arduino Uno": { clockMHz: 16, adcBits: 10, adcVref: 5.0, maxSps: 9_600, sampleRate: 9_600 },
  "Arduino Nano": { clockMHz: 16, adcBits: 10, adcVref: 5.0, maxSps: 9_600, sampleRate: 9_600 },
  "Raspberry Pi Pico": { clockMHz: 133, adcBits: 12, adcVref: 3.3, maxSps: 500_000, sampleRate: 100_000 },
};
const BOARDS = [...Object.keys(BOARD_PROFILES), "Custom"];

export function MCUInspector() {
  const board = useDraftField<string>("mcu_board");
  const clock = useDraftField<number>("mcu_clock_MHz");
  const adcBits = useDraftField<number>("adc_bits");
  const adcVref = useDraftField<number>("adc_vref");
  const sampleRate = useDraftField<number>("mcu_sample_rate_hz");

  // Selecting a known board auto-fills its realistic clock / ADC / sample rate.
  function onBoardChange(next: string) {
    board.set(next);
    const p = BOARD_PROFILES[next];
    if (p) {
      clock.set(p.clockMHz);
      adcBits.set(p.adcBits);
      adcVref.set(p.adcVref);
      sampleRate.set(p.sampleRate);
    }
  }

  const maxSps = BOARD_PROFILES[board.value ?? ""]?.maxSps;

  // Keep the current value selectable even if it's a custom string.
  const boardOptions =
    board.value && !BOARDS.includes(board.value) ? [board.value, ...BOARDS] : BOARDS;

  return (
    <div className="space-y-5">
      <Section
        title="Microcontroller"
        description="The node's controller — reads the receiver and runs the demod DSP. ADC resolution and sample rate affect the simulation; board selection fills realistic defaults."
      >
        <Select label="Board" value={board.value} onChange={onBoardChange} options={boardOptions} />
        <NumberField
          label="Clock"
          unit="MHz"
          step={1}
          min={1}
          value={clock.value}
          onChange={clock.set}
        />
        <NumberField
          label="ADC resolution"
          unit="bits"
          step={1}
          min={6}
          max={24}
          value={adcBits.value}
          onChange={adcBits.set}
        />
        <NumberField
          label="Sample rate"
          unit="Hz"
          step={1000}
          min={0}
          max={maxSps}
          value={sampleRate.value}
          onChange={sampleRate.set}
          hint={
            maxSps
              ? `ADC sampling for demod (max ${maxSps.toLocaleString()} Hz on this board)`
              : "ADC sampling for demod (0 = ideal ADC)"
          }
        />
      </Section>
    </div>
  );
}

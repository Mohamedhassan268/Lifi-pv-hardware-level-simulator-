# Project Status — Hardware-Faithful LiFi/PV Simulator

**Last updated:** 2026-06-01

---

## 1. Purpose

A SPICE-grade co-simulator for visible-light communication systems that double
as photovoltaic energy harvesters. The simulator validates the full
TX → optical channel → RX chain against six peer-reviewed papers (Kadirvelu
2021, Sarwar 2017, Xu 2024, Oliveira 2024, González 2024, Correa 2025) and is
intended to support a Cairo hardware deployment scheduled for July 2026.

Design principles:

- **Hardware-faithful** — every component value comes from a datasheet, not
  empirical fitting. 14 specific parts (Philips LXM5-PD01 LED, IXYS
  KXOB25-04X3F GaAs solar cell, TI INA322 amp, TI TLV7011 comparator, etc.)
  have explicit Python classes with `get_parameters()`.
- **Physics-first** — Lambertian emission, Beer-Lambert atmospheric attenuation,
  Johnson-Nyquist + shot + ambient + amplifier + ADC + processing noise (the
  six-source noise model).
- **Dual engine** — Python for fast iteration; SPICE (LTspice / ngspice) for
  full transient circuit analysis. Auto-falls back to Python when no SPICE
  binary is found.
- **Paper-agnostic** — papers are validation targets, not the architecture.
  Adding a paper is a JSON preset + a 30-line Python file.

---

## 2. Migration status

The project is mid-migration from a PyQt6 desktop GUI to a React/Tauri desktop
app. Original simulation core remains untouched; only the shell is changing.

| Phase | Description | Status |
|---|---|---|
| **1** | FastAPI REST skeleton + React/Vite Setup view + Zustand stores + hero animation | ✅ Done |
| **2** | Pipeline WebSocket + live step cards + Plotly results (waveforms, bits compare) | ✅ Done |
| **3** | Workbench (14-component browser, filter+search, "Use as…" slot wiring) | ✅ Done |
| **4** | Sweep WebSocket (thermal + Monte Carlo, live streaming Plotly) | ✅ Done |
| **5** | Tauri packaging (sidecar lifecycle, PyInstaller spec, MSI installer) | ✅ Done |
| **6** | UI redesign: Landing → Choice modal → modal-based BuilderWorkspace (react-flow schematic, NoiseOverlay, ResultsMiniPanel, ExportButton, EngineRoute results expansion) | ✅ Done |
| **7** | EDA-grade upgrades: noise breakdown, eye/jitter metrics, analytical BER + Wilson CI, BER-vs-SNR / BER-vs-distance sweeps, n_bits bump (100→10k), AC analysis (Bode + GM/PM), per-component tolerance metadata, IEEE 802.15.7 PHY-I compliance | ✅ Done |
| **8** | Standards-grade additions: IEEE 802.11bb HE PHY preset (20 MHz, MCS 7), generic LDPC FEC layer (`cosim/fec.py` via pyldpc), OFDM SNR-injection bug fix (DC mean → AC swing RMS) | ✅ Done |
| **9** | Parameter validation guardrails: `cosim/validation.py` with 5 rule families (physical sanity, Nyquist, datasheet, link budget, statistical) + structured `IssueLevel`/`ValidationIssue`; `SystemConfig.validate()`; `POST /api/config/validate` returns `issues[]`; BuilderRail color-coded panel; `cli.py validate-config` subcommand + pre-flight gate on `cmd_pipeline` | ✅ Done |
| **10** | KiCad UI + Inspector dock + real schematics: `/api/kicad/export` + KicadExportCard; `/inspector` route with 4 tabs (Schematics / Validation / Messages / Probes) + messagesStore ring buffer; `/api/schematic/{preset}/{idx}` serves schemdraw SVG for all 34 hand-tuned drawings | ✅ Done |
| **11** | Quick-win cluster: PM/GM convention note on BodePanel; `n_bits` bump on `kadirvelu2021` (100→10000) and `xu2024` (20→200) — flipped Xu to PASS and exposed a real Kadirvelu BER mismatch previously masked by small-sample noise | ✅ Done |
| **12** | Demodulator filter bugfix: `_demodulate_ook` / `_demodulate_manchester` HPF was using scipy `butter(...)` in (b, a) polynomial form at extreme low normalized frequencies (`wn ~ 4e-5` for 2.5 kbps @ 1.25 MHz fs) — filter coefficients silently diverged, output ~7e8, BER pinned to coin-flip. Switched to SOS form (`output='sos'` + `sosfiltfilt`). Kadirvelu 2021 BER **0.50 → 0.0009** (within 9% of paper target 1.008e-3). Pipeline comparator now **8/8 PASS** | ✅ Done |
| **13** | Soft demap for FEC: per-bit max-log-MAP LLRs in `cosim/modulation.py::_qam_demap_soft` (closed-form QPSK, min-distance loop for 16/64-QAM with cached constellation) + `_demodulate_ofdm(llr_out=...)` plumbing + empirical N₀ from nearest-neighbour distance + new `LDPCCodec.decode_llrs` that passes true LLRs to pyldpc via `snr=0, y=LLRs/2`. python_engine wires the soft path only when `fec_enable && OFDM`; hard-bit fallback retained for non-OFDM FEC. **ieee_802_11bb post-FEC BER 2e-3 → 0/51490 errors** (Wilson UB ≈ 7.5e-5). Comparator stays 8/8 PASS via the existing Wilson-UB gate; no comparator surgery needed | ✅ Done |
| **14** | OFDM through the analog RX chain: replace the digital-bypass path with `V_rx`-fed demodulation. New `cosim/ofdm_equalizer.py` provides `chain_response_at(cfg, freqs)` and `subcarrier_frequencies(fs, n_fft, n_sc)`. `_modulate_ofdm` gains an `n_subcarriers` param (TX/RX now consistent) and uses FFT-based upsampling (scipy `resample`) instead of `np.interp` to avoid triangular-filter amplitude roll-off on high subcarriers. `python_engine.py` ADC-resamples `V_rx_ac` to the OFDM digital rate before demodulation; equalizer is applied only for non-direct topologies (direct topology is `V_rx = I_ph × R_sense`, a scalar — no RC filter in sim). Validator: `nyquist.ofdm_bpf_incompatible` WARNING when OFDM + bpf_stages > 0. Tests: `tests/test_ofdm_chain.py` (8 tests). Comparator: **8/8 PASS** | ✅ Done |
| **15** | Drag-and-drop schematic editor + component-driven sim (React Flow editor, `graph → SPICE netlist`, ERC, in-process libngspice via PySpice) | ✅ Done |
| **16** | **Single-canvas TX→Channel→RX co-simulator**: full two-pass engine (TX SPICE → Python channel → RX SPICE → Python post). All 5 modulations close on the canvas — OOK + Manchester (in-circuit comparator slice + Python decode), BFSK + OFDM (analog front-end + Python DSP demod, pilot-based equalizer). Honest 6-source noise in post (proven to bite with distance). Explicit MCU (ESP32) node. UI: power-rail flags, component-relative node sizing, optical-beam channel, New/Open-project → Schematic/Block/Both onboarding with a workspace form-toggle. `.exe` + installers rebuilt. Tests: `tests/test_schematic_cosim.py` (7) | ✅ Done |
| **17** | **Multi-window launcher + EDA-grade schematic + two-system block diagram** (frontend-only, no Rust/backend change). Landing = launcher; each task spawns a real **separate OS window** (Tauri JS `WebviewWindow`, hash-routed boot, `backend_port` command for the port). Strict per-task tabs + progressive reveal (Engine/Inspector on run, Sweeps/AC on completion). Schematic editor: per-component anchored pin terminals, symbol-keyed sizing, Proteus boxed grid, card-free glyphs. **Two-system block diagram**: A/B systems with an active-system switcher, coupling modes (none / duplex cross-links / shared channel), shared-config electrical tie, two-pass duplex run, per-system results panel. New "build your own" opens **blank** (no preset numbers). **MCU block (Tier A)**: Controller category + inspector + per-system canvas node. `tsc`/`vite build` clean; verified live in `tauri:dev`. **UNCOMMITTED; installers stale (frontend post-dates the Phase-16 build).** | ✅ Done |
| **18** | **MCU node made load-bearing**: board profiles auto-fill realistic clock/ADC/Vref/sample-rate; `mcu_sample_rate_hz` band-limits `V_rx` before demod (finite-rate ADC — sub-Nyquist aliases, BER degrades; 0 = ideal = no-op); `_rule_mcu` validator (Nyquist + clock-budget warnings). Tests `tests/test_mcu_adc.py` (7); `compare` 8/8. **Two-stage rebuild run 2026-06-02 → `.exe`/MSI/NSIS ship Phase 17+18.** | ✅ Done |
| **19** | **Feature extraction (ML/data-engineering) + duplex draft-leak fix**: `cosim/features.py` (~33 documented features — link quality, throughput, 3 latencies, power/energy, signal stats, noise) feeding 3 surfaces — `/api/features` + `/dataset` (CSV/JSONL), `metrics.features` WS payload, and a `LinkAnalyticsPanel` (table + export + sweep). Fixed `setActiveSystem` leaving a stale draft that leaked System A edits into B. Tests `tests/test_features.py` (6); 13/13, `compare` 8/8. | ✅ Done |

**Phase 5 final artifacts** (built 2026-05-16, Rust compile 8m 06s — these
predate the Phase 6/7 work and need to be rebuilt for distribution):

| Artifact | Size | Path |
|---|---|---|
| Portable Tauri shell | 12.83 MB | `dist\installer\LiFi PV Simulator.exe` |
| MSI installer (WiX 3.14) | 73.99 MB | `dist\installer\LiFi PV Simulator_0.1.0_x64_en-US.msi` |
| NSIS .exe installer | 72.81 MB | `dist\installer\LiFi PV Simulator_0.1.0_x64-setup.exe` |

Build notes:
- MSVC toolchain: VS Build Tools 2026 (v18.6.11806.211) at `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\`. `cl.exe` at `VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\`. Activate via `vcvars64.bat` before building.
- `CARGO_TARGET_DIR=D:\cargo-target\lifi-pv` — keep cargo target off C: (only ~9 GB free) and out of OneDrive (repo lives in OneDrive, which causes intermittent file locks during build).
- **Avast shields must be disabled during the build.** Avast's `aswMonFlt` filter driver races cargo on every `.exe` write/spawn and produces intermittent `os error 5` failures (most commonly on the `proc-macro2` build script). Defender exclusions don't help — Avast scans separately. Disable via the tray icon for 10 minutes, build, then re-enable.

The legacy PyQt6 GUI (`python cli.py gui`) **remains fully functional** and
is the recommended day-to-day driver until the Tauri build lands.

### Phase 6 — UI redesign (2026-05-18 → 2026-05-19)

The Setup/Engine/Workbench/Build tabbed shell from Phase 1-4 was reworked
into a launch flow that matches the "design then simulate" mental model:

  Landing splash  →  "Builder" CTA  →  LaunchChoiceModal (preset vs own design)
                  →  BuilderWorkspace (category rail + live react-flow schematic)
                  →  Simulate (no gate)  →  ResultsMiniPanel
                  →  "Open full results →" → EngineRoute (all Plotly panels)

What changed:

- **LandingRoute** + **LaunchChoiceModal** replace the old direct-into-Setup
  default. Presets are *only* offered at the Landing step; the Simulate
  button inside the workspace runs straight through.
- **BuilderWorkspace** with a left **CategoryRail** (Transmitter, Receiver,
  Geometry, Noise — Modulation folded into Transmitter) and a right
  react-flow **SchematicCanvas** with three SVG-rich custom nodes (TxNode,
  ChannelNode, RxNode). A **NoiseOverlay** layer renders animated grain
  and beam distortion on the channel when noise is applied.
- Per-category **ModalShell** wraps the existing TransmitterTab /
  ReceiverTab / ChannelTab / NoiseTab form bodies — no duplicate form
  logic.
- **builderUIStore** drives the `configured[category]` flags that gate
  the schematic visuals + the rail's checkmarks.
- **ResultsMiniPanel** appears inline over the schematic after Simulate;
  the "Open full results →" link is held back until both metrics AND
  waveforms have arrived (avoids the race where EngineRoute mounts with
  half-populated stores).
- **ExportButton** in EngineRoute uses Tauri `dialog.open()` +
  `fs.writeBinaryFile()` to save `config.json` + a PNG per Plotly panel
  to a user-picked folder; disabled outside Tauri.
- Backend bug fix: `pipeline_ws.py` previously only emitted a `waveforms`
  WS message when the Python engine ran. `backend/spice_waveforms.py`
  closes the SPICE path by parsing the `.raw` file and emitting the same
  payload shape, so EngineRoute panels populate regardless of engine.

### Phase 7 — EDA-grade upgrades (2026-05-19 → 2026-05-20)

Eight upgrades shipped across one PR-sized block of work. The three
"engine already exists" wins arrived first; the heavier additions
(AC analysis, standards) followed.

- **Per-stage noise breakdown** (`NoiseBreakdownPanel`) — backend attaches
  `NoiseModel.compute_noise().as_dict()` to every metrics message; the
  panel renders a stacked horizontal bar and a numeric table.
- **Eye opening + jitter σ** (`lib/eyeMetrics.ts`) — pure frontend math
  on `V_rx + bits_tx`. Four chips on the EyeDiagramPanel header.
- **Analytical BER companion** (`lib/dsp.ts::analyticalBer`) — erfc
  (Abramowitz-Stegun 7.1.26), Q-function, Wilson 95% CI helpers. The
  MetricsCard now shows "Measured vs Theory" with Δ in dB.
- **BER vs SNR / BER vs distance sweeps** (`cosim/ber_sweep.py`) — two
  new `/ws/sweep` kinds (`ber_snr`, `ber_distance`), Wilson-CI error bars
  in the SweepChart, analytical theory curve overlaid as a dashed line.
- **Default `n_bits` 100 → 10000** + a **Precision run** button
  (`n_bits = 100000`) in the rail. Paper validation harnesses keep their
  original values for reproducibility.
- **AC analysis route** (`cosim/ac_analysis.py` + `routes/AcRoute.tsx`
  + `BodePanel.tsx` + `useAcSocket.ts`) — closed-form Python cascade of
  sense R → INA (single-pole) → BPF (HP+LP per stage). Reports midband,
  3-dB bandwidth, unity-gain frequency, 180° crossing, GM, PM. Streams
  via `/ws/ac`. **No SPICE round-trip** — runs in <50 ms.
- **Per-component tolerance metadata** (`cosim/tolerance.py`) — registry
  keyed by `(picker_field, part_name) → [ToleranceSpec]`. Monte Carlo
  picks up an `auto_tolerance: true` flag and the backend streams a
  `tolerance_resolved` message with the resolved table.
- **IEEE 802.15.7 PHY-I compliance** (`cosim/standards/`) — first MVP
  profile (OOK / Manchester, 100 kbps anchor, BER ≤ 1e-6 at SNR ≥ 11 dB).
  `pipeline_ws.py` accepts `profile_id` on the start frame and emits a
  `compliance` payload after metrics. UI: `StandardsModal` in the rail,
  `ComplianceBanner` above the ResultsPanel.

New frontend dependencies: `@xyflow/react@^12`, `@tauri-apps/api@^1`. No
others.

### Phase 8 — Standards & FEC (2026-05-27)

Three changes landed as a single increment, with the bug fix exposed by the
new preset:

- **IEEE 802.11bb-2023 preset** (`presets/ieee_802_11bb.json`) — HE PHY at
  20 MHz channel BW, MCS 7 (64-QAM, code rate 5/6). DCO-OFDM with Hermitian
  symmetry; 256-FFT, CP=16, 117 data subcarriers. Standards-locked PHY
  parameters reproduce the standard's published symbol-budget math to 0.1%.
  Registered in `papers/pipeline_validation.py::_PAPER_METRICS` so it
  participates in `cli.py compare` alongside the academic-paper presets.
- **Generic LDPC FEC layer** (`cosim/fec.py`) — opt-in via new SimConfig
  fields (`fec_enable`, `fec_rate_num/den`, `fec_codeword_n`, `fec_d_v`,
  `fec_max_iter`, `fec_decode_snr_db`, `fec_seed`). `LDPCCodec` class wraps
  `pyldpc` (lazy import); encode/decode wires around the existing
  modulate/demodulate at `python_engine.py:179-189` and `:393-405`. Pre-FEC
  channel BER is preserved on the result dict as `ber_uncoded` for
  diagnostics. Caveats documented in the module docstring: pyldpc generates
  regular Gallager LDPC matrices (not 802.11n QC-LDPC base matrices), and
  the current integration feeds hard-decision bits to BP rather than per-bit
  LLRs from the QAM constellation (BSC-class performance; soft demap is a
  follow-up).
- **OFDM SNR-injection bug fix** (`cosim/python_engine.py:354-368`) — the
  link-budget SNR used to scale OFDM noise computed signal current as
  `R × mean(P_rx)`, which is the DC bias photocurrent, not the AC-modulated
  swing that actually carries OFDM payload. This over-stated delivered SNR
  by ~6-12 dB depending on `modulation_depth`, masking distance and
  QAM-order effects in BER until SNR dropped extremely low. Fix uses
  `np.std(I_ph)` — the AC RMS photocurrent — as the signal numerator.
  After the fix, at d=1.5 m / SNR=14 dB the BER for QPSK/16-QAM/64-QAM
  goes 0 / 1.3% / 11% (matches AWGN theory); previously all three reported
  ~0. Sarwar 2017 and Oliveira 2024 retain PASS (their link budgets have
  large SNR margin); the 802.11bb preset was retuned to d=1.0 m so the
  link operates at MCS 7's intended sensitivity (~21 dB).

802.11bb final metrics with LDPC active (seed 42, deterministic):
- Uncoded channel BER: 6.0 × 10⁻³ (at MCS 7 sensitivity)
- Post-FEC BER: 2.02 × 10⁻³ (~3× reduction; soft demap would push lower)
- Payload data rate: 40.47 Mb/s (target 40.5, matches to 0.07%)

New runtime dependency: `pyldpc==0.7.9` (pulls `numba`, `llvmlite`). Install
into the project venv with `--no-build-isolation` so pyldpc's setup sees
the existing numpy. The dependency is lazy-imported in `cosim/fec.py` so
the rest of the simulator runs without it installed.

### Phase 15 — Component-driven sim + drag-and-drop schematic editor (2026-05-30 → 2026-05-31)

Two threads landed this block. **All work is currently UNCOMMITTED** (large
working-tree diff across ~25 files, 6 new). Validation gate held throughout:
`cli.py compare` **8/8 PASS**, `cli.py test` **14/14**, frontend `tsc`/`vite
build` clean.

**Thread A — component objects now drive the simulation** (committed-quality,
verified):
- The pipeline RX subcircuits now source their parameters from the selected
  datasheet component objects instead of raw config numbers:
  `_subckt_ina` ← `INA322.voltage_gain/gain_bandwidth_product`,
  `_subckt_comparator` ← `TLV7011.propagation_delay_s`,
  `_subckt_solar_cell` ← the photodetector's `responsivity/capacitance/
  shunt/series` (via a new `_rx_component(part)` helper in `cosim/pipeline.py`).
  Unknown parts fall back to config. Kadirvelu BER unchanged (1.2e-3, the
  values matched).
- TX DC optical power now sourced from the LED component
  (`LXM5_PD01.radiated_power_at_operating_point()`).
- **Unit bug fixed:** `components/solar_cells.py` KXOB25 had junction
  capacitance `798 pF` and shunt `138.8 Ω` — both 1000× off the paper-canonical
  `798 nF` / `138.8 kΩ` (confirmed against `papers/kadirvelu_2021.py` and the
  `# paper says nF!` note in `systems/kadirvelu2021.py`). Corrected.

**Thread B — drag-and-drop schematic editor** (functional, with one honest gap):
- New EDA-style editor on the **"Schematic"** TopBar tab (React Flow v12):
  `frontend/src/features/schematic/` (Palette, PartNode with role-based pin
  handles, SchematicWorkspace, presets, symbols) + `store/schematicStore.ts`
  (union-find net resolver → CircuitGraph JSON) + `routes/SchematicRoute.tsx`.
- **Per-component SPICE pin model:** `components/pins.py` (`Port`, `PortRole`);
  `spice_ports()` on amp/comparator/photodetector/LED/MOSFET base classes,
  reconciled to match each `.SUBCKT` header. Exposed via `/api/components/{part}`.
- **Wireable subcircuits added** to photodetector / LED / MOSFET base classes
  (previously `.MODEL`-only): `spice_subcircuit()` with named ports
  (anode/cathode/photo_in; anode/cathode; drain/gate/source). SPICE-validated.
- **INA322 hardened** with a REF pin (datasheet-correct) so it level-shifts
  correctly under arbitrary single-supply wiring, not just the pipeline topology.
- **Friendly display names** (`/api/components` `display_name`: "Instrumentation
  Amp", "Photodiode", …) with the part code kept searchable; **schematic symbols**
  per category (`symbols.tsx`); **Output**/**Ground** label terminals (named-net
  markers, no SPICE element).
- **graph → SPICE netlist** generator (`cosim/graph_netlist.py`), **ERC**
  (`cosim/erc.py` — floating pins, no ground/supply → 422), and
  **`POST /api/schematic-sim`** endpoint (`backend/routers/schematic_sim.py`).
- **Engine: schematic-sim runs in-process via libngspice (PySpice)**, not LTspice
  subprocess. `cosim/pyspice_graph_runner.py`. Reason: LTspice batch mode
  **nondeterministically hangs** on the behavioral component subcircuits (~30%
  of runs spin to 1000s+ CPU; same deck solves in ~2 s the rest of the time —
  unfixable from Python). libngspice is in-process, killable, deterministic
  (~3 s/run), and is what the `.exe` already bundles. This **solved the
  stability blocker.** (Also hardened `cosim/ltspice_runner.py` along the way:
  pre-run stray-process kill, `.raw` flush poll, Popen+kill timeout — and fixed
  that `ltspice_timeout_s` was silently dropped because it isn't a SystemConfig
  field.)

**Open gap (Thread B):** the example breadboard preset *loads, wires, and
simulates* (real libngspice, real node voltages) but **does not produce a
correct BER** (~0.5). Root-caused through the full chain: (a) sim window must
cover the scored bits [fixed], (b) the PV needs a harvest return path to ground
[the freeform preset omitted it], (c) the INA output swing is small on a large
DC offset with no AC-coupling/BPF stage, and (d) **Kadirvelu is Manchester-coded
OOK**, which the simple bit-center slicer in `calculate_ber_from_transient`
cannot demodulate. A freeform preset that *truly closes the link* needs real
analog design (gain/bias/AC-coupling) **plus** a demodulator matched to the
modulation — substantially more than a preset drop. The editor + engine + ERC +
round-trip are correct; the "honest BER from an arbitrary user circuit" is the
remaining work. Decision pending on scope (commit wins + report node-voltages
vs. invest in one fully-demodulating preset).

Not yet done from the agreed plan: React Flow `ConnectionMode.Loose` (so
target↔target pins render — some preset edges currently drop); the dual-engine
**TX SPICE deck** (LED+MOSFET driver → I(LED) → Python channel → RX) and the
single-canvas TX→Channel-bridge→RX layout. **All three landed in Phase 16.**

### Phase 16 — Single-canvas TX→Channel→RX co-simulator (2026-05-31 → 2026-06-01)

The mission locked here: a user draws TX → optical channel → RX on **one
canvas** and runs a hardware-faithful link. Because the channel is Python, this
is a **multi-pass co-simulation**, not one SPICE pass:

```
[ TX analog: SPICE ] → [ comms: Python ] → [ RX analog: SPICE ] → [ post: Python ]
  drive→MOSFET→LED      I(LED)→P_tx→        PV/photodiode→amp→      demod→BER/eye
  probe I(LED)          channel→P_rx        probe V(out)            (the MCU/DSP)
```

**Engine choice = in-process libngspice (PySpice), not LTspice** — LTspice batch
mode nondeterministically hangs on the behavioral subcircuits; libngspice is
in-process, killable, deterministic, and is what the `.exe` bundles. **Noise =
Python post** (apply the validated 6-source model to the probed signal; same
simplification the validated pipeline already makes).

New `cosim/` modules: `tx_spice_runner` (TX deck, probes `I(LED)` via an
inserted 0 V ammeter), `optical_bridge` (`I(LED)→optical→channel→P_rx`),
`graph_partition` (split a drawn graph into TX/RX islands — rails, `CHANNEL` and
`MCU` excluded so they can't merge islands), `pyspice_graph_runner` (RX deck;
returns probe nets; lenient when there's no `dout`), and `schematic_cosim`
(orchestrator). `graph_netlist` / `erc` from Phase 15 extended (C primitive,
`DRIVE`/`CHANNEL`/`MCU`/rail markers, optical-pin ERC).

**Modulations — the analog ↔ digital boundary mirrors real hardware** (analog
front-end in SPICE; (de)modulation DSP in Python = the MCU/DSP):

- **OOK / Manchester** (`run_two_pass`): 2-level line codes — the comparator
  *demodulates in-circuit*; Python does the decode (Manchester only). Key RX
  design: an **average-value (DC-restoration) RC slicer** feeds the comparator
  threshold, NOT `vref` (tying to `vref` lands the threshold on the bit-0 level
  → coin flip — the root of the old Phase-15 "BER≈0.5 from an arbitrary circuit"
  gap). Noise bandwidth follows the symbol rate (Manchester pays its honest 2×
  penalty).
- **BFSK / OFDM** (`run_analog_link`): continuous-waveform, no in-circuit
  slicing. Python linear-driver TX + channel → SPICE RX analog front-end →
  Python DSP demod (Goertzel / FFT). **OFDM needs a fast photodiode (BPW34,
  72 pF — NOT the 798 nF solar cell) + a passive transimpedance (photodiode +
  load R)** — an active OPA380 TIA does **not** converge in libngspice feedback
  (documented limitation). OFDM closes via a **pilot-based per-subcarrier
  equalizer** measured from the drawn circuit (a 1–2 symbol training preamble;
  resolves the inverting front-end's 180° phase). BER 0 at 200 kbps.

**Honest noise verified**: swept link distance — BER climbs 0 → 0.33 as `P_rx`
falls, while the noiseless in-circuit recovery stays clean (errors are noise,
not a circuit failure). Guarded by `tests/test_schematic_cosim.py`.

**MCU (ESP32) node**: the digital baseband made explicit — a label-only node
whose `adc` pin marks the DSP sample point (`_find_mcu_adc`).

**Frontend** (`features/schematic/`): React Flow editor — palette, power-rail
flags (VCC/VEE/VREF/GND), MCU/Channel/DataSource markers, OOK + OFDM/BFSK link
presets, `ConnectionMode.Loose`, component-relative node sizing, optical-beam
channel symbol, visible wiring. **Onboarding reworked**: Builder → New/Open
project → Schematic / Block diagram / Both, with a workspace form-toggle
(`FormToggle`) and TopBar tabs filtered to the chosen forms; results still pop
up as before.

**Also**: fixed `BSD235N`/MOSFET `.MODEL` for ngspice (explicit terminal caps —
`CGS`/`CGD` aren't level-1 MOSFET params); KXOB25 junction-cap/shunt unit fix.

Validation gate held throughout: `cli.py test` **14/14**,
`tests/test_schematic_cosim.py` **7/7**, frontend `tsc`/`vite build` clean.
`.exe` + MSI/NSIS installers rebuilt 2026-06-01 (frozen sidecar smoke-tested:
boots + runs the libngspice canvas sim).

### Phase 17 — Multi-window launcher · EDA schematic · two-system block diagram (2026-06-01)

**All frontend-only — no Rust shell or backend changes.** All work is currently
**UNCOMMITTED**. `tsc -b` + `vite build` clean throughout; each increment was
verified live in `tauri:dev`. The Phase-16 `.exe`/installers therefore **predate
this UI and are stale** for distribution (a `tauri:build` will fold it in).

**Multi-window launcher** — the Landing page is now purely a launcher. Each
choice opens a **real separate OS window** instead of swapping the route in
place. Implemented entirely with Tauri v1's JS `WebviewWindow` (allowlist
`all:true`), so **no `main.rs` change was needed**:
- `lib/workspace.ts` — `WorkspaceSpec`, hash encode/parse, `applyWorkspaceSpec`
  (per-window store setup), `openWorkspaceWindow` (spawns the OS window with the
  task in the URL hash, e.g. `index.html#/schematic`, `#/both`,
  `#/preset/<name>`). Falls back to same-window route swap outside Tauri.
- `App.tsx` boots from the hash; a spawned window builds its own stores and
  lands directly in its task (launcher stays on Landing).
- `api/backend.ts` resolves the port via `invoke("backend_port")` (injection
  reaches only the `main` window; spawned windows ask the already-registered
  Rust command).
- **Strict per-task tabs + progressive reveal** (`uiStore` gains sticky
  `revealed` flags): Schematic-only shows just Schematic; Block shows
  Builder + Setup; Engine/Inspector appear on run start, Sweeps/AC on completion
  (`useSimulationSocket` reveals them).

**Schematic editor — anchored terminals + Proteus grid + card-free glyphs**
(`features/schematic/`):
- `PartNode.tsx` — per-symbol pin **anchor map** (`SYMBOL_PINS`, glyph 64×40
  coords keyed by role, consumed in pin order): handles now sit on the drawn
  terminal stubs (op-amp INP/INN left, OUT right, VCC/VEE on added rail stubs;
  photodiode/solar-cell anode-left/cathode-right/optical-top; MOSFET
  gate-left/drain-source-right). Symbol-keyed footprint sizing (xs→xl) so parts
  differ by real-world size (resistor vs ESP32). The per-component **card was
  removed** — bare glyph on the grid, selection shown by a glyph-tracing glow.
- `SchematicWorkspace.tsx` — `BackgroundVariant.Lines` two-layer boxed grid
  (fine 20px + bold 100px).
- `symbols.tsx` — VCC/VEE rail stubs added to amp/comparator glyphs.

**Two-system block diagram** (`features/builder/`, `store/`):
- `configStore` holds **two systems** (`systems.{A,B}`) + `activeSystem`;
  `config` mirrors the active one so every existing single-system consumer is
  unchanged. `coupling ∈ {none, duplex, shared}`; `loadBlank()` for new systems.
- `builderUIStore` tracks `configuredBySystem` (per A/B); `useActiveConfigured()`.
- `SchematicCanvas` renders one TX→Channel→RX→MCU row per system; **duplex**
  draws violet cross-links (A.TX→B.RX, B.TX→A.RX); **shared** re-lays-out to one
  channel fed by both TX and feeding both RX.
- **Electrical tie (option a)**: when linked, shared rail/MCU fields
  (`vcc_volts`, `adc_vref`, `adc_bits`) stay in lockstep A↔B; shared-channel mode
  also syncs the geometry/channel fields.
- **Two-pass duplex run** (`useDuplexRun` + `duplexStore`): runs A then B over
  `/ws/pipeline`, **per-system results side by side** in `DuplexResultsPanel`.
  The single-system run path is untouched.
- `CategoryRail` gains the A/B `SystemSwitcher` (Add/Remove B, Link toggle,
  Duplex|Shared selector) and routes Simulate to the duplex runner when B exists.

**Blank new system** — "build your own" opens with empty fields (`loadBlank`
instead of `loadDefaults`); the backend normalizes unset fields at run time.

**MCU block (Tier A)** — a 5th **Controller** category (`builderUIStore`
`BuilderCategory += "mcu"`): `MCUInspector` (board ESP32/Arduino/…, clock,
ADC bits, sample rate) writing `mcu_*` config keys, and a per-system `McuNode`
wired RX→MCU on the canvas. Representational only — `mcu_*` are unknown keys the
backend **silently ignores** (`system_config.py:346`), so runs are unaffected.
`.ino` firmware execution explicitly deferred (would need an MCU instruction
emulator); a future Tier B can parse a sketch for declared parameters.

### Phase 18 — MCU node made load-bearing (2026-06-02)

The Tier-A MCU block (Phase 17, representational only) became a configurable
ESP whose settings drive the simulation. Three threads:

- **Board parameter profiles** (`MCUInspector.tsx`): selecting
  ESP32 / ESP32-S3 / Arduino Uno / Nano / Pico auto-fills realistic clock,
  ADC resolution, Vref, and sample rate, and caps the sample-rate field at the
  board's real ADC ceiling (ESP32 → 2 MSps, Uno → 9.6 kSps). "Custom" leaves
  fields free.
- **Sample rate drives DSP**: three real `SimConfig` fields (`mcu_board`,
  `mcu_clock_MHz`, `mcu_sample_rate_hz`) replace the silently-ignored `mcu_*`
  keys. A new `_adc_sample` stage in `python_engine.py` band-limits `V_rx` to
  `mcu_sample_rate_hz` right before demodulation (sample at the ADC instants,
  hold back onto the physics grid) — one scheme-agnostic insertion point for
  OOK / Manchester / BFSK / OFDM. A too-slow ADC aliases and BER degrades
  faithfully (kadirvelu OOK_Manchester @ 2500 bps: ideal → BER 0; 2 kHz ADC →
  0.34; 500 Hz → 0.45). `mcu_sample_rate_hz = 0` (the default) is a strict
  no-op, so every existing preset is unaffected.
- **Validator MCU checks** (`cosim/validation.py` `_rule_mcu`): WARNING when the
  ADC rate is sub-Nyquist for the line-code symbol rate (`mcu.adc_under_nyquist`)
  and when the clock leaves < 20 cycles per ADC sample (`mcu.clock_budget_tight`).

Tests: `tests/test_mcu_adc.py` (7). Validation gate held: `cli.py test` 14/14,
`cli.py compare` 8/8, frontend `tsc -b` + `vite build` clean. Phase 18 touched
the backend (`cosim/`), so a full two-stage rebuild was run 2026-06-02 (sidecar
re-frozen 125.9 MB + `tauri:build`); `.exe`/MSI/NSIS now ship Phase 17+18.

### Phase 19 — Feature extraction (ML/data-engineering) + duplex draft-leak fix (2026-06-02)

**Duplex bug fix**: editing System A's receiver/channel leaked into System B.
Root cause — `setActiveSystem` swapped the `config` mirror but left the
in-progress Inspector `draft` alive; since the Inspector re-seeds a draft
whenever one is open, A's draft survived the switch and committed into B.
`configStore.setActiveSystem`/`removeSystemB` now discard the draft on switch.
The intentional 3-field electrical tie (`vcc_volts`/`adc_vref`/`adc_bits` when
coupled) is unaffected.

**Feature extraction** — one source of truth, three surfaces:
- `cosim/features.py` `extract_features(result, cfg)` → a NaN-safe, documented
  record of ~33 features in 6 groups (link quality incl. Wilson CI / Q / EVM;
  throughput incl. goodput / spectral efficiency; **three latencies** —
  propagation, DSP/processing, frame; power & energy incl. harvested PV power /
  energy-per-bit; signal stats incl. PAPR; total noise σ). Each feature carries
  label/unit/group/desc in `FEATURES`.
- `cosim/feature_sweep.py` sweeps one of 11 numeric fields → one feature row per
  point → `to_csv`/`to_jsonl` ML datasets.
- `backend/routers/features.py`: `POST /api/features`, `GET /api/features/schema`,
  `POST /api/features/dataset` (json/csv/jsonl). Single-run record is also
  attached to the `/ws/pipeline` `metrics` payload.
- Frontend `LinkAnalyticsPanel`: grouped feature table + Run CSV/JSON export +
  a dataset-sweep control (field + min/max/points → CSV/JSONL download).

Tests: `tests/test_features.py` (6). Gate held: features+mcu **13/13**,
`cli.py test` **14/14**, `cli.py compare` **8/8**, frontend `tsc`+`vite build`
clean. Backend changed (`cosim/`,`backend/`) → sidecar/installers stale until a
two-stage rebuild.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  LiFi PV Simulator.exe   (Tauri Rust shell — Windows)                    │
│                                                                          │
│  ┌──────────────────────────┐    spawn      ┌──────────────────────────┐ │
│  │  WebView (Chromium)      │ ─────────────►│  lifi-backend.exe        │ │
│  │  React + Vite + Tailwind │               │  PyInstaller-bundled     │ │
│  │  + Framer Motion         │ ◄── REST ─────│  FastAPI + uvicorn       │ │
│  │  + Plotly + Zustand      │ ◄── WS    ────│  wraps cosim/* unmodified│ │
│  └──────────────────────────┘               └────────────┬─────────────┘ │
│                                                          │               │
└──────────────────────────────────────────────────────────┼───────────────┘
                                                           │
                                       ┌───────────────────┴────────────────┐
                                       │ Python simulation core             │
                                       │ ┌────────────────────────────────┐ │
                                       │ │ cosim/pipeline.py              │ │
                                       │ │   TX → Channel → RX orchestr.  │ │
                                       │ └─────┬──────────────────┬───────┘ │
                                       │       │                  │         │
                                       │  ┌────▼────┐       ┌─────▼─────┐   │
                                       │  │ SPICE   │       │ Python    │   │
                                       │  │ engine  │       │ engine    │   │
                                       │  │ (.cir)  │       │ (NumPy)   │   │
                                       │  └────┬────┘       └─────┬─────┘   │
                                       │       └────────┬─────────┘         │
                                       │                │                   │
                                       │  ┌─────────────▼─────────────────┐ │
                                       │  │ Shared modules                │ │
                                       │  │ channel.py · noise.py         │ │
                                       │  │ modulation.py · components/   │ │
                                       │  └───────────────────────────────┘ │
                                       └────────────────────────────────────┘
```

### Why this shape

- **Tauri over Electron:** ~40–80 MB smaller binary, native Rust shell, same
  Chromium webview.
- **Sidecar over PyO3-embedded Python:** the Python world (GIL, NumPy MKL,
  matplotlib) is isolated from the webview process. A crash in cosim can't
  take down the UI.
- **WebSocket for streaming:** pipeline progress, thermal sweeps, and Monte
  Carlo trials all stream point-by-point. The existing
  `SimulationPipeline.on_progress` callback in `cosim/pipeline.py` plugs
  straight into the WS without refactoring simulation code.
- **Zustand over Redux Toolkit:** the source of truth lives in Python; the
  frontend just mirrors it. Less ceremony.
- **Plotly over Recharts/uPlot:** eye diagrams + log-y BER + zoom/pan are
  free out of the box. Heavier bundle but acceptable for a desktop app.

---

## 4. Directory layout

```
hardware_faithful_simulator/
├── cli.py                       # legacy CLI entrypoint (test, gui, validate…)
├── CLAUDE.md                    # project guide for Claude Code
├── TODO.md                      # phase tracker
├── README.md
├── docs/                        # PACKAGING.md, PLAN.md, SETUP_GUIDE.txt, etc.
│
├── cosim/                       # Simulation infrastructure
│   ├── pipeline.py              #   TX → Channel → RX orchestrator
│   ├── system_config.py         #   SystemConfig dataclass + presets loader
│   │                            #   (n_bits default bumped 100 → 10000 in Phase 7)
│   ├── channel.py               #   Lambertian + Beer-Lambert + multipath
│   ├── noise.py                 #   6-source physical noise model
│   ├── modulation.py            #   5 modulation schemes + BER (predict_ber)
│   ├── python_engine.py         #   pure-Python RX simulation
│   ├── ltspice_runner.py        #   LTspice subprocess wrapper
│   ├── ngspice_runner.py        #   ngspice subprocess wrapper
│   ├── thermal_sweep.py         #   sweep over temperature
│   ├── monte_carlo.py           #   yield analysis
│   ├── ber_sweep.py             #   NEW (Phase 7): BER-vs-SNR + BER-vs-distance
│   │                            #   generators with Wilson 95% CI
│   ├── ac_analysis.py           #   NEW (Phase 7): closed-form Bode + GM/PM
│   │                            #   for the RX cascade (no SPICE round-trip)
│   ├── tolerance.py             #   NEW (Phase 7): per-component tolerance
│   │                            #   registry + derive_tolerance_spec()
│   ├── fec.py                   #   NEW (Phase 8): generic FEC interface;
│   │                            #   LDPCCodec wrapping pyldpc (lazy import);
│   │                            #   wired around modulate/demod in python_engine
│   ├── standards/               #   NEW (Phase 7): PHY compliance profiles
│   │   ├── __init__.py          #     PROFILES registry + evaluate()
│   │   ├── types.py             #     PhyProfile / Criterion / CriterionResult
│   │   └── ieee_802_15_7_phy_i.py  # IEEE 802.15.7 PHY-I (OOK/Manchester) MVP
│   ├── session.py               #   session directory mgmt
│   ├── pwl_writer.py            #   SPICE PWL bridge files
│   ├── raw_parser.py            #   .raw file parser
│   ├── sim_result.py            #   SimulationResult dataclass
│   ├── spice_extract.py         #   BER extraction from SPICE output
│   └── spice_finder.py          #   cross-platform SPICE auto-detect
│
├── components/                  # 14 hardware components  (unchanged)
│   ├── base.py                  #   abstract PhotodetectorBase / LEDBase / AmplifierBase
│   ├── solar_cells.py           #   KXOB25-04X3F, SM141K, GenericGaAsPV
│   ├── photodiodes.py           #   BPW34, SFH206K, VEMD5510
│   ├── leds.py                  #   LXM5-PD01 (Philips + Fraen lens)
│   ├── amplifiers.py            #   INA322, TLV2379, ADA4891, OPA380
│   ├── comparators.py           #   TLV7011
│   ├── mosfets.py               #   BSD235N, NTS4409
│   ├── power.py                 #   BQ25570 (TI energy harvester)
│   └── adc.py                   #   STM32H7_ADC
│
├── physics/                     # PN junction, photodetection, LED emission
├── materials/                   # GaAs / Si / InGaP semiconductor models
├── systems/                     # paper-specific system definitions
├── papers/                      # 6 paper validation scripts
├── presets/                     # 6 JSON configurations  (kadirvelu, sarwar, …)
├── spice_libs/                  # bundled SPICE component libraries
├── ngspice-45.2_64/             # bundled ngspice for Windows
│
├── gui/                         # PyQt6 GUI  (legacy, still functional)
│   ├── main_window.py
│   ├── top_bar.py
│   ├── channel_canvas.py        #   QPainter TX→RX diagram
│   ├── tab_simulation_engine.py
│   ├── tab_results.py
│   ├── tab_system_setup.py
│   ├── tab_component_library.py
│   ├── tab_validation.py
│   ├── tab_message.py
│   ├── tab_schematics.py
│   ├── workbench_window.py
│   └── theme.py
│
├── backend/                     # NEW — FastAPI wrapper
│   ├── __init__.py
│   ├── __main__.py              #   `python -m backend --port N` entrypoint
│   ├── app.py                   #   FastAPI app, CORS, route registration
│   ├── schemas.py               #   Pydantic request/response models
│   ├── requirements.txt
│   ├── lifi-backend.spec        #   PyInstaller spec for sidecar build
│   ├── spice_waveforms.py       #   NEW (Phase 6): parses SPICE .raw and
│   │                            #   emits the same waveforms payload as the
│   │                            #   Python path — fixes empty-results bug.
│   └── routers/
│       ├── presets.py           #   GET /api/presets, /api/presets/{name}
│       ├── config.py            #   POST /api/config/validate
│       ├── components.py        #   GET /api/components, /categories, /{part}
│       ├── link_budget.py       #   POST /api/link-budget
│       ├── standards.py         #   NEW (Phase 7): GET /api/standards, /{id}
│       ├── pipeline_ws.py       #   WS /ws/pipeline. Now emits noise_breakdown
│       │                        #   on metrics + compliance after metrics if a
│       │                        #   profile_id is supplied on the start frame
│       ├── sweep_ws.py          #   WS /ws/sweep — thermal · MC (auto-tol) ·
│       │                        #   ber_snr · ber_distance
│       └── ac_ws.py             #   NEW (Phase 7): WS /ws/ac (Bode batch)
│
├── frontend/                    # NEW — Vite + React + TS + Tailwind + Framer Motion
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json · tsconfig.node.json
│   ├── tailwind.config.ts · postcss.config.js
│   ├── index.html
│   ├── src/
│   │   ├── App.tsx              #   route shell with LayoutGroup + AnimatePresence
│   │   │                        #   (top bar hidden on the landing route)
│   │   ├── main.tsx             #   imports @xyflow/react/dist/style.css
│   │   ├── index.css            #   Tailwind base
│   │   ├── vite-env.d.ts        #   ImportMetaEnv types
│   │   ├── lib/
│   │   │   ├── motion.ts        #   EASE + DUR tokens (ease-in-out)
│   │   │   ├── dsp.ts           #   NEW: radix-2 FFT, erfc, Q, Wilson CI,
│   │   │   │                    #   analyticalBer (mirrors cosim.modulation)
│   │   │   ├── eyeMetrics.ts    #   NEW: eye opening + jitter σ from V_rx + bits
│   │   │   ├── tauriBridge.ts   #   NEW: dialog/fs feature-detected wrapper
│   │   │   └── exporter.ts      #   NEW: writes config.json + plot PNGs to a
│   │   │                        #   user-picked folder via Plotly.toImage
│   │   ├── api/
│   │   │   ├── backend.ts       #   getBackendBase, waitForBackend (Tauri-aware)
│   │   │   ├── client.ts        #   REST wrapper (incl. listStandards/getStandard)
│   │   │   └── ws.ts            #   WebSocket URL helper
│   │   ├── store/
│   │   │   ├── configStore.ts   #   SystemConfig mirror + draft slice
│   │   │   ├── uiStore.ts       #   route ∈ {landing, setup, engine, sweeps,
│   │   │   │                    #            builder, ac} — default "landing"
│   │   │   ├── builderUIStore.ts#   NEW: openCategory, configured[], schematicRev
│   │   │   ├── pipelineStore.ts #   3-step status (TX/Channel/RX)
│   │   │   ├── resultsStore.ts  #   metrics (+ noise_breakdown) + waveforms
│   │   │   ├── sweepStore.ts    #   thermal/MC/BER points + toleranceRows
│   │   │   ├── acStore.ts       #   NEW: AC analysis result + running flag
│   │   │   └── standardsStore.ts#   NEW: active profile + last compliance result
│   │   ├── hooks/
│   │   │   ├── useSimulationSocket.ts  # narrowed message writes; forwards
│   │   │   │                           # profile_id; handles `compliance`
│   │   │   ├── useSweepSocket.ts       # handles `tolerance_resolved`
│   │   │   └── useAcSocket.ts          # NEW: /ws/ac
│   │   ├── primitives/
│   │   │   ├── Card.tsx · Button.tsx · Slider.tsx · StatusDot.tsx
│   │   │   └── PlotCanvas.tsx   #   Plotly wrapper (never inside motion.div+layout)
│   │   ├── components/
│   │   │   └── Hero.tsx         #   shared hero motion
│   │   ├── features/
│   │   │   ├── topbar/TopBar.tsx
│   │   │   ├── landing/LaunchChoiceModal.tsx   # NEW
│   │   │   ├── builder/                        # NEW (Phase 6)
│   │   │   │   ├── BuilderWorkspace.tsx
│   │   │   │   ├── CategoryRail.tsx
│   │   │   │   ├── SchematicCanvas.tsx         # react-flow + custom nodes
│   │   │   │   ├── ResultsMiniPanel.tsx        # inline mini + "Open full →"
│   │   │   │   ├── ModalShell.tsx              # shared draft→commit lifecycle
│   │   │   │   ├── NoiseOverlay.tsx            # animated grain + beam distortion
│   │   │   │   ├── nodes/{TxNode,ChannelNode,RxNode,NodeShell,nodeTypes}
│   │   │   │   └── modals/{Transmitter,Receiver,Geometry,Noise}Modal.tsx
│   │   │   ├── setup/{ChannelCanvas,GeometrySliders,LinkBudgetTable}.tsx
│   │   │   ├── engine/{StepCard,PipelineCards,MetricsCard}.tsx
│   │   │   │   #   MetricsCard now shows Throughput + Theory companion + Wilson CI
│   │   │   ├── results/
│   │   │   │   ├── WaveformsPanel.tsx
│   │   │   │   ├── BitsComparePanel.tsx
│   │   │   │   ├── EyeDiagramPanel.tsx          # NEW (eye + metric chips)
│   │   │   │   ├── ConstellationPanel.tsx      # NEW (decision scatter)
│   │   │   │   ├── SpectrumPanel.tsx           # NEW (FFT power spectrum)
│   │   │   │   ├── NoiseBreakdownPanel.tsx     # NEW (per-source bar + table)
│   │   │   │   ├── BodePanel.tsx               # NEW (AC: mag/phase + GM/PM)
│   │   │   │   ├── ExportButton.tsx            # NEW (Tauri-only save-to-folder)
│   │   │   │   └── ResultsPanel.tsx
│   │   │   ├── workbench/{ComponentLibrary,ComponentParamTable,SystemSetupForm}.tsx
│   │   │   ├── sweeps/{SweepControls,SweepProgress,SweepChart,ToleranceTable}.tsx
│   │   │   │   #   four sweep kinds; auto-tolerance table when active
│   │   │   └── standards/                       # NEW (Phase 7)
│   │   │       ├── StandardsModal.tsx
│   │   │       └── ComplianceBanner.tsx
│   │   └── routes/
│   │       ├── LandingRoute.tsx     # NEW — default
│   │       ├── BuilderRoute.tsx     # NEW — wraps BuilderWorkspace
│   │       ├── SetupRoute.tsx       # legacy, still reachable
│   │       ├── EngineRoute.tsx
│   │       ├── AcRoute.tsx          # NEW
│   │       └── SweepsRoute.tsx
│   └── src-tauri/               #   Rust shell
│       ├── Cargo.toml
│       ├── build.rs
│       ├── tauri.conf.json      #   window config, sidecar allowlist, CSP
│       ├── icons/               #   32x32.png, 128x128.png, icon.ico, icon.icns
│       ├── binaries/            #   lifi-backend-<target-triple>.exe (built)
│       └── src/main.rs          #   sidecar lifecycle (spawn, health, inject port)
│
├── scripts/
│   └── build-sidecar.ps1        #   PyInstaller build wrapper
│
├── workspace/                   # session output dirs (gitignored)
└── venv/                        # Python virtual env
```

---

## 5. Module responsibilities (quick reference)

### Backend (FastAPI)

| Endpoint | Maps to |
|---|---|
| `GET /api/presets` | `SystemConfig.list_presets()` |
| `GET /api/presets/{name}` | `SystemConfig.from_preset(name).to_dict()` |
| `POST /api/config/validate` | `SystemConfig._from_flat_dict(payload)` |
| `GET /api/components` | iterate `COMPONENT_REGISTRY` |
| `GET /api/components/categories` | unique categories from `get_parameters()['type']` |
| `GET /api/components/{part}` | `get_component(part).get_parameters()` |
| `POST /api/link-budget` | `OpticalChannel(...).propagate(P_tx)` + SNR calc |
| `WS /ws/pipeline` | `SimulationPipeline(cfg, ...).run_all()` in thread pool. Streams `step` / `metrics` (incl. `noise_breakdown`) / `waveforms` / `compliance` (if `profile_id` was on the start frame) |
| `WS /ws/sweep` | Thermal, Monte Carlo (with optional `auto_tolerance` + `tolerance_resolved` message), `ber_snr`, `ber_distance` generators |
| `WS /ws/ac` | `cosim/ac_analysis.py::run_ac_analysis(cfg, …)` — Bode + GM/PM in one batch payload |
| `GET /api/standards` | List of PHY compliance profiles (only IEEE 802.15.7 PHY-I today) |
| `GET /api/standards/{id}` | Full profile incl. overrides + criteria labels |
| `GET /health` | sidecar readiness probe (used by Tauri) |

### Frontend (React)

| Route | Purpose |
|---|---|
| `landing` (default) | Splash with single **Builder** CTA. Opens the LaunchChoiceModal (preset vs build-your-own). Hides the top bar. |
| `builder` | CategoryRail (TX / RX / Geometry / Noise) + react-flow SchematicCanvas + ResultsMiniPanel + Standards picker + Simulate / Precision-run buttons |
| `setup` | Legacy: hero + channel canvas + sliders + link budget. Still reachable from top bar. |
| `engine` | Streaming step cards + MetricsCard (incl. Theory companion + Wilson CI + Throughput) + ResultsPanel (Waveforms, Eye, ConstellationDecisionScatter, Spectrum, NoiseBreakdown, BitsCompare). ExportButton + ComplianceBanner in the header. |
| `ac` | "Run AC sweep" → BodePanel (mag/phase subplots, GM/PM/BW chips, crosshairs at f_unity and f_180) |
| `sweeps` | Thermal · Monte Carlo (with auto-tolerance) · BER vs SNR · BER vs distance — all live-streaming with Wilson 95% CI bars and analytical overlay |

### Animation contract

All routes animate via `opacity` + `y` (both transforms) inside
`<LayoutGroup>` + `<AnimatePresence mode="wait">`. Shared `EASE` token
(cubic-bezier(0.4, 0, 0.2, 1)) and `DUR` tokens (`fast: 0.18`, `base: 0.28`,
`slow: 0.45`).

`PlotCanvas` is intentionally **not** wrapped in a `motion.div` with `layout`
to avoid Plotly's internal ResizeObserver firing every animation frame.

---

## 6. Build status

| Artifact | Status | Location |
|---|---|---|
| Python backend module (`python -m backend`) | ✅ Works | `backend/__main__.py` |
| FastAPI app smoke test | ✅ All 12 routes register, REST + WS verified | — |
| Frozen sidecar binary | ✅ Rebuilt 2026-06-02 (125.9 MB, incl. Phase 18 MCU backend + bundled libngspice); smoke-tested: boots + `/health` ok | `dist/lifi-backend.exe` |
| Tauri sidecar binary (with target triple) | ✅ Copied | `frontend/src-tauri/binaries/lifi-backend-x86_64-pc-windows-msvc.exe` |
| Frontend Vite build (`npm run build`) | ✅ Builds cleanly | `frontend/dist/` |
| Tauri icons | ✅ Generated from placeholder source.png | `frontend/src-tauri/icons/` |
| Rust toolchain | ✅ rustc 1.95.0, cargo 1.95.0 | — |
| MSVC linker (`link.exe`) | ✅ Available via `vcvars64.bat` (MSVC 14.51.36231 + Windows 11 SDK 10.0.26100.0) | — |
| Portable app | ✅ Built 2026-06-02 (13.8 MB) | `D:\cargo-target\lifi-pv\release\OptiSim.exe` |
| Final `.msi` installer | ✅ Built 2026-06-02 (131.8 MB) | `D:\cargo-target\lifi-pv\release\bundle\msi\OptiSim_0.1.0_x64_en-US.msi` |
| NSIS `.exe` installer (incl. uninstaller) | ✅ Built 2026-06-02 (128.0 MB) | `D:\cargo-target\lifi-pv\release\bundle\nsis\OptiSim_0.1.0_x64-setup.exe` |

> **Note:** all artifacts above were rebuilt 2026-06-02 with the full two-stage
> build (sidecar re-frozen for the Phase 18 backend, then `tauri:build`), so they
> ship the current Phase 17+18 UI. Rust compiled in 53.86s with no Avast `os
> error 5`.

### Reproducing the build

# Two stages. Stage 1 re-freezes the backend (must run after any cosim/backend
# change so the .exe ships the current Python); Stage 2 builds the Tauri app.
```powershell
# 0. Disable Avast shields for 10 minutes (tray icon → Avast shields control)

# Stage 1 — sidecar (PyInstaller): rebuild + copy under the target triple
.\scripts\build-sidecar.ps1
#   → dist\lifi-backend.exe  +  frontend\src-tauri\binaries\lifi-backend-x86_64-pc-windows-msvc.exe

# Stage 2 — Tauri (.exe + installers). Run in ONE cmd shell so vcvars persists.
#   NOTE: use the QUOTED set form — `set "VAR=value"` — or cmd captures the
#   space before `&&` into CARGO_TARGET_DIR and cargo fails with os error 3.
cmd /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && set "CARGO_TARGET_DIR=D:\cargo-target\lifi-pv" && cd /d frontend && npm run tauri:build'
# Output (2026-06-01, incremental Rust ~41 s):
#   D:\cargo-target\lifi-pv\release\bundle\msi\OptiSim_0.1.0_x64_en-US.msi
#   D:\cargo-target\lifi-pv\release\bundle\nsis\OptiSim_0.1.0_x64-setup.exe
#   D:\cargo-target\lifi-pv\release\OptiSim.exe (portable)
```

---

## 7. How to run today

### Legacy GUI (still the most-validated path)

```powershell
.\venv\Scripts\python.exe cli.py gui
```

### New React UI in dev mode (browser, two-terminal)

```powershell
# Terminal 1 — backend
.\venv\Scripts\python.exe -m uvicorn backend.app:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm run dev
# → http://localhost:5173
```

### CLI (always available)

```powershell
.\venv\Scripts\python.exe cli.py test          # 14 module self-tests
.\venv\Scripts\python.exe cli.py validate      # generate 23 paper figures
.\venv\Scripts\python.exe cli.py pipeline --preset kadirvelu2021
.\venv\Scripts\python.exe cli.py compare       # cross-preset validation
.\venv\Scripts\python.exe cli.py components    # list 14 components
```

---

## 8. Validation reference points

| Preset | Modulation | Data rate | Target BER | Notes |
|---|---|---|---|---|
| `kadirvelu2021` | OOK_Manchester | 2.5 kbps | < 1e-3 | Primary reference design |
| `sarwar2017` | OFDM (256-FFT, 16-QAM) | ~5 Mbps | — | 80 subcarriers |
| `xu2024` | BFSK | — | — | Multi-cell reconfigurable PV |
| `oliveira2024` | OFDM | — | — | Enhanced RX with notch filtering |
| `gonzalez2024` | OOK_Manchester | — | — | Temperature-compensated |
| `correa2025` | PWM_ASK | — | — | PWM-ASK with DC-DC integration |
| `lifi_poc_breadboard` | PWM_ASK | 1 kbps | 1e-3 | Starter-kit build: 5 mm LED + small PV + TL072 dual-stage + ESP32. `rx_topology=amp_slicer`, `bpf_stages=0`, `dcdc_enable=false`. `n_bits=20` so BER is informational only — bump n_bits for any publishable claim. See [lifi_poc_breadboard_simulation_notes.txt](docs/lifi_poc_breadboard_simulation_notes.txt) for a per-block parameter walkthrough. |
| `ieee_802_11bb` | OFDM (256-FFT, 64-QAM) + LDPC 5/6 | 40.5 Mbps payload | 2e-3 post-FEC | **Standards preset** (not a paper). IEEE 802.11bb-2023 HE PHY at 20 MHz, MCS 7. Link tuned to MCS 7 sensitivity (~21 dB SNR). Uncoded channel BER ~6e-3; LDPC 5/6 (pyldpc, regular Gallager construction) reduces to ~2e-3 with hard-decision input. Suite's only quantitatively informative OFDM benchmark (other OFDM presets pass via Wilson CI with BER=0). |

Pipeline validation: **20% error threshold** for PASS. Per CLAUDE.md, 5 of 7
pipeline validations currently fail (independent bugs flagged as next-after-
refactor work).

---

## 9. Known gaps / next work

### Resolved since 2026-05-16
- ✅ **Eye-diagram + BER-vs-SNR + spectrum + decision-scatter + noise
  breakdown** Plotly panels — all landed in Phase 7.
- ✅ **SPICE results-panel bug** — `backend/spice_waveforms.py` now parses
  the `.raw` file and emits the same `waveforms` shape as the Python path,
  so EngineRoute populates regardless of engine.
- ✅ **AC analysis route** — closed-form Bode + GM/PM via
  `cosim/ac_analysis.py` (no SPICE round-trip needed).
- ✅ **PHY compliance picker (MVP)** — IEEE 802.15.7 PHY-I OOK profile +
  PASS/FAIL banner; the architecture is open for additional PHY modes.
- ✅ **IEEE 802.11bb preset** (Phase 8, 2026-05-27) — HE PHY 20 MHz MCS 7
  preset registered, both as a standards anchor and as the suite's first
  quantitatively informative OFDM benchmark.
- ✅ **LDPC FEC layer** (Phase 8) — `cosim/fec.py` generic interface,
  pyldpc-backed. Opt-in via SimConfig.fec_enable; other presets unaffected.
- ✅ **OFDM SNR-injection bug** (Phase 8) — `python_engine.py` was using
  DC photocurrent as the OFDM signal, overstating SNR by 6-12 dB and
  hiding distance/QAM-order sensitivity in BER. Fixed to use the AC swing
  (`np.std(I_ph)`); BER now matches AWGN theory across the QAM grid.

### Resolved 2026-05-27 (Phase 9–11)
- ✅ **Parameter validation guardrails** (Phase 9) — `cosim/validation.py`
  with 5 rule families; `SystemConfig.validate()`; structured `issues[]` on
  the `/api/config/validate` endpoint; BuilderRail color-coded panel; new
  `python cli.py validate-config` subcommand + pre-flight on `cmd_pipeline`.
- ✅ **KiCad export wired to UI** (Phase 10) — `/api/kicad/{presets,export,
  download/...}` + KicadExportCard in BuilderRail.
- ✅ **InspectorDock** (Phase 10) — `/inspector` route with 4 tabs:
  Schematics (in-page schemdraw SVG render), Validation (grouped issues),
  Messages (WS ring-buffer log), Probes (signal stats + Plotly trace).
- ✅ **Real schematics in-page** (Phase 10) — `/api/schematic/{preset}/{idx}`
  renders the existing 34 hand-tuned schemdraw drawings to SVG; embedded
  via `<img>` in the InspectorRoute Schematics tab.
- ✅ **PM/GM convention note** (Phase 11) — informational caveat in the
  BodePanel header explaining open-loop band-pass conventions.
- ✅ **n_bits bump in validation harnesses** (Phase 11) — `kadirvelu2021`
  100→10000, `xu2024` 20→200. Xu flipped to PASS; Kadirvelu now exposes
  a real BER mismatch (49% vs target 1e-3) that was previously masked by
  small-sample noise — tracked as a new gap #3 below.
- ✅ **Rebuild the .exe / MSI** — rebuilt 2026-05-27 with Phases 9 only;
  Phases 10–12 are post-build so the on-disk artifacts are stale again.
  Per current policy, defer next rebuild until the whole agenda lands.
- ✅ **Constellation diagram for OFDM/BFSK** — already implemented in
  `ConstellationPanel.tsx` (OFDM: I/Q scatter from `ofdm_iq_real`/
  `ofdm_iq_imag`; BFSK: `(E0, E1)` Goertzel energy plane with bit colouring;
  OOK/Manchester/PWM-ASK: 1-D decision scatter). Backend already ships the
  payload via `pipeline_ws.py`. STATUS gap entry was stale — verified end
  to end across sarwar2017 / oliveira2024 / ieee_802_11bb / xu2024 on
  2026-05-27.
- ✅ **Soft demap for FEC** (Phase 13) — per-bit max-log-MAP LLRs feed
  pyldpc's BP decoder; ieee_802_11bb post-FEC BER dropped 2e-3 → 0 errors
  in 51490 message bits (Wilson UB ≈ 7.5e-5). Pre-FEC `ber_uncoded`
  unchanged at 6.17e-3. Hard-bit fallback retained for non-OFDM users.

### Still open
1. **Code signing** — the current MSI/NSIS/EXE are unsigned; Windows shows
   a SmartScreen warning on first launch. For public distribution this
   needs an Authenticode certificate (~$200-400/yr from DigiCert, Sectigo,
   etc.) and the `windows.certificateThumbprint` / `timestampUrl` fields
   filled in `frontend\src-tauri\tauri.conf.json`.
2. **SPICE-via-sidecar** — the bundled `lifi-backend.exe` does not include
   LTspice or ngspice. The pipeline auto-falls back to Python; full SPICE
   simulation requires the user to have LTspice or ngspice installed
   separately. Acceptable degradation for the first .exe but should be
   addressed (option: ship ngspice in `bundle.resources`).
3. **Pipeline validation comparator: 8/8 PASS (2026-05-27 after Phase 12).**
   Two pre-existing flagged follow-ups remain (TODO(verify) inline in
   `_PAPER_METRICS`): Kadirvelu P_rx/I_ph ~16% above target despite
   channel_gain matching to 0.1% (suspect P_tx unit issue or responsivity-
   tempco interaction); Oliveira data_rate ~19.5% above target (unmodelled
   OFDM overhead beyond cyclic prefix — pilots, headers). Both are
   amplitude/throughput slips on already-PASSing presets, not test-status
   failures.
4. **Additional PHY profiles** — IEEE 802.15.7 PHY-II/III, ITU-T G.9991.
   IEEE 802.11bb landed in Phase 8 as a *preset* (not yet as a
   `cosim/standards/` PHY profile with PASS/FAIL banner); wiring it through
   the `PhyProfile` system so it surfaces in the StandardsModal is a ~1 day
   follow-up. Each remaining profile is ~1 day on top of the existing
   architecture.
5. **802.11n QC-LDPC base matrices** — Phase 8 uses pyldpc's regular
   Gallager construction at the standardized rates and codeword lengths.
   Using the literal 802.11n quasi-cyclic base matrices is bit-exact
   standards compliance; the code rate and decoder are already
   standards-class but the parity structure differs. Future thesis work.
6. **Monte Carlo inner-noise seeding (latent bug exposed 2026-05-27).**
   `cosim/monte_carlo.py::run_monte_carlo` seeds the tolerance
   perturbation but not the inner simulator's noise model, so repeated
   runs at zero tolerance produce non-identical BER samples. Before the
   Phase 12 SOS-filter fix this was hidden (Kadirvelu always returned
   BER ≈ 0.5 regardless of seed). The `test_zero_tolerance_is_deterministic`
   test in `tests/test_paper_validation.py:500` now fails for this
   reason. Fix is a 1-line `np.random.seed(per_run_seed)` or equivalent
   inside the MC loop; left as a separate gap so we don't entangle FEC
   work with MC infrastructure.
---

## 10. Key conventions

- Paper parameters are **locked** — never modify values from publications.
- Component values come from datasheets, not approximations.
- All figures publication-quality (≥150 DPI, matplotlib).
- Validation threshold: **20% error** for PASS.
- `SystemConfig.from_preset(name)` is the canonical config entrypoint.
- React animations: transform/opacity only (never width/height/top/left).
- `PlotCanvas` never inside `motion.div` with `layout`.

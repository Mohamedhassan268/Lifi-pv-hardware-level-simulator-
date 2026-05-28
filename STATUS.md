# Project Status — Hardware-Faithful LiFi/PV Simulator

**Last updated:** 2026-05-27

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
| Frozen sidecar binary | ✅ Built (71.8 MB) | `dist/lifi-backend.exe` |
| Tauri sidecar binary (with target triple) | ✅ Copied | `frontend/src-tauri/binaries/lifi-backend-x86_64-pc-windows-msvc.exe` |
| Frontend Vite build (`npm run build`) | ✅ Builds cleanly (after `@types/node` + `vite-env.d.ts` fixes) | `frontend/dist/` |
| Tauri icons | ✅ Generated from placeholder source.png | `frontend/src-tauri/icons/` |
| Rust toolchain | ✅ rustc 1.95.0, cargo 1.95.0 | — |
| MSVC linker (`link.exe`) | ✅ Available via `vcvars64.bat` (MSVC 14.51.36231 + Windows 11 SDK 10.0.26100.0) | — |
| Final `.msi` installer | ✅ Built | `dist\installer\LiFi PV Simulator_0.1.0_x64_en-US.msi` |
| NSIS `.exe` installer | ✅ Built (bonus output from `targets: all`) | `dist\installer\LiFi PV Simulator_0.1.0_x64-setup.exe` |

### Reproducing the build

```powershell
# 1. Disable Avast shields for 10 minutes (tray icon → Avast shields control)
# 2. From repo root:
& "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$env:CARGO_TARGET_DIR = "D:\cargo-target\lifi-pv"
cd frontend
npm run tauri:build
# Output:
#   D:\cargo-target\lifi-pv\release\bundle\msi\LiFi PV Simulator_0.1.0_x64_en-US.msi
#   D:\cargo-target\lifi-pv\release\bundle\nsis\LiFi PV Simulator_0.1.0_x64-setup.exe
#   D:\cargo-target\lifi-pv\release\LiFi PV Simulator.exe (portable)
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

# Phase 5 — Multi-Paper Pipeline Validation ✅ COMPLETED

All 5 tasks done. Created `papers/pipeline_validation.py` with `validate_preset()`,
`validate_all()`, `cross_validate()`. Added CLI `compare` command with `--preset`,
`--cross` flags. Generates 8 figures: BER summary, per-paper detail (×6), radar
summary, cross-validation. 14 test groups pass, 6/6 standalone validations pass.
2/7 pipeline presets pass (González, Xu) — remaining 5 need SNR/noise tuning
in future phases.

---
---

# Phase 4 — Generalized Configuration & Paper-Agnostic Pipeline ✅ COMPLETED

All 6 tasks done. Added `rx_topology` and `dcdc_enable` to SystemConfig with
auto-detection. Removed Kadirvelu-specific hardcoding from pipeline — netlist
generation is now config-driven for all 3 topologies (ina_bpf_comp, amp_slicer,
direct). Python engine respects topology. All 7 presets load and run. 14 test
groups pass (71 pytest).

## Summary

Make the pipeline truly paper-agnostic: any preset should flow through the same
pipeline without hardcoded paper-specific imports. Expand SystemConfig to describe
the full RX architecture, update all 7 presets to be self-describing, and add
a configurable RX chain topology so papers without INA/BPF/comparator/DCDC
work correctly through both Python and SPICE paths.

---

## Task 1 — Expand SystemConfig for Generalized Architecture
**Status:** `[ ]`
**Files:** `cosim/system_config.py`

The current SystemConfig has fields for Kadirvelu's architecture but doesn't
properly describe other papers' RX chains (e.g., González uses voltage amp
instead of INA, Sarwar/Xu/Oliveira have no BPF or comparator).

- [ ] **1.1** Add `rx_topology` field: `'ina_bpf_comp'` (Kadirvelu),
      `'amp_slicer'` (González), `'direct'` (Sarwar/Xu/Oliveira)
      - `ina_bpf_comp`: R_sense → INA → BPF(×N) → Comparator
      - `amp_slicer`: R_sense → voltage amp → notch → slicer
      - `direct`: R_sense → (optional BPF) → demodulator (digital domain)
- [ ] **1.2** Add `dcdc_enable` field (bool, default True for Kadirvelu,
      False for others that set dcdc_fsw_kHz=0)
- [ ] **1.3** Add `vcc_volts` default derivation: use existing value or 3.3
- [ ] **1.4** Derive `rx_topology` automatically in `__post_init__` from existing
      fields when not explicitly set (backward compat):
      - If `ina_gain_dB > 0` and `bpf_stages > 0` → `ina_bpf_comp`
      - If `amp_gain_linear > 1` and `bpf_stages == 0` → `amp_slicer`
      - Otherwise → `direct`
- [ ] **1.5** Auto-set `dcdc_enable` from `dcdc_fsw_kHz > 0` when not explicit
- [ ] **1.6** Validate new fields in `__post_init__`

---

## Task 2 — Remove Kadirvelu-Specific Hardcoding from Pipeline
**Status:** `[ ]`
**Files:** `cosim/pipeline.py`

The current `_generate_rx_netlist()` imports `KadirveluParams` and
`SubcircuitLibrary` directly. This must be generalized.

- [ ] **2.1** Replace `from systems.kadirvelu2021_netlist import SubcircuitLibrary`
      with a config-driven netlist generator
- [ ] **2.2** Generate subcircuit definitions directly from SystemConfig fields:
      - Solar cell: from `sc_*` fields
      - INA: from `ina_gain_dB`, `ina_gbw_kHz` (skip if `ina_gain_dB == 0`)
      - BPF: from `bpf_*` fields (skip if `bpf_stages == 0`)
      - Comparator: from `comparator_*` fields (skip if `comparator_part == 'N/A'`)
      - DC-DC: from `dcdc_*` fields (skip if `dcdc_enable == False`)
- [ ] **2.3** Netlist topology adapts to `rx_topology`:
      - `ina_bpf_comp`: Full chain as before
      - `amp_slicer`: R_sense → E-source amp → comparator
      - `direct`: R_sense → output (for SPICE validation only)
- [ ] **2.4** Both inline and subcircuit paths use config-driven values
- [ ] **2.5** Backward compatible: Kadirvelu preset produces identical netlist

---

## Task 3 — Config-Driven Python RX Chain
**Status:** `[ ]`
**Files:** `cosim/python_engine.py`

The Python engine needs to respect `rx_topology` to handle papers with
different receiver architectures.

- [ ] **3.1** In `run_python_simulation()`, branch on `cfg.rx_topology`:
      - `ina_bpf_comp`: Use existing ReceiverChain (if rx_chain_enable)
        or TIA+BPF path
      - `amp_slicer`: Apply voltage amplifier gain + notch + threshold
      - `direct`: Skip analog chain, pass I_ph directly to demodulator
- [ ] **3.2** For `direct` topology, compute V_rx = I_ph * R_sense
      (no TIA/INA/BPF — demodulation works in current/voltage domain)
- [ ] **3.3** For `amp_slicer`, use existing `amp_gain_linear` and
      `notch_freq_hz` fields already in SystemConfig
- [ ] **3.4** Backward compatible: existing behavior unchanged when
      `rx_topology='ina_bpf_comp'`

---

## Task 4 — Update All 7 Presets
**Status:** `[ ]`
**Files:** `presets/*.json`

Ensure all presets are self-describing with the new fields and
correct `simulation_engine` settings.

- [ ] **4.1** `kadirvelu2021.json`: Add `rx_topology: "ina_bpf_comp"`,
      `dcdc_enable: true` (explicit)
- [ ] **4.3** `gonzalez2024.json`: Add `rx_topology: "amp_slicer"`,
      `dcdc_enable: false`
- [ ] **4.4** `correa2025.json`: Add `rx_topology: "amp_slicer"`,
      `dcdc_enable: false`
- [ ] **4.5** `sarwar2017.json`: Add `rx_topology: "direct"`,
      `dcdc_enable: false`
- [ ] **4.6** `xu2024.json`: Add `rx_topology: "direct"`,
      `dcdc_enable: false`
- [ ] **4.7** `oliveira2024.json`: Add `rx_topology: "direct"`,
      `dcdc_enable: false`

---

## Task 5 — Preset Smoke Tests
**Status:** `[ ]`
**Files:** `tests/test_suite.py`

- [ ] **5.1** Add test: load all 7 presets via `SystemConfig.from_preset()`
- [ ] **5.2** Add test: run `run_python_simulation()` for each preset
      (verify it returns valid result dict with required keys)
- [ ] **5.3** Add test: verify BER is within order of magnitude of target
      for presets that have `target_ber` set
- [ ] **5.4** Add test: `rx_topology` auto-detection matches explicit values

---

## Task 6 — Regression & Validation
**Status:** `[ ]`

- [ ] **6.1** All existing 13 test groups pass
- [ ] **6.2** New preset smoke tests pass (all 7 presets)
- [ ] **6.3** Kadirvelu preset produces identical pipeline output
- [ ] **6.4** Each paper's Python simulation runs without errors
- [ ] **6.5** Update CLAUDE.md Phase 4 status to Done

---

## Execution Order

```
Task 1 (SystemConfig) ──→ Task 4 (update presets)
                       ──→ Task 2 (pipeline netlist)
                       ──→ Task 3 (python engine)
                       ──→ Task 5 (smoke tests)
                       ──→ Task 6 (regression)
```

Start with Task 1 (config fields), then Tasks 2-4 in parallel
(independent changes), then Task 5 (tests), Task 6 (validate).

---
---

# Phase 1 — Wireless/Channel Engine (Python) ✅ COMPLETED

All 7 tasks done. Created `cosim/channel.py`, `cosim/noise.py`, `cosim/modulation.py`.
Rewired `python_engine.py` and `pipeline.py`. All 13 tests pass.

---

# Phase 2 — Enhanced Python RX Engine ✅ COMPLETED

All 8 tasks done. Created `cosim/pv_model.py`, `cosim/rx_chain.py`,
`cosim/dcdc_model.py`, `cosim/tx_model.py`, `cosim/engine_compare.py`.
Rewired `python_engine.py` with feature flags. All 13 tests pass.

---

## Task 1 — `cosim/pv_model.py` : PV Cell ODE with Voltage-Dependent Capacitance
**Status:** `[x]`
**Files:** new `cosim/pv_model.py`, modify `cosim/python_engine.py`

The current `PVReceiver.optical_to_current()` is a simple `I = R * P_rx` —
no junction dynamics. The system-level sim used a full ODE solver.

**What exists already:**
- `physics/pn_junction.py:109-143` — `junction_capacitance(V, area, Na, Nd, material)`
- `components/solar_cells.py:235-251` — `C(V) = C_j0 / (1 + V/V_bi)^M`
- SPICE subcircuit: `Gph`, `Rs`, `Cj`, `Rsh`, `D1` (in `kadirvelu2021_netlist.py`)

**What's missing:** ODE transient solver coupling I_ph, V_cell, C_j(V).

- [ ] **1.1** Single-diode equivalent circuit ODE:
      ```
      C_j(V) · dV/dt = I_ph(t) - I_dark(V) - V/R_sh - I_load
      I_dark = I_s · (exp(V / n·V_T) - 1)
      C_j(V) = C_j0 / (1 - V/V_bi)^m     (m = 0.5 for abrupt junction)
      ```
- [ ] **1.2** Use `scipy.integrate.solve_ivp` with Radau (stiff) solver
- [ ] **1.3** Safe exponential: `exp(min(V/nV_T, 40))` to prevent overflow
- [ ] **1.4** Output: `V_cell(t)`, `I_cell(t)` time-domain waveforms
- [ ] **1.5** `from_config(cfg)` factory using SystemConfig PV parameters
- [ ] **1.6** Validate against SPICE: compare V_cell transient for step input
- [ ] **1.7** Unit tests in `tests/test_pv_model.py`

**Source ref:** `simulator/receiver.py` — PV cell ODE with Radau/Euler fallback

---

## Task 2 — `cosim/rx_chain.py` : Receiver Signal Chain (INA + BPF + Comparator)
**Status:** `[ ]`
**Files:** new `cosim/rx_chain.py`, modify `cosim/python_engine.py`

The current chain is: `apply_tia()` (1-pole LPF) → `demodulate()`. Missing the
actual INA gain stage, 2-stage BPF, and comparator with propagation delay.

**What exists already:**
- `PVReceiver.apply_tia()` — single-pole Butterworth (too simple)
- `PVReceiver.apply_bandpass()` — generic Butterworth BPF
- SPICE subcircuits: INA322 (2-pole, Vref offset), BPF_STAGE (HP + active LP), COMPARATOR (RC delay)

- [ ] **2.1** `RsenseStage`: I_cell → V_sense = I_cell × R_sense
- [ ] **2.2** `INAStage`: 2-pole frequency response matching INA322 subcircuit
      - Gain = 100.5 (40 dB), GBW = 700 kHz → f_3dB = 7 kHz
      - Pole 1 (dominant): 6965 Hz, Pole 2: 69652 Hz
      - Vref offset: V_out = G·(V_inp - V_inn) + Vref
      - Rail clamping: `clip(V_out, Vee+0.05, Vcc-0.05)`
- [ ] **2.3** `BPFStage`: Match the TLV2379-based active BPF subcircuit
      - High-pass: C_hp (470nF) + R_hp (100k) → f_HP = 3.4 Hz
      - Low-pass (inverting): R_in (10k) + R_fb (10k) + C_fb (1.5nF) → f_LP = 10.6 kHz
      - Passband gain: ×1 (inverting)
      - Cascadable: `bpf_stages` config field (default: 2)
- [ ] **2.4** `ComparatorStage`: Match TLV7011 subcircuit
      - Decision: V_out = Vcc if V_inp > V_inn, else Vee
      - Propagation delay: 260 ns (RC model: R=1k, C=260p)
      - Hysteresis: optional configurable threshold offset
- [ ] **2.5** `ReceiverChain` class chaining all stages:
      `R_sense → INA → BPF(×N) → Comparator → digital output`
- [ ] **2.6** `from_config(cfg)` factory
- [ ] **2.7** Validate: compare Python chain output vs SPICE waveforms at each node
- [ ] **2.8** Unit tests in `tests/test_rx_chain.py`

**Source ref:** `simulator/receiver.py` — TIA, INA, BPF chain; SPICE subcircuits in netlist

---

## Task 3 — `cosim/dcdc_model.py` : Boost DC-DC Converter (Python)
**Status:** `[ ]`
**Files:** new `cosim/dcdc_model.py`, modify `cosim/python_engine.py`

The Python engine currently has no DC-DC model at all (SPICE-only).

**What exists already:**
- SPICE subcircuit: `BOOST_DCDC` with L, NMOS switch, Schottky, caps, Rload
- `SystemConfig`: `dcdc_fsw_kHz`, `dcdc_l_uH`, `dcdc_cp_uF`, `dcdc_cl_uF`, `r_load_ohm`

- [ ] **3.1** Steady-state duty cycle: `D = 1 - V_in/V_out`
- [ ] **3.2** CCM/DCM boundary detection:
      `I_LB = V_in·D·(1-D)² / (2·L·f_sw)` — CCM if I_load > I_LB
- [ ] **3.3** CCM output voltage: `V_out = V_in / (1-D)`
- [ ] **3.4** DCM output voltage (from energy balance)
- [ ] **3.5** Loss model:
      - Conduction: `P_cond = I_rms² × R_ds_on` (NTS4409: 52 mΩ)
      - Switching: `P_sw = 0.5 × V_out × I_L × (t_r + t_f) × f_sw`
      - Diode: `P_diode = V_f × I_out` (Schottky: V_f ≈ 0.3V)
      - Inductor DCR: `P_dcr = I_rms² × R_dcr`
- [ ] **3.6** Efficiency: `η = P_out / (P_out + P_losses)`
- [ ] **3.7** Time-domain inductor current waveform (triangular in CCM)
- [ ] **3.8** `from_config(cfg)` factory
- [ ] **3.9** Validate: compare η vs SPICE at 50/100/200 kHz switching
- [ ] **3.10** Unit tests in `tests/test_dcdc.py`

**Source ref:** `simulator/dc_dc_converter.py` — CCM/DCM, loss model, efficiency curves

---

## Task 4 — `cosim/tx_model.py` : LED Transmitter Model
**Status:** `[ ]`
**Files:** new `cosim/tx_model.py`, modify `cosim/python_engine.py`

The current modulation dispatch (`cosim/modulation.py`) converts bits to
`P_tx(t)` but doesn't model the LED's frequency response or driver dynamics.

**What exists already:**
- `components/leds.py` — LXM5-PD01 with `modulation_bandwidth()` (RC + carrier)
- `components/base.py:332-361` — `LEDBase.modulation_bandwidth(R_drive)`
- SPICE: TX_DRIVER subcircuit with ADA4891 + LED + optical output

- [ ] **4.1** LED frequency response: single-pole LPF at `f_3dB` from component
      `f_3dB = modulation_bandwidth(R_drive)` — typically ~6.6 MHz for LXM5-PD01
- [ ] **4.2** LED I-V → optical power: `P_opt = η_LED × I_LED` with clamping
- [ ] **4.3** Lens transmittance: `P_tx = T_lens × P_opt`
- [ ] **4.4** Apply bandwidth limit to modulated signal before channel
- [ ] **4.5** `from_config(cfg)` factory using LED part from COMPONENT_REGISTRY
- [ ] **4.6** Unit tests in `tests/test_tx_model.py`

**Source ref:** `simulator/transmitter.py` — LED bandwidth limiting

---

## Task 5 — Update `SystemConfig` for Phase 2
**Status:** `[ ]`
**Files:** `cosim/system_config.py`

- [ ] **5.1** PV ODE fields: `pv_ode_enable` (bool), `pv_dark_current_A` (I_s),
      `pv_ideality_factor` (n), `pv_vbi_V` (built-in voltage)
- [ ] **5.2** DC-DC fields: `dcdc_rds_on_mohm`, `dcdc_diode_vf_V`, `dcdc_inductor_dcr_ohm`
- [ ] **5.3** TX fields: `led_bandwidth_limit_enable` (bool)
- [ ] **5.4** All new fields default to values that reproduce Phase 1 behavior
      (e.g., `pv_ode_enable=False` keeps the simple `I=R×P` model)
- [ ] **5.5** Update `__post_init__` validation

---

## Task 6 — Rewire `run_python_simulation()` to Use New Models
**Status:** `[ ]`
**Files:** `cosim/python_engine.py`

- [ ] **6.1** Replace `PVReceiver` with `PVCellModel` (ODE) when `pv_ode_enable=True`
- [ ] **6.2** Replace simple TIA+demod chain with `ReceiverChain` pipeline
- [ ] **6.3** Add TX bandwidth limiting via `TXModel` when `led_bandwidth_limit_enable=True`
- [ ] **6.4** Add DC-DC output computation via `BoostConverter`
- [ ] **6.5** Return expanded result dict with per-node waveforms:
      `V_sense`, `V_ina`, `V_bpf1`, `V_bpf2`, `V_comp`, `V_dcdc`
- [ ] **6.6** Backward compatible: with all new features disabled, output matches Phase 1

---

## Task 7 — SPICE vs Python Comparison Tool
**Status:** `[ ]`
**Files:** new `cosim/engine_compare.py`

- [ ] **7.1** Run same config through both engines
- [ ] **7.2** Overlay waveforms at each node (SPICE vs Python)
- [ ] **7.3** Report per-node RMS error and correlation
- [ ] **7.4** Target: <5% RMS error at each node for default Kadirvelu config

---

## Task 8 — Regression & Validation
**Status:** `[ ]`

- [ ] **8.1** All existing tests pass (13 groups)
- [ ] **8.2** New unit tests for pv_model, rx_chain, dcdc, tx_model all pass
- [ ] **8.3** `run_python_simulation()` with features disabled matches Phase 1 output
- [ ] **8.4** Python vs SPICE waveform comparison within tolerance
- [ ] **8.5** GUI still launches and runs both engines

---

## Execution Order

```
Task 5 (SystemConfig)  ──→  Task 1 (pv_model.py)     ──→  Task 6 (rewire python_engine)
                         ──→  Task 2 (rx_chain.py)     ──→  Task 7 (SPICE comparison)
                         ──→  Task 3 (dcdc_model.py)   ──→  Task 8 (regression)
                         ──→  Task 4 (tx_model.py)
```

Start with Task 5 (config), then Tasks 1-4 in parallel (independent models),
then Task 6 wires them into the engine, Task 7 validates against SPICE,
and Task 8 ensures nothing broke.

---

## Key Formulas Reference

### PV Cell ODE
```
C_j(V) · dV/dt = I_ph(t) - I_s·(exp(V/nV_T) - 1) - V/R_sh - I_load
C_j(V) = C_j0 / (1 - V/V_bi)^0.5
```

### INA322 Transfer Function
```
H(s) = G / ((1 + s/ω₁)(1 + s/ω₂))
G = 100.5,  ω₁ = 2π×6965,  ω₂ = 2π×69652
```

### BPF Stage Transfer Function
```
H_HP(s) = s·R_hp·C_hp / (1 + s·R_hp·C_hp)         [f_HP = 3.4 Hz]
H_LP(s) = -1 / (1 + s·R_fb·C_fb)                    [f_LP = 10.6 kHz]
H_BPF = H_HP × H_LP
```

### Boost Converter
```
CCM: V_out = V_in / (1 - D)
CCM boundary: I_LB = V_in·D·(1-D)² / (2·L·f_sw)
Efficiency: η = P_out / (P_out + P_cond + P_sw + P_diode + P_dcr)
```

---
---

# Phase 3 — Unified SPICE <-> Python Pipeline ✅ COMPLETED

All 8 tasks done. Created `cosim/ngspice_runner.py`, `cosim/spice_extract.py`,
`cosim/sim_result.py`. Refactored `cosim/pipeline.py` for hybrid mode,
multi-modulation SPICE, PWL noise injection. Enhanced `cosim/engine_compare.py`
with per-node FFT comparison. All 13 tests pass + 8 regression checks.

---

## Task 1 — Unified Pipeline Orchestrator
**Status:** `[ ]`
**Files:** modify `cosim/pipeline.py`

The current `SimulationPipeline` has two separate paths: `run_all()` (SPICE)
and `run_python_engine()` (Python). These need to merge into a single flow
that picks the best engine per stage.

- [ ] **1.1** Refactor `run_all()` into a hybrid pipeline:
      - TX: always Python (modulation dispatch)
      - Channel: always Python (optical propagation)
      - RX: SPICE if available + `simulation_engine='spice'`, else Python
      - DC-DC: SPICE if in netlist, else Python `BoostConverter`
      - BER: always Python (works with both engine outputs)
- [ ] **1.2** New `run_hybrid()` method that:
      1. Runs Python TX+channel to produce P_rx(t)
      2. Writes P_rx PWL bridge file
      3. Generates SPICE netlist with noise injection
      4. Runs SPICE for RX circuit
      5. Parses .raw, extracts V(dout)
      6. Runs Python BER computation
- [ ] **1.3** Unified result dict: same keys regardless of engine
      ```python
      {'time', 'P_tx', 'P_rx', 'I_ph', 'V_rx', 'V_comp',
       'bits_tx', 'bits_rx', 'ber', 'snr_est_dB', 'engine',
       'channel_gain', 'P_rx_avg_uW', 'I_ph_avg_uA'}
      ```
- [ ] **1.4** `StepResult` gains an `engine` field ('spice'|'python'|'hybrid')
- [ ] **1.5** Backward compatible: existing `run_all()` and `run_python_engine()`
      still work as before

---

## Task 2 — SPICE Noise Injection via PWL
**Status:** `[ ]`
**Files:** modify `cosim/pipeline.py`, `cosim/pwl_writer.py`

The current `_noise_section()` uses behavioral `white()` noise sources which
are LTspice-specific and not physically calibrated. Replace with PWL noise
injection using the Python `NoiseModel` (which is calibrated to 6 sources).

- [ ] **2.1** `write_noise_pwl()` in `pwl_writer.py`:
      - Takes `NoiseModel`, time array, photocurrent level
      - Generates time-domain noise samples
      - Writes as PWL current source file
      - Downsamples to max 10000 points for SPICE efficiency
- [ ] **2.2** Update `_generate_rx_netlist()` to use PWL noise:
      - Replace `Bn_sum` behavioral source with `Inoise ... PWL file="noise.pwl"`
      - Noise injected at INA input (current domain) rather than output (voltage)
      - This matches the physical noise injection point
- [ ] **2.3** Seed management: use `config.random_seed` for reproducible noise
      across both engines
- [ ] **2.4** Toggle: when `noise_enable=False`, no noise PWL is generated

---

## Task 3 — Multi-Modulation SPICE Support
**Status:** `[ ]`
**Files:** modify `cosim/pipeline.py`

Currently the SPICE pipeline only supports OOK. The Python engine handles
all 5 modulation schemes. Phase 3 extends the SPICE path to accept any
modulation by generating the TX waveform in Python and feeding it via PWL.

- [ ] **3.1** Replace `generate_prbs + generate_ook_waveform` in `run_step_tx()`
      with the unified `modulate()` dispatch from `cosim.modulation`
- [ ] **3.2** Write modulated P_tx(t) as PWL regardless of scheme
- [ ] **3.3** For OFDM/BFSK/PWM-ASK: demodulation still uses Python
      (SPICE only processes the analog RX chain)
- [ ] **3.4** Store `bits_tx` for BER computation after SPICE RX
- [ ] **3.5** Test: OOK through SPICE pipeline matches previous behavior

---

## Task 4 — SPICE Result Extraction & Alignment
**Status:** `[ ]`
**Files:** modify `cosim/pipeline.py`, `cosim/raw_parser.py`

The SPICE .raw output and Python output need alignment for comparison
and for the demodulator to consume SPICE waveforms.

- [ ] **4.1** `extract_spice_waveforms()`: parse .raw → dict with standard keys
      `{'time', 'V_sense', 'V_ina', 'V_bpf1', 'V_bpf2', 'V_comp', 'V_dcdc'}`
      Maps SPICE node names to standardized keys
- [ ] **4.2** Time-domain resampling: SPICE uses variable timestep,
      Python uses uniform — interpolate SPICE to uniform grid for BER
- [ ] **4.3** `compute_ber_from_spice()`: extract V(dout), resample to
      uniform grid, threshold at Vcc/2, compare to `bits_tx`
      - Handles polarity inversion (BPF may invert)
      - Handles settling time (skip first N bits)
- [ ] **4.4** SNR estimation from SPICE waveforms:
      `SNR = (V_bpf_pp / 2)² / var(V_bpf - V_bpf_ideal)`

---

## Task 5 — ngspice Runner
**Status:** `[ ]`
**Files:** new `cosim/ngspice_runner.py`

The current ngspice path imports from `simulation.ngspice_runner` which may
not exist. Create a proper ngspice runner in cosim/ that mirrors LTSpiceRunner.

- [ ] **5.1** `NgSpiceRunner` class:
      - `__init__(path=None)`: auto-detect via `spice_finder.find_ngspice()`
      - `available` property
      - `run(cir_path, timeout_s=120) -> bool`
      - `get_raw_path() -> Optional[str]`
- [ ] **5.2** ngspice batch command: `ngspice -b -r output.raw <file.cir>`
- [ ] **5.3** ngspice .raw format differs from LTspice — handle both in
      `raw_parser.py` (ngspice writes all-float64 binary)
- [ ] **5.4** Netlist compatibility: ngspice uses `.control`/`.endc` blocks
      instead of LTspice `.MEAS` — generate compatible netlists
- [ ] **5.5** Update `pipeline.py` to use `NgSpiceRunner` directly from cosim
      (remove `simulation.ngspice_runner` dependency)

---

## Task 6 — Per-Node Waveform Export
**Status:** `[ ]`
**Files:** modify `cosim/pipeline.py`, `cosim/python_engine.py`

Both engines should export the same per-node waveform data for plotting
and comparison.

- [ ] **6.1** Define `SimulationResult` dataclass:
      ```python
      @dataclass
      class SimulationResult:
          time: np.ndarray
          P_tx: np.ndarray
          P_rx: np.ndarray
          I_ph: np.ndarray
          V_sense: Optional[np.ndarray]
          V_ina: Optional[np.ndarray]
          V_bpf: Optional[list]       # [V_bpf1, V_bpf2, ...]
          V_comp: Optional[np.ndarray]
          V_dcdc: Optional[np.ndarray]
          bits_tx: np.ndarray
          bits_rx: np.ndarray
          ber: float
          snr_est_dB: float
          engine: str                 # 'spice', 'python', 'hybrid'
          channel_gain: float
          dcdc_efficiency: Optional[float]
          noise_breakdown: Optional[dict]
      ```
- [ ] **6.2** `run_python_simulation()` returns `SimulationResult`
- [ ] **6.3** SPICE pipeline returns `SimulationResult` (via .raw extraction)
- [ ] **6.4** `save_results()` method: saves waveforms as `.npz` and
      summary as `.json` (replaces `_save_python_results()`)
- [ ] **6.5** Backward compat: `SimulationResult.to_dict()` returns the
      existing dict format

---

## Task 7 — Enhanced Engine Comparison
**Status:** `[ ]`
**Files:** modify `cosim/engine_compare.py`

Upgrade the Phase 2 comparison tool to work with the unified pipeline.

- [ ] **7.1** Compare per-node waveforms (not just I_ph and V_rx):
      V_sense, V_ina, V_bpf1, V_bpf2, V_comp
- [ ] **7.2** Time-alignment: handle different timestep grids between engines
- [ ] **7.3** Frequency-domain comparison: FFT of each node, overlay spectra
- [ ] **7.4** Eye diagram comparison at BPF output
- [ ] **7.5** Generate multi-page comparison report (PDF or multi-plot PNG)
- [ ] **7.6** CLI command: `python cli.py compare --preset kadirvelu2021`

---

## Task 8 — Regression & Validation
**Status:** `[ ]`

- [ ] **8.1** All existing 13 test groups pass
- [ ] **8.2** Python engine with all Phase 2 features still matches Phase 2 output
- [ ] **8.3** SPICE pipeline with OOK still produces valid .raw and BER
- [ ] **8.4** Hybrid pipeline (Python TX+Channel → SPICE RX → Python BER) works
- [ ] **8.5** Noise injection via PWL matches behavioral noise within 10%
- [ ] **8.6** All 5 modulation schemes work through hybrid pipeline
- [ ] **8.7** `SimulationResult` serialization round-trips correctly
- [ ] **8.8** GUI still launches and runs all engine modes

---

## Execution Order

```
Task 5 (ngspice runner)  ──→  Task 1 (unified pipeline)
Task 2 (noise PWL)       ──→  Task 3 (multi-mod SPICE)
Task 4 (result extraction)──→  Task 6 (SimulationResult)
                           ──→  Task 7 (enhanced compare)
                           ──→  Task 8 (regression)
```

Start with Tasks 2, 4, 5 in parallel (independent infrastructure),
then Task 1 + 3 (pipeline rewiring depends on 2/4/5),
then Task 6 (unified result format), Task 7 (comparison),
and Task 8 (validate everything).

---

## Key Integration Points

### PWL Bridge Files
```
Python TX → P_tx.pwl (optical power)
Python Channel → optical_power.pwl (received power at PV)
Python Noise → noise.pwl (calibrated noise current)
                ↓
        SPICE RX Circuit
                ↓
        .raw output → raw_parser → SimulationResult
```

### Engine Selection Logic
```python
if simulation_engine == 'python':
    # Pure Python: all stages in Python
elif simulation_engine == 'spice' and spice_available():
    # Hybrid: Python TX/Channel/Noise → SPICE RX → Python BER
else:
    # Fallback: Pure Python with warning
```

### Netlist Node Mapping
```
SPICE node          → Standard key
V(sc_anode)         → V_cell
V(sense_lo)         → V_sense  (but sense_lo = 0 due to Vgnd_ref)
I(Rsense)           → I_sense
V(ina_out)          → V_ina
V(bpf1_out)         → V_bpf[0]
V(bpf_out)          → V_bpf[1]
V(dout)             → V_comp
V(dcdc_out)         → V_dcdc
```

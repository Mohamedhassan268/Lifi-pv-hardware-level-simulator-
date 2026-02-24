# Integration Plan: System-Level Simulator → Hardware-Faithful Simulator

## Goal
Integrate 6-paper knowledge, presets, and validation targets from `C:\Users\HP OMEN\lifi_pv_simulator`
into the hardware-faithful simulator. Use a hybrid approach: SPICE pipeline for OOK-compatible papers,
Python system-level engine as fallback for OFDM/BFSK/PWM-ASK/MIMO papers.

---

## Phase 1: Extend SystemConfig for Multi-Paper Support

**File: `cosim/system_config.py`**

Add new fields to the SystemConfig dataclass:
- `simulation_engine`: `"spice"` | `"python"` — selects which engine runs the simulation
- `modulation`: extend from just `"OOK"` to also support `"OOK_Manchester"`, `"OFDM"`, `"BFSK"`, `"PWM_ASK"`
- `humidity_rh`: `Optional[float]` — relative humidity (0-1) for Beer-Lambert (Correa 2025)
- `n_cells_series`: `int = 1` — for multi-cell modules (Xu 2024)
- `ofdm_nfft`: `int = 256` — FFT size for OFDM papers
- `ofdm_qam_order`: `int = 16` — QAM modulation order
- `ofdm_n_subcarriers`: `int = 80` — active subcarriers
- `bfsk_f0_hz`: `float = 1600` — BFSK frequency 0
- `bfsk_f1_hz`: `float = 2000` — BFSK frequency 1
- `amp_gain_linear`: `float = 1.0` — voltage amplifier gain (González)
- `notch_freq_hz`: `Optional[float] = None` — notch filter frequency (González)
- `target_data_rate_mbps`: `Optional[float] = None` — for OFDM papers
- `target_fec_threshold`: `Optional[float] = None` — FEC BER threshold

---

## Phase 2: Create 5 New Paper Presets

**Directory: `presets/`**

### 2a. `sarwar2017.json` — OFDM 16-QAM over solar panel
- simulation_engine: "python"
- modulation: "OFDM"
- distance_m: 2.0, led_radiated_power_mW: 3000 (3W LED)
- sc_area_cm2: 7.5, sc_cj_nF: 0.1 (100 pF), sc_rsh_kOhm: 10
- ofdm_nfft: 256, ofdm_qam_order: 16, ofdm_n_subcarriers: 80
- data_rate_bps: 15030000, target_ber: 0.0016883

### 2b. `correa2025.json` — Greenhouse VLC with PWM-ASK
- simulation_engine: "python"
- modulation: "PWM_ASK"
- distance_m: 0.85, led_radiated_power_mW: 3000 (30W @ 10% eff)
- sc_area_cm2: 66.0, sc_responsivity: 0.5, sc_cj_nF: 50, sc_rsh_kOhm: 200
- humidity_rh: 0.5, r_sense_ohm: 220
- data_rate_bps: 10000

### 2c. `xu2024.json` — Sunlight-Duo reconfigurable BFSK
- simulation_engine: "python"
- modulation: "BFSK"
- n_cells_series: 2 (2s-8p default config)
- sc_area_cm2: 16.0 (16 cells × 1cm²), sc_responsivity: 0.457
- bfsk_f0_hz: 1600, bfsk_f1_hz: 2000
- data_rate_bps: 400

### 2d. `oliveira2024.json` — MIMO OFDM reconfigurable
- simulation_engine: "python"
- modulation: "OFDM"
- ofdm_nfft: 1024, ofdm_qam_order: 64, ofdm_n_subcarriers: 500
- sc_area_cm2: 0.693 (9 small PDs total), sc_responsivity: 0.36
- data_rate_bps: 25700000, target_data_rate_mbps: 21.3

### 2e. `gonzalez2024.json` — Low-cost Manchester OOK (SPICE-compatible!)
- simulation_engine: "spice" (can use hardware-faithful pipeline!)
- modulation: "OOK_Manchester"
- distance_m: 0.60, led_radiated_power_mW: 3000
- sc_area_cm2: 66.0, sc_responsivity: 0.4, sc_cj_nF: 14.5, sc_rsh_kOhm: 200
- amp_gain_linear: 165, notch_freq_hz: 100
- data_rate_bps: 4800, target_ber: 0.0

---

## Phase 3: Integrate Python System-Level Engine

**New file: `cosim/python_engine.py`**

Copy and adapt the core simulation classes from `lifi_pv_simulator/simulator/`:
- `Transmitter` class (OOK, Manchester, OFDM, PWM-ASK, BFSK modulation)
- `OpticalChannel` class (Lambertian + Beer-Lambert + MIMO H-matrix)
- `PVReceiver` class (photocurrent, PV junction ODE, TIA, BPF, notch filter)
- `NoiseModel` class (6-source noise)
- `Demodulator` class (HPF/LPF, Manchester decode, threshold decision)
- BER prediction functions (OOK, BPSK, M-QAM)

Key adapter function:
```python
def run_python_simulation(config: SystemConfig) -> dict:
    """Run system-level Python simulation using SystemConfig parameters."""
    # Map SystemConfig fields to simulator class params
    # Return dict compatible with pipeline StepResult format
```

---

## Phase 4: Dual-Engine Pipeline

**File: `cosim/pipeline.py`**

Modify `SimulationPipeline.run_all()`:
- Check `config.simulation_engine`
- If `"spice"`: use existing 3-step TX→Channel→RX SPICE flow (unchanged)
- If `"python"`: call `run_python_simulation()` from python_engine.py
- Both paths produce compatible result dicts with BER, waveforms, timing

**File: `cosim/pipeline.py` — new method:**
```python
def run_python_engine(self) -> Dict[str, StepResult]:
    """Run all-Python simulation for non-SPICE-compatible papers."""
```

---

## Phase 5: GUI Integration

### 5a. System Setup Tab (`gui/tab_system_setup.py`)
- Preset dropdown auto-populates with all 7 presets (2 existing + 5 new)
- Add "Simulation Engine" indicator (shows SPICE vs Python based on preset)
- Add paper-specific parameter fields that show/hide based on modulation type:
  - OFDM group: FFT size, QAM order, subcarriers
  - BFSK group: f0, f1
  - Environment group: humidity slider (Correa)
  - Notch filter: frequency (González)

### 5b. Simulation Engine Tab (`gui/tab_simulation_engine.py`)
- Detect engine type from config and run appropriate pipeline
- Python engine: show TX/Channel/RX steps inline (no LTspice needed)
- SPICE engine: existing flow (unchanged)
- Both produce waveform plots in the 4-panel preview

### 5c. Results Tab (`gui/tab_results.py`)
- Add validation comparison panel: simulated vs paper target values
- Show paper reference and key metrics
- BER vs Distance plot works for all papers (uses analytical model)

---

## Phase 6: Validation & Testing

### 6a. Per-paper validation tests (`tests/test_paper_validation.py`)
- For each paper preset, run Python engine and check against targets
- Kadirvelu: BER ≈ 1e-3, harvested power ≈ 223 µW
- Sarwar: data rate ≈ 15 Mbps, BER < 3.8e-3
- Correa: BER curves match Fig. 7 trends
- Xu: charging time ≈ 40 min, PSR > 90%
- Oliveira: SISO ≈ 21.3 Mbps net
- González: error-free at 4.8 kBd, 60 cm

### 6b. Integration tests
- Verify all 7 presets load correctly
- Verify SPICE pipeline still works for kadirvelu2021, fakidis2020, gonzalez2024
- Verify Python pipeline works for sarwar2017, correa2025, xu2024, oliveira2024

---

## File Summary

| Action | File | Description |
|--------|------|-------------|
| Modify | `cosim/system_config.py` | Add multi-paper fields |
| Create | `presets/sarwar2017.json` | OFDM paper preset |
| Create | `presets/correa2025.json` | Greenhouse paper preset |
| Create | `presets/xu2024.json` | Sunlight-Duo paper preset |
| Create | `presets/oliveira2024.json` | MIMO paper preset |
| Create | `presets/gonzalez2024.json` | Low-cost VLC preset |
| Create | `cosim/python_engine.py` | System-level Python sim engine |
| Modify | `cosim/pipeline.py` | Dual-engine dispatch |
| Modify | `gui/tab_system_setup.py` | Engine indicator + paper fields |
| Modify | `gui/tab_simulation_engine.py` | Python engine integration |
| Modify | `gui/tab_results.py` | Validation comparison panel |
| Create | `tests/test_paper_validation.py` | Per-paper validation tests |

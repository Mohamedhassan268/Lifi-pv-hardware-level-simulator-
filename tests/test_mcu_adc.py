"""MCU node: finite-rate ADC sampling + validator checks (Phase 18).

The MCU's `mcu_sample_rate_hz` band-limits V_rx before demodulation, modelling
the controller's real ADC. 0 = ideal ADC (no-op; all existing presets keep
their behaviour). The validator warns when the rate is sub-Nyquist or the clock
leaves too few cycles per sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from cosim.python_engine import _adc_sample, run_python_simulation  # noqa: E402
from cosim.system_config import SystemConfig  # noqa: E402
from cosim.validation import IssueLevel, validate_config  # noqa: E402


# ---------------------------------------------------------------------------
# 1. _adc_sample helper — no-op when fast, band-limits when slow.
# ---------------------------------------------------------------------------

def test_adc_sample_noop_when_faster_than_grid():
    t = np.linspace(0, 1e-3, 1000)
    v = np.sin(2 * np.pi * 1000 * t)
    # ADC far above the sim grid rate -> nothing to lose.
    out = _adc_sample(v, t, fs_adc=1e12)
    assert np.array_equal(out, v)


def test_adc_sample_zero_disables():
    t = np.linspace(0, 1e-3, 1000)
    v = np.sin(2 * np.pi * 1000 * t)
    assert np.array_equal(_adc_sample(v, t, fs_adc=0.0), v)


def test_adc_sample_aliases_sub_nyquist_tone():
    # A tone above the ADC Nyquist can't be reconstructed: the held-back signal
    # differs sharply from the original (aliasing), while a fast ADC tracks it.
    fs_sim = 1_000_000
    t = np.arange(0, 0.01, 1 / fs_sim)
    tone_hz = 40_000
    v = np.sin(2 * np.pi * tone_hz * t)

    slow = _adc_sample(v, t, fs_adc=10_000)   # Nyquist 5 kHz << 40 kHz -> aliases
    fast = _adc_sample(v, t, fs_adc=400_000)  # Nyquist 200 kHz >> 40 kHz -> tracks
    assert np.std(slow - v) > 0.5 * np.std(v), "sub-Nyquist tone should not reconstruct"
    assert np.std(fast - v) < 0.1 * np.std(v), "well-sampled tone should reconstruct"


# ---------------------------------------------------------------------------
# 2. End-to-end: a sub-Nyquist MCU ADC degrades BER; ideal ADC closes.
# ---------------------------------------------------------------------------

def test_slow_adc_degrades_ber():
    base = SystemConfig.from_preset('kadirvelu2021')
    base.simulation_engine = 'python'
    base.n_bits = 2000
    base.random_seed = 0
    ber_ideal = run_python_simulation(base)['ber']
    assert ber_ideal < 0.05, f"ideal-ADC kadirvelu should close, got {ber_ideal}"

    slow = SystemConfig.from_preset('kadirvelu2021')
    slow.simulation_engine = 'python'
    slow.n_bits = 2000
    slow.random_seed = 0
    # 500 Hz is far below the 5000-baud Manchester symbol rate -> heavy aliasing.
    slow.mcu_sample_rate_hz = 500.0
    ber_slow = run_python_simulation(slow)['ber']
    assert ber_slow > ber_ideal, "slow ADC must not improve BER"
    assert ber_slow > 0.1, f"sub-Nyquist ADC should clearly degrade BER, got {ber_slow}"


# ---------------------------------------------------------------------------
# 3. Validator — sub-Nyquist warning, clock-budget warning, and clean default.
# ---------------------------------------------------------------------------

def _ids(cfg):
    return {i.rule_id for i in validate_config(cfg)}


def test_validator_warns_sub_nyquist():
    cfg = SystemConfig.from_preset('kadirvelu2021')  # OOK_Manchester @ 2500 bps
    cfg.mcu_sample_rate_hz = 3000.0  # below Nyquist (2 x 5000 = 10000 Hz)
    issues = validate_config(cfg)
    hit = [i for i in issues if i.rule_id == 'mcu.adc_under_nyquist']
    assert hit and hit[0].level == IssueLevel.WARNING


def test_validator_warns_tight_clock_budget():
    cfg = SystemConfig.from_preset('kadirvelu2021')
    cfg.mcu_sample_rate_hz = 2_000_000.0  # above Nyquist (no alias warning)
    cfg.mcu_clock_MHz = 16.0              # 16 MHz / 2 MHz = 8 cycles/sample
    ids = _ids(cfg)
    assert 'mcu.clock_budget_tight' in ids
    assert 'mcu.adc_under_nyquist' not in ids


def test_validator_silent_when_mcu_unset():
    cfg = SystemConfig.from_preset('kadirvelu2021')  # mcu_sample_rate_hz = 0
    ids = _ids(cfg)
    assert 'mcu.adc_under_nyquist' not in ids
    assert 'mcu.clock_budget_tight' not in ids

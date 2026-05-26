"""Tests for the non-AWGN real-world noise sources added on top of the
6-source AWGN model: AC mains flicker and amplifier 1/f (pink) noise.

These complement test_probes.py and test_paper_validation.py — they target
the noise generators themselves rather than the pipeline that consumes them.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosim.noise import NoiseModel, _pink_voss


# ---------------------------------------------------------------------------
# Mains flicker
# ---------------------------------------------------------------------------

def _flicker_model(freq_hz=100.0, depth=0.05, lux=200.0):
    return NoiseModel(
        R_load=10e3,
        ambient_illuminance_lux=lux,
        rx_area_cm2=25.0,
        responsivity=0.40,
        enable_mains_flicker=True,
        mains_flicker_freq_hz=freq_hz,
        mains_flicker_depth=depth,
    )


def test_mains_flicker_is_off_by_default():
    """Constructor defaults must leave the new sources disabled — opt-in only."""
    nm = NoiseModel(R_load=10e3, ambient_illuminance_lux=200.0,
                    rx_area_cm2=25.0, responsivity=0.4)
    t = np.linspace(0, 0.1, 1000)
    out = nm.mains_flicker_current(t)
    assert np.array_equal(out, np.zeros_like(t))


def test_mains_flicker_amplitude_scales_with_ambient_and_depth():
    """Peak amplitude ≈ depth · I_ambient · (1.0 + 0.30 + 0.10) when phases align."""
    nm = _flicker_model(depth=0.10, lux=500.0)
    # I_amb = 0.40 * (500 * 1.46e-6 * 25) = 7.3 mA
    expected_I_amb = 0.40 * 500 * 1.46e-6 * 25.0
    rng = np.random.default_rng(0)
    t = np.linspace(0, 0.5, 50000)
    sig = nm.mains_flicker_current(t, rng=rng)
    # Sum of cosines with random phases — peak is bounded by Σ |a_k| = 1.40,
    # typical peak hits within 70% of that. Test the conservative upper bound.
    peak = float(np.max(np.abs(sig)))
    assert peak <= 0.10 * expected_I_amb * 1.40 * 1.01    # ≤ analytical max
    assert peak >= 0.10 * expected_I_amb * 0.5            # ≥ 50% of expected


def test_mains_flicker_spectrum_peaks_at_fundamental_and_harmonics():
    """FFT of the generated signal must show energy at f, 2f, 3f (and nowhere else)."""
    f0 = 100.0
    nm = _flicker_model(freq_hz=f0, depth=0.10, lux=500.0)
    fs = 10_000.0
    n = 50_000
    t = np.arange(n) / fs
    sig = nm.mains_flicker_current(t, rng=np.random.default_rng(7))

    # FFT magnitude
    mag = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(n, d=1 / fs)

    def _energy_near(target_hz, tol_hz=2.0):
        mask = np.abs(freqs - target_hz) <= tol_hz
        return float(np.sum(mag[mask] ** 2))

    e1 = _energy_near(f0)
    e2 = _energy_near(2 * f0)
    e3 = _energy_near(3 * f0)
    # Off-tone bin: a frequency we should NOT see energy at
    e_off = _energy_near(150.0)

    # Harmonics descend a_1=1.0, a_2=0.30, a_3=0.10 → energy ratios 1 : 0.09 : 0.01
    assert e1 > 1e-3                                     # fundamental is present
    assert 0.05 < e2 / e1 < 0.20                         # ~0.09
    assert e3 / e1 < 0.05                                # ~0.01
    assert e_off / e1 < 1e-3                             # no leakage into 150 Hz


def test_mains_flicker_returns_zero_when_ambient_is_zero():
    nm = _flicker_model(lux=0.0)
    t = np.linspace(0, 0.1, 1000)
    out = nm.mains_flicker_current(t, rng=np.random.default_rng(0))
    assert np.array_equal(out, np.zeros_like(t))


def test_mains_flicker_repeatable_with_seeded_rng():
    nm = _flicker_model()
    t = np.linspace(0, 0.1, 1000)
    a = nm.mains_flicker_current(t, rng=np.random.default_rng(42))
    b = nm.mains_flicker_current(t, rng=np.random.default_rng(42))
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Amplifier 1/f (pink) noise
# ---------------------------------------------------------------------------

def _pink_model(e_n_nV=18.0, R_load=10e3, f_c=100.0):
    return NoiseModel(
        R_load=R_load,
        ina_noise_nV_rtHz=e_n_nV,
        enable_amp_flicker=True,
        amp_flicker_corner_hz=f_c,
    )


def test_pink_noise_is_off_by_default():
    nm = NoiseModel(R_load=10e3)
    out = nm.amp_flicker_current(1000, dt=1e-5)
    assert np.array_equal(out, np.zeros(1000))


def test_pink_noise_sigma_matches_analytical_within_tolerance():
    """σ_pink ≈ e_n · √(f_c · ln(Bn/f_low)) / R_load — within ±15%."""
    e_n_nV = 25.0
    R = 5_000.0
    f_c = 200.0
    dt = 1e-5                # fs = 100 kHz → Bn = 50 kHz
    n = 100_000              # T_total = 1 s → f_low ≈ 1 Hz
    nm = _pink_model(e_n_nV=e_n_nV, R_load=R, f_c=f_c)

    samples = nm.amp_flicker_current(n, dt=dt, rng=np.random.default_rng(1))
    measured = float(np.std(samples))
    expected = (e_n_nV * 1e-9) * np.sqrt(f_c * np.log(50_000 / 1.0)) / R

    assert measured == pytest.approx(expected, rel=0.15), (
        f"measured σ={measured:.3e} vs expected σ={expected:.3e}"
    )


def test_pink_noise_spectrum_slopes_down_at_one_over_f():
    """Crude 1/f slope check: low-frequency bins must hold more energy than high."""
    nm = _pink_model()
    n = 50_000
    samples = nm.amp_flicker_current(n, dt=1e-5, rng=np.random.default_rng(2))

    fft = np.abs(np.fft.rfft(samples))
    # Compare the lowest-decade energy to the top-decade energy.
    low_band = fft[1:50] ** 2          # ~2-100 Hz
    high_band = fft[5000:10000] ** 2   # ~10-20 kHz
    assert np.mean(low_band) > np.mean(high_band) * 5    # ≥ ×5 stronger at LF


def test_pink_voss_helper_returns_correct_length():
    rng = np.random.default_rng(3)
    out = _pink_voss(2048, rng)
    assert out.shape == (2048,)
    assert out.dtype == np.float64
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Integration — the pipeline path
# ---------------------------------------------------------------------------

def test_pipeline_sums_all_noise_sources_into_I_ph_noisy():
    """End-to-end: when noise is enabled, I_ph_noisy must differ from I_ph."""
    from cosim.python_engine import run_python_simulation
    from cosim.probes import ProbeCapture
    from cosim.system_config import SystemConfig

    cfg = SystemConfig.from_preset('lifi_poc_breadboard')
    cfg.simulation_engine = 'python'
    cfg.n_bits = 200           # small for speed
    cfg.random_seed = 42

    cap = ProbeCapture()
    run_python_simulation(cfg, probes=cap)

    I_ph = cap.get("rx.I_ph")
    I_ph_noisy = cap.get("rx.I_ph_noisy")
    flicker = cap.get("rx.noise.mains_flicker")
    pink = cap.get("rx.noise.pink")

    assert I_ph is not None and I_ph_noisy is not None
    assert not np.allclose(I_ph, I_ph_noisy), "noise had no effect on photocurrent"
    # Both new sources should be captured and non-trivial
    assert flicker is not None and np.std(flicker) > 0
    assert pink is not None and np.std(pink) > 0

"""
Paper validation tests for the multi-paper LiFi-PV simulator.

Tests that:
1. All 7 presets load correctly and have valid parameters
2. Python simulation engine runs for each Python-engine paper
3. BER results are physically reasonable for each paper
4. Channel gain calculations match expected orders of magnitude
5. SystemConfig extensions work correctly

Run with:  pytest tests/test_paper_validation.py -v
"""

import sys
import os
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================================
# 1. Preset Loading
# ============================================================================

class TestPresetLoading:
    """All 6 presets load without error and have required fields."""

    ALL_PRESETS = [
        'kadirvelu2021', 'sarwar2017',
        'correa2025', 'xu2024', 'oliveira2024', 'gonzalez2024',
    ]

    @pytest.mark.parametrize("preset_name", ALL_PRESETS)
    def test_preset_loads(self, preset_name):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset(preset_name)
        assert cfg.preset_name == preset_name

    @pytest.mark.parametrize("preset_name", ALL_PRESETS)
    def test_preset_has_valid_distance(self, preset_name):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset(preset_name)
        assert cfg.distance_m > 0

    @pytest.mark.parametrize("preset_name", ALL_PRESETS)
    def test_preset_has_valid_data_rate(self, preset_name):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset(preset_name)
        assert cfg.data_rate_bps > 0

    @pytest.mark.parametrize("preset_name", ALL_PRESETS)
    def test_preset_has_valid_modulation(self, preset_name):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset(preset_name)
        valid_mods = {'OOK', 'OOK_Manchester', 'OFDM', 'BFSK', 'PWM_ASK'}
        assert cfg.modulation in valid_mods, (
            f"{preset_name}: modulation '{cfg.modulation}' not in {valid_mods}")

    @pytest.mark.parametrize("preset_name", ALL_PRESETS)
    def test_preset_has_engine_type(self, preset_name):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset(preset_name)
        assert cfg.simulation_engine in ('spice', 'python')

    def test_list_presets_has_all_7(self):
        from cosim.system_config import SystemConfig
        presets = SystemConfig.list_presets()
        for name in self.ALL_PRESETS:
            assert name in presets, f"Preset '{name}' not found in list_presets()"

    @pytest.mark.parametrize("preset_name", ALL_PRESETS)
    def test_preset_roundtrip_json(self, preset_name, tmp_path):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset(preset_name)
        path = tmp_path / f'{preset_name}_test.json'
        cfg.save(path)
        loaded = SystemConfig.load(path)
        assert loaded.preset_name == preset_name
        assert loaded.distance_m == pytest.approx(cfg.distance_m)
        assert loaded.data_rate_bps == pytest.approx(cfg.data_rate_bps)


# ============================================================================
# 2. Engine Assignment
# ============================================================================

class TestEngineAssignment:
    """Papers are assigned to the correct simulation engine."""

    def test_kadirvelu_uses_spice(self):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('kadirvelu2021')
        assert cfg.simulation_engine == 'spice'

    def test_gonzalez_uses_python(self):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('gonzalez2024')
        assert cfg.simulation_engine == 'python'

    def test_sarwar_uses_python(self):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('sarwar2017')
        assert cfg.simulation_engine == 'python'

    def test_correa_uses_python(self):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('correa2025')
        assert cfg.simulation_engine == 'python'

    def test_xu_uses_python(self):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('xu2024')
        assert cfg.simulation_engine == 'python'

    def test_oliveira_uses_python(self):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('oliveira2024')
        assert cfg.simulation_engine == 'python'


# ============================================================================
# 3. Python Engine Simulation Tests
# ============================================================================

class TestPythonEngine:
    """Run the Python simulation engine with each Python-engine paper."""

    def test_engine_import(self):
        from cosim.python_engine import run_python_simulation  # noqa: F401

    def test_sarwar2017_ofdm(self):
        """Sarwar 2017: OFDM 16-QAM, 15 Mbps, 2m distance."""
        from cosim.system_config import SystemConfig
        from cosim.python_engine import run_python_simulation
        cfg = SystemConfig.from_preset('sarwar2017')
        # OFDM needs enough bits: n_data_carriers * bps per symbol
        # sarwar: nfft=64, n_data=31, bps=4 -> 124 bits/symbol, need >= 124
        cfg = cfg.__class__(**{**cfg.__dict__, 'n_bits': 248})
        result = run_python_simulation(cfg)
        assert 'ber' in result
        assert 'snr_est_dB' in result
        assert result['engine'] == 'python'
        assert result['modulation'] == 'OFDM'
        assert result['ber'] >= 0
        assert result['P_rx_avg_uW'] > 0

    def test_correa2025_pwm_ask(self):
        """Correa 2025: PWM-ASK, greenhouse, humidity model."""
        from cosim.system_config import SystemConfig
        from cosim.python_engine import run_python_simulation
        cfg = SystemConfig.from_preset('correa2025')
        cfg = cfg.__class__(**{**cfg.__dict__, 'n_bits': 20})
        result = run_python_simulation(cfg)
        assert result['engine'] == 'python'
        assert result['modulation'] == 'PWM_ASK'
        assert result['ber'] >= 0
        assert result['channel_gain'] > 0

    def test_xu2024_bfsk(self):
        """Xu 2024: BFSK, sunlight + LC shutter, 400 bps."""
        from cosim.system_config import SystemConfig
        from cosim.python_engine import run_python_simulation
        cfg = SystemConfig.from_preset('xu2024')
        cfg = cfg.__class__(**{**cfg.__dict__, 'n_bits': 20})
        result = run_python_simulation(cfg)
        assert result['engine'] == 'python'
        assert result['modulation'] == 'BFSK'
        assert result['ber'] >= 0
        assert result['P_rx_avg_uW'] > 0

    def test_oliveira2024_ofdm(self):
        """Oliveira 2024: OFDM 64-QAM, MIMO, 25.7 Mbps."""
        from cosim.system_config import SystemConfig
        from cosim.python_engine import run_python_simulation
        cfg = SystemConfig.from_preset('oliveira2024')
        # oliveira: nfft=1024, n_subcarriers=500, bps=6 -> 3000 bits/symbol
        cfg = cfg.__class__(**{**cfg.__dict__, 'n_bits': 3000})
        result = run_python_simulation(cfg)
        assert result['engine'] == 'python'
        assert result['modulation'] == 'OFDM'
        assert result['ber'] >= 0


# ============================================================================
# 4. Channel Gain Validation
# ============================================================================

class TestChannelGain:
    """Channel gain should be physically reasonable for each paper."""

    def test_kadirvelu_short_range(self):
        """Kadirvelu 2021: 32.5 cm, should have high channel gain."""
        from cosim.python_engine import OpticalChannel
        ch = OpticalChannel(distance_m=0.325, led_half_angle_deg=60,
                            rx_area_cm2=9.0)
        g = ch.channel_gain()
        assert g > 1e-4  # short range: decent gain
        assert g < 1.0   # still < 1 (physical)

    def test_sarwar_medium_range(self):
        """Sarwar 2017: 2m distance, lower gain."""
        from cosim.python_engine import OpticalChannel
        ch = OpticalChannel(distance_m=2.0, led_half_angle_deg=60,
                            rx_area_cm2=7.5)
        g = ch.channel_gain()
        assert g > 1e-6
        assert g < 0.1

    def test_xu_long_range(self):
        """Xu 2024: 5m, sunlight, large PV array."""
        from cosim.python_engine import OpticalChannel
        ch = OpticalChannel(distance_m=5.0, led_half_angle_deg=90,
                            rx_area_cm2=16.0)
        g = ch.channel_gain()
        assert g > 1e-6
        assert g < 0.1

    def test_humidity_attenuation(self):
        """Correa 2025: humidity should reduce channel gain."""
        from cosim.python_engine import OpticalChannel
        ch_dry = OpticalChannel(distance_m=0.85, led_half_angle_deg=60,
                                rx_area_cm2=66.0, humidity_rh=0.0,
                                beer_lambert_enabled=True)
        ch_wet = OpticalChannel(distance_m=0.85, led_half_angle_deg=60,
                                rx_area_cm2=66.0, humidity_rh=0.8,
                                beer_lambert_enabled=True)
        assert ch_wet.channel_gain() < ch_dry.channel_gain()


# ============================================================================
# 5. BER Prediction Functions
# ============================================================================

class TestBERPrediction:
    """BER prediction functions should give correct values."""

    def test_ook_ber_at_snr0(self):
        from cosim.python_engine import predict_ber_ook
        ber = predict_ber_ook(1.0)  # SNR_linear = 1 (0 dB)
        assert 0.1 < ber < 0.3  # ~0.159

    def test_ook_ber_at_high_snr(self):
        from cosim.python_engine import predict_ber_ook
        ber = predict_ber_ook(100.0)  # ~20 dB
        assert ber < 1e-6

    def test_bfsk_ber_at_snr0(self):
        from cosim.python_engine import predict_ber_bfsk
        ber = predict_ber_bfsk(1.0)
        assert 0.2 < ber < 0.5  # ~0.303

    def test_mqam_ber_16qam(self):
        from cosim.python_engine import predict_ber_mqam
        ber = predict_ber_mqam(100.0, 16)  # 20 dB, 16-QAM
        assert ber < 0.01

    def test_mqam_ber_64qam_harder(self):
        """64-QAM needs higher SNR than 16-QAM."""
        from cosim.python_engine import predict_ber_mqam
        snr = 50.0  # ~17 dB
        ber_16 = predict_ber_mqam(snr, 16)
        ber_64 = predict_ber_mqam(snr, 64)
        assert ber_64 > ber_16  # 64-QAM is harder


# ============================================================================
# 6. Noise Model
# ============================================================================

class TestNoiseModel:
    """Noise model should produce physically reasonable noise levels."""

    def test_noise_std_positive(self):
        from cosim.python_engine import NoiseModel
        nm = NoiseModel(temperature_K=300, R_load=50)
        sigma = nm.total_noise_std(I_ph=1e-6, bandwidth=5e3)
        assert sigma > 0

    def test_noise_increases_with_bandwidth(self):
        from cosim.python_engine import NoiseModel
        nm = NoiseModel(temperature_K=300, R_load=50)
        s1 = nm.total_noise_std(1e-6, 1e3)
        s2 = nm.total_noise_std(1e-6, 1e6)
        assert s2 > s1

    def test_noise_samples_length(self):
        from cosim.python_engine import NoiseModel
        nm = NoiseModel()
        noise = nm.generate_time_domain(1000, 1e-6, 5e3)
        assert len(noise) == 1000


# ============================================================================
# 7. Manchester Codec
# ============================================================================

class TestManchesterCodec:
    """Manchester encode/decode roundtrip."""

    def test_encode_length(self):
        from cosim.python_engine import manchester_encode
        bits = np.array([1, 0, 1, 1, 0])
        symbols = manchester_encode(bits)
        assert len(symbols) == 10  # 2x bits

    def test_encode_decode_roundtrip(self):
        from cosim.python_engine import manchester_encode, manchester_decode
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        symbols = manchester_encode(bits)
        recovered = manchester_decode(symbols.astype(float))
        np.testing.assert_array_equal(bits, recovered)


# ============================================================================
# 8. SystemConfig Extensions
# ============================================================================

class TestSystemConfigExtensions:
    """Test new fields added for multi-paper support."""

    def test_simulation_engine_field(self):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig()
        assert hasattr(cfg, 'simulation_engine')
        assert cfg.simulation_engine in ('spice', 'python')

    def test_ofdm_fields(self):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('sarwar2017')
        assert cfg.ofdm_nfft > 0
        assert cfg.ofdm_qam_order in (4, 16, 64)

    def test_bfsk_fields(self):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('xu2024')
        assert cfg.bfsk_f0_hz > 0
        assert cfg.bfsk_f1_hz > 0
        assert cfg.bfsk_f0_hz != cfg.bfsk_f1_hz

    def test_temperature_field(self):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig()
        assert cfg.temperature_K == 300.0

    def test_gonzalez_amp_gain(self):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('gonzalez2024')
        assert cfg.amp_gain_linear > 1
        assert cfg.notch_freq_hz > 0

    def test_oliveira_ofdm_params(self):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('oliveira2024')
        assert cfg.ofdm_nfft == 1024
        assert cfg.ofdm_qam_order == 64
        assert cfg.ofdm_n_subcarriers == 500


# ============================================================================
# 9. Pipeline Dual-Engine Dispatch
# ============================================================================

class TestDualEngine:
    """Pipeline correctly dispatches to Python engine."""

    def test_python_engine_pipeline(self, tmp_path):
        """Run full pipeline with Python engine."""
        from cosim.pipeline import SimulationPipeline
        from cosim.system_config import SystemConfig

        cfg = SystemConfig.from_preset('xu2024')
        # Use fewer bits for speed
        cfg = cfg.replace(n_bits=10)

        for sub in ('pwl', 'netlists', 'raw'):
            (tmp_path / sub).mkdir()

        pipe = SimulationPipeline(cfg, tmp_path)
        results = pipe.run_all()

        assert 'RX' in results
        rx = results['RX']
        assert rx.status == 'done'
        assert 'python_result' in rx.outputs
        pr = rx.outputs['python_result']
        assert pr['engine'] == 'python'
        assert pr['ber'] >= 0

    def test_spice_engine_still_generates_tx(self, tmp_path):
        """SPICE engine papers still generate TX PWL."""
        from cosim.pipeline import SimulationPipeline
        from cosim.system_config import SystemConfig

        cfg = SystemConfig.from_preset('kadirvelu2021')
        cfg = cfg.replace(n_bits=10)

        for sub in ('pwl', 'netlists', 'raw'):
            (tmp_path / sub).mkdir()

        pipe = SimulationPipeline(cfg, tmp_path)
        tx = pipe.run_step_tx()
        assert tx.status == 'done'
        assert 'P_tx_pwl' in tx.outputs


# ============================================================================
# 10. Thermal Modelling (Phase 6C)
# ============================================================================

class TestThermalModel:
    """Temperature-dependent dark current and responsivity."""

    def test_dark_current_ref_zero_returns_zero(self):
        from cosim.noise import NoiseModel
        nm = NoiseModel(temperature_K=320.0, dark_current_ref_A=0.0)
        assert nm.dark_current_at_T() == 0.0

    def test_dark_current_doubles_per_10K(self):
        from cosim.noise import NoiseModel
        I0 = 1e-10
        nm_ref = NoiseModel(temperature_K=300.0, dark_current_ref_A=I0,
                            dark_current_ref_T_K=300.0)
        nm_hot = NoiseModel(temperature_K=310.0, dark_current_ref_A=I0,
                            dark_current_ref_T_K=300.0)
        assert nm_ref.dark_current_at_T() == pytest.approx(I0, rel=1e-6)
        assert nm_hot.dark_current_at_T() == pytest.approx(2 * I0, rel=1e-6)

    def test_dark_current_50K_gives_32x(self):
        from cosim.noise import NoiseModel
        I0 = 1e-10
        nm = NoiseModel(temperature_K=350.0, dark_current_ref_A=I0,
                        dark_current_ref_T_K=300.0)
        # 50 K / 10 K per doubling = 2^5 = 32x
        assert nm.dark_current_at_T() == pytest.approx(32 * I0, rel=1e-6)

    def test_dark_current_cold_decreases(self):
        """Below reference temperature, dark current shrinks."""
        from cosim.noise import NoiseModel
        I0 = 1e-10
        nm_cold = NoiseModel(temperature_K=290.0, dark_current_ref_A=I0,
                             dark_current_ref_T_K=300.0)
        # -10 K => 0.5 x
        assert nm_cold.dark_current_at_T() == pytest.approx(0.5 * I0, rel=1e-6)

    def test_responsivity_tempco_increases_with_T(self):
        """Si/GaAs responsivity has positive tempco (~4e-4/K)."""
        from cosim.python_engine import PVReceiver
        rx_cold = PVReceiver(responsivity=0.5, temperature_K=300.0,
                             responsivity_ref_T_K=300.0,
                             responsivity_tempco_per_K=4e-4)
        rx_hot = PVReceiver(responsivity=0.5, temperature_K=350.0,
                            responsivity_ref_T_K=300.0,
                            responsivity_tempco_per_K=4e-4)
        # Delta T = 50 K, tempco = 4e-4/K => +2% => 0.510
        assert rx_cold.R == pytest.approx(0.5, rel=1e-6)
        assert rx_hot.R == pytest.approx(0.5 * (1 + 4e-4 * 50), rel=1e-6)

    def test_responsivity_zero_tempco_invariant(self):
        from cosim.python_engine import PVReceiver
        rx = PVReceiver(responsivity=0.5, temperature_K=400.0,
                        responsivity_ref_T_K=300.0,
                        responsivity_tempco_per_K=0.0)
        assert rx.R == pytest.approx(0.5, rel=1e-6)

    def test_thermal_sweep_runs(self, tmp_path):
        """run_thermal_sweep returns one ThermalPoint per temperature."""
        from cosim.thermal_sweep import run_thermal_sweep, ThermalPoint
        temps = [25.0, 65.0]
        points = run_thermal_sweep('kadirvelu2021', temps,
                                   output_dir=str(tmp_path), verbose=False)
        assert len(points) == 2
        assert all(isinstance(p, ThermalPoint) for p in points)
        # Hot point should have higher dark current
        assert points[1].dark_current_A > points[0].dark_current_A
        # Responsivity should be > 0 at both temperatures
        assert points[0].responsivity_A_per_W > 0
        assert points[1].responsivity_A_per_W > 0

    def test_thermal_config_fields_default(self):
        """SystemConfig exposes thermal configuration fields."""
        from cosim.system_config import SystemConfig
        cfg = SystemConfig()
        assert cfg.pv_dark_current_ref_T_K == 300.0
        assert cfg.dark_current_doubling_dT_K == 10.0
        assert cfg.sc_responsivity_ref_T_K == 300.0
        assert cfg.sc_responsivity_tempco_per_K > 0


# ============================================================================
# 11. Monte Carlo Tolerance Analysis (Phase 6C)
# ============================================================================

class TestMonteCarlo:
    """Production yield estimation via Monte Carlo tolerance sweep."""

    def test_zero_tolerance_is_deterministic(self):
        """With tol=0, every run should produce the same BER."""
        from cosim.monte_carlo import run_monte_carlo
        spec = {'sc_responsivity': 0.0, 'distance_m': 0.0}
        r = run_monte_carlo('kadirvelu2021', spec, n_runs=5,
                            seed=1, verbose=False)
        assert len(r.ber_samples) == 5
        assert len(set(r.ber_samples)) == 1, \
            "Zero tolerance should yield identical BER across runs"

    def test_nonzero_tolerance_produces_variation(self):
        """With a wide tolerance, SNR should vary across runs."""
        from cosim.monte_carlo import run_monte_carlo
        spec = {'distance_m': 0.30}  # +/-30% placement error
        r = run_monte_carlo('kadirvelu2021', spec, n_runs=20,
                            seed=7, verbose=False)
        snr = np.array(r.snr_samples)
        # Spread should exceed numerical jitter
        assert snr.std() > 0.05, \
            f"Expected SNR spread from 30% distance tolerance, got std={snr.std()}"

    def test_unknown_field_raises(self):
        """Referencing a non-existent field should error clearly."""
        from cosim.monte_carlo import run_monte_carlo
        with pytest.raises(ValueError, match="unknown field"):
            run_monte_carlo('kadirvelu2021',
                            {'not_a_real_field': 0.1},
                            n_runs=1, verbose=False)

    def test_yield_fraction_bounds(self):
        """Yield fraction must lie in [0, 1]."""
        from cosim.monte_carlo import run_monte_carlo
        spec = {'sc_responsivity': 0.05}
        r = run_monte_carlo('kadirvelu2021', spec, n_runs=10,
                            seed=3, verbose=False)
        assert 0.0 <= r.yield_fraction <= 1.0

    def test_reproducible_parameter_samples(self):
        """Same seed => identical sampled parameter values.

        Note: SNR/BER estimates include internal channel noise
        randomness not tied to the MC seed, so we check the sampled
        parameter sequence rather than downstream metrics.
        """
        import numpy as np
        from cosim.monte_carlo import _sample_param
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        seq1 = [_sample_param(0.5, 0.1, rng1) for _ in range(10)]
        seq2 = [_sample_param(0.5, 0.1, rng2) for _ in range(10)]
        assert seq1 == seq2

    def test_sample_stays_within_bounds(self):
        """Sampled values must fall within +/-tol of nominal."""
        from cosim.system_config import SystemConfig
        from cosim.monte_carlo import _sample_param
        import numpy as np
        rng = np.random.default_rng(0)
        nominal = 0.5
        tol = 0.1
        for _ in range(100):
            v = _sample_param(nominal, tol, rng)
            assert nominal * (1 - tol) <= v <= nominal * (1 + tol)

    def test_figure_is_written(self, tmp_path):
        """With output_dir set, a histogram PNG should appear."""
        from cosim.monte_carlo import run_monte_carlo
        spec = {'sc_responsivity': 0.05}
        run_monte_carlo('kadirvelu2021', spec, n_runs=5,
                        seed=1, output_dir=str(tmp_path), verbose=False)
        assert (tmp_path / 'monte_carlo_kadirvelu2021.png').exists()


# ============================================================================
# 12. New Components (Phase 6C)
# ============================================================================

class TestNewComponents:
    """OPA380 TIA, BQ25570 MPPT charger, STM32H7 ADC."""

    def test_opa380_construction(self):
        from components import OPA380
        opa = OPA380(R_feedback=100e3, C_feedback=2.5e-12)
        assert opa.name == 'OPA380'
        assert opa.transimpedance_V_per_A == pytest.approx(100e3)
        assert opa.gain_bandwidth_product == pytest.approx(90e6)

    def test_opa380_bandwidth_scales_inversely_with_rf_cf(self):
        from components import OPA380
        opa1 = OPA380(R_feedback=100e3, C_feedback=2.5e-12)
        opa2 = OPA380(R_feedback=200e3, C_feedback=2.5e-12)
        # Doubling Rf halves bandwidth
        assert opa2.tia_bandwidth_Hz == pytest.approx(opa1.tia_bandwidth_Hz / 2, rel=1e-6)

    def test_opa380_spice_subcircuit(self):
        from components import OPA380
        opa = OPA380()
        sub = opa.spice_subcircuit()
        assert '.SUBCKT OPA380' in sub
        assert '.ENDS OPA380' in sub

    def test_bq25570_construction(self):
        from components import BQ25570
        chg = BQ25570()
        assert chg.name == 'BQ25570'
        assert chg.v_in_min_startup == 0.6
        assert chg.v_in_min_operating == 0.1

    def test_bq25570_cold_start_threshold(self):
        from components import BQ25570
        chg = BQ25570()
        assert chg.cold_start_supported(0.8) is True
        assert chg.cold_start_supported(0.5) is False

    def test_bq25570_mppt_voltage(self):
        from components import BQ25570
        chg = BQ25570(mppt_ratio=0.80)
        assert chg.mppt_voltage(1.1) == pytest.approx(0.88)

    def test_bq25570_efficiency_monotonic(self):
        """Efficiency should not decrease as V_IN rises from 0.5 to 1.0 V."""
        from components import BQ25570
        chg = BQ25570()
        assert chg.boost_efficiency(0.5) <= chg.boost_efficiency(0.75)
        assert chg.boost_efficiency(0.75) <= chg.boost_efficiency(1.0)

    def test_bq25570_below_operating_returns_zero(self):
        from components import BQ25570
        chg = BQ25570()
        assert chg.boost_efficiency(0.05) == 0.0
        assert chg.output_power(1e-3, 0.05) == 0.0

    def test_stm32h7_adc_resolution_validation(self):
        from components import STM32H7_ADC
        with pytest.raises(ValueError):
            STM32H7_ADC(resolution_bits=11)
        with pytest.raises(ValueError):
            STM32H7_ADC(v_ref_V=5.0)
        with pytest.raises(ValueError):
            STM32H7_ADC(oversampling=2048)

    def test_stm32h7_adc_lsb_matches_vref(self):
        from components import STM32H7_ADC
        adc = STM32H7_ADC(resolution_bits=12, v_ref_V=3.3)
        assert adc.n_codes == 4096
        assert adc.lsb_V == pytest.approx(3.3 / 4096)

    def test_stm32h7_adc_quantization_reduced_by_oversampling(self):
        from components import STM32H7_ADC
        a1 = STM32H7_ADC(resolution_bits=12, v_ref_V=3.3, oversampling=1)
        a4 = STM32H7_ADC(resolution_bits=12, v_ref_V=3.3, oversampling=4)
        # Oversampling x4 reduces sigma by sqrt(4)
        assert a4.quantization_noise_std_V == pytest.approx(
            a1.quantization_noise_std_V / 2, rel=1e-6)

    def test_stm32h7_adc_sample_round_trip_within_one_lsb(self):
        from components import STM32H7_ADC
        adc = STM32H7_ADC(resolution_bits=12, v_ref_V=3.3)
        v_in = 1.65
        code = adc.sample(v_in)
        v_out = adc.code_to_voltage(code)
        assert abs(v_out - v_in) <= adc.lsb_V

    def test_stm32h7_adc_clamps_to_range(self):
        from components import STM32H7_ADC
        adc = STM32H7_ADC(resolution_bits=12, v_ref_V=3.3)
        assert adc.sample(-1.0) == 0
        assert adc.sample(10.0) == adc.n_codes - 1

    def test_new_components_registered(self):
        from components import COMPONENT_REGISTRY, get_component
        for name in ('OPA380', 'BQ25570', 'STM32H7_ADC'):
            assert name in COMPONENT_REGISTRY
            # get_component normalizes names via upper + underscore
            inst = get_component(name)
            assert inst is not None


# ============================================================================
# 13. Pipeline-validation comparator (BER=0 + Wilson-CI gating)
# ============================================================================

class TestComparatorBerWilson:
    """
    The comparator at papers/pipeline_validation.py must only claim
    "BER below target" when n_bits is large enough for the Wilson 95%
    upper bound to fall below the target.
    """

    @staticmethod
    def _stub_result(ber, n_bits, n_errors=0):
        """Build a result dict shaped like run_python_simulation's output."""
        return {
            'ber': ber,
            'n_bits_tested': n_bits,
            'ber_n_errors': n_errors,
            'channel_gain': 0.0773,
            'P_rx_avg_uW': 718.9,
            'I_ph_avg_uA': 328.5,
            'snr_est_dB': 30.0,
        }

    def test_ber_zero_with_sufficient_bits_passes(self, monkeypatch):
        """n=10000 errors=0 vs target 1e-2 -> Wilson UB << target -> PASS."""
        import papers.pipeline_validation as pv

        monkeypatch.setattr(
            pv, 'run_python_simulation',
            lambda cfg: self._stub_result(ber=0.0, n_bits=10000))

        # Correa target is 0.01 — easy to reject with 10k bits.
        result = pv.validate_preset('correa2025', verbose=False)
        ber_comp = next(c for c in result['comparisons'] if c['metric'] == 'BER')
        assert ber_comp['status'] == 'PASS'
        assert result['passed'] is True

    def test_ber_zero_with_insufficient_bits_flagged(self, monkeypatch):
        """n=20 errors=0 vs target 1e-3 -> Wilson UB >> target -> INSUFFICIENT_BITS."""
        import papers.pipeline_validation as pv

        monkeypatch.setattr(
            pv, 'run_python_simulation',
            lambda cfg: self._stub_result(ber=0.0, n_bits=20))

        # Kadirvelu target is ~1e-3 — cannot be rejected from only 20 bits.
        result = pv.validate_preset('kadirvelu2021', verbose=False)
        ber_comp = next(c for c in result['comparisons'] if c['metric'] == 'BER')
        assert ber_comp['status'] == 'INSUFFICIENT_BITS'
        assert result['passed'] is False

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
    """All 7 presets load without error and have required fields."""

    ALL_PRESETS = [
        'kadirvelu2021', 'fakidis2020', 'sarwar2017',
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

    def test_fakidis_uses_spice(self):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('fakidis2020')
        assert cfg.simulation_engine == 'spice'

    def test_gonzalez_uses_spice(self):
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('gonzalez2024')
        assert cfg.simulation_engine == 'spice'

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
        ch = OpticalChannel(distance_m=0.325, beam_half_angle_deg=60,
                            rx_area_cm2=9.0)
        g = ch.channel_gain()
        assert g > 1e-4  # short range: decent gain
        assert g < 1.0   # still < 1 (physical)

    def test_sarwar_medium_range(self):
        """Sarwar 2017: 2m distance, lower gain."""
        from cosim.python_engine import OpticalChannel
        ch = OpticalChannel(distance_m=2.0, beam_half_angle_deg=60,
                            rx_area_cm2=7.5)
        g = ch.channel_gain()
        assert g > 1e-6
        assert g < 0.1

    def test_xu_long_range(self):
        """Xu 2024: 5m, sunlight, large PV array."""
        from cosim.python_engine import OpticalChannel
        ch = OpticalChannel(distance_m=5.0, beam_half_angle_deg=90,
                            rx_area_cm2=16.0)
        g = ch.channel_gain()
        assert g > 1e-6
        assert g < 0.1

    def test_humidity_attenuation(self):
        """Correa 2025: humidity should reduce channel gain."""
        from cosim.python_engine import OpticalChannel
        ch_dry = OpticalChannel(distance_m=0.85, beam_half_angle_deg=60,
                                rx_area_cm2=66.0, humidity=0.0)
        ch_wet = OpticalChannel(distance_m=0.85, beam_half_angle_deg=60,
                                rx_area_cm2=66.0, humidity=0.8)
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
        noise = nm.generate_noise(1000, 1e-6, 5e3)
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
        import dataclasses
        cfg = dataclasses.replace(cfg, n_bits=10)

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
        import dataclasses
        cfg = dataclasses.replace(cfg, n_bits=10)

        for sub in ('pwl', 'netlists', 'raw'):
            (tmp_path / sub).mkdir()

        pipe = SimulationPipeline(cfg, tmp_path)
        tx = pipe.run_step_tx()
        assert tx.status == 'done'
        assert 'P_tx_pwl' in tx.outputs

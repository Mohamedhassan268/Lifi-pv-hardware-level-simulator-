# cosim/pipeline.py
"""
Unified Simulation Pipeline: TX -> Channel -> RX

Orchestrates the full co-simulation flow with automatic engine selection:
    Step 1 (TX):      Python — modulate() for all 5 schemes
    Step 2 (Channel): Python — OpticalChannel propagation + noise PWL
    Step 3 (RX):      SPICE (if available) or Python — receiver circuit

Supports three modes:
    - 'python':  All-Python simulation (always available)
    - 'spice':   Python TX+Channel → SPICE RX → Python BER (hybrid)
    - fallback:  If SPICE requested but unavailable, degrades to Python

Usage:
    from cosim.pipeline import SimulationPipeline

    pipe = SimulationPipeline(config, session_dir)
    result = pipe.run_all()
    # or individually:
    pipe.run_step_tx()
    pipe.run_step_channel()
    pipe.run_step_rx()
"""

import logging
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable
import time as _time

# Add project root to sys.path so we can import sibling packages (simulation/, systems/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from .system_config import SystemConfig
from .pwl_writer import write_photocurrent_pwl
from .ltspice_runner import LTSpiceRunner
from .ngspice_runner import NgSpiceRunner
from .raw_parser import LTSpiceRawParser
from .spice_finder import spice_available
from .channel import OpticalChannel
from .modulation import modulate, demodulate, calculate_ber

logger = logging.getLogger(__name__)

# Samples per bit for TX waveform generation (SPICE pipeline)
_SAMPLES_PER_BIT = 100


class StepResult:
    """Result of a single pipeline step."""

    def __init__(self, name: str):
        self.name = name
        self.status = 'pending'       # pending | running | done | error
        self.message = ''
        self.duration_s = 0.0
        self.outputs = {}             # key -> file path or value

    def __repr__(self):
        return f"StepResult({self.name}: {self.status})"


class SimulationPipeline:
    """3-step TX->Channel->RX simulation orchestrator."""

    def __init__(self, config: SystemConfig, session_dir: Path,
                 ltspice_runner: Optional[LTSpiceRunner] = None,
                 on_progress: Optional[Callable] = None):
        """
        Initialize pipeline.

        Args:
            config: System configuration
            session_dir: Session directory (with netlists/, pwl/, raw/ subdirs)
            ltspice_runner: LTSpice runner (auto-created if None)
            on_progress: Optional callback(step_name, status, message)
        """
        self.config = config
        self.session_dir = Path(session_dir)
        self.ltspice = ltspice_runner or LTSpiceRunner()
        self.on_progress = on_progress

        # Step results
        self.step_tx = StepResult('TX')
        self.step_channel = StepResult('Channel')
        self.step_rx = StepResult('RX')

        # Intermediate data
        self._time = None
        self._P_tx = None
        self._P_rx = None
        self._I_ph = None
        self._tx_bits = None

    def _notify(self, step_name: str, status: str, message: str = ''):
        """Send progress notification."""
        if self.on_progress:
            self.on_progress(step_name, status, message)

    # -------------------------------------------------------------------------
    # Step 1: TX — generate modulated optical power waveform
    # -------------------------------------------------------------------------

    def run_step_tx(self) -> StepResult:
        """
        Generate transmitted optical power waveform P_tx(t).

        Uses the unified modulate() dispatch to support all 5 modulation
        schemes (OOK, Manchester, OFDM, BFSK, PWM-ASK) for both SPICE
        and Python engine paths.
        """
        self.step_tx.status = 'running'
        cfg = self.config
        mod_scheme = cfg.modulation.upper().replace('-', '_')
        self._notify('TX', 'running', f'Generating {mod_scheme} waveform...')
        t0 = _time.time()

        try:
            # Generate time array
            bit_period = 1.0 / cfg.data_rate_bps
            n_samples = cfg.n_bits * _SAMPLES_PER_BIT
            dt = bit_period / _SAMPLES_PER_BIT
            time_arr = np.arange(n_samples) * dt

            # Generate TX bits
            if cfg.random_seed is not None:
                np.random.seed(cfg.random_seed)
            bits = np.random.randint(0, 2, cfg.n_bits)
            self._tx_bits = bits

            # Modulate using unified dispatch (supports all 5 schemes)
            P_tx = modulate(mod_scheme, bits, time_arr, config=cfg)
            self._time = time_arr
            self._P_tx = P_tx

            # Save TX waveform as PWL for reference
            tx_pwl_path = self.session_dir / 'pwl' / 'P_tx.pwl'
            from .pwl_writer import write_voltage_pwl
            write_voltage_pwl(time_arr, P_tx, tx_pwl_path)

            P_dc = cfg.led_radiated_power_mW * 1e-3
            self.step_tx.status = 'done'
            self.step_tx.duration_s = _time.time() - t0
            self.step_tx.message = (
                f'{cfg.n_bits} bits @ {cfg.data_rate_bps/1e3:.0f} kbps, '
                f'{mod_scheme} modulation, P_dc={P_dc*1e3:.1f} mW'
            )
            self.step_tx.outputs = {
                'P_tx_pwl': str(tx_pwl_path),
                'n_bits': len(bits),
                'P_dc_mW': P_dc * 1e3,
                'modulation': mod_scheme,
            }
            self._notify('TX', 'done', self.step_tx.message)

        except Exception as e:
            self.step_tx.status = 'error'
            self.step_tx.message = str(e)
            self.step_tx.duration_s = _time.time() - t0
            self._notify('TX', 'error', str(e))

        return self.step_tx

    # -------------------------------------------------------------------------
    # Step 2: Channel — apply optical channel and write i_ph.pwl
    # -------------------------------------------------------------------------

    def run_step_channel(self) -> StepResult:
        """
        Apply channel model to P_tx(t) and write PWL bridge files.

        Writes:
            - optical_power.pwl: P_rx(t) for SOLAR_CELL subcircuit
            - noise.pwl: Calibrated noise current (if noise_enable=True)
        """
        self.step_channel.status = 'running'
        self._notify('Channel', 'running', 'Computing channel response...')
        t0 = _time.time()

        try:
            if self._time is None or self._P_tx is None:
                raise RuntimeError('Run TX step first')

            cfg = self.config
            channel = OpticalChannel.from_config(cfg)
            G_ch = channel.channel_gain()
            R_lambda = cfg.sc_responsivity

            # Compute received optical power P_rx(t) = G_ch * P_tx(t)
            P_rx = channel.propagate(self._P_tx)
            self._P_rx = P_rx
            self._I_ph = R_lambda * P_rx

            # Write PWL bridge file: P_rx(t) in watts (voltage representation)
            pwl_path = self.session_dir / 'pwl' / 'optical_power.pwl'
            from .pwl_writer import write_optical_power_pwl
            write_optical_power_pwl(
                self._time, P_rx,
                filename=pwl_path,
                metadata={
                    'LED': cfg.led_part,
                    'PV': cfg.pv_part,
                    'Distance': f'{cfg.distance_m*100:.1f} cm',
                    'G_ch': f'{G_ch:.6e}',
                    'R_lambda': f'{R_lambda:.4f} A/W',
                    'P_rx range': f'[{P_rx.min()*1e6:.2f}, {P_rx.max()*1e6:.2f}] uW',
                    'I_ph range': f'[{self._I_ph.min()*1e6:.2f}, {self._I_ph.max()*1e6:.2f}] uA',
                },
            )

            P_rx_avg = np.mean(P_rx)
            I_ph_avg = np.mean(self._I_ph)

            # Generate calibrated noise PWL if noise is enabled
            noise_pwl_path = None
            if cfg.noise_enable:
                from .noise import NoiseModel
                from .pwl_writer import write_noise_pwl
                noise_model = NoiseModel.from_config(cfg)
                bandwidth = cfg.data_rate_bps / 2
                noise_pwl_path = self.session_dir / 'pwl' / 'noise.pwl'
                write_noise_pwl(
                    noise_model, self._time, I_ph_avg,
                    bandwidth=bandwidth,
                    filename=noise_pwl_path,
                    random_seed=cfg.random_seed,
                )
                logger.info("Noise PWL written: %s", noise_pwl_path)

            self.step_channel.status = 'done'
            self.step_channel.duration_s = _time.time() - t0
            self.step_channel.message = (
                f'G_ch={G_ch:.4e}, '
                f'P_rx_avg={P_rx_avg*1e6:.2f} uW, '
                f'I_ph_avg={I_ph_avg*1e6:.2f} uA'
            )
            self.step_channel.outputs = {
                'optical_pwl': str(pwl_path),
                'G_ch': G_ch,
                'P_rx_avg_uW': P_rx_avg * 1e6,
                'I_ph_avg_uA': I_ph_avg * 1e6,
            }
            if noise_pwl_path:
                self.step_channel.outputs['noise_pwl'] = str(noise_pwl_path)
            self._notify('Channel', 'done', self.step_channel.message)

        except Exception as e:
            self.step_channel.status = 'error'
            self.step_channel.message = str(e)
            self.step_channel.duration_s = _time.time() - t0
            self._notify('Channel', 'error', str(e))

        return self.step_channel

    # -------------------------------------------------------------------------
    # Step 3: RX — generate netlist, run SPICE, parse results
    # -------------------------------------------------------------------------

    def run_step_rx(self) -> StepResult:
        """
        Generate receiver netlist with PWL source, run SPICE, parse .raw.
        """
        self.step_rx.status = 'running'
        self._notify('RX', 'running', 'Running SPICE simulation...')
        t0 = _time.time()

        try:
            pwl_path = self.step_channel.outputs.get('optical_pwl')
            if not pwl_path:
                raise RuntimeError('Run Channel step first')

            cfg = self.config
            noise_pwl = self.step_channel.outputs.get('noise_pwl')

            # Generate receiver netlist with PWL photocurrent source
            netlist = self._generate_rx_netlist(pwl_path, noise_pwl_path=noise_pwl)

            # Save netlist
            cir_path = self.session_dir / 'netlists' / 'receiver.cir'
            cir_path.write_text(netlist, encoding='utf-8')

            # Try LTspice first, fallback to ngspice
            raw_data = None
            sim_engine = 'none'

            if self.ltspice.available:
                self._notify('RX', 'running', 'Running LTspice...')
                ok = self.ltspice.run_transient(str(cir_path), timeout_s=120)
                if ok:
                    sim_engine = 'LTspice'
                    raw_path = self.ltspice.get_raw_path()
                    if raw_path:
                        parser = LTSpiceRawParser(raw_path)
                        raw_data = parser.to_dict()
                        # Copy .raw to session
                        import shutil
                        dest = self.session_dir / 'raw' / 'receiver.raw'
                        shutil.copy2(raw_path, dest)
                        self.step_rx.outputs['raw_file'] = str(dest)
                        self.step_rx.outputs['traces'] = parser.list_traces()

            if raw_data is None:
                # Fallback: try ngspice (from cosim module)
                self._notify('RX', 'running', 'Trying ngspice fallback...')
                try:
                    ngrunner = NgSpiceRunner()
                    if ngrunner.available:
                        ok = ngrunner.run_transient(str(cir_path))
                        if ok:
                            sim_engine = 'ngspice'
                            ng_raw_path = ngrunner.get_raw_path()
                            if ng_raw_path:
                                parser = LTSpiceRawParser(ng_raw_path)
                                raw_data = parser.to_dict()
                                import shutil
                                dest = self.session_dir / 'raw' / 'receiver.raw'
                                shutil.copy2(ng_raw_path, dest)
                                self.step_rx.outputs['raw_file'] = str(dest)
                                self.step_rx.outputs['traces'] = parser.list_traces()
                except Exception as e:
                    logger.warning("ngspice simulation failed: %s", e)

            if sim_engine == 'none':
                self.step_rx.status = 'error'
                self.step_rx.message = 'No SPICE engine available (LTspice or ngspice)'
                self.step_rx.duration_s = _time.time() - t0
                self._notify('RX', 'error', self.step_rx.message)
                return self.step_rx

            self.step_rx.status = 'done'
            self.step_rx.duration_s = _time.time() - t0
            self.step_rx.message = f'Simulation complete ({sim_engine}, {self.step_rx.duration_s:.1f}s)'
            self.step_rx.outputs['engine'] = sim_engine
            self.step_rx.outputs['netlist'] = str(cir_path)
            self._notify('RX', 'done', self.step_rx.message)

        except Exception as e:
            self.step_rx.status = 'error'
            self.step_rx.message = str(e)
            self.step_rx.duration_s = _time.time() - t0
            self._notify('RX', 'error', str(e))

        return self.step_rx

    # -------------------------------------------------------------------------
    # Run all 3 steps
    # -------------------------------------------------------------------------

    def run_all(self) -> Dict[str, StepResult]:
        """
        Run complete TX -> Channel -> RX pipeline.

        Automatically selects engine based on config.simulation_engine:
            - 'python': All-Python simulation
            - 'spice':  Hybrid (Python TX+Channel → SPICE RX → Python BER)
            - fallback: SPICE unavailable → Python with warning

        Returns:
            Dict with 'TX', 'Channel', 'RX' StepResult objects
        """
        engine = getattr(self.config, 'simulation_engine', 'spice')

        if engine == 'python':
            return self.run_python_engine()

        # Graceful degradation: if SPICE requested but unavailable, fall back
        if engine == 'spice' and not self.ltspice.available and not spice_available():
            fallback_msg = (
                "WARNING: SPICE engine requested but no SPICE simulator found "
                "(LTspice/ngspice). Falling back to Python engine. "
                "Install LTspice or ngspice for full SPICE-level simulation."
            )
            logger.warning(fallback_msg)
            self._notify('RX', 'running', fallback_msg)
            return self.run_python_engine()

        # Hybrid pipeline: Python TX+Channel → SPICE RX → Python BER
        return self.run_hybrid()

    def run_hybrid(self) -> Dict[str, StepResult]:
        """
        Run hybrid pipeline: Python TX+Channel → SPICE RX → Python BER.

        This is the primary SPICE execution path. Uses Python's unified
        modulate() for TX (supporting all 5 schemes), Python channel model
        for propagation, and SPICE for the analog receiver circuit.
        BER is computed in Python from the SPICE comparator output.
        """
        self.run_step_tx()
        if self.step_tx.status != 'done':
            return self._results_dict()

        self.run_step_channel()
        if self.step_channel.status != 'done':
            return self._results_dict()

        self.run_step_rx()

        # Auto-compute BER if .raw data available
        if self.step_rx.status == 'done':
            self.compute_ber()
            self.step_rx.outputs['engine'] = 'hybrid'

        return self._results_dict()

    def run_python_engine(self) -> Dict[str, StepResult]:
        """
        Run all-Python system-level simulation for non-SPICE papers.

        Uses the Python simulation engine (cosim.python_engine) which
        supports OFDM, BFSK, PWM-ASK, Manchester, and standard OOK.
        """
        from .python_engine import run_python_simulation

        t0 = _time.time()
        cfg = self.config

        # TX step
        self.step_tx.status = 'running'
        self._notify('TX', 'running', f'Python engine: {cfg.modulation} modulation...')

        try:
            result = run_python_simulation(cfg)

            # TX done
            self.step_tx.status = 'done'
            self.step_tx.duration_s = _time.time() - t0
            self.step_tx.message = (
                f'{cfg.n_bits} bits @ {cfg.data_rate_bps/1e3:.0f} kbps, '
                f'{cfg.modulation} modulation'
            )
            self.step_tx.outputs = {
                'n_bits': cfg.n_bits,
                'P_dc_mW': np.mean(result['P_tx']) * 1e3,
            }
            self._tx_bits = result['bits_tx']
            self._notify('TX', 'done', self.step_tx.message)

            # Channel done
            self.step_channel.status = 'done'
            self.step_channel.duration_s = 0.0
            self.step_channel.message = (
                f'G_ch={result["channel_gain"]:.4e}, '
                f'P_rx_avg={result["P_rx_avg_uW"]:.2f} uW'
            )
            self.step_channel.outputs = {
                'G_ch': result['channel_gain'],
                'P_rx_avg_uW': result['P_rx_avg_uW'],
                'I_ph_avg_uA': result['I_ph_avg_uA'],
            }
            self._notify('Channel', 'done', self.step_channel.message)

            # RX done
            self.step_rx.status = 'done'
            self.step_rx.duration_s = _time.time() - t0
            ber = result['ber']
            self.step_rx.message = (
                f'Python sim complete ({self.step_rx.duration_s:.2f}s) | '
                f'BER={ber:.4e} ({result["n_errors"]}/{result["n_bits_tested"]})'
            )
            self.step_rx.outputs = {
                'engine': 'python',
                'ber': ber,
                'ber_n_errors': result['n_errors'],
                'ber_n_bits': result['n_bits_tested'],
                'snr_est_dB': result['snr_est_dB'],
                'python_result': result,
            }
            self._notify('RX', 'done', self.step_rx.message)

            # Save results to session directory
            self._save_python_results(result)

        except Exception as e:
            self.step_rx.status = 'error'
            self.step_rx.message = f'Python engine error: {e}'
            self.step_rx.duration_s = _time.time() - t0
            self._notify('RX', 'error', str(e))

        return self._results_dict()

    def _save_python_results(self, result: Dict) -> None:
        """Save Python engine results to session directory."""
        try:
            # Save waveform data as numpy arrays
            data_dir = self.session_dir / 'raw'
            data_dir.mkdir(parents=True, exist_ok=True)

            np.savez(
                data_dir / 'python_results.npz',
                time=result['time'],
                P_tx=result['P_tx'],
                P_rx=result['P_rx'],
                I_ph=result['I_ph'],
                V_rx=result['V_rx'],
                bits_tx=result['bits_tx'],
                bits_rx=result['bits_rx'],
            )

            # Save summary as JSON
            import json
            summary = {
                'engine': 'python',
                'modulation': result['modulation'],
                'ber': result['ber'],
                'n_errors': result['n_errors'],
                'n_bits_tested': result['n_bits_tested'],
                'snr_est_dB': result['snr_est_dB'],
                'channel_gain': result['channel_gain'],
                'P_rx_avg_uW': result['P_rx_avg_uW'],
                'I_ph_avg_uA': result['I_ph_avg_uA'],
            }
            (data_dir / 'python_summary.json').write_text(
                json.dumps(summary, indent=2), encoding='utf-8')
        except Exception as e:
            logger.warning("Failed to save Python results: %s", e)

    def compute_ber(self) -> Optional[Dict]:
        """
        Compute BER from V(dout) in .raw file vs TX bits.

        Uses cosim.spice_extract for result extraction, resampling,
        and BER computation. Handles polarity inversion automatically.

        Returns:
            BER result dict or None
        """
        raw_path = self.step_rx.outputs.get('raw_file')
        if not raw_path or self._tx_bits is None:
            return None

        try:
            self._notify('RX', 'done', 'Computing BER...')
            cfg = self.config

            from .spice_extract import compute_ber_from_spice
            ber_result = compute_ber_from_spice(
                raw_path,
                bits_tx=self._tx_bits,
                data_rate_bps=cfg.data_rate_bps,
                vcc=cfg.vcc_volts,
                skip_bits=2,
                sample_offset=0.5,
            )

            # Store in RX outputs
            self.step_rx.outputs['ber'] = ber_result['ber']
            self.step_rx.outputs['ber_n_errors'] = ber_result['n_errors']
            self.step_rx.outputs['ber_n_bits'] = ber_result['n_bits_tested']
            self.step_rx.outputs['snr_est_dB'] = ber_result['snr_est_dB']
            self.step_rx.outputs['bits_rx'] = ber_result['bits_rx']

            ber = ber_result['ber']
            inv_str = ' (inverted)' if ber_result.get('inverted') else ''
            self.step_rx.message += (
                f' | BER={ber:.4e} '
                f'({ber_result["n_errors"]}/{ber_result["n_bits_tested"]}){inv_str}'
            )
            self._notify('RX', 'done', self.step_rx.message)

            return ber_result

        except Exception as e:
            self._notify('RX', 'done',
                         f'{self.step_rx.message} | BER error: {e}')
            return None

    def _results_dict(self) -> Dict[str, StepResult]:
        return {
            'TX': self.step_tx,
            'Channel': self.step_channel,
            'RX': self.step_rx,
        }

    # -------------------------------------------------------------------------
    # Netlist generation helpers
    # -------------------------------------------------------------------------

    def _noise_section(self, cfg, noise_pwl_path: Optional[str] = None) -> str:
        """Generate noise source SPICE lines if noise is enabled.

        Phase 3: Uses calibrated PWL noise current source from the Python
        6-source NoiseModel, injected at the sense resistor input (current domain).
        Falls back to behavioral white() sources if no PWL is available.
        """
        if not cfg.noise_enable:
            return '* (noise sources disabled)'

        lines = ['* === NOISE SOURCES ===']

        # Prefer PWL noise injection (calibrated, 6-source)
        if noise_pwl_path:
            pwl_str = str(Path(noise_pwl_path).resolve()).replace('\\', '/')
            lines.append('* Calibrated 6-source noise (from cosim.noise.NoiseModel)')
            lines.append(f'* PWL file: {pwl_str}')
            lines.append(f'Inoise sc_cathode sense_lo PWL file="{pwl_str}"')
            return '\n'.join(lines)

        # Fallback: behavioral white() noise at INA output
        gain = 10 ** (cfg.ina_gain_dB / 20)
        noise_terms = []
        lines.append('* Behavioral noise (fallback, LTspice-specific)')

        en = cfg.ina_noise_nV_rtHz * 1e-9
        en_out = en * gain
        noise_terms.append(f'{en_out:.4e} * white(13579)')

        if cfg.shot_noise_enable:
            q = 1.602e-19
            I_ph_est = cfg.photocurrent_A()
            i_shot = np.sqrt(2 * q * I_ph_est)
            v_shot_out = i_shot * cfg.r_sense_ohm * gain
            noise_terms.append(f'{v_shot_out:.4e} * white(24680)')

        if cfg.thermal_noise_enable:
            kT = 1.38e-23 * 300
            v_thermal = np.sqrt(4 * kT * cfg.r_sense_ohm)
            v_thermal_out = v_thermal * gain
            noise_terms.append(f'{v_thermal_out:.4e} * white(97531)')

        noise_expr = ' + '.join(noise_terms)
        lines.append(f'Bn_sum ina_out 0 V = {{V(ina_out_clean) + {noise_expr}}}')

        return '\n'.join(lines)

    # -------------------------------------------------------------------------
    # Config-driven subcircuit generators
    # -------------------------------------------------------------------------

    @staticmethod
    def _subckt_solar_cell(cfg) -> str:
        """Generate SOLAR_CELL subcircuit from SystemConfig fields."""
        Cj = cfg.sc_cj_nF * 1e-9
        Rsh = cfg.sc_rsh_kOhm * 1e3
        R_lambda = cfg.sc_responsivity
        Rs = getattr(cfg, 'pv_series_resistance_ohm', 2.5)
        return f"""\
.SUBCKT SOLAR_CELL anode cathode photo_in
Gph cathode anode_int VALUE = {{V(photo_in) * {R_lambda}}}
Rs anode_int anode {Rs}
Cj anode_int cathode {Cj:.6e}
Rsh anode_int cathode {Rsh:.1f}
D1 anode_int cathode SOLAR_D
.MODEL SOLAR_D D(IS=1e-10 N=1.5 RS=0.01)
.ENDS SOLAR_CELL
"""

    @staticmethod
    def _subckt_ina(cfg) -> str:
        """Generate INA subcircuit from SystemConfig fields."""
        gain = 10 ** (cfg.ina_gain_dB / 20)
        gbw = cfg.ina_gbw_kHz * 1e3
        f_3dB = gbw / gain
        f_p2 = f_3dB * 10
        C_p1 = 1 / (2 * np.pi * f_3dB * 1e3)
        C_p2 = 1 / (2 * np.pi * f_p2 * 1e3)
        return f"""\
.SUBCKT INA INP INN OUT VCC VEE REF
Rinp INP 0 1G
Rinn INN 0 1G
Rref REF 0 1G
Ediff diff_int 0 INP INN {gain:.2f}
Rp1 diff_int p1 1k
Cp1 p1 0 {C_p1:.6e}
Rp2 p1 p2 1k
Cp2 p2 0 {C_p2:.6e}
Bout OUT 0 V = {{MAX(MIN(V(p2) + V(REF), V(VCC)-0.05), V(VEE)+0.05)}}
.ENDS INA
"""

    @staticmethod
    def _subckt_bpf(cfg) -> str:
        """Generate BPF_STAGE subcircuit from SystemConfig fields."""
        Rhp = cfg.bpf_rhp
        Chp = cfg.bpf_chp_pF * 1e-12
        Rlp = cfg.bpf_rlp
        Clp = cfg.bpf_clf_nF * 1e-9
        return f"""\
.SUBCKT BPF_STAGE inp out vcc vee vref
Chp inp hp_out {Chp:.6e}
Rhp hp_out vref {Rhp:.0f}
Rin hp_out opamp_inn {Rlp:.0f}
Rfb opamp_inn out {Rlp:.0f}
Cfb opamp_inn out {Clp:.6e}
Ediff_oa oa_diff 0 vref opamp_inn 100000
Rpole_oa oa_diff oa_pole 1k
Cpole_oa oa_pole 0 1.59n
Bout_oa out 0 V = {{MAX(MIN(V(oa_pole), V(vcc)-0.02), V(vee)+0.02)}}
.ENDS BPF_STAGE
"""

    @staticmethod
    def _subckt_comparator(cfg) -> str:
        """Generate COMPARATOR subcircuit from SystemConfig fields."""
        delay_ns = getattr(cfg, 'comparator_prop_delay_ns', 260.0)
        C_del = delay_ns  # pF with 1k resistor gives tau = delay_ns
        return f"""\
.SUBCKT COMPARATOR INP INN OUT VCC VEE
Rinp INP 0 1T
Rinn INN 0 1T
Bcomp comp_int 0 V = {{(V(VCC)+V(VEE))/2 + (V(VCC)-V(VEE))/2 * tanh(1e4*(V(INP)-V(INN)))}}
Rdel comp_int del_out 1k
Cdel del_out 0 {C_del:.0f}p
Eout OUT 0 del_out 0 1
.ENDS COMPARATOR
"""

    @staticmethod
    def _subckt_dcdc(cfg) -> str:
        """Generate BOOST_DCDC subcircuit from SystemConfig fields."""
        L = cfg.dcdc_l_uH * 1e-6
        Cp = cfg.dcdc_cp_uF * 1e-6
        Cl = cfg.dcdc_cl_uF * 1e-6
        Rload = cfg.r_load_ohm
        dcr = getattr(cfg, 'dcdc_inductor_dcr_ohm', 0.5)
        return f"""\
.SUBCKT BOOST_DCDC vin vout gnd phi
Cp vin gnd {Cp:.6e}
L1 vin sw {L:.6e}
R_dcr sw sw2 {dcr}
M1 sw2 phi gnd gnd BOOST_SW W=1m L=1u
.MODEL BOOST_SW NMOS(VTO=0.8 KP=200m RD=0.026 RS=0.026)
Ds sw2 vout SCHOTTKY_BOOST
.MODEL SCHOTTKY_BOOST D(IS=1e-5 N=1.05 RS=0.1 CJO=50p VJ=0.3 BV=40)
Cl vout gnd {Cl:.6e}
Rload vout gnd {Rload:.0f}
.ENDS BOOST_DCDC
"""

    def _generate_rx_netlist(self, pwl_path: str,
                             noise_pwl_path: Optional[str] = None) -> str:
        """
        Generate receiver SPICE netlist from SystemConfig fields.

        Topology adapts to cfg.rx_topology:
            - ina_bpf_comp: Solar cell → R_sense → INA → BPF(×N) → Comparator
            - amp_slicer: Solar cell → R_sense → E-source amp → Comparator
            - direct: Solar cell → R_sense → output

        All subcircuit parameters are derived from SystemConfig — no
        paper-specific imports.
        """
        cfg = self.config
        pwl_abs = Path(pwl_path).resolve()
        if not pwl_abs.exists():
            raise FileNotFoundError(f"PWL file not found: {pwl_abs}")
        pwl_str = str(pwl_abs).replace('\\', '/')
        t_stop = cfg.t_stop_s
        t_step = t_stop / 1000
        topology = getattr(cfg, 'rx_topology', 'ina_bpf_comp')

        _ch = OpticalChannel.from_config(cfg)
        G_ch = _ch.channel_gain()
        P_rx_avg = cfg.led_radiated_power_mW * 1e-3 * G_ch

        # Build subcircuit definitions based on topology
        subckt_defs = self._subckt_solar_cell(cfg)
        if topology == 'ina_bpf_comp':
            subckt_defs += '\n' + self._subckt_ina(cfg)
            if cfg.bpf_stages > 0:
                subckt_defs += '\n' + self._subckt_bpf(cfg)
            if cfg.comparator_part != 'N/A':
                subckt_defs += '\n' + self._subckt_comparator(cfg)
        if cfg.dcdc_enable:
            subckt_defs += '\n' + self._subckt_dcdc(cfg)

        # Build data path based on topology
        data_path_lines = []
        data_path_lines.append('* --- Solar Cell ---')
        data_path_lines.append('Xsc sc_anode sc_cathode optical_power SOLAR_CELL')
        data_path_lines.append('')
        data_path_lines.append(f'* --- Current Sense Resistor ---')
        data_path_lines.append(f'Rsense sc_cathode sense_lo {cfg.r_sense_ohm}')
        data_path_lines.append('')
        data_path_lines.append('* --- Ground reference ---')
        data_path_lines.append('Vgnd_ref sense_lo 0 DC 0')

        if topology == 'ina_bpf_comp':
            # INA → BPF(×N) → Comparator
            ina_out_node = 'ina_out_clean' if cfg.noise_enable and not noise_pwl_path else 'ina_out'
            data_path_lines.append('')
            data_path_lines.append(f'* --- Instrumentation Amplifier ({cfg.ina_gain_dB:.0f} dB) ---')
            data_path_lines.append(f'Xina sense_lo sc_cathode {ina_out_node} vcc vee vref INA')
            data_path_lines.append(self._noise_section(cfg, noise_pwl_path))

            # BPF stages
            prev_node = 'ina_out'
            for i in range(cfg.bpf_stages):
                out_node = f'bpf{i+1}_out' if i < cfg.bpf_stages - 1 else 'bpf_out'
                data_path_lines.append(f'Xbpf{i+1} {prev_node} {out_node} vcc vee vref BPF_STAGE')
                prev_node = out_node

            # Comparator
            comp_input = 'bpf_out' if cfg.bpf_stages > 0 else 'ina_out'
            if cfg.comparator_part != 'N/A':
                data_path_lines.append(f'Xcomp {comp_input} vref dout vcc vee COMPARATOR')
            else:
                data_path_lines.append(f'* No comparator — use BPF output as dout')
                data_path_lines.append(f'Eout_buf dout 0 {comp_input} 0 1')

        elif topology == 'amp_slicer':
            # Simple voltage amplifier + comparator/slicer
            gain = 10 ** (cfg.ina_gain_dB / 20) if cfg.ina_gain_dB > 0 else cfg.amp_gain_linear
            data_path_lines.append('')
            data_path_lines.append(f'* --- Voltage Amplifier (gain={gain:.1f}) ---')
            data_path_lines.append(f'Eamp amp_out 0 VALUE = {{V(sense_lo) * {-gain:.2f} + {cfg.vcc_volts/2}}}')
            data_path_lines.append(self._noise_section(cfg, noise_pwl_path))
            data_path_lines.append(f'* --- Slicer/Comparator ---')
            data_path_lines.append(f'Bcomp dout 0 V = {{{cfg.vcc_volts} * (1 + tanh(1000 * (V(amp_out) - {cfg.vcc_volts/2}))) / 2}}')

        else:  # direct
            # Just output sense voltage
            data_path_lines.append('')
            data_path_lines.append(f'* --- Direct output (no analog chain) ---')
            data_path_lines.append(f'Eout_buf dout 0 sense_lo 0 {-1}')
            data_path_lines.append(self._noise_section(cfg, noise_pwl_path))

        data_path_lines.append('')
        data_path_lines.append('* --- Output measurement ---')
        data_path_lines.append('Rout_data dout 0 1MEG')

        data_path = '\n'.join(data_path_lines)

        # Build DC-DC section
        dcdc_section = ''
        if cfg.dcdc_enable:
            fsw = cfg.dcdc_fsw_kHz * 1e3
            dcdc_section = f"""\
* =====================================================================
* DC-DC CONVERTER (Power Path)
* =====================================================================
Vphi phi 0 PULSE(0 {cfg.vcc_volts} 0 10n 10n {0.5/fsw:.6e} {1/fsw:.6e})
Xdcdc sc_anode dcdc_out 0 phi BOOST_DCDC
Rout_dcdc dcdc_out 0 1MEG
"""

        # Build measurements
        meas_lines = [f'.MEAS TRAN v_sc_avg AVG V(sc_anode) FROM={t_stop/2:.2e} TO={t_stop:.2e}']
        if topology == 'ina_bpf_comp':
            meas_lines.append(f'.MEAS TRAN v_ina_rms RMS V(ina_out) FROM={t_stop/2:.2e} TO={t_stop:.2e}')
            if cfg.bpf_stages > 0:
                meas_lines.append(f'.MEAS TRAN v_bpf_rms RMS V(bpf_out) FROM={t_stop/2:.2e} TO={t_stop:.2e}')
                meas_lines.append(f'.MEAS TRAN v_bpf_pp PP V(bpf_out) FROM={t_stop/2:.2e} TO={t_stop:.2e}')
        if cfg.dcdc_enable:
            meas_lines.append(f'.MEAS TRAN v_dcdc_avg AVG V(dcdc_out) FROM={t_stop/2:.2e} TO={t_stop:.2e}')
        measurements = '\n'.join(meas_lines)

        netlist = f"""\
* =====================================================================
* LiFi-PV Receiver — Co-Simulation Netlist (config-driven)
* Config: {cfg.preset_name or 'custom'}
* Topology: {topology}
* =====================================================================
* Channel gain:    G_ch = {G_ch:.6e}
* P_rx (avg):      {P_rx_avg*1e6:.2f} uW
* I_ph (avg):      {P_rx_avg*cfg.sc_responsivity*1e6:.2f} uA
* PWL source:      {pwl_str}
* =====================================================================

.TITLE LiFi_PV_CoSim_{cfg.preset_name or 'custom'}

* =====================================================================
* SUBCIRCUIT DEFINITIONS
* =====================================================================
{subckt_defs}

* =====================================================================
* POWER SUPPLIES
* =====================================================================
Vcc vcc 0 DC {cfg.vcc_volts}
Vee vee 0 DC 0
Vref vref 0 DC {cfg.vcc_volts / 2}

* =====================================================================
* OPTICAL INPUT (PWL bridge from channel model)
* =====================================================================
Voptical optical_power 0 PWL file="{pwl_str}"

* =====================================================================
* RECEIVER - DATA PATH
* =====================================================================
{data_path}

{dcdc_section}
* =====================================================================
* SIMULATION COMMANDS
* =====================================================================
.tran {t_step:.2e} {t_stop:.2e} 0 {t_step:.2e}
.OPTIONS reltol=0.001 abstol=1e-12 vntol=1e-6

* Measurements
{measurements}

.END
"""
        return netlist

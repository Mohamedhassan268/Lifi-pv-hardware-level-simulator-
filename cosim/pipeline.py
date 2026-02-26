# cosim/pipeline.py
"""
3-Step Simulation Pipeline: TX -> Channel -> RX

Orchestrates the full co-simulation flow:
    Step 1 (TX):      Generate P_tx(t) — OOK modulated optical power waveform
    Step 2 (Channel): Apply channel model → write i_ph.pwl bridge file
    Step 3 (RX):      Generate receiver netlist → run SPICE → parse results

Each step is independent and re-runnable.

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
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable
import time as _time

from .system_config import SystemConfig
from .pwl_writer import write_photocurrent_pwl
from .ltspice_runner import LTSpiceRunner
from .raw_parser import LTSpiceRawParser
from .spice_finder import spice_available

logger = logging.getLogger(__name__)


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
        self._I_ph = None
        self._tx_bits = None

    def _notify(self, step_name: str, status: str, message: str = ''):
        """Send progress notification."""
        if self.on_progress:
            self.on_progress(step_name, status, message)

    # -------------------------------------------------------------------------
    # Step 1: TX — generate OOK optical power waveform
    # -------------------------------------------------------------------------

    def run_step_tx(self) -> StepResult:
        """
        Generate transmitted optical power waveform P_tx(t).

        Uses PRBS generator to create OOK-modulated waveform.
        """
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self.step_tx.status = 'running'
        self._notify('TX', 'running', 'Generating OOK waveform...')
        t0 = _time.time()

        try:
            from simulation.prbs_generator import generate_prbs, generate_ook_waveform

            cfg = self.config

            # Generate PRBS bit sequence
            bits = generate_prbs(order=cfg.prbs_order, n_bits=cfg.n_bits)
            self._tx_bits = bits

            # Convert to OOK optical power waveform
            P_dc = cfg.led_radiated_power_mW * 1e-3  # W
            time_arr, P_norm = generate_ook_waveform(
                bits,
                bit_rate=cfg.data_rate_bps,
                samples_per_bit=100,
                modulation_depth=cfg.modulation_depth,
                dc_level=1.0,
            )
            # Scale normalized waveform to actual optical power
            P_tx = P_norm * P_dc
            self._time = time_arr
            self._P_tx = P_tx

            # Save TX waveform as PWL for reference
            tx_pwl_path = self.session_dir / 'pwl' / 'P_tx.pwl'
            from .pwl_writer import write_voltage_pwl
            write_voltage_pwl(time_arr, P_tx, tx_pwl_path)

            self.step_tx.status = 'done'
            self.step_tx.duration_s = _time.time() - t0
            self.step_tx.message = (
                f'{cfg.n_bits} bits @ {cfg.data_rate_bps/1e3:.0f} kbps, '
                f'P_dc={P_dc*1e3:.1f} mW, mod={cfg.modulation_depth}'
            )
            self.step_tx.outputs = {
                'P_tx_pwl': str(tx_pwl_path),
                'n_bits': len(bits),
                'P_dc_mW': P_dc * 1e3,
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
        Apply channel model to P_tx(t) and write optical power PWL bridge file.

        The PWL file contains P_rx(t) = G_ch * P_tx(t) in watts.
        This feeds into SOLAR_CELL subcircuit as V(photo_in) where 1V = 1W.
        The subcircuit internally computes I_ph = R_lambda * V(photo_in).
        """
        self.step_channel.status = 'running'
        self._notify('Channel', 'running', 'Computing channel response...')
        t0 = _time.time()

        try:
            if self._time is None or self._P_tx is None:
                raise RuntimeError('Run TX step first')

            cfg = self.config
            G_ch = cfg.optical_channel_gain()
            R_lambda = cfg.sc_responsivity

            # Compute received optical power P_rx(t) = G_ch * P_tx(t)
            P_rx = G_ch * self._P_tx
            self._I_ph = R_lambda * P_rx

            # Write PWL bridge file: P_rx(t) in watts (voltage representation)
            # The SOLAR_CELL subcircuit uses V(photo_in) * R_lambda for Iph
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

            # Generate receiver netlist with PWL photocurrent source
            netlist = self._generate_rx_netlist(pwl_path)

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
                # Fallback: try ngspice
                self._notify('RX', 'running', 'Trying ngspice fallback...')
                try:
                    import sys, os
                    sys.path.insert(0, os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))))
                    from simulation.ngspice_runner import NgSpiceRunner
                    ngrunner = NgSpiceRunner()
                    if ngrunner.available:
                        ok = ngrunner.run(str(cir_path))
                        if ok:
                            sim_engine = 'ngspice'
                            self.step_rx.outputs['engine'] = 'ngspice'
                except ImportError:
                    pass

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

        Automatically selects SPICE or Python engine based on
        config.simulation_engine setting.

        Returns:
            Dict with 'TX', 'Channel', 'RX' StepResult objects
        """
        engine = getattr(self.config, 'simulation_engine', 'spice')

        if engine == 'python':
            return self.run_python_engine()

        # Graceful degradation: if SPICE requested but unavailable, fall back
        if engine == 'spice' and not self.ltspice.available and not spice_available():
            logger.warning(
                "SPICE engine requested but no SPICE simulator found. "
                "Falling back to Python engine."
            )
            self._notify('RX', 'running',
                         'No SPICE engine found — falling back to Python engine')
            return self.run_python_engine()

        # Default: SPICE pipeline
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

        Samples comparator output at bit centers and compares
        against transmitted PRBS sequence.

        Returns:
            BER result dict or None
        """
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))

        raw_path = self.step_rx.outputs.get('raw_file')
        if not raw_path or self._tx_bits is None:
            return None

        try:
            self._notify('RX', 'done', 'Computing BER...')

            parser = LTSpiceRawParser(raw_path)
            time = parser.get_time()
            dout = parser.get_trace('V(dout)')

            cfg = self.config
            bit_period = 1.0 / cfg.data_rate_bps
            threshold = 1.65  # VCC/2

            # Only use bits that fit within the simulation time
            t_max = time[-1]
            max_bits = int(t_max / bit_period)
            tx_bits_clipped = self._tx_bits[:max_bits]

            from simulation.analysis import calculate_ber_from_transient
            ber_result = calculate_ber_from_transient(
                tx_bits_clipped, dout, time,
                threshold=threshold,
                bit_period=bit_period,
                sample_offset=0.5,
                skip_bits=2,
            )

            # If BER > 0.4, try inverted polarity (in case BPF inverts)
            if ber_result['ber'] > 0.4:
                inv_bits = 1 - tx_bits_clipped
                inv_result = calculate_ber_from_transient(
                    inv_bits, dout, time,
                    threshold=threshold,
                    bit_period=bit_period,
                    sample_offset=0.5,
                    skip_bits=2,
                )
                if inv_result['ber'] < ber_result['ber']:
                    ber_result = inv_result
                    ber_result['inverted'] = True

            # Store in RX outputs
            self.step_rx.outputs['ber'] = ber_result['ber']
            self.step_rx.outputs['ber_n_errors'] = ber_result['n_errors']
            self.step_rx.outputs['ber_n_bits'] = ber_result['n_bits_tested']
            self.step_rx.outputs['snr_est_dB'] = ber_result['snr_est_dB']
            self.step_rx.outputs['rx_decisions'] = ber_result['rx_decisions']

            ber = ber_result['ber']
            self.step_rx.message += (
                f' | BER={ber:.4e} '
                f'({ber_result["n_errors"]}/{ber_result["n_bits_tested"]})'
            )
            self._notify('RX', 'done', self.step_rx.message)

            # Also compute BER from BPF output (pre-comparator SNR)
            try:
                v_bpf = parser.get_trace('V(bpf_out)')
                bpf_ber = calculate_ber_from_transient(
                    tx_bits_clipped, v_bpf, time,
                    threshold=1.65,
                    bit_period=bit_period,
                    sample_offset=0.5,
                    skip_bits=2,
                )
                self.step_rx.outputs['bpf_snr_dB'] = bpf_ber['snr_est_dB']
            except KeyError:
                pass

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

    def _noise_section(self, cfg) -> str:
        """Generate noise source SPICE lines if noise is enabled.

        When enabled, changes the INA output from 'ina_out' to 'ina_out_clean',
        then creates 'ina_out' = 'ina_out_clean' + noise terms. This injects
        noise at the INA output without modifying subcircuit topology.
        """
        if not cfg.noise_enable:
            return '* (noise sources disabled)'

        gain = 10 ** (cfg.ina_gain_dB / 20)
        noise_terms = []
        lines = ['* === NOISE SOURCES (injected at INA output) ===']

        # INA322 input-referred voltage noise (~45 nV/√Hz)
        en = cfg.ina_noise_nV_rtHz * 1e-9  # V/√Hz
        en_out = en * gain  # Output-referred
        lines.append(f'* INA input noise: {cfg.ina_noise_nV_rtHz} nV/rtHz '
                      f'-> {en_out*1e6:.2f} uV/rtHz at output (gain={gain:.0f})')
        noise_terms.append(f'{en_out:.4e} * white(13579)')

        # Shot noise on photocurrent -> voltage noise via Rsense * gain
        if cfg.shot_noise_enable:
            q = 1.602e-19
            I_ph_est = cfg.photocurrent_A()
            i_shot = np.sqrt(2 * q * I_ph_est)  # A/√Hz
            v_shot_out = i_shot * cfg.r_sense_ohm * gain  # V/√Hz at INA output
            lines.append(f'* Shot noise: i_n={i_shot:.2e} A/rtHz '
                          f'-> {v_shot_out*1e6:.2f} uV/rtHz at output')
            noise_terms.append(f'{v_shot_out:.4e} * white(24680)')

        # Thermal noise from Rsense -> voltage noise via gain
        if cfg.thermal_noise_enable:
            kT = 1.38e-23 * 300
            v_thermal = np.sqrt(4 * kT * cfg.r_sense_ohm)  # V/√Hz
            v_thermal_out = v_thermal * gain  # V/√Hz at INA output
            lines.append(f'* Rsense thermal: v_n={v_thermal:.2e} V/rtHz '
                          f'-> {v_thermal_out*1e6:.4f} uV/rtHz at output')
            noise_terms.append(f'{v_thermal_out:.4e} * white(97531)')

        # Build summing node: ina_out = ina_out_clean + noise
        noise_expr = ' + '.join(noise_terms)
        lines.append(f'Bn_sum ina_out 0 V = V(ina_out_clean) + {noise_expr}')

        return '\n'.join(lines)

    def _generate_rx_netlist(self, pwl_path: str) -> str:
        """
        Generate receiver SPICE netlist using PWL photocurrent source.

        Uses FullSystemNetlist subcircuit library for proper behavioral
        models (INA322 with 2-pole GBW, active BPF, comparator with delay).
        Falls back to inline netlist if FullSystemNetlist is unavailable.
        """
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))

        cfg = self.config
        pwl_abs = Path(pwl_path).resolve()
        # LTspice needs forward slashes or escaped backslashes in paths
        pwl_str = str(pwl_abs).replace('\\', '/')
        t_stop = cfg.t_stop_s
        t_step = t_stop / 1000

        try:
            from systems.kadirvelu2021_netlist import SubcircuitLibrary
            from systems.kadirvelu2021 import KadirveluParams
            lib = SubcircuitLibrary(KadirveluParams())
            subckt_defs = (lib.solar_cell() + '\n' +
                           lib.ina322() + '\n' +
                           lib.bpf_stage() + '\n' +
                           lib.comparator() + '\n' +
                           lib.dcdc_boost())
            use_subcircuits = True
        except ImportError:
            subckt_defs = ''
            use_subcircuits = False

        G_ch = cfg.optical_channel_gain()
        P_rx_avg = cfg.led_radiated_power_mW * 1e-3 * G_ch

        if use_subcircuits:
            # Full netlist using proper subcircuit models
            fsw = cfg.dcdc_fsw_kHz * 1e3
            netlist = f"""\
* =====================================================================
* LiFi-PV Receiver — Co-Simulation Netlist
* Generated by cosim.pipeline (using FullSystemNetlist subcircuits)
* Config: {cfg.preset_name or 'custom'}
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
Vcc vcc 0 DC 3.3
Vee vee 0 DC 0
Vref vref 0 DC 1.65

* =====================================================================
* OPTICAL INPUT (PWL bridge from channel model)
* =====================================================================
* V(optical_power) represents received optical power in W (1V = 1W)
Voptical optical_power 0 PWL file="{pwl_str}"

* =====================================================================
* RECEIVER - DATA PATH
* =====================================================================

* --- Solar Cell ---
Xsc sc_anode sc_cathode optical_power SOLAR_CELL

* --- Current Sense Resistor ---
Rsense sc_cathode sense_lo {cfg.r_sense_ohm}

* --- Ground reference ---
Vgnd_ref sense_lo 0 DC 0

* --- INA322 Instrumentation Amplifier (40 dB) ---
* INP=sense_lo (0V), INN=sc_cathode (negative) → positive output + Vref offset
Xina sense_lo sc_cathode {'ina_out_clean' if cfg.noise_enable else 'ina_out'} vcc vee vref INA322
{self._noise_section(cfg)}

* --- Band-Pass Filter Stage 1 ---
Xbpf1 ina_out bpf1_out vcc vee vref BPF_STAGE

* --- Band-Pass Filter Stage 2 ---
Xbpf2 bpf1_out bpf_out vcc vee vref BPF_STAGE

* --- Comparator (Data Recovery) ---
Xcomp bpf_out vref dout vcc vee COMPARATOR

* --- Output measurement ---
Rout_data dout 0 1MEG

* =====================================================================
* DC-DC CONVERTER (Power Path)
* =====================================================================
* Switching clock (fsw = {cfg.dcdc_fsw_kHz:.0f} kHz)
Vphi phi 0 PULSE(0 3.3 0 10n 10n {0.5/fsw:.6e} {1/fsw:.6e})

* Boost converter
Xdcdc sc_anode dcdc_out 0 phi BOOST_DCDC

* DC-DC output measurement
Rout_dcdc dcdc_out 0 1MEG

* =====================================================================
* SIMULATION COMMANDS
* =====================================================================
.tran {t_step:.2e} {t_stop:.2e} 0 {t_step:.2e}
.OPTIONS reltol=0.001 abstol=1e-12 vntol=1e-6

* Measurements
.MEAS TRAN v_ina_rms RMS V(ina_out) FROM={t_stop/2:.2e} TO={t_stop:.2e}
.MEAS TRAN v_bpf_rms RMS V(bpf_out) FROM={t_stop/2:.2e} TO={t_stop:.2e}
.MEAS TRAN v_bpf_pp PP V(bpf_out) FROM={t_stop/2:.2e} TO={t_stop:.2e}
.MEAS TRAN v_sc_avg AVG V(sc_anode) FROM={t_stop/2:.2e} TO={t_stop:.2e}
.MEAS TRAN v_dcdc_avg AVG V(dcdc_out) FROM={t_stop/2:.2e} TO={t_stop:.2e}

.END
"""
        else:
            # Fallback: inline simplified netlist
            Cj = cfg.sc_cj_nF * 1e-9
            Rsh = cfg.sc_rsh_kOhm * 1e3
            gain = 10 ** (cfg.ina_gain_dB / 20)
            gbw = cfg.ina_gbw_kHz * 1e3
            f0 = gbw / gain
            R_hp = cfg.bpf_rhp
            C_hp = cfg.bpf_chp_pF * 1e-12
            R_lp = cfg.bpf_rlp
            C_lp = cfg.bpf_clf_nF * 1e-9

            netlist = f"""\
* LiFi-PV Receiver — Co-Simulation (inline models)
* Config: {cfg.preset_name or 'custom'}

.OPTIONS reltol=0.001 abstol=1e-12 vntol=1e-6

Iph 0 sc_anode PWL file="{pwl_str}"
Cj sc_anode 0 {Cj}
Rsh sc_anode 0 {Rsh}
Rsense sc_anode ina_inp {cfg.r_sense_ohm}

Eina ina_raw 0 VALUE={{V(ina_inp) * {gain}}}
Rina ina_raw ina_out {1/(2*3.14159*f0):.6e}
Cina ina_out 0 1

Chp1 ina_out hp1_out {C_hp}
Rhp1 hp1_out 0 {R_hp}
Rlp1 hp1_out lp1_out {R_lp}
Clp1 lp1_out 0 {C_lp}

Chp2 lp1_out hp2_out {C_hp}
Rhp2 hp2_out 0 {R_hp}
Rlp2 hp2_out bpf_out {R_lp}
Clp2 bpf_out 0 {C_lp}

Bcomp dout 0 V = {{3.3 * (1 + tanh(1000 * V(bpf_out))) / 2}}
Rout_data dout 0 1MEG

.tran {t_step:.2e} {t_stop:.2e} uic
.END
"""
        return netlist

"""PySpice + libngspice RX-chain runner.

Replaces the LTspice/ngspice subprocess path (`cosim/ltspice_runner.py` +
`cosim/ngspice_runner.py` + `cosim/raw_parser.py` + `cosim/pwl_writer.py`)
with an in-process libngspice driver. The numerical kernel is the same
SPICE3 implementation as before; the IO is just numpy arrays in and out.

Usage:
    from cosim.pyspice_runner import PySpiceRxRunner

    runner = PySpiceRxRunner(cfg)
    if runner.available:
        wf = runner.run_transient(
            optical_t=t_arr, optical_v=p_rx_arr,
            duration_s=cfg.t_stop_s,
        )
        v_dout = wf.nodes['dout']

Phase A status: runner is functional but not yet wired into the main
pipeline. Use via the `engine_compare` dual-run flag (added in next
chunk) for verification against LTspice before flipping the default.
"""
from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .pyspice_subckts import (
    subckt_solar_cell,
    subckt_ina,
    subckt_comparator,
)
# BPF is intentionally NOT imported from pyspice_subckts: ngspice 34
# cannot converge the 100k-gain behavioral op-amp model under transient.
# Instead, BPFStage from cosim.rx_chain (scipy IIR) is spliced between
# two SPICE runs (INA -> scipy BPF -> Comparator).

logger = logging.getLogger(__name__)


# Max PWL points before decimation. ngspice handles longer but the netlist
# string balloons (~25 chars/point) and parse time grows linearly.
_MAX_PWL_POINTS = 8000


@dataclass
class RxWaveforms:
    """Output of a PySpice RX-chain transient.

    `nodes` is keyed by lowercase ngspice node name (e.g. 'sc_anode',
    'sense_lo', 'ina_out', 'bpf_out', 'dout'). Only nodes that exist in
    the simulated topology are present.
    """
    t: np.ndarray
    nodes: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    def __contains__(self, key: str) -> bool:
        return key.lower() in self.nodes

    def get(self, key: str, default=None):
        return self.nodes.get(key.lower(), default)


class PySpiceRxRunner:
    """Run the RX-chain transient via libngspice in-process.

    Construction is cheap — the actual DLL load happens on first
    `run_transient()` call (or via `selftest()`).
    """

    _dll_check_cache: Optional[bool] = None

    def __init__(self, cfg, verbose: bool = False):
        self.config = cfg
        self.verbose = verbose
        self._last_meta: Dict[str, object] = {}

    # ------------------------------------------------------------------ #
    # Availability                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def available(cls) -> bool:
        """Check (and cache) whether NgSpiceShared can be instantiated."""
        if cls._dll_check_cache is not None:
            return cls._dll_check_cache
        try:
            from PySpice.Spice.NgSpice.Shared import NgSpiceShared
            ngs = NgSpiceShared.new_instance()
            _ = ngs.ngspice_version
            cls._dll_check_cache = True
        except Exception as e:
            logger.info("PySpice / libngspice unavailable: %s", e)
            cls._dll_check_cache = False
        return cls._dll_check_cache

    @property
    def version_string(self) -> str:
        if not self.available():
            return 'PySpice / libngspice unavailable'
        try:
            from PySpice.Spice.NgSpice.Shared import NgSpiceShared
            return f'libngspice {NgSpiceShared.new_instance().ngspice_version}'
        except Exception as e:
            return f'libngspice load error: {e}'

    # ------------------------------------------------------------------ #
    # Transient                                                          #
    # ------------------------------------------------------------------ #

    def run_transient(
        self,
        optical_t: np.ndarray,
        optical_v: np.ndarray,
        noise_t: Optional[np.ndarray] = None,
        noise_v: Optional[np.ndarray] = None,
        duration_s: Optional[float] = None,
        t_step_s: Optional[float] = None,
    ) -> RxWaveforms:
        """Hybrid two-stage RX-chain transient.

        Pipeline:
            Stage A (SPICE):  optical -> SOLAR_CELL -> Rsense -> INA -> V(ina_out)
            Stage B (scipy):  V(ina_out) -> BPFStage.process() x N -> V(bpf_out)
            Stage C (SPICE):  PWL(V(bpf_out)) -> COMPARATOR -> V(dout)

        For `amp_slicer` and `direct` topologies, only Stage A runs (no BPF,
        no comparator subckt -- the slicer is baked into Stage A's netlist).

        Args:
            optical_t, optical_v: optical-power PWL (V at `optical_power` node;
                interpreted internally as W * responsivity = photocurrent).
            noise_t, noise_v: optional injected noise (added between INA and BPF).
            duration_s: defaults to cfg.t_stop_s.
            t_step_s: defaults to duration_s / 1000.
        """
        if not self.available():
            raise RuntimeError(
                'PySpice / libngspice not available. Run scripts/install_pyspice.py')

        cfg = self.config
        t_stop = float(duration_s if duration_s is not None else cfg.t_stop_s)
        t_step = float(t_step_s if t_step_s is not None else t_stop / 1000.0)
        topology = getattr(cfg, 'rx_topology', 'ina_bpf_comp')

        # -- Stage A: SPICE through INA (or full slicer chain for non-INA topologies) --
        t0 = _time.time()
        stage_a_circuit = self._build_circuit_stage_a(
            cfg=cfg, topology=topology,
            optical_t=optical_t, optical_v=optical_v,
            noise_t=noise_t, noise_v=noise_v,
        )
        stage_a_nodes, t_sim = self._simulate_circuit(stage_a_circuit, t_step, t_stop)
        elapsed_a = _time.time() - t0

        nodes: Dict[str, np.ndarray] = dict(stage_a_nodes)
        elapsed_b = 0.0
        elapsed_c = 0.0

        if topology == 'ina_bpf_comp':
            # -- Stage B: scipy BPF (cascaded) --
            if cfg.bpf_stages > 0 and 'ina_out' in nodes:
                tb0 = _time.time()
                from .rx_chain import BPFStage
                bpf = BPFStage.from_config(cfg)
                v_bpf = nodes['ina_out']
                for i in range(cfg.bpf_stages):
                    v_bpf = bpf.process(v_bpf, t_sim)
                    nodes[f'bpf{i+1}_out'] = v_bpf
                nodes['bpf_out'] = v_bpf
                elapsed_b = _time.time() - tb0

            # -- Stage C: SPICE comparator --
            if cfg.comparator_part != 'N/A':
                tc0 = _time.time()
                comp_in_v = nodes.get('bpf_out', nodes.get('ina_out'))
                if comp_in_v is None:
                    raise RuntimeError(
                        'Stage C cannot run: neither bpf_out nor ina_out available')
                try:
                    stage_c_circuit = self._build_circuit_comparator(
                        cfg, t_sim, comp_in_v,
                    )
                    stage_c_nodes, t_c = self._simulate_circuit(
                        stage_c_circuit, t_step, t_stop,
                    )
                    if 'dout' in stage_c_nodes:
                        # Stage C uses ngspice's adaptive step internally, so
                        # its returned t_c may have a different length from
                        # t_sim. Resample dout onto Stage A's grid so all
                        # nodes share one time vector downstream.
                        v_dout_c = stage_c_nodes['dout']
                        if len(t_c) != len(t_sim) or not np.allclose(t_c, t_sim):
                            v_dout_c = np.interp(t_sim, t_c, v_dout_c)
                        nodes['dout'] = v_dout_c
                except Exception as e:
                    logger.warning(
                        'SPICE comparator stage failed (%s); falling back to '
                        'scipy ComparatorStage', e)
                    from .rx_chain import ComparatorStage
                    comp = ComparatorStage(
                        prop_delay_ns=getattr(cfg, 'comparator_prop_delay_ns', 260.0),
                        V_cc=cfg.vcc_volts, V_ee=0.0,
                    )
                    v_ref_arr = np.full_like(comp_in_v, cfg.vcc_volts / 2)
                    nodes['dout'] = comp.process(comp_in_v, v_ref_arr, t_sim)
                elapsed_c = _time.time() - tc0

        return RxWaveforms(
            t=t_sim,
            nodes=nodes,
            meta={
                'engine': 'pyspice+scipy_bpf',
                'topology': topology,
                'duration_s': elapsed_a + elapsed_b + elapsed_c,
                'duration_stage_a_s': elapsed_a,
                'duration_stage_b_s': elapsed_b,
                'duration_stage_c_s': elapsed_c,
                'n_points': len(t_sim),
            },
        )

    def _simulate_circuit(self, circuit, t_step: float, t_stop: float):
        """Run a PySpice Circuit transient and return (nodes_dict, t_vector)."""
        try:
            simulator = circuit.simulator(
                temperature=25, nominal_temperature=25,
                simulator='ngspice-shared',
            )
            simulator.options('reltol=0.001', 'abstol=1e-12', 'vntol=1e-6')
            analysis = simulator.transient(step_time=t_step, end_time=t_stop)
        except Exception as e:
            raise RuntimeError(f'libngspice transient failed: {e}') from e

        t_vec = np.array(analysis.time)
        wanted = [
            'sc_anode', 'sc_cathode', 'sense_lo', 'optical_power',
            'ina_out', 'ina_out_clean', 'amp_out',
            'dout',
        ]
        nodes: Dict[str, np.ndarray] = {}
        for name in wanted:
            try:
                nodes[name] = np.array(analysis[name])
            except (KeyError, IndexError):
                continue
        return nodes, t_vec

    # ------------------------------------------------------------------ #
    # Circuit construction                                               #
    # ------------------------------------------------------------------ #

    def _build_circuit_stage_a(
        self,
        cfg,
        topology: str,
        optical_t: np.ndarray,
        optical_v: np.ndarray,
        noise_t: Optional[np.ndarray],
        noise_v: Optional[np.ndarray],
    ):
        """Stage A circuit: optical -> SOLAR_CELL -> Rsense -> (INA | amp_slicer | direct).

        Stops at V(ina_out) for ina_bpf_comp topology. For amp_slicer and
        direct topologies, the full chain stays in SPICE (no BPF involved).

        DC-DC converter is NOT included here - it runs as scipy BoostConverter
        in parallel on V(sc_anode).
        """
        from PySpice.Spice.Netlist import Circuit

        title = f"LiFi_PV_PySpice_A_{getattr(cfg, 'preset_name', None) or 'custom'}"
        circuit = Circuit(title)

        circuit.raw_spice += subckt_solar_cell(cfg) + '\n'
        if topology == 'ina_bpf_comp':
            circuit.raw_spice += subckt_ina(cfg) + '\n'

        # --- Power supplies (always needed for INA rails) ---
        circuit.V('cc', 'vcc', circuit.gnd, f'DC {cfg.vcc_volts}')
        circuit.V('ee', 'vee', circuit.gnd, 'DC 0')
        circuit.V('ref', 'vref', circuit.gnd, f'DC {cfg.vcc_volts / 2}')

        # --- Optical input as PWL voltage source on 'optical_power' node ---
        pwl_str = _pwl_string(optical_t, optical_v)
        circuit.raw_spice += f'Voptical optical_power 0 DC 0 PWL({pwl_str})\n'

        # --- Solar cell + sense resistor ---
        # sc_anode needs an external harvest path; without it, photocurrent
        # can't return, the cell pins at Voc, and the INA sees nothing.
        # We model the DC-DC's input load as a parallel pair:
        #   R_harvest  — average DC current draw (cell at its MPP point)
        #   Idcdc      — pulsed AC current at f_sw representing the boost
        #                converter's chopper behaviour at vin
        # The Idcdc PULSE is what gives LTspice its characteristic 50 kHz
        # ripple at V(sc_anode); without it Phase B bit-level agreement
        # caps at ~75%. With it (Phase B.2), expected agreement is >95%.
        # If dcdc_enable=False, only R_harvest is emitted.
        v_mpp = getattr(cfg, 'sc_vmpp_mV', 740.0) * 1e-3
        i_mpp = getattr(cfg, 'sc_impp_uA', 470.0) * 1e-6
        r_harvest = v_mpp / i_mpp if i_mpp > 0 else 1e6
        circuit.raw_spice += 'Xsc sc_anode sc_cathode optical_power SOLAR_CELL\n'
        circuit.raw_spice += f'R_harvest sc_anode 0 {r_harvest:.6e}\n'
        circuit.raw_spice += f'Rsense sc_cathode sense_lo {cfg.r_sense_ohm}\n'
        circuit.raw_spice += 'Vgnd_ref sense_lo 0 DC 0\n'

        # --- Synthetic DC-DC switching ---
        # First attempt at injection (Phase B.2) made bit-level agreement
        # WORSE: a pulse current sink at f_sw with DCM peak amplitude
        # (~1.15 mA) overshot the cell's I_sc and dragged V(sc_anode)
        # across the I-V curve. Bit agreement dropped 75% -> 50%.
        # Removed pending a better model. Two options for a future pass:
        #   1. Pre-record V(sc_anode) from one LTspice run, replay it as
        #      a forcing PWL voltage source here (highest fidelity).
        #   2. Couple cosim/dcdc_model.BoostConverter dynamically by
        #      iterating cell operating point + injecting only the AC
        #      delta (current-limited to I_sc).
        # For now: no synthetic injection. PySpice represents the
        # 'optical chain without DC-DC noise' view; LTspice keeps the
        # full 'with DC-DC noise' view. Engines test different physics
        # by design.

        if topology == 'ina_bpf_comp':
            self._stage_a_ina(circuit, cfg, noise_t, noise_v)
        elif topology == 'amp_slicer':
            self._stage_a_amp_slicer(circuit, cfg, noise_t, noise_v)
        else:  # 'direct'
            self._stage_a_direct(circuit, cfg, noise_t, noise_v)

        return circuit

    def _stage_a_ina(self, circuit, cfg, noise_t, noise_v):
        """Stage A tail: drive Xina, optionally adding noise via PWL B-source."""
        has_noise_pwl = noise_t is not None and noise_v is not None
        ina_out_node = 'ina_out_clean' if has_noise_pwl else 'ina_out'
        circuit.raw_spice += (
            f'Xina sense_lo sc_cathode {ina_out_node} vcc vee vref INA\n'
        )
        if has_noise_pwl:
            n_pwl = _pwl_string(noise_t, noise_v)
            circuit.raw_spice += (
                f'Vnoise_inj noise_inj 0 DC 0 PWL({n_pwl})\n'
                f'Bnoise ina_out 0 V = V(ina_out_clean) + V(noise_inj)\n'
            )
        # Light load on ina_out for measurement / DC reference.
        circuit.raw_spice += 'Rina_load ina_out 0 1MEG\n'

    def _stage_a_amp_slicer(self, circuit, cfg, noise_t, noise_v):
        self._build_amp_slicer(circuit, cfg, noise_t, noise_v)
        circuit.raw_spice += 'Rout_data dout 0 1MEG\n'

    def _stage_a_direct(self, circuit, cfg, noise_t, noise_v):
        self._build_direct(circuit, cfg, noise_t, noise_v)
        circuit.raw_spice += 'Rout_data dout 0 1MEG\n'

    def _build_circuit_comparator(self, cfg, t_arr: np.ndarray, v_in: np.ndarray):
        """Stage C circuit: PWL(v_in) -> COMPARATOR -> V(dout).

        Tiny netlist - just the comparator subckt fed by a PWL voltage source.
        Runs cheaply (<50 ms typical) against the BPF-filtered waveform from
        Stage B.
        """
        from PySpice.Spice.Netlist import Circuit

        title = f"LiFi_PV_PySpice_C_{getattr(cfg, 'preset_name', None) or 'custom'}"
        circuit = Circuit(title)
        circuit.raw_spice += subckt_comparator(cfg) + '\n'

        circuit.V('cc', 'vcc', circuit.gnd, f'DC {cfg.vcc_volts}')
        circuit.V('ee', 'vee', circuit.gnd, 'DC 0')
        circuit.V('ref', 'vref', circuit.gnd, f'DC {cfg.vcc_volts / 2}')

        pwl_str = _pwl_string(t_arr, v_in)
        circuit.raw_spice += (
            f'Vcomp_in comp_in 0 DC {cfg.vcc_volts / 2} PWL({pwl_str})\n'
            f'Xcomp comp_in vref dout vcc vee COMPARATOR\n'
            f'Rout_data dout 0 1MEG\n'
        )
        return circuit

    def _build_amp_slicer(self, circuit, cfg, noise_t, noise_v):
        """E-source amp + tanh slicer data path (no INA subckt)."""
        gain = 10 ** (cfg.ina_gain_dB / 20) if cfg.ina_gain_dB > 0 \
            else getattr(cfg, 'amp_gain_linear', 1.0)
        circuit.raw_spice += (
            f'Eamp amp_out 0 VALUE = '
            f'{{V(sense_lo) * {-gain:.4f} + {cfg.vcc_volts/2}}}\n'
        )
        if noise_t is not None and noise_v is not None:
            n_pwl = _pwl_string(noise_t, noise_v)
            circuit.raw_spice += f'Vnoise_inj noise_inj 0 DC 0 PWL({n_pwl})\n'
            circuit.raw_spice += (
                'Bamp_noisy amp_out_noisy 0 V = V(amp_out) + V(noise_inj)\n'
            )
            slicer_in = 'amp_out_noisy'
        else:
            slicer_in = 'amp_out'
        circuit.raw_spice += (
            f'Bcomp dout 0 V = {{{cfg.vcc_volts} * '
            f'(1 + tanh(1000 * (V({slicer_in}) - {cfg.vcc_volts/2}))) / 2}}\n'
        )

    def _build_direct(self, circuit, cfg, noise_t, noise_v):
        """Direct sense voltage to output (no amplification)."""
        circuit.raw_spice += 'Eout_buf dout 0 sense_lo 0 -1\n'
        if noise_t is not None and noise_v is not None:
            n_pwl = _pwl_string(noise_t, noise_v)
            circuit.raw_spice += f'Vnoise_inj noise_inj 0 DC 0 PWL({n_pwl})\n'

    # ------------------------------------------------------------------ #
    # Self-test (for CI / install verification)                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def selftest(cls) -> int:
        """Run a trivial RC transient through libngspice. Returns 0 on PASS."""
        if not cls.available():
            print('FAIL: libngspice not available')
            return 1
        try:
            from PySpice.Spice.Netlist import Circuit
            from PySpice.Unit import u_us, u_ms, u_uF, u_Ohm
            c = Circuit('selftest_rc')
            c.V('src', 'src', c.gnd, 'PULSE(0 1 0 1u 1u 0.5m 1m)')
            c.R('1', 'src', 'load', 1@u_Ohm * 1000)
            c.C('1', 'load', c.gnd, 1@u_uF * 0.1)
            sim = c.simulator(simulator='ngspice-shared')
            a = sim.transient(step_time=1@u_us, end_time=1@u_ms)
            v = np.array(a['load'])
            if v[-1] < 0.05:   # tau=100us, 5tau after falling edge -> ~0
                print(f'OK PySpiceRxRunner.selftest passed (V_out[-1]={v[-1]:.4f})')
                return 0
            print(f'FAIL: V_out[-1]={v[-1]:.3f} (expected near 0)')
            return 2
        except Exception as e:
            print(f'FAIL: {e}')
            return 3


# -------------------------------------------------------------------------- #
# Helpers                                                                    #
# -------------------------------------------------------------------------- #


def _pwl_string(t: np.ndarray, v: np.ndarray) -> str:
    """Format (t, v) pairs as an ngspice PWL argument list.

    Decimates uniformly if longer than _MAX_PWL_POINTS to keep the netlist
    string manageable.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if t.shape != v.shape:
        raise ValueError(f'PWL t and v shapes differ: {t.shape} vs {v.shape}')
    n = len(t)
    if n > _MAX_PWL_POINTS:
        idx = np.linspace(0, n - 1, _MAX_PWL_POINTS).astype(int)
        t = t[idx]
        v = v[idx]
    # ngspice prefers scientific notation; %.6e is precise and compact.
    return ' '.join(f'{ti:.6e} {vi:.6e}' for ti, vi in zip(t, v))


if __name__ == '__main__':
    import sys
    sys.exit(PySpiceRxRunner.selftest())

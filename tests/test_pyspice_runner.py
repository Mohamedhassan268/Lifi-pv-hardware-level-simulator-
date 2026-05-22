"""Tests for cosim/pyspice_runner.py.

Most tests skip cleanly if libngspice is unavailable (so CI doesn't fail
on machines that haven't run scripts/install_pyspice.py). The
PySpice/libngspice path is the future default for SPICE simulation in
this codebase; the LTspice subprocess path is being retired.
"""
from __future__ import annotations

import numpy as np
import pytest


# Sentinel: skip the whole module if PySpice/libngspice can't load.
pyspice_unavailable = pytest.mark.skipif(
    not __import__('cosim.pyspice_runner', fromlist=['PySpiceRxRunner'])
       .PySpiceRxRunner.available(),
    reason='libngspice DLL not loadable (run scripts/install_pyspice.py)',
)


class TestPySpiceImports:
    """Imports must succeed even without libngspice."""

    def test_runner_imports(self):
        from cosim.pyspice_runner import PySpiceRxRunner, RxWaveforms  # noqa: F401

    def test_subckts_import(self):
        from cosim.pyspice_subckts import (  # noqa: F401
            subckt_solar_cell, subckt_ina, subckt_bpf,
            subckt_comparator, subckt_dcdc,
        )

    def test_runner_construction_without_dll(self):
        """Constructor must not require the DLL to load."""
        from cosim.pyspice_runner import PySpiceRxRunner
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('kadirvelu2021')
        runner = PySpiceRxRunner(cfg)
        assert runner is not None


@pyspice_unavailable
class TestPySpiceSelftest:
    """Minimal libngspice load and RC transient."""

    def test_selftest_returns_zero(self):
        from cosim.pyspice_runner import PySpiceRxRunner
        assert PySpiceRxRunner.selftest() == 0

    def test_version_string_includes_libngspice(self):
        from cosim.pyspice_runner import PySpiceRxRunner
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('kadirvelu2021')
        runner = PySpiceRxRunner(cfg)
        assert 'libngspice' in runner.version_string


@pyspice_unavailable
class TestPWLEncoding:
    """In-memory PWL formatting helper."""

    def test_pwl_short(self):
        from cosim.pyspice_runner import _pwl_string
        t = np.array([0.0, 1e-6, 2e-6])
        v = np.array([0.0, 0.5, 1.0])
        s = _pwl_string(t, v)
        assert '0.000000e+00' in s
        assert '2.000000e-06' in s

    def test_pwl_decimates_long(self):
        """PWL > 8000 points should be decimated to keep netlist manageable."""
        from cosim.pyspice_runner import _pwl_string, _MAX_PWL_POINTS
        t = np.linspace(0, 1e-3, 20000)
        v = np.sin(2 * np.pi * 5000 * t)
        s = _pwl_string(t, v)
        # Each point is '<t> <v> ' ~ 28 chars; decimated to _MAX_PWL_POINTS.
        n_points = s.count(' ') // 2 + 1
        assert n_points <= _MAX_PWL_POINTS + 1
        assert n_points >= _MAX_PWL_POINTS - 1

    def test_pwl_shape_mismatch_raises(self):
        from cosim.pyspice_runner import _pwl_string
        with pytest.raises(ValueError):
            _pwl_string(np.array([0.0, 1e-6]), np.array([0.0]))


@pyspice_unavailable
class TestStageATransient:
    """Stage A (SOLAR_CELL + Rsense + INA) end-to-end via run_transient.

    This is the topology variant proven to converge in ngspice (see
    Phase A bisect). Confirms the runner produces real waveforms, not
    just an empty result.
    """

    def _make_input(self):
        t_stop = 1e-3
        t = np.linspace(0, t_stop, 1000)
        # 5 kHz OOK at 1 mW peak
        p_rx = np.where(np.sin(2 * np.pi * 5e3 * t) > 0, 1e-3, 0.0)
        return t, p_rx, t_stop

    def test_full_chain_converges_and_returns_nodes(self):
        from cosim.pyspice_runner import PySpiceRxRunner
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('kadirvelu2021')
        if cfg.rx_topology == 'auto':
            cfg.rx_topology = 'ina_bpf_comp'
        t, p_rx, t_stop = self._make_input()
        runner = PySpiceRxRunner(cfg)
        wf = runner.run_transient(
            optical_t=t, optical_v=p_rx,
            duration_s=t_stop, t_step_s=t_stop / 5000.0,
        )
        # Time vector + standard nodes must exist
        assert len(wf.t) > 100
        for node in ('sc_anode', 'sc_cathode', 'ina_out', 'bpf_out', 'dout'):
            assert node in wf.nodes, f'missing node {node!r}'
        # V(sc_anode) must respond to the input (Voc swings under light)
        assert np.ptp(wf.nodes['sc_anode']) > 0.05

    def test_meta_reports_hybrid_engine(self):
        from cosim.pyspice_runner import PySpiceRxRunner
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('kadirvelu2021')
        if cfg.rx_topology == 'auto':
            cfg.rx_topology = 'ina_bpf_comp'
        t, p_rx, t_stop = self._make_input()
        runner = PySpiceRxRunner(cfg)
        wf = runner.run_transient(
            optical_t=t, optical_v=p_rx,
            duration_s=t_stop, t_step_s=t_stop / 5000.0,
        )
        assert wf.meta['engine'] == 'pyspice+scipy_bpf'
        # All three stages should contribute non-negative time
        assert wf.meta['duration_stage_a_s'] >= 0
        assert wf.meta['duration_stage_b_s'] >= 0
        assert wf.meta['duration_stage_c_s'] >= 0


@pyspice_unavailable
class TestStageAAlternateTopologies:
    """amp_slicer and direct topologies bypass the BPF/comparator entirely."""

    def _make_input(self):
        t_stop = 1e-3
        t = np.linspace(0, t_stop, 1000)
        p_rx = np.where(np.sin(2 * np.pi * 5e3 * t) > 0, 1e-3, 0.0)
        return t, p_rx, t_stop

    def test_amp_slicer_runs(self):
        from cosim.pyspice_runner import PySpiceRxRunner
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('kadirvelu2021')
        cfg.rx_topology = 'amp_slicer'
        t, p_rx, t_stop = self._make_input()
        wf = PySpiceRxRunner(cfg).run_transient(
            optical_t=t, optical_v=p_rx,
            duration_s=t_stop, t_step_s=t_stop / 5000.0,
        )
        assert 'dout' in wf.nodes

    def test_direct_runs(self):
        from cosim.pyspice_runner import PySpiceRxRunner
        from cosim.system_config import SystemConfig
        cfg = SystemConfig.from_preset('kadirvelu2021')
        cfg.rx_topology = 'direct'
        t, p_rx, t_stop = self._make_input()
        wf = PySpiceRxRunner(cfg).run_transient(
            optical_t=t, optical_v=p_rx,
            duration_s=t_stop, t_step_s=t_stop / 5000.0,
        )
        assert 'dout' in wf.nodes

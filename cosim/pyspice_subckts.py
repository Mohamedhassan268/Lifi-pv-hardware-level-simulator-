"""SPICE subcircuit text builders for the PySpice RX-chain engine.

These mirror the LTspice-flavor subcircuits previously embedded in
`cosim/pipeline.py` (`_subckt_solar_cell`, `_subckt_ina`, `_subckt_bpf`,
`_subckt_comparator`, `_subckt_dcdc`). They emit ngspice-compatible SPICE
text that PySpice attaches to a `Circuit` via `circuit.raw_spice`.

Why text and not PySpice's `SubCircuitFactory` object API:
  - The B-source expressions (`MAX(MIN(...))`, `tanh(...)`) are already
    debugged in ngspice syntax; rebuilding them through SubCircuitFactory
    just adds a translation layer.
  - The RX chain only has ~5 fixed subckts. The object API earns its keep
    when you generate many circuits programmatically.

All builders are pure functions of a SystemConfig-like object (only
attribute access is required, not the full SystemConfig class).
"""
from __future__ import annotations

import math


def subckt_solar_cell(cfg) -> str:
    """Photovoltaic cell: photocurrent source + shunt diode + parasitics.

    Matches the topology of pipeline._subckt_solar_cell. Node order:
        anode (output high), cathode (output low), photo_in (V driven by PWL).
    """
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


def subckt_ina(cfg) -> str:
    """Instrumentation amplifier: 2-pole linear gain with rail-to-rail clipping.

    f_3dB derived from GBW/gain. Output limited to vee+50mV .. vcc-50mV via
    ngspice-native MAX/MIN math (no codemodel needed).
    """
    gain = 10 ** (cfg.ina_gain_dB / 20)
    gbw = cfg.ina_gbw_kHz * 1e3
    f_3dB = gbw / gain
    f_p2 = f_3dB * 10
    # Time constants chosen so cutoff hits f_3dB and f_p2 with 1k poles.
    # tau = R*C, f = 1/(2*pi*tau) -> C = 1/(2*pi*f*R), with R=1k:
    C_p1 = 1.0 / (2 * math.pi * f_3dB * 1e3)
    C_p2 = 1.0 / (2 * math.pi * f_p2 * 1e3)
    return f"""\
.SUBCKT INA INP INN OUT VCC VEE REF
Rinp INP 0 1G
Rinn INN 0 1G
Rref REF 0 1G
Ediff diff_int 0 INP INN {gain:.4f}
Rp1 diff_int p1 1k
Cp1 p1 0 {C_p1:.6e}
Rp2 p1 p2 1k
Cp2 p2 0 {C_p2:.6e}
Bout OUT 0 V = {{MAX(MIN(V(p2) + V(REF), V(VCC)-0.05), V(VEE)+0.05)}}
.ENDS INA
"""


def subckt_bpf(cfg) -> str:
    """Single bandpass stage: AC-coupling HPF + active LPF feedback.

    Op-amp open-loop gain is 10,000 (vs LTspice's 100,000). Closed-loop
    transfer here is set by Rfb/Rin and is essentially insensitive to
    open-loop gain above ~1000; lowering it dramatically improves
    ngspice transient convergence by reducing initial-condition swing
    on the high-impedance internal node.
    """
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
Ediff_oa oa_diff 0 vref opamp_inn 10000
Rpole_oa oa_diff oa_pole 1k
Cpole_oa oa_pole 0 1.59n
Bout_oa out 0 V = {{MAX(MIN(V(oa_pole), V(vcc)-0.02), V(vee)+0.02)}}
.ENDS BPF_STAGE
"""


def subckt_comparator(cfg) -> str:
    """tanh-based slicer with RC propagation delay.

    Cdel (in pF) with the 1k Rdel gives tau = delay_ns when read as
    "delay_ns picofarads against 1 kOhm" (1e-12 * 1e3 = 1e-9 = ns).
    """
    delay_ns = getattr(cfg, 'comparator_prop_delay_ns', 260.0)
    return f"""\
.SUBCKT COMPARATOR INP INN OUT VCC VEE
Rinp INP 0 1T
Rinn INN 0 1T
Bcomp comp_int 0 V = {{(V(VCC)+V(VEE))/2 + (V(VCC)-V(VEE))/2 * tanh(1e4*(V(INP)-V(INN)))}}
Rdel comp_int del_out 1k
Cdel del_out 0 {delay_ns:.0f}p
Eout OUT 0 del_out 0 1
.ENDS COMPARATOR
"""


def subckt_dcdc(cfg) -> str:
    """Boost DC-DC: L + switching NMOS + Schottky + output filter."""
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

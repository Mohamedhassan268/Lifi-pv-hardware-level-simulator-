# systems/kadirvelu2021_netlist.py
"""
Kadirvelu 2021 - Unified SPICE Netlist Generator
=================================================

Generates complete, simulation-ready SPICE netlists for the full LiFi-PV
system from Kadirvelu et al. (2021).

All component values are sourced from KadirveluParams (single source of truth),
ensuring consistency between SPICE netlists and visual schematics.

System Blocks:
    TX:   LED driver (ADA4891 + BSD235N + LXM5-PD01)
    CH:   Optical channel (Lambertian, behavioral gain)
    RX:   Solar cell (KXOB25 equivalent) + R_sense + INA322 + 2x BPF + TLV7011
    PWR:  DC-DC boost converter (L + NTS4409 + Schottky + caps)

Usage:
    from systems.kadirvelu2021_netlist import FullSystemNetlist

    gen = FullSystemNetlist()
    netlist = gen.generate(source_type='ook', f_signal=5e3)
    gen.save('kadirvelu2021_full.cir')
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.kadirvelu2021 import KadirveluParams


# =============================================================================
# SPICE MODEL PATHS
# =============================================================================

_BASE_DIR = Path(__file__).parent.parent
_SPICE_MODELS_DIR = _BASE_DIR / 'spice_models'


def _include_path(model_name: str) -> str:
    """Get .include line for a SPICE model file."""
    path = _SPICE_MODELS_DIR / f"{model_name}.lib"
    return f".include {path}"


# =============================================================================
# SUBCIRCUIT GENERATORS
# =============================================================================

class SubcircuitLibrary:
    """
    Generates SPICE subcircuits from KadirveluParams.

    Each method returns a string containing one .SUBCKT definition.
    All component values come from the params object.
    """

    def __init__(self, params: KadirveluParams = None):
        self.p = params or KadirveluParams()

    def solar_cell(self) -> str:
        """
        Solar cell equivalent circuit.

        Topology: Iph || Cj || Rsh || D1 + Rs

        Nodes: anode, cathode, photo_in
            photo_in is a voltage node: V(photo_in) = received optical power (W)
        """
        Cj = self.p.SC_CJ_nF * 1e-9          # F
        Rsh = self.p.SC_RSH_kOhm * 1e3        # ohm
        R_lambda = self.p.SC_RESPONSIVITY      # A/W
        Rs = 2.5                                # ohm (series resistance)

        return f"""\
* =============================================================
* Solar Cell Equivalent Circuit (GaAs, KXOB25-04X3F equivalent)
* Cj = {self.p.SC_CJ_nF} nF, Rsh = {self.p.SC_RSH_kOhm} kohm
* R_lambda = {R_lambda} A/W
* =============================================================
.SUBCKT SOLAR_CELL anode cathode photo_in
* photo_in: voltage representing received optical power (1V = 1W)
* Photocurrent: Iph = R_lambda * P_optical

* Photocurrent source (controlled by optical input)
Gph cathode anode_int VALUE = {{V(photo_in) * {R_lambda}}}

* Series resistance
Rs anode_int anode {Rs}

* Junction capacitance
Cj anode_int cathode {Cj:.6e}

* Shunt resistance
Rsh anode_int cathode {Rsh:.1f}

* Junction diode (single-diode model)
D1 anode_int cathode SOLAR_D
.MODEL SOLAR_D D(IS=1e-10 N=1.5 RS=0.01)

.ENDS SOLAR_CELL
"""

    def ina322(self) -> str:
        """
        INA322 instrumentation amplifier (behavioral).

        Gain = 5 + 5*(R1/R2) = 100.5 ≈ 100 (40 dB)
        GBW = 700 kHz => f_3dB = 7 kHz at gain=100

        The REF pin sets the output DC level:
            V_out = G*(V_INP - V_INN) + V_REF
        """
        R1 = self.p.INA_R1
        R2 = self.p.INA_R2
        gain = 5 + 5 * (R1 / R2)
        gbw = self.p.INA_GBW_kHz * 1e3
        f_3dB = gbw / gain
        f_p2 = f_3dB * 10  # non-dominant pole

        # RC values for poles
        C_p1 = 1 / (2 * np.pi * f_3dB * 1e3)   # R=1k
        C_p2 = 1 / (2 * np.pi * f_p2 * 1e3)     # R=1k

        return f"""\
* =============================================================
* INA322 Instrumentation Amplifier - Behavioral Model
* Gain = {gain:.1f} ({20*np.log10(gain):.1f} dB)
* GBW = {gbw/1e3:.0f} kHz, f_3dB = {f_3dB/1e3:.1f} kHz
* R1 = {R1/1e3:.0f}k, R2 = {R2/1e3:.0f}k
* REF pin sets output common-mode (typically Vref = VCC/2)
* =============================================================
.SUBCKT INA322 INP INN OUT VCC VEE REF

* High input impedance
Rinp INP 0 1G
Rinn INN 0 1G
Rref REF 0 1G

* Differential gain stage
Ediff diff_int 0 INP INN {gain:.2f}

* Pole 1 (dominant): f = {f_3dB:.0f} Hz
Rp1 diff_int p1 1k
Cp1 p1 0 {C_p1:.6e}

* Pole 2 (non-dominant): f = {f_p2:.0f} Hz
Rp2 p1 p2 1k
Cp2 p2 0 {C_p2:.6e}

* Output buffer with REF offset and rail clamping
* V_out = G*(Vinp - Vinn) + V(REF), clamped to rails
Bout OUT 0 V = MAX(MIN(V(p2) + V(REF), V(VCC)-0.05), V(VEE)+0.05)

.ENDS INA322
"""

    def bpf_stage(self) -> str:
        """
        Band-pass filter stage (active, using TLV2379).

        Topology:
            Input -> Chp -> node_hp
            node_hp -> Rhp -> Vref  (high-pass)
            node_hp -> Rin -> opamp_inn
            opamp_inn -> Rfb || Cfb -> output  (low-pass active)

        HP corner: f_HP = 1/(2*pi*Rhp*Chp)
        LP corner: f_LP = 1/(2*pi*Rfb*Cfb)
        Passband gain: -Rfb/Rin
        """
        Rhp = self.p.BPF_RHP
        Chp = self.p.BPF_CHP_pF * 1e-12
        Rlp = self.p.BPF_RLP       # Rin = Rfb = Rlp
        Clp = self.p.BPF_CLF_nF * 1e-9

        f_hp = 1 / (2 * np.pi * Rhp * Chp)
        f_lp = 1 / (2 * np.pi * Rlp * Clp)
        gain_passband = -Rlp / Rlp  # = -1 (unity, inverting)

        return f"""\
* =============================================================
* Band-Pass Filter Stage (TLV2379-based)
* HP corner: {f_hp:.0f} Hz (Chp={self.p.BPF_CHP_pF}pF, Rhp={Rhp/1e3:.0f}k)
* LP corner: {f_lp:.0f} Hz (Cfb={self.p.BPF_CLF_nF}nF, Rfb={Rlp/1e3:.0f}k)
* Passband gain: {abs(gain_passband):.0f}x (inverting)
* =============================================================
.SUBCKT BPF_STAGE inp out vcc vee vref

* --- High-pass section (AC coupling) ---
Chp inp hp_out {Chp:.6e}
Rhp hp_out vref {Rhp:.0f}

* --- Active low-pass filter (inverting) ---
Rin hp_out opamp_inn {Rlp:.0f}
Rfb opamp_inn out {Rlp:.0f}
Cfb opamp_inn out {Clp:.6e}

* --- Op-amp: TLV2379 (GBW=100kHz) ---
* Simplified single-pole behavioral model
Ediff_oa oa_diff 0 vref opamp_inn 100000
Rpole_oa oa_diff oa_pole 1k
Cpole_oa oa_pole 0 1.59n
Bout_oa out 0 V = MAX(MIN(V(oa_pole), V(vcc)-0.02), V(vee)+0.02)

.ENDS BPF_STAGE
"""

    def comparator(self) -> str:
        """
        TLV7011 nanopower comparator (behavioral).
        Propagation delay: 260 ns
        """
        return """\
* =============================================================
* TLV7011 Comparator - Behavioral Model
* Propagation delay: 260 ns
* =============================================================
.SUBCKT COMPARATOR INP INN OUT VCC VEE

* High input impedance
Rinp INP 0 1T
Rinn INN 0 1T

* Comparator decision
Bcomp comp_int 0 V = IF(V(INP)-V(INN) > 0, V(VCC), V(VEE))

* Propagation delay (RC: tau = 260ns)
Rdel comp_int del_out 1k
Cdel del_out 0 260p

* Output buffer
Eout OUT 0 del_out 0 1

.ENDS COMPARATOR
"""

    def tx_driver(self) -> str:
        """
        LED transmitter driver.

        Topology:
            Vmod -> ADA4891 buffer -> BSD235N MOSFET -> LED
            Optical output = GLED * I(LED) * T_lens
        """
        Re = self.p.LED_DRIVER_RE
        GLED = self.p.LED_GLED
        T_lens = self.p.LENS_TRANSMITTANCE

        return f"""\
* =============================================================
* LED Transmitter Driver
* ADA4891 + BSD235N + LXM5-PD01 + Fraen Lens
* Re = {Re} ohm, GLED = {GLED} W/A, T_lens = {T_lens}
* =============================================================
.SUBCKT TX_DRIVER vin_mod optical_out gnd vcc

* LED drive resistance
Re vin_mod gate_drive {Re}

* MOSFET switch (BSD235N, simplified)
M1 led_cathode gate_drive gnd gnd BSD235N_SW W=1m L=1u
.MODEL BSD235N_SW NMOS(VTO=1.2 KP=150m RD=0.09 RS=0.09)

* LED (voltage source models Vf drop)
Vled vcc led_cathode DC 3.2

* Optical output: P_opt = GLED * I(Vled) * T_lens
Bopt optical_out gnd V = {GLED * T_lens} * I(Vled)

.ENDS TX_DRIVER
"""

    def dcdc_boost(self) -> str:
        """
        Boost DC-DC converter.

        Components: L=22uH, NTS4409, Schottky, Cp=10uF, CL=47uF
        """
        L = self.p.DCDC_L_uH * 1e-6
        Cp = self.p.DCDC_CP_uF * 1e-6
        Cl = self.p.DCDC_CL_uF * 1e-6
        Rload = self.p.DCDC_RLOAD_kOhm * 1e3

        return f"""\
* =============================================================
* Boost DC-DC Converter
* L = {self.p.DCDC_L_uH} uH, Cp = {self.p.DCDC_CP_uF} uF
* CL = {self.p.DCDC_CL_uF} uF, Rload = {self.p.DCDC_RLOAD_kOhm} kohm
* Switch: NTS4409 (Rds_on = 52 mohm)
* =============================================================
.SUBCKT BOOST_DCDC vin vout gnd phi

* Input capacitor
Cp vin gnd {Cp:.6e}

* Inductor (with DCR = 0.5 ohm)
L1 vin sw {L:.6e}
R_dcr sw sw2 0.5

* NMOS switch (NTS4409)
M1 sw2 phi gnd gnd NTS4409_SW W=1m L=1u
.MODEL NTS4409_SW NMOS(VTO=0.8 KP=200m RD=0.026 RS=0.026)

* Schottky diode
Ds sw2 vout SCHOTTKY_BOOST
.MODEL SCHOTTKY_BOOST D(IS=1e-5 N=1.05 RS=0.1 CJO=50p VJ=0.3 BV=40)

* Output capacitor
Cl vout gnd {Cl:.6e}

* Load resistor
Rload vout gnd {Rload:.0f}

.ENDS BOOST_DCDC
"""


# =============================================================================
# FULL SYSTEM NETLIST GENERATOR
# =============================================================================

class FullSystemNetlist:
    """
    Generates complete SPICE netlists for the Kadirvelu 2021 system.

    All values sourced from KadirveluParams for consistency with schematics.
    """

    def __init__(self, params: KadirveluParams = None):
        self.params = params or KadirveluParams()
        self.lib = SubcircuitLibrary(self.params)
        self._last_netlist = None

    def generate(self,
                 source_type: str = 'sine',
                 f_signal: float = 5e3,
                 modulation_depth: float = 0.33,
                 vmod_dc: float = 0.12,
                 fsw: float = 50e3,
                 t_stop: float = 2e-3,
                 t_step: float = 1e-7,
                 pwl_file: str = None,
                 include_tx: bool = True,
                 include_dcdc: bool = True,
                 ac_analysis: bool = True) -> str:
        """
        Generate complete system netlist.

        Args:
            source_type: 'sine', 'pulse' (OOK), or 'pwl' (arbitrary)
            f_signal: Signal frequency (Hz) for sine/pulse
            modulation_depth: Modulation depth (0-1)
            vmod_dc: DC bias of LED modulation (V)
            fsw: DC-DC switching frequency (Hz)
            t_stop: Transient simulation end time (s)
            t_step: Transient time step (s)
            pwl_file: PWL file path (for source_type='pwl')
            include_tx: Include TX driver subcircuit
            include_dcdc: Include DC-DC converter
            ac_analysis: Include .AC analysis

        Returns:
            Complete SPICE netlist as string
        """
        p = self.params

        # Calculate channel gain
        Gop = p.optical_channel_gain()
        G0 = Gop * p.SC_RESPONSIVITY
        P_optical_dc = (vmod_dc / p.LED_DRIVER_RE * p.LED_GLED *
                        p.LENS_TRANSMITTANCE * Gop)
        P_optical_ac = P_optical_dc * modulation_depth

        # Build signal source
        if source_type == 'sine':
            source_line = (f"Voptical optical_power 0 "
                          f"SIN({P_optical_dc:.6e} {P_optical_ac:.6e} {f_signal})")
        elif source_type == 'pulse':
            period = 1.0 / f_signal
            source_line = (f"Voptical optical_power 0 "
                          f"PULSE(0 {P_optical_dc + P_optical_ac:.6e} "
                          f"0 1n 1n {period/2:.6e} {period:.6e})")
        elif source_type == 'pwl':
            if pwl_file is None:
                raise ValueError("pwl_file required for source_type='pwl'")
            source_line = f"Voptical optical_power 0 PWL FILE = '{pwl_file}'"
        else:
            raise ValueError(f"Unknown source_type: {source_type}")

        # Build TX section
        tx_section = ""
        if include_tx:
            tx_section = f"""\
* === TRANSMITTER (informational — channel gain applied directly) ===
* LED: LXM5-PD01, Pe = {p.LED_RADIATED_POWER_mW} mW
* Driver: ADA4891 + BSD235N, Re = {p.LED_DRIVER_RE} ohm
* Lens: Fraen FLP-S9-SP, T = {p.LENS_TRANSMITTANCE}
* GLED = {p.LED_GLED} W/A
*
* NOTE: TX is modeled as a behavioral optical power source
* (channel gain Gop = {Gop:.6e} is pre-computed)
"""

        # Build DC-DC section
        dcdc_section = ""
        if include_dcdc:
            dcdc_section = f"""\
* === DC-DC CONVERTER (Power Path) ===
* Switching clock (fsw = {fsw/1e3:.0f} kHz)
Vphi phi 0 PULSE(0 3.3 0 10n 10n {0.5/fsw:.6e} {1/fsw:.6e})

* Boost converter
Xdcdc sc_anode dcdc_out 0 phi BOOST_DCDC

* DC-DC output measurement
Rout_dcdc dcdc_out 0 1MEG
"""

        # Build AC analysis
        ac_section = ""
        if ac_analysis:
            ac_section = """\
* AC analysis (frequency response)
.AC DEC 100 10 100k
"""

        # Assemble full netlist
        netlist = f"""\
* =====================================================================
* Kadirvelu 2021 - Full LiFi-PV System
* "A Circuit for Simultaneous Reception of Data and Power Using a Solar Cell"
* IEEE Trans. Green Communications and Networking, Vol. 5, No. 4, Dec 2021
* =====================================================================
* Generated by Hardware-Faithful LiFi-PV Simulator
* =====================================================================
*
* SYSTEM PARAMETERS:
*   Distance:        {p.DISTANCE_M*100:.1f} cm
*   Optical gain:    Gop = {Gop:.6e}
*   Channel gain:    G0 = Gop * R = {G0:.6e} A/W
*   P_optical (DC):  {P_optical_dc*1e6:.2f} uW
*   Signal freq:     {f_signal/1e3:.1f} kHz
*   Mod depth:       {modulation_depth*100:.0f}%
*   DC-DC fsw:       {fsw/1e3:.0f} kHz
*
* VALIDATION TARGETS:
*   Harvested power: {p.TARGET_HARVESTED_POWER_uW} uW
*   BER:             {p.TARGET_BER:.3e}
*   Noise RMS:       {p.TARGET_NOISE_RMS_mV} mV
* =====================================================================

.TITLE Kadirvelu2021_LiFi_PV_Full_System

* =====================================================================
* SUBCIRCUIT DEFINITIONS
* =====================================================================
{self.lib.solar_cell()}
{self.lib.ina322()}
{self.lib.bpf_stage()}
{self.lib.comparator()}
{self.lib.dcdc_boost()}

* =====================================================================
* POWER SUPPLIES
* =====================================================================
Vcc vcc 0 DC 3.3
Vee vee 0 DC 0
Vref vref 0 DC 1.65

{tx_section}
* =====================================================================
* OPTICAL INPUT (represents received optical power)
* =====================================================================
{source_line}

* =====================================================================
* RECEIVER - DATA PATH
* =====================================================================

* --- Solar Cell ---
Xsc sc_anode sc_cathode optical_power SOLAR_CELL

* --- Current Sense Resistor ---
Rsense sc_cathode sense_lo {p.R_SENSE}

* --- Ground reference ---
Vgnd_ref sense_lo 0 DC 0

* --- INA322 Instrumentation Amplifier (40 dB) ---
* Measures voltage across Rsense: V_sense = I_ph * R_sense
* INP=sense_lo (0V), INN=sc_cathode (negative) → V_out = G*(0 - (-Vsense)) + Vref
Xina sense_lo sc_cathode ina_out vcc vee vref INA322

* --- Band-Pass Filter Stage 1 ---
Xbpf1 ina_out bpf1_out vcc vee vref BPF_STAGE

* --- Band-Pass Filter Stage 2 ---
Xbpf2 bpf1_out bpf_out vcc vee vref BPF_STAGE

* --- Comparator (Data Recovery) ---
Xcomp bpf_out vref dout vcc vee COMPARATOR

* --- Output measurement ---
Rout_data dout 0 1MEG

{dcdc_section}
* =====================================================================
* SIMULATION COMMANDS
* =====================================================================

* Transient analysis
.TRAN {t_step:.2e} {t_stop:.2e} 0 {t_step:.2e}

* Operating point
.OP

{ac_section}
* Measurements
.MEAS TRAN v_ina_rms RMS V(ina_out) FROM={t_stop/2:.2e} TO={t_stop:.2e}
.MEAS TRAN v_bpf_rms RMS V(bpf_out) FROM={t_stop/2:.2e} TO={t_stop:.2e}
.MEAS TRAN v_bpf_pp PP V(bpf_out) FROM={t_stop/2:.2e} TO={t_stop:.2e}
.MEAS TRAN v_sc_avg AVG V(sc_anode) FROM={t_stop/2:.2e} TO={t_stop:.2e}

.END
"""
        self._last_netlist = netlist
        return netlist

    def generate_receiver_only(self,
                                f_signal: float = 5e3,
                                P_optical_uW: float = 100.0,
                                modulation_depth: float = 0.33,
                                t_stop: float = 2e-3,
                                t_step: float = 1e-7) -> str:
        """
        Generate receiver-only netlist (no TX, no DC-DC).

        Useful for focused signal chain analysis.
        """
        return self.generate(
            source_type='sine',
            f_signal=f_signal,
            modulation_depth=modulation_depth,
            vmod_dc=P_optical_uW * 1e-6 / self.params.SC_RESPONSIVITY,
            t_stop=t_stop,
            t_step=t_step,
            include_tx=False,
            include_dcdc=False,
        )

    def generate_ook_test(self,
                          bit_rate: float = 5e3,
                          n_bits: int = 100,
                          modulation_depth: float = 0.33,
                          fsw: float = 50e3) -> str:
        """
        Generate OOK test netlist with pulse source.

        Args:
            bit_rate: Bit rate in bps
            n_bits: Number of bits to simulate
            modulation_depth: OOK modulation depth
            fsw: DC-DC switching frequency

        Returns:
            SPICE netlist string
        """
        t_stop = n_bits / bit_rate  # Simulate enough for all bits

        return self.generate(
            source_type='pulse',
            f_signal=bit_rate,
            modulation_depth=modulation_depth,
            fsw=fsw,
            t_stop=t_stop,
            t_step=1 / (bit_rate * 100),  # 100 samples per bit
        )

    def save(self, filename: str, netlist: str = None) -> str:
        """
        Save netlist to file.

        Args:
            filename: Output file path
            netlist: Netlist string (default: last generated)

        Returns:
            Absolute path to saved file
        """
        if netlist is None:
            netlist = self._last_netlist
        if netlist is None:
            raise RuntimeError("No netlist generated. Call generate() first.")

        filepath = os.path.abspath(filename)
        with open(filepath, 'w') as f:
            f.write(netlist)

        print(f"Netlist saved: {filepath}")
        return filepath

    def get_node_list(self) -> dict:
        """
        Return key node names for post-processing.

        These match the node names in the generated netlist.
        """
        return {
            'optical_input': 'optical_power',
            'solar_cell_anode': 'sc_anode',
            'solar_cell_cathode': 'sc_cathode',
            'sense_resistor_lo': 'sense_lo',
            'ina_output': 'ina_out',
            'bpf_stage1_output': 'bpf1_out',
            'bpf_output': 'bpf_out',
            'comparator_output': 'dout',
            'dcdc_output': 'dcdc_out',
            'supply_vcc': 'vcc',
            'reference': 'vref',
        }

    def get_system_parameters(self) -> dict:
        """
        Return computed system parameters for documentation/validation.
        """
        p = self.params
        Gop = p.optical_channel_gain()
        m = p.lambertian_order()

        return {
            'lambertian_order': m,
            'optical_channel_gain': Gop,
            'G0_AW': Gop * p.SC_RESPONSIVITY,
            'distance_m': p.DISTANCE_M,
            'sc_area_cm2': p.SC_AREA_CM2,
            'responsivity_AW': p.SC_RESPONSIVITY,
            'capacitance_nF': p.SC_CJ_nF,
            'shunt_resistance_kohm': p.SC_RSH_kOhm,
            'ina_gain_dB': p.INA_GAIN_DB,
            'ina_gbw_kHz': p.INA_GBW_kHz,
            'bpf_passband': '700 Hz - 10 kHz',
            'bpf_stages': p.BPF_STAGES,
            'target_harvested_uW': p.TARGET_HARVESTED_POWER_uW,
            'target_ber': p.TARGET_BER,
            'target_noise_rms_mV': p.TARGET_NOISE_RMS_mV,
        }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate Kadirvelu 2021 SPICE netlist')
    parser.add_argument('-o', '--output', type=str,
                        default='kadirvelu2021_full.cir',
                        help='Output file')
    parser.add_argument('--source', type=str, default='sine',
                        choices=['sine', 'pulse', 'pwl'],
                        help='Signal source type')
    parser.add_argument('--freq', type=float, default=5e3,
                        help='Signal frequency (Hz)')
    parser.add_argument('--mod-depth', type=float, default=0.33,
                        help='Modulation depth (0-1)')
    parser.add_argument('--fsw', type=float, default=50e3,
                        help='DC-DC switching frequency (Hz)')
    parser.add_argument('--tstop', type=float, default=2e-3,
                        help='Simulation time (s)')
    parser.add_argument('--rx-only', action='store_true',
                        help='Receiver only (no TX/DC-DC)')
    parser.add_argument('--info', action='store_true',
                        help='Print system parameters')

    args = parser.parse_args()

    gen = FullSystemNetlist()

    if args.info:
        params = gen.get_system_parameters()
        print("=" * 50)
        print("SYSTEM PARAMETERS")
        print("=" * 50)
        for k, v in params.items():
            print(f"  {k}: {v}")
        print()

    if args.rx_only:
        netlist = gen.generate_receiver_only(f_signal=args.freq)
    else:
        netlist = gen.generate(
            source_type=args.source,
            f_signal=args.freq,
            modulation_depth=args.mod_depth,
            fsw=args.fsw,
            t_stop=args.tstop,
        )

    gen.save(args.output)

    # Print summary
    nodes = gen.get_node_list()
    print(f"\nKey nodes for probing:")
    for desc, node in nodes.items():
        print(f"  V({node})  — {desc}")

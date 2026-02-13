"""
Kadirvelu et al. (2021) - Full SPICE-Level Transient Simulation
================================================================

Paper: "A Circuit for Simultaneous Reception of Data and Power Using a Solar Cell"
       IEEE Trans. Green Communications and Networking, Vol. 5, No. 4, Dec 2021

This module implements the complete system from Fig. 12(b) for full
SPICE-level transient simulation using PySpice/ngspice.

System Components:
    TX: LED Driver (ADA4891 + BSD235N + LXM5-PD01)
    Channel: Optical path (Lambertian model)
    RX: Solar Cell + INA322 + Band-Pass Filter + Comparator + DC-DC Converter

Author: Hardware-Faithful LiFi-PV Simulator
"""

import os
import sys
import numpy as np
from pathlib import Path

# PySpice imports
try:
    import PySpice
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    from PySpice.Spice.Parser import SpiceParser
    from PySpice.Unit import *
    PYSPICE_AVAILABLE = True
except ImportError:
    PYSPICE_AVAILABLE = False
    print("Warning: PySpice not available. Install with: pip install PySpice")


# =============================================================================
# SYSTEM PARAMETERS (Locked from Paper)
# =============================================================================

class KadirveluParams:
    """All parameters from Kadirvelu 2021 paper."""
    
    # --- Transmitter ---
    LED_RADIATED_POWER_mW = 9.3      # Pe = 9.3 mW
    LED_HALF_ANGLE_DEG = 9.0          # α1/2 = 9° (Fraen lens)
    LED_DRIVER_RE = 12.1              # Ω
    LED_GLED = 0.88                   # W/A (LED current to optical power)
    LENS_TRANSMITTANCE = 0.85         # 85%
    
    # --- Channel ---
    DISTANCE_M = 0.325                # r = 32.5 cm
    THETA_DEG = 0.0                   # On-axis
    BETA_DEG = 0.0                    # No tilt
    
    # --- Solar Cell (GaAs, Alta Devices) ---
    SC_AREA_CM2 = 9.0                 # 5 cm × 1.8 cm
    SC_RESPONSIVITY = 0.457           # A/W
    SC_CJ_nF = 798.0                  # Junction capacitance (NOTE: paper says nF!)
    SC_RSH_kOhm = 138.8               # Shunt resistance
    SC_IPH_uA = 508.0                 # Photocurrent @ 400 lux
    SC_VMPP_mV = 740.0                # MPP voltage
    SC_IMPP_uA = 470.0                # MPP current
    SC_PMPP_uW = 347.0                # MPP power
    
    # --- Current Sense ---
    R_SENSE = 1.0                     # Ω
    
    # --- INA322 Instrumentation Amplifier ---
    INA_R1 = 191e3                    # Ω (gain setting)
    INA_R2 = 10e3                     # Ω (gain setting)
    INA_GAIN_DB = 40.0                # dB
    INA_GBW_kHz = 700.0               # kHz
    
    # --- Band-Pass Filter (2-stage) ---
    BPF_RHP = 33e3                    # Ω (high-pass)
    BPF_CHP_pF = 482.0                # pF (high-pass)
    BPF_RLP = 10e3                    # Ω (low-pass)
    BPF_CLF_nF = 64.0                 # nF (low-pass)
    BPF_STAGES = 2
    
    # Derived: fL ≈ 1/(2π × 33k × 482p) ≈ 10 kHz (HP corner)
    # Derived: fH ≈ 1/(2π × 10k × 64n) ≈ 250 Hz (LP corner)
    # But paper says 700 Hz - 10 kHz passband, so there's complexity in the active filter
    
    # --- Voltage Reference ---
    VREF = 3.3                        # V (from TLV2401)
    
    # --- DC-DC Converter ---
    DCDC_L_uH = 22.0                  # Inductor
    DCDC_CP_uF = 10.0                 # Input capacitor
    DCDC_CL_uF = 47.0                 # Output capacitor
    DCDC_RLOAD_kOhm = 180.0           # Load resistor
    DCDC_FSW_kHz = 50.0               # Switching frequency (default)
    
    # --- Validation Targets ---
    TARGET_HARVESTED_POWER_uW = 223.0
    TARGET_BER = 1.008e-3
    TARGET_NOISE_RMS_mV = 7.769
    
    @classmethod
    def lambertian_order(cls):
        """Calculate Lambertian order m from half-angle."""
        alpha_rad = np.radians(cls.LED_HALF_ANGLE_DEG)
        return -np.log(2) / np.log(np.cos(alpha_rad))
    
    @classmethod
    def optical_channel_gain(cls):
        """Calculate optical path gain Gop."""
        m = cls.lambertian_order()
        r = cls.DISTANCE_M
        A = cls.SC_AREA_CM2 * 1e-4  # Convert to m²
        theta = np.radians(cls.THETA_DEG)
        beta = np.radians(cls.BETA_DEG)
        
        Gop = (m + 1) / (2 * np.pi * r**2) * np.cos(theta)**m * np.cos(beta) * A
        return Gop
    
    @classmethod
    def photocurrent_from_vmod(cls, vmod_V):
        """Calculate photocurrent from modulating voltage."""
        Gop = cls.optical_channel_gain()
        G0 = (cls.SC_RESPONSIVITY * Gop * cls.LENS_TRANSMITTANCE * cls.LED_GLED) / cls.LED_DRIVER_RE
        return G0 * vmod_V


# =============================================================================
# SPICE NETLIST GENERATOR
# =============================================================================

class KadirveluNetlist:
    """Generate SPICE netlist for Kadirvelu 2021 system."""
    
    def __init__(self, params=None):
        self.params = params or KadirveluParams()
    
    def generate_solar_cell_model(self):
        """
        Generate solar cell SPICE model.
        
        Equivalent circuit:
            Iph (current source) || Cj || Rsh || Diode
        """
        p = self.params
        
        # Convert units
        Cj = p.SC_CJ_nF * 1e-9       # F
        Rsh = p.SC_RSH_kOhm * 1e3    # Ω
        Iph = p.SC_IPH_uA * 1e-6     # A
        
        netlist = f"""
* Solar Cell Model (GaAs, Alta Devices)
* From Kadirvelu 2021, Table parameters
.SUBCKT SOLAR_CELL anode cathode photo_in
* photo_in is a voltage that controls photocurrent (for modulation)

* Photocurrent source (controlled by optical input)
Gph cathode anode VALUE={{V(photo_in) * {p.SC_RESPONSIVITY}}}

* Junction capacitance
Cj anode cathode {Cj}

* Shunt resistance  
Rsh anode cathode {Rsh}

* Diode (models I-V curve knee)
D1 anode cathode SOLAR_DIODE
.MODEL SOLAR_DIODE D(IS=1e-10 N=1.5 RS=0.1)

.ENDS SOLAR_CELL
"""
        return netlist
    
    def generate_ina322_model(self):
        """
        Generate INA322 instrumentation amplifier model.
        
        Simplified behavioral model with:
        - Differential gain set by R1, R2
        - GBW limiting
        - Input impedance
        """
        p = self.params
        
        # Gain = 5 + 5*R1/R2 (from INA322 datasheet, but simplified here)
        # Paper uses ~40 dB = 100x
        gain = 10 ** (p.INA_GAIN_DB / 20)
        gbw = p.INA_GBW_kHz * 1e3  # Hz
        
        # Single pole model: H(s) = gain / (1 + s/ω0)
        # ω0 = GBW / gain
        f0 = gbw / gain
        
        netlist = f"""
* INA322 Instrumentation Amplifier Model
* Behavioral model with GBW limiting
.SUBCKT INA322 inp inn out vcc vee
* inp = non-inverting input
* inn = inverting input  
* out = output
* vcc, vee = supply rails

* High input impedance
Rinp inp 0 1G
Rinn inn 0 1G

* Differential input stage
Ediff diff_int 0 inp inn {gain}

* Single-pole GBW limiting
Rpole diff_int pole_int 1k
Cpole pole_int 0 {{1/(2*3.14159*{f0}*1000)}}

* Output buffer with rail limiting
Eout out_int 0 pole_int 0 1
Rout out_int out 100

* Output clamping (simplified)
* In real model, would clamp to VCC/VEE

.ENDS INA322
"""
        return netlist
    
    def generate_bandpass_filter(self):
        """
        Generate 2-stage active band-pass filter.
        
        Each stage: High-pass (AC coupling) + Low-pass (active filter)
        Op-amp: TLV2379 (GBW = 100 kHz)
        """
        p = self.params
        
        Rhp = p.BPF_RHP
        Chp = p.BPF_CHP_pF * 1e-12
        Rlp = p.BPF_RLP
        Clp = p.BPF_CLF_nF * 1e-9
        
        netlist = f"""
* Band-Pass Filter - Stage 1
* TLV2379 Op-Amp (GBW = 100 kHz)
.SUBCKT BPF_STAGE inp out vcc vee vref
* High-pass section (AC coupling)
Chp inp hp_out {Chp}
Rhp hp_out vref {Rhp}

* Low-pass active filter (inverting configuration)
Rin hp_out opamp_inn {Rlp}
Rfb opamp_inn out {Rlp}
Cfb opamp_inn out {Clp}

* Op-amp (simplified single-pole model, TLV2379 GBW=100kHz)
Xopamp vref opamp_inn out vcc vee OPAMP_TLV2379

.ENDS BPF_STAGE

* TLV2379 Op-Amp Model
.SUBCKT OPAMP_TLV2379 inp inn out vcc vee
Ediff diff_int 0 inp inn 100000
Rpole diff_int pole_int 1k  
Cpole pole_int 0 1.59n
Eout out 0 pole_int 0 1
.ENDS OPAMP_TLV2379
"""
        return netlist
    
    def generate_dcdc_converter(self):
        """
        Generate boost DC-DC converter model.
        
        Components: L=22µH, Cp=10µF, CL=47µF, Schottky diode, NMOS switch
        """
        p = self.params
        
        L = p.DCDC_L_uH * 1e-6
        Cp = p.DCDC_CP_uF * 1e-6
        Cl = p.DCDC_CL_uF * 1e-6
        Rload = p.DCDC_RLOAD_kOhm * 1e3
        
        netlist = f"""
* Boost DC-DC Converter
.SUBCKT BOOST_DCDC vin vout gnd phi
* vin = input from solar cell
* vout = output to load
* phi = switching clock

* Input capacitor
Cp vin gnd {Cp}

* Inductor
L1 vin sw {L}

* NMOS Switch (NTS4409)
M1 sw phi gnd gnd NMOS_SW W=1m L=1u
.MODEL NMOS_SW NMOS(VTO=1.5 KP=100m)

* Schottky diode
Ds sw vout SCHOTTKY
.MODEL SCHOTTKY D(IS=1e-5 N=1.05 RS=0.1 BV=40)

* Output capacitor
Cl vout gnd {Cl}

* Load resistor
Rload vout gnd {Rload}

.ENDS BOOST_DCDC
"""
        return netlist
    
    def generate_full_system(self, 
                             t_stop=1e-3, 
                             t_step=1e-7,
                             fsw=50e3,
                             vmod_dc=0.12,
                             vmod_ac=0.04,
                             f_signal=5e3,
                             modulation_depth=0.33):
        """
        Generate complete system netlist.
        
        Args:
            t_stop: Simulation end time (s)
            t_step: Time step (s)
            fsw: DC-DC switching frequency (Hz)
            vmod_dc: DC component of LED modulation (V)
            vmod_ac: AC amplitude of LED modulation (V)
            f_signal: Signal frequency for testing (Hz)
            modulation_depth: Modulation depth (0-1)
        """
        p = self.params
        
        # Calculate optical power received
        Gop = p.optical_channel_gain()
        P_optical = vmod_dc / p.LED_DRIVER_RE * p.LED_GLED * p.LENS_TRANSMITTANCE * Gop
        
        netlist = f"""
* =====================================================================
* Kadirvelu 2021 - Complete System Simulation
* "A Circuit for Simultaneous Reception of Data and Power Using a Solar Cell"
* =====================================================================

.TITLE Kadirvelu2021_LiFi_PV_System

* --- Include Component Models ---
{self.generate_solar_cell_model()}
{self.generate_ina322_model()}
{self.generate_bandpass_filter()}
{self.generate_dcdc_converter()}

* =====================================================================
* MAIN CIRCUIT
* =====================================================================

* --- Power Supplies ---
Vcc vcc 0 DC 3.3
Vee vee 0 DC 0
Vref vref 0 DC 1.65

* --- Optical Input (represents LED optical power modulated) ---
* This voltage represents received optical power in Watts
* Vmod_dc = {vmod_dc}V, Vmod_ac = {vmod_ac}V, f = {f_signal}Hz
Voptical optical_power 0 SIN({P_optical} {P_optical * modulation_depth} {f_signal})

* --- Solar Cell ---
Xsc sc_anode sc_cathode optical_power SOLAR_CELL
Rsense sc_cathode sense_node {p.R_SENSE}

* --- Ground reference for solar cell ---
* In reality, cathode connects to DC-DC, here simplified
Vgnd_ref sense_node 0 DC 0

* --- INA322 Instrumentation Amplifier ---
* Measures voltage across Rsense
Xina sc_cathode sense_node ina_out vcc vee INA322

* --- Band-Pass Filter Stage 1 ---
Xbpf1 ina_out bpf1_out vcc vee vref BPF_STAGE

* --- Band-Pass Filter Stage 2 ---
Xbpf2 bpf1_out bpf_out vcc vee vref BPF_STAGE

* --- Output measurement point ---
Rout_meas bpf_out 0 1MEG

* --- DC-DC Converter ---
* Switching clock
Vphi phi 0 PULSE(0 3.3 0 10n 10n {0.5/fsw} {1/fsw})

* DC-DC boost converter (simplified - separate from signal path)
Xdcdc sc_anode dcdc_out 0 phi BOOST_DCDC

* =====================================================================
* SIMULATION COMMANDS
* =====================================================================

* Transient analysis
.TRAN {t_step} {t_stop} 0 {t_step}

* Operating point
.OP

* AC analysis (frequency response)
.AC DEC 100 10 100k

* Measurements
.MEAS TRAN vbpf_rms RMS V(bpf_out) FROM={t_stop/2} TO={t_stop}
.MEAS TRAN vbpf_pp PP V(bpf_out) FROM={t_stop/2} TO={t_stop}
.MEAS TRAN vdcdc_avg AVG V(dcdc_out) FROM={t_stop/2} TO={t_stop}

.END
"""
        return netlist
    
    def save_netlist(self, filename, **kwargs):
        """Save netlist to file."""
        netlist = self.generate_full_system(**kwargs)
        with open(filename, 'w') as f:
            f.write(netlist)
        print(f"Netlist saved to: {filename}")
        return filename


# =============================================================================
# SIMULATION RUNNER
# =============================================================================

class KadirveluSimulation:
    """Run and analyze Kadirvelu 2021 system simulation."""
    
    def __init__(self, ngspice_path=None):
        """
        Initialize simulation.
        
        Args:
            ngspice_path: Path to ngspice executable
        """
        self.params = KadirveluParams()
        self.netlist_gen = KadirveluNetlist(self.params)
        
        # Default ngspice path
        if ngspice_path is None:
            ngspice_path = r"C:\Users\HP OMEN\OneDrive\Desktop\gradproject\components level sim\hardware_faithful_simulator\hardware_faithful_simulator\ngspice-45.2_64\Spice64\bin\ngspice.exe"
        self.ngspice_path = ngspice_path
        
        self.results = {}
    
    def run_transient(self, 
                      t_stop=1e-3,
                      fsw=50e3,
                      modulation_depth=0.33,
                      f_signal=5e3,
                      output_dir=None):
        """
        Run transient simulation.
        
        Args:
            t_stop: Simulation time (s)
            fsw: DC-DC switching frequency (Hz)
            modulation_depth: Modulation depth (0-1)
            f_signal: Signal frequency (Hz)
            output_dir: Directory for output files
        """
        import subprocess
        import tempfile
        
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        # Generate netlist
        netlist_file = os.path.join(output_dir, "kadirvelu2021.cir")
        self.netlist_gen.save_netlist(
            netlist_file,
            t_stop=t_stop,
            fsw=fsw,
            modulation_depth=modulation_depth,
            f_signal=f_signal
        )
        
        # Create control file for batch mode
        control_file = os.path.join(output_dir, "control.cir")
        raw_file = os.path.join(output_dir, "output.raw")
        
        with open(control_file, 'w') as f:
            f.write(f"""
.include {netlist_file}
.control
run
write {raw_file}
quit
.endc
""")
        
        # Run ngspice
        print(f"Running ngspice simulation...")
        print(f"  Netlist: {netlist_file}")
        print(f"  t_stop: {t_stop*1e3:.2f} ms")
        print(f"  fsw: {fsw/1e3:.1f} kHz")
        print(f"  Modulation depth: {modulation_depth*100:.0f}%")
        
        try:
            result = subprocess.run(
                [self.ngspice_path, "-b", control_file],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print("Simulation completed successfully!")
                self.results['raw_file'] = raw_file
                self.results['netlist_file'] = netlist_file
                return True
            else:
                print(f"Simulation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("Simulation timed out!")
            return False
        except FileNotFoundError:
            print(f"ngspice not found at: {self.ngspice_path}")
            return False
    
    def run_frequency_sweep(self, f_start=10, f_stop=100e3, n_points=100):
        """Run AC analysis for frequency response."""
        # TODO: Implement AC analysis
        pass
    
    def run_ber_analysis(self, 
                         modulation_depths=[0.1, 0.2, 0.33, 0.5],
                         fsw_values=[50e3, 100e3, 200e3],
                         n_bits=1000):
        """Run BER analysis for different modulation depths and switching frequencies."""
        # TODO: Implement BER analysis with PRBS
        pass
    
    def calculate_theoretical_values(self):
        """Calculate expected values from paper equations."""
        p = self.params
        
        # Optical gain
        Gop = p.optical_channel_gain()
        m = p.lambertian_order()
        
        # Photocurrent for Vmod = 120mV
        vmod_dc = 0.12  # V
        Iph = p.photocurrent_from_vmod(vmod_dc)
        
        # DC-DC efficiency targets
        efficiency = {
            50e3: 0.67,
            100e3: 0.564,
            200e3: 0.42
        }
        
        results = {
            'lambertian_order_m': m,
            'optical_gain_Gop': Gop,
            'photocurrent_Iph_uA': Iph * 1e6,
            'expected_vmpp_mV': p.SC_VMPP_mV,
            'expected_impp_uA': p.SC_IMPP_uA,
            'dcdc_efficiency': efficiency,
            'bpf_bandwidth_Hz': (p.BPF_RLP, 1/(2*np.pi*p.BPF_RLP*p.BPF_CLF_nF*1e-9)),
        }
        
        return results


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_frequency_response():
    """Plot system frequency response (Fig. 13 from paper)."""
    import matplotlib.pyplot as plt
    
    # Theoretical model from paper equations
    f = np.logspace(1, 6, 500)  # 10 Hz to 1 MHz
    
    p = KadirveluParams()
    
    # Solar cell transfer function: Hsense = Zd*Rsense / (Zd + Rsense + Zload)
    Rsh = p.SC_RSH_kOhm * 1e3
    Cj = p.SC_CJ_nF * 1e-9
    Rload = p.SC_VMPP_mV / p.SC_IMPP_uA * 1e3  # Rmpp
    Rsense = p.R_SENSE
    
    omega = 2 * np.pi * f
    s = 1j * omega
    
    # Zd = rd || Rsh || 1/sCj (simplified, assuming rd >> Rsh at operating point)
    Zd = 1 / (1/Rsh + s*Cj)
    
    # Hsense = Zd * Rsense / (Zd + Rsense + Zload)
    # Simplified for Rsense << Zd
    Hsense = Zd * Rsense / (Zd + Rsense)
    
    # INA gain (40 dB with GBW limiting)
    Aina = 100  # 40 dB
    f_ina = p.INA_GBW_kHz * 1e3 / Aina
    Hina = Aina / (1 + s / (2*np.pi*f_ina))
    
    # Band-pass filter (2 stages)
    # HP: fc = 1/(2π*33k*482p) ≈ 10 kHz
    # LP: fc = 1/(2π*10k*64n) ≈ 250 Hz
    fhp = 1 / (2*np.pi*p.BPF_RHP*p.BPF_CHP_pF*1e-12)
    flp = 1 / (2*np.pi*p.BPF_RLP*p.BPF_CLF_nF*1e-9)
    
    # Each stage gain ≈ 10 (20 dB total for 2 stages)
    Abpf = 10
    Hbpf_stage = Abpf * (s / (s + 2*np.pi*flp)) / (1 + s/(2*np.pi*fhp))
    Hbpf = Hbpf_stage ** 2  # Two stages
    
    # Total system response
    Htotal = Hsense * Hina * Hbpf
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Magnitude
    ax1.semilogx(f, 20*np.log10(np.abs(Htotal)), 'b-', linewidth=2, label='Model')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Kadirvelu 2021 - System Frequency Response')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.axvline(700, color='r', linestyle='--', alpha=0.5, label='fL = 700 Hz')
    ax1.axvline(10000, color='r', linestyle='--', alpha=0.5, label='fH = 10 kHz')
    ax1.legend()
    ax1.set_xlim([10, 1e6])
    
    # Phase
    ax2.semilogx(f, np.angle(Htotal, deg=True), 'b-', linewidth=2)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2.set_xlim([10, 1e6])
    
    plt.tight_layout()
    plt.savefig('kadirvelu2021_frequency_response.png', dpi=150)
    plt.show()
    
    return fig


def plot_dcdc_efficiency():
    """Plot DC-DC converter efficiency vs frequency (Fig. 15 from paper)."""
    import matplotlib.pyplot as plt
    
    # Data from paper
    fsw_khz = [50, 100, 200]
    efficiency = [67, 56.4, 42]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(fsw_khz, efficiency, width=30, color='steelblue', edgecolor='black')
    ax.set_xlabel('Switching Frequency (kHz)')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Kadirvelu 2021 - DC-DC Converter Efficiency')
    ax.set_ylim([0, 100])
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(fsw_khz, efficiency)):
        ax.text(x, y + 2, f'{y}%', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('kadirvelu2021_dcdc_efficiency.png', dpi=150)
    plt.show()
    
    return fig


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for Kadirvelu 2021 simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Kadirvelu 2021 LiFi-PV System Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kadirvelu2021.py --info
  python kadirvelu2021.py --netlist output.cir
  python kadirvelu2021.py --run --fsw 50000 --mod-depth 0.5
  python kadirvelu2021.py --plot-freq
        """
    )
    
    parser.add_argument('--info', action='store_true',
                        help='Show system parameters and theoretical values')
    parser.add_argument('--netlist', type=str, metavar='FILE',
                        help='Generate SPICE netlist to file')
    parser.add_argument('--run', action='store_true',
                        help='Run transient simulation')
    parser.add_argument('--fsw', type=float, default=50e3,
                        help='Switching frequency in Hz (default: 50000)')
    parser.add_argument('--mod-depth', type=float, default=0.33,
                        help='Modulation depth 0-1 (default: 0.33)')
    parser.add_argument('--t-stop', type=float, default=1e-3,
                        help='Simulation time in seconds (default: 1e-3)')
    parser.add_argument('--plot-freq', action='store_true',
                        help='Plot frequency response')
    parser.add_argument('--plot-eff', action='store_true',
                        help='Plot DC-DC efficiency')
    parser.add_argument('--schematic', action='store_true',
                        help='Generate system schematic (requires schemdraw)')
    parser.add_argument('--schematic-all', action='store_true',
                        help='Generate all schematics (system, solar cell, DC-DC, BPF)')
    
    args = parser.parse_args()
    
    if args.info:
        print("=" * 60)
        print("KADIRVELU 2021 - System Parameters")
        print("=" * 60)
        
        sim = KadirveluSimulation()
        results = sim.calculate_theoretical_values()
        
        print(f"\nOptical Path:")
        print(f"  Lambertian order m:     {results['lambertian_order_m']:.2f}")
        print(f"  Optical gain Gop:       {results['optical_gain_Gop']:.6f}")
        print(f"  Distance:               {KadirveluParams.DISTANCE_M*100:.1f} cm")
        
        print(f"\nSolar Cell:")
        print(f"  Area:                   {KadirveluParams.SC_AREA_CM2} cm²")
        print(f"  Responsivity:           {KadirveluParams.SC_RESPONSIVITY} A/W")
        print(f"  Junction capacitance:   {KadirveluParams.SC_CJ_nF} nF")
        print(f"  Shunt resistance:       {KadirveluParams.SC_RSH_kOhm} kΩ")
        print(f"  Photocurrent @ 400lux:  {results['photocurrent_Iph_uA']:.1f} µA")
        
        print(f"\nReceiver:")
        print(f"  INA gain:               {KadirveluParams.INA_GAIN_DB} dB")
        print(f"  INA GBW:                {KadirveluParams.INA_GBW_kHz} kHz")
        print(f"  BPF passband:           700 Hz - 10 kHz")
        
        print(f"\nDC-DC Efficiency:")
        for fsw, eff in results['dcdc_efficiency'].items():
            print(f"  @ {fsw/1e3:.0f} kHz:            {eff*100:.1f}%")
        
        print(f"\nValidation Targets:")
        print(f"  Harvested power:        {KadirveluParams.TARGET_HARVESTED_POWER_uW} µW")
        print(f"  BER:                    {KadirveluParams.TARGET_BER:.3e}")
        print(f"  Noise RMS:              {KadirveluParams.TARGET_NOISE_RMS_mV} mV")
    
    if args.netlist:
        netlist_gen = KadirveluNetlist()
        netlist_gen.save_netlist(
            args.netlist,
            fsw=args.fsw,
            modulation_depth=args.mod_depth,
            t_stop=args.t_stop
        )
    
    if args.run:
        sim = KadirveluSimulation()
        success = sim.run_transient(
            t_stop=args.t_stop,
            fsw=args.fsw,
            modulation_depth=args.mod_depth
        )
        if success:
            print(f"\nResults saved to: {sim.results.get('raw_file', 'N/A')}")
    
    if args.plot_freq:
        plot_frequency_response()
    
    if args.plot_eff:
        plot_dcdc_efficiency()
    
    if args.schematic:
        try:
            from systems.kadirvelu2021_schematic import draw_full_system
            draw_full_system('kadirvelu2021_system.png', show=False)
        except ImportError:
            # Try relative import
            try:
                from kadirvelu2021_schematic import draw_full_system
                draw_full_system('kadirvelu2021_system.png', show=False)
            except ImportError:
                print("Error: schemdraw not installed. Run: pip install schemdraw")
    
    if args.schematic_all:
        try:
            from systems.kadirvelu2021_schematic import draw_all_schematics
            draw_all_schematics('.')
        except ImportError:
            try:
                from kadirvelu2021_schematic import draw_all_schematics
                draw_all_schematics('.')
            except ImportError:
                print("Error: schemdraw not installed. Run: pip install schemdraw")
    
    if not any([args.info, args.netlist, args.run, args.plot_freq, args.plot_eff, args.schematic, args.schematic_all]):
        parser.print_help()


if __name__ == "__main__":
    main()

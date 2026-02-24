"""Quick test for all project modules."""
import sys, numpy as np
sys.path.insert(0, '.')

def test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {e}")

print("=" * 50)
print("PROJECT TEST SUITE")
print("=" * 50)

# Materials
test("Materials", lambda: (
    __import__('materials', fromlist=['GAAS','SILICON','get_material']),
))

# Components
test("Solar Cells", lambda: (
    __import__('components.solar_cells', fromlist=['KXOB25_04X3F']).KXOB25_04X3F().get_parameters(),
))
test("LEDs", lambda: (
    __import__('components.leds', fromlist=['LXM5_PD01']).LXM5_PD01().get_parameters(),
))
test("Amplifiers", lambda: [
    getattr(__import__('components.amplifiers', fromlist=[c]), c)().get_parameters()
    for c in ['INA322','TLV2379','ADA4891']
])
test("Comparators", lambda: (
    __import__('components.comparators', fromlist=['TLV7011']).TLV7011().get_parameters(),
))
test("MOSFETs", lambda: [
    getattr(__import__('components.mosfets', fromlist=[c]), c)().get_parameters()
    for c in ['BSD235N','NTS4409']
])
test("Registry", lambda: (
    __import__('components', fromlist=['get_component']).get_component('KXOB25-04X3F'),
))

# Simulation
test("PRBS-7", lambda: (
    None if np.sum(__import__('simulation.prbs_generator', fromlist=['generate_prbs']).generate_prbs(7, 127)) == 64
    else (_ for _ in ()).throw(AssertionError("bad balance"))
))
test("OOK waveform", lambda: (
    __import__('simulation.prbs_generator', fromlist=['generate_ook_waveform']).generate_ook_waveform(
        np.array([1,0,1,1,0]), bit_rate=5e3, modulation_depth=0.33, samples_per_bit=100),
))
test("BER theory", lambda: (
    None if 1e-4 < __import__('simulation.analysis', fromlist=['theoretical_ber_ook']).theoretical_ber_ook(10) < 1e-2
    else (_ for _ in ()).throw(AssertionError("BER out of range"))
))

# Systems
test("Channel", lambda: (
    __import__('systems.kadirvelu2021_channel', fromlist=['OpticalChannel']).OpticalChannel().link_budget(),
))
test("Netlist gen", lambda: (
    __import__('systems.kadirvelu2021_netlist', fromlist=['FullSystemNetlist']).FullSystemNetlist().generate(),
))
test("Simulation class", lambda: (
    __import__('systems.kadirvelu2021', fromlist=['KadirveluSimulation']).KadirveluSimulation().calculate_theoretical_values(),
))

print("=" * 50)
print("DONE")

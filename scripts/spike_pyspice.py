"""PySpice spike — verify libngspice loads and a trivial RC transient runs.

Run: .venv\\Scripts\\python.exe scripts\\spike_pyspice.py
"""
import sys
import traceback


def main() -> int:
    print(f"Python: {sys.version}")
    try:
        import PySpice
        print(f"PySpice: {PySpice.__version__} at {PySpice.__file__}")
    except Exception as e:
        print(f"FAIL import PySpice: {e}")
        return 1

    try:
        from PySpice.Spice.NgSpice.Shared import NgSpiceShared
        print("NgSpiceShared imported OK")
    except Exception as e:
        print(f"FAIL import NgSpiceShared: {e}")
        traceback.print_exc()
        return 2

    try:
        ngs = NgSpiceShared.new_instance()
        print(f"NgSpiceShared instantiated: ngspice version = {ngs.ngspice_version}")
    except Exception as e:
        print(f"FAIL instantiate NgSpiceShared: {e}")
        traceback.print_exc()
        return 3

    try:
        from PySpice.Spice.Netlist import Circuit
        from PySpice.Unit import u_V, u_Ohm, u_uF, u_us, u_ms

        circuit = Circuit('rc_smoke')
        circuit.V('in', 'in', circuit.gnd, 'PULSE(0 1 0 1u 1u 0.5m 1m)')
        circuit.R('1', 'in', 'out', 1@u_Ohm * 1000)
        circuit.C('1', 'out', circuit.gnd, 1@u_uF * 0.1)
        print("Circuit built:")
        print(str(circuit))

        simulator = circuit.simulator(temperature=25, nominal_temperature=25,
                                      simulator='ngspice-shared')
        analysis = simulator.transient(step_time=1@u_us, end_time=1@u_ms)
        t = analysis.time.as_ndarray()
        v_out = analysis['out'].as_ndarray()
        print(f"Transient ran: {len(t)} points, t[-1]={t[-1]*1e3:.3f} ms, "
              f"V_out[-1]={v_out[-1]:.3f} V")
        # Pulse is HIGH 0-0.5ms then LOW 0.5-1ms; with tau=100us,
        # V_out at t=1ms should be near 0 (5tau after falling edge).
        assert v_out[-1] < 0.05, f"V_out[-1]={v_out[-1]:.3f} should be ~0"
    except Exception as e:
        print(f"FAIL transient: {e}")
        traceback.print_exc()
        return 4

    # B-source smoke test: comparator-like tanh(). Needs analog.cm codemodel.
    try:
        from PySpice.Spice.Netlist import Circuit
        from PySpice.Unit import u_V, u_us, u_ms

        circuit = Circuit('bsrc_smoke')
        circuit.V('p', 'p', circuit.gnd, 'PULSE(-0.1 0.1 0 1u 1u 0.5m 1m)')
        circuit.V('n', 'n', circuit.gnd, 0@u_V)
        # Behavioral comparator: V_out = 1.65 * (1 + tanh(200*(V(p)-V(n))))
        circuit.B('comp', 'dout', circuit.gnd,
                  v='1.65*(1+tanh(200*(V(p)-V(n))))')

        simulator = circuit.simulator(simulator='ngspice-shared')
        analysis = simulator.transient(step_time=1@u_us, end_time=1@u_ms)
        v_dout = analysis['dout'].as_ndarray()
        v_p = analysis['p'].as_ndarray()
        # When p > 0 -> dout near 3.3; when p < 0 -> dout near 0
        hi = v_dout[v_p > 0.05].mean() if (v_p > 0.05).any() else float('nan')
        lo = v_dout[v_p < -0.05].mean() if (v_p < -0.05).any() else float('nan')
        print(f"B-source transient: dout HI={hi:.3f}V (expect ~3.3), "
              f"LO={lo:.3f}V (expect ~0)")
        assert 3.0 < hi < 3.4, f"comparator HI off: {hi}"
        assert lo < 0.3, f"comparator LO off: {lo}"
        print("B-source / analog.cm codemodel: PASS")
    except Exception as e:
        print(f"FAIL B-source: {e}")
        traceback.print_exc()
        return 5

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == '__main__':
    sys.exit(main())

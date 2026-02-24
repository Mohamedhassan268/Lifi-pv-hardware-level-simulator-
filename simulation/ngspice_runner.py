# simulation/ngspice_runner.py
"""
ngspice Batch Simulation Runner

Provides a robust interface for running ngspice simulations and
parsing output files.

Usage:
    runner = NgSpiceRunner()
    results = runner.run_transient('circuit.cir', ['V(out)', 'I(Rsense)'])
    time = results['time']
    vout = results['V(out)']
"""

import os
import sys
import subprocess
import tempfile
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Auto-detect bundled ngspice
_BASE_DIR = Path(__file__).parent.parent
_DEFAULT_NGSPICE = _BASE_DIR / 'ngspice-45.2_64' / 'Spice64' / 'bin' / 'ngspice.exe'


class NgSpiceRunner:
    """
    Interface to ngspice batch-mode simulation.

    Handles:
        - Running netlists in batch mode
        - Parsing ASCII and binary .raw output files
        - Extracting node voltages and currents
    """

    def __init__(self, ngspice_path: str = None):
        """
        Initialize ngspice runner.

        Args:
            ngspice_path: Path to ngspice executable.
                          Auto-detects bundled version if None.
        """
        if ngspice_path is None:
            if _DEFAULT_NGSPICE.exists():
                ngspice_path = str(_DEFAULT_NGSPICE)
            else:
                ngspice_path = 'ngspice'  # Hope it's on PATH

        self.ngspice_path = ngspice_path
        self._verify_ngspice()

    def _verify_ngspice(self):
        """Check that ngspice is accessible."""
        try:
            result = subprocess.run(
                [self.ngspice_path, '--version'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[0]
                print(f"ngspice found: {version}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"Warning: ngspice not found at {self.ngspice_path}")

    def run_transient(self, netlist_path: str,
                      output_nodes: List[str] = None,
                      output_dir: str = None,
                      timeout: int = 120) -> Optional[Dict[str, np.ndarray]]:
        """
        Run transient simulation and return waveforms.

        Args:
            netlist_path: Path to .cir netlist file
            output_nodes: List of node names to extract (e.g., ['V(out)', 'V(bpf_out)'])
            output_dir: Directory for output files (default: temp)
            timeout: Simulation timeout in seconds

        Returns:
            Dict mapping node names to numpy arrays, or None on failure.
            Always includes 'time' key.
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix='ngspice_')

        raw_file = os.path.join(output_dir, 'output.raw')
        data_file = os.path.join(output_dir, 'output.txt')

        # Build control file for batch mode
        # Use wrdata for ASCII output (more reliable parsing)
        if output_nodes:
            wrdata_vars = ' '.join(output_nodes)
            wrdata_line = f"wrdata {data_file} {wrdata_vars}"
        else:
            wrdata_line = f"write {raw_file}"

        control_content = f"""\
.include {netlist_path}
.control
run
{wrdata_line}
quit
.endc
"""
        control_file = os.path.join(output_dir, 'control.cir')
        with open(control_file, 'w') as f:
            f.write(control_content)

        # Run ngspice
        print(f"Running ngspice transient simulation...")
        print(f"  Netlist: {netlist_path}")

        try:
            result = subprocess.run(
                [self.ngspice_path, '-b', control_file],
                capture_output=True, text=True, timeout=timeout,
                cwd=output_dir
            )

            if result.returncode != 0:
                print(f"ngspice error (rc={result.returncode}):")
                # Print last few lines of stderr
                stderr_lines = result.stderr.strip().split('\n')
                for line in stderr_lines[-10:]:
                    print(f"  {line}")
                return None

            print("Simulation completed successfully.")

            # Parse output
            if output_nodes and os.path.exists(data_file):
                return self._parse_wrdata(data_file, output_nodes)
            elif os.path.exists(raw_file):
                return self._parse_raw_ascii(raw_file)
            else:
                print("Warning: No output file found")
                return None

        except subprocess.TimeoutExpired:
            print(f"Simulation timed out after {timeout}s")
            return None
        except FileNotFoundError:
            print(f"ngspice not found: {self.ngspice_path}")
            return None

    def run_ac(self, netlist_path: str,
               output_nodes: List[str] = None,
               output_dir: str = None,
               timeout: int = 60) -> Optional[Dict[str, np.ndarray]]:
        """
        Run AC analysis and return frequency response.

        Returns:
            Dict with 'frequency' and complex-valued node arrays.
        """
        # AC analysis uses the same flow, but output is complex
        # For now, use wrdata which outputs magnitude and phase
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix='ngspice_ac_')

        data_file = os.path.join(output_dir, 'ac_output.txt')

        if output_nodes:
            wrdata_vars = ' '.join(output_nodes)
        else:
            wrdata_vars = 'all'

        control_content = f"""\
.include {netlist_path}
.control
ac dec 100 10 100k
wrdata {data_file} {wrdata_vars}
quit
.endc
"""
        control_file = os.path.join(output_dir, 'ac_control.cir')
        with open(control_file, 'w') as f:
            f.write(control_content)

        print("Running ngspice AC analysis...")

        try:
            result = subprocess.run(
                [self.ngspice_path, '-b', control_file],
                capture_output=True, text=True, timeout=timeout,
                cwd=output_dir
            )

            if result.returncode != 0:
                print(f"ngspice AC error: {result.stderr[-200:]}")
                return None

            if os.path.exists(data_file):
                return self._parse_wrdata(data_file, output_nodes, is_ac=True)
            return None

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"AC analysis failed: {e}")
            return None

    def _parse_wrdata(self, filepath: str,
                      node_names: List[str],
                      is_ac: bool = False) -> Dict[str, np.ndarray]:
        """
        Parse ngspice wrdata ASCII output.

        wrdata format: columns of numbers, first column is independent variable.
        For AC: columns alternate between real and imaginary parts.
        """
        try:
            data = np.loadtxt(filepath)
        except Exception as e:
            print(f"Error parsing wrdata file: {e}")
            return None

        if data.ndim == 1:
            data = data.reshape(1, -1)

        result = {}

        if is_ac:
            result['frequency'] = data[:, 0]
            for i, name in enumerate(node_names):
                col_re = 1 + 2 * i
                col_im = 2 + 2 * i
                if col_im < data.shape[1]:
                    result[name] = data[:, col_re] + 1j * data[:, col_im]
                elif col_re < data.shape[1]:
                    result[name] = data[:, col_re]
        else:
            result['time'] = data[:, 0]
            for i, name in enumerate(node_names):
                col = i + 1
                if col < data.shape[1]:
                    result[name] = data[:, col]

        return result

    def _parse_raw_ascii(self, filepath: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Parse ngspice ASCII .raw file.

        Format:
            Title: ...
            Plotname: ...
            Flags: real/complex
            No. Variables: N
            No. Points: M
            Variables:
                0  var0  type
                1  var1  type
                ...
            Values:
                index  val0
                    val1
                    val2
                ...
        """
        try:
            with open(filepath, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading raw file: {e}")
            return None

        lines = content.split('\n')
        n_vars = 0
        n_points = 0
        var_names = []
        is_complex = False
        data_start = 0

        for i, line in enumerate(lines):
            if line.startswith('Flags:'):
                is_complex = 'complex' in line.lower()
            elif line.startswith('No. Variables:'):
                n_vars = int(line.split(':')[1].strip())
            elif line.startswith('No. Points:'):
                n_points = int(line.split(':')[1].strip())
            elif line.startswith('Variables:'):
                for j in range(n_vars):
                    parts = lines[i + 1 + j].strip().split()
                    if len(parts) >= 2:
                        var_names.append(parts[1])
            elif line.strip() == 'Values:' or line.strip() == 'Binary:':
                data_start = i + 1
                break

        if n_vars == 0 or n_points == 0:
            print("Could not parse raw file header")
            return None

        # Parse data values
        data = np.zeros((n_points, n_vars))
        point_idx = 0
        var_idx = 0

        for line in lines[data_start:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # Check if this is a new point (starts with index)
            if len(parts) == 2 and parts[0].isdigit():
                point_idx = int(parts[0])
                val = parts[1]
                if ',' in val:
                    val = val.split(',')[0]  # Real part for complex
                data[point_idx, 0] = float(val)
                var_idx = 1
            elif len(parts) == 1:
                val = parts[0]
                if ',' in val:
                    val = val.split(',')[0]
                if var_idx < n_vars and point_idx < n_points:
                    data[point_idx, var_idx] = float(val)
                var_idx += 1

        # Build result dict
        result = {}
        for i, name in enumerate(var_names):
            result[name] = data[:, i]

        # Ensure 'time' key exists if first variable is time
        if var_names and 'time' not in result:
            result['time'] = data[:, 0]

        return result

    def generate_and_run(self, netlist_str: str,
                         output_nodes: List[str] = None,
                         timeout: int = 120) -> Optional[Dict[str, np.ndarray]]:
        """
        Write netlist string to temp file and run simulation.

        Args:
            netlist_str: Complete SPICE netlist as string
            output_nodes: Nodes to extract
            timeout: Timeout in seconds

        Returns:
            Simulation results dict
        """
        tmpdir = tempfile.mkdtemp(prefix='ngspice_')
        netlist_path = os.path.join(tmpdir, 'circuit.cir')

        with open(netlist_path, 'w') as f:
            f.write(netlist_str)

        return self.run_transient(netlist_path, output_nodes, tmpdir, timeout)


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    runner = NgSpiceRunner()
    print(f"ngspice path: {runner.ngspice_path}")
    print(f"Available: {os.path.exists(runner.ngspice_path)}")

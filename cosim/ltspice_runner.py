# cosim/ltspice_runner.py
"""
LTspice Runner - Auto-detect and batch execution.

Finds LTspice XVII/24 on Windows, runs .asc/.cir files in batch mode,
and returns paths to generated .raw / .log files.

Also supports ngspice as fallback via simulation.ngspice_runner.

Usage:
    from cosim.ltspice_runner import LTSpiceRunner

    runner = LTSpiceRunner()          # auto-detect
    runner = LTSpiceRunner('/path/to/LTspice.exe')  # explicit
    ok = runner.run_transient('circuit.cir', timeout_s=120)
"""

import subprocess
import time as _time
from pathlib import Path
from typing import Optional, List

# Delegate discovery to centralized spice_finder module
from .spice_finder import find_ltspice


class LTSpiceRunner:
    """Run LTspice simulations in batch mode."""

    def __init__(self, ltspice_path: str = None):
        """
        Initialize runner.

        Args:
            ltspice_path: Explicit path to LTspice executable.
                          If None, auto-detects.
        """
        if ltspice_path:
            self.exe = Path(ltspice_path)
        else:
            found = find_ltspice()
            self.exe = Path(found) if found else None

        self._last_log = ''
        self._last_raw = ''

    @property
    def available(self) -> bool:
        """Check if LTspice is available."""
        return self.exe is not None and self.exe.exists()

    @property
    def version_string(self) -> str:
        """Return a description of the detected LTspice."""
        if not self.available:
            return 'LTspice not found'
        return f'LTspice at {self.exe}'

    @property
    def last_log(self) -> str:
        """Path to last simulation .log file."""
        return self._last_log

    @property
    def last_raw(self) -> str:
        """Path to last simulation .raw file."""
        return self._last_raw

    def run(self, cir_path, timeout_s: float = 120) -> bool:
        """
        Run a .cir or .asc file in batch mode.

        LTspice batch: LTspice.exe -Run -b <file>

        Args:
            cir_path: Path to circuit file (.cir or .asc)
            timeout_s: Timeout in seconds

        Returns:
            True if simulation completed successfully
        """
        if not self.available:
            raise RuntimeError(
                'LTspice not found. Set path with LTSpiceRunner("/path/to/LTspice.exe")')

        cir_path = Path(cir_path).resolve()
        if not cir_path.exists():
            raise FileNotFoundError(f'Circuit file not found: {cir_path}')

        # Expected output files
        raw_path = cir_path.with_suffix('.raw')
        log_path = cir_path.with_suffix('.log')

        # Remove old outputs to detect fresh run
        for p in [raw_path, log_path]:
            if p.exists():
                p.unlink()

        cmd = [str(self.exe), '-Run', '-b', str(cir_path)]

        try:
            result = subprocess.run(
                cmd,
                timeout=timeout_s,
                capture_output=True,
                cwd=str(cir_path.parent),
            )
        except subprocess.TimeoutExpired:
            self._last_log = ''
            self._last_raw = ''
            return False

        self._last_raw = str(raw_path) if raw_path.exists() else ''
        self._last_log = str(log_path) if log_path.exists() else ''

        return raw_path.exists()

    def run_transient(self, cir_path, timeout_s: float = 120) -> bool:
        """Run transient analysis. Alias for run()."""
        return self.run(cir_path, timeout_s)

    def run_ac(self, cir_path, timeout_s: float = 120) -> bool:
        """Run AC analysis. Alias for run()."""
        return self.run(cir_path, timeout_s)

    def read_log(self) -> str:
        """Read the last simulation log file contents."""
        if self._last_log and Path(self._last_log).exists():
            return Path(self._last_log).read_text(encoding='utf-8', errors='replace')
        return ''

    def get_raw_path(self) -> Optional[str]:
        """Return path to last .raw file, or None."""
        if self._last_raw and Path(self._last_raw).exists():
            return self._last_raw
        return None

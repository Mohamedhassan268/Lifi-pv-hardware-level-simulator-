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

        # LTspice is single-instance: a leftover/wedged batch worker from a prior
        # run makes a new '-b' invocation hand off and exit without producing a
        # .raw (the cause of intermittent "no engine" failures). Clearing strays
        # first guarantees every run starts from the clean state that always
        # succeeds. (Trade-off: also closes an interactive LTspice GUI, which is
        # acceptable for a simulation backend.)
        self._kill_stray_workers()

        # '-b' is LTspice's self-contained batch mode (runs then exits). '-Run'
        # is a GUI flag; combining it can hand the job to a resident GUI
        # instance that exits without producing a .raw, so we use '-b' alone.
        cmd = [str(self.exe), '-b', str(cir_path)]

        # Use Popen so a hung/slow LTspice can be force-killed on timeout. On
        # Windows subprocess.run(timeout=) can leave the child alive; a lingering
        # batch process blocks the next run (LTspice is single-instance).
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(cir_path.parent),
            )
        except (OSError, PermissionError):
            self._last_log = ''
            self._last_raw = ''
            return False

        try:
            proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.communicate(timeout=10)
            except Exception:  # noqa: BLE001
                pass
            self._last_log = ''
            self._last_raw = ''
            return False

        # On Windows the LTspice '-b' launcher spawns a worker and the parent
        # exits almost immediately, so communicate() returning does NOT mean the
        # simulation is done. Wait (up to timeout_s) for the .raw to appear AND
        # stop growing (write complete) before declaring success/failure. If the
        # deadline passes (a wedged solve never stabilises), kill any stray
        # LTspice worker by name so it can't poison the next run.
        deadline = _time.time() + timeout_s
        last_size = -1
        stable_count = 0
        completed = False
        while _time.time() < deadline:
            if raw_path.exists():
                size = raw_path.stat().st_size
                if size > 0 and size == last_size:
                    stable_count += 1
                    if stable_count >= 3:  # size unchanged ~0.3 s -> flushed
                        completed = True
                        break
                else:
                    stable_count = 0
                last_size = size
            _time.sleep(0.1)

        if not completed:
            self._kill_stray_workers()

        self._last_raw = str(raw_path) if raw_path.exists() else ''
        self._last_log = str(log_path) if log_path.exists() else ''

        return completed and raw_path.exists()

    @staticmethod
    def _kill_stray_workers() -> None:
        """Force-kill any LTspice processes (the '-b' worker detaches from the
        launcher we spawned, so proc.kill() can't reach a wedged solve)."""
        import sys
        if sys.platform != 'win32':
            return
        for name in ('XVIIx64.exe', 'LTspice.exe', 'ngspice.exe'):
            try:
                subprocess.run(['taskkill', '/F', '/IM', name],
                               capture_output=True, timeout=10)
            except Exception:  # noqa: BLE001
                pass

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

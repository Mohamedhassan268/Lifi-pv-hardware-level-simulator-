# cosim/ngspice_runner.py
"""
ngspice Runner — Batch execution and result parsing.

Mirrors LTSpiceRunner interface for drop-in substitution.
Supports both bundled (Windows) and system-installed ngspice.

ngspice batch mode: ngspice -b -r output.raw input.cir

Usage:
    from cosim.ngspice_runner import NgSpiceRunner

    runner = NgSpiceRunner()          # auto-detect
    runner = NgSpiceRunner('/usr/bin/ngspice')  # explicit
    ok = runner.run('circuit.cir', timeout_s=120)
    raw_path = runner.get_raw_path()
"""

import subprocess
import logging
import time as _time
from pathlib import Path
from typing import Optional

from .spice_finder import find_ngspice

logger = logging.getLogger(__name__)


class NgSpiceRunner:
    """Run ngspice simulations in batch mode."""

    def __init__(self, ngspice_path: str = None):
        """
        Initialize runner.

        Args:
            ngspice_path: Explicit path to ngspice executable.
                          If None, auto-detects via spice_finder.
        """
        if ngspice_path:
            self.exe = Path(ngspice_path)
        else:
            found = find_ngspice()
            self.exe = Path(found) if found else None

        self._last_raw = ''
        self._last_log = ''
        self._last_stdout = ''
        self._last_stderr = ''

    @property
    def available(self) -> bool:
        """Check if ngspice is available."""
        return self.exe is not None and self.exe.exists()

    @property
    def version_string(self) -> str:
        """Return description of detected ngspice."""
        if not self.available:
            return 'ngspice not found'
        return f'ngspice at {self.exe}'

    @property
    def last_raw(self) -> str:
        """Path to last simulation .raw file."""
        return self._last_raw

    @property
    def last_log(self) -> str:
        """Path to last simulation .log file."""
        return self._last_log

    def run(self, cir_path, timeout_s: float = 120) -> bool:
        """
        Run a .cir file in batch mode.

        ngspice batch: ngspice -b -r output.raw input.cir

        Args:
            cir_path: Path to circuit file (.cir)
            timeout_s: Timeout in seconds

        Returns:
            True if simulation completed and .raw file was produced
        """
        if not self.available:
            raise RuntimeError(
                'ngspice not found. Install ngspice or set path explicitly.')

        cir_path = Path(cir_path).resolve()
        if not cir_path.exists():
            raise FileNotFoundError(f'Circuit file not found: {cir_path}')

        # Output file paths
        raw_path = cir_path.with_suffix('.raw')
        log_path = cir_path.with_suffix('.log')

        # Remove old outputs
        for p in [raw_path, log_path]:
            if p.exists():
                p.unlink()

        # Build command
        cmd = [
            str(self.exe),
            '-b',                      # Batch mode (no interactive)
            '-r', str(raw_path),       # Raw output file
            '-o', str(log_path),       # Log output file
            str(cir_path),             # Input circuit
        ]

        logger.debug("Running ngspice: %s", ' '.join(cmd))

        try:
            result = subprocess.run(
                cmd,
                timeout=timeout_s,
                capture_output=True,
                text=True,
                cwd=str(cir_path.parent),
            )
            self._last_stdout = result.stdout
            self._last_stderr = result.stderr

            if result.returncode != 0:
                logger.warning("ngspice returned code %d: %s",
                               result.returncode, result.stderr[:500])

        except subprocess.TimeoutExpired:
            logger.error("ngspice timed out after %ds", timeout_s)
            self._last_raw = ''
            self._last_log = ''
            return False
        except (OSError, PermissionError) as e:
            logger.error("ngspice execution error: %s", e)
            self._last_raw = ''
            self._last_log = ''
            return False

        self._last_raw = str(raw_path) if raw_path.exists() else ''
        self._last_log = str(log_path) if log_path.exists() else ''

        success = raw_path.exists()
        if success:
            logger.info("ngspice simulation complete: %s", raw_path)
        else:
            logger.warning("ngspice did not produce .raw file")
            # Check if log has error info
            if log_path.exists():
                log_text = log_path.read_text(encoding='utf-8', errors='replace')
                if 'Error' in log_text or 'error' in log_text:
                    logger.warning("ngspice log errors: %s",
                                   log_text[-500:])

        return success

    def run_transient(self, cir_path, timeout_s: float = 120) -> bool:
        """Run transient analysis. Alias for run()."""
        return self.run(cir_path, timeout_s)

    def get_raw_path(self) -> Optional[str]:
        """Return path to last .raw file, or None."""
        if self._last_raw and Path(self._last_raw).exists():
            return self._last_raw
        return None

    def read_log(self) -> str:
        """Read the last simulation log file contents."""
        if self._last_log and Path(self._last_log).exists():
            return Path(self._last_log).read_text(
                encoding='utf-8', errors='replace')
        return ''

    def generate_compatible_netlist(self, ltspice_netlist: str) -> str:
        """
        Convert LTspice-specific netlist syntax to ngspice-compatible.

        Key differences:
            - LTspice .MEAS → ngspice .measure
            - LTspice PWL file="..." → ngspice PWL file '...'
            - LTspice behavioral B source syntax may differ
            - ngspice needs .control/.endc for post-processing

        Args:
            ltspice_netlist: Original LTspice netlist string

        Returns:
            ngspice-compatible netlist string
        """
        lines = ltspice_netlist.splitlines()
        out_lines = []
        has_end = False

        for line in lines:
            stripped = line.strip()

            # Skip LTspice-specific options that ngspice doesn't understand
            if stripped.upper().startswith('.OPTIONS'):
                # Keep basic options, filter LTspice-specific ones
                out_lines.append(line)
                continue

            # Convert PWL file= syntax
            # LTspice: PWL file="path"  →  ngspice: PWL file 'path'
            if 'PWL file=' in line or 'PWL FILE=' in line:
                import re
                line = re.sub(
                    r'PWL\s+file="([^"]+)"',
                    r"PWL file '\1'",
                    line, flags=re.IGNORECASE
                )

            # Convert .MEAS to .measure (ngspice accepts both, but be safe)
            if stripped.upper().startswith('.MEAS '):
                line = line.replace('.MEAS ', '.measure ', 1)
                line = line.replace('.meas ', '.measure ', 1)

            # Track .END
            if stripped.upper() == '.END':
                has_end = True

            out_lines.append(line)

        # Ensure .end exists
        if not has_end:
            out_lines.append('.end')

        return '\n'.join(out_lines)

    def __repr__(self) -> str:
        status = "available" if self.available else "not found"
        return f"NgSpiceRunner({status}: {self.exe})"

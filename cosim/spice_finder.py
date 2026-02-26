# cosim/spice_finder.py
"""
Centralized SPICE Engine Discovery

Single source of truth for finding ngspice and LTspice executables.
Cross-platform: Windows, Linux, macOS.

Usage:
    from cosim.spice_finder import find_ngspice, find_ltspice, spice_available

    ngspice_path = find_ngspice()   # str or None
    ltspice_path = find_ltspice()   # str or None

    if not spice_available():
        print("No SPICE engine found — using Python engine")
"""

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Project root: hardware_faithful_simulator/
_PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# ngspice Discovery
# =============================================================================

def find_ngspice() -> Optional[str]:
    """
    Find ngspice executable, searching in order:
      1. Bundled binary (Windows: ngspice-45.2_64/)
      2. Common install paths (Linux, macOS, Windows)
      3. System PATH

    Returns:
        Absolute path to ngspice, or None if not found.
    """
    # 1. Bundled (Windows) — prefer ngspice_con.exe (console/batch, no GUI window)
    bundled_dir = _PROJECT_ROOT / 'ngspice-45.2_64' / 'Spice64' / 'bin'
    bundled_con = bundled_dir / 'ngspice_con.exe'
    bundled_gui = bundled_dir / 'ngspice.exe'
    if bundled_con.exists():
        logger.debug("ngspice found (bundled console): %s", bundled_con)
        return str(bundled_con)
    if bundled_gui.exists():
        logger.debug("ngspice found (bundled GUI): %s", bundled_gui)
        return str(bundled_gui)

    # 2. Common install paths
    search_paths = []

    if sys.platform == 'win32':
        search_paths.extend([
            r'C:\Program Files\ngspice\bin\ngspice.exe',
            r'C:\Program Files (x86)\ngspice\bin\ngspice.exe',
        ])
    elif sys.platform == 'darwin':
        search_paths.extend([
            '/opt/homebrew/bin/ngspice',      # Homebrew (Apple Silicon)
            '/usr/local/bin/ngspice',          # Homebrew (Intel)
            '/opt/local/bin/ngspice',          # MacPorts
        ])
    else:  # Linux
        search_paths.extend([
            '/usr/bin/ngspice',
            '/usr/local/bin/ngspice',
            '/snap/bin/ngspice',
        ])

    for path_str in search_paths:
        p = Path(path_str)
        if p.exists():
            logger.debug("ngspice found (system): %s", p)
            return str(p)

    # 3. System PATH
    found = shutil.which('ngspice')
    if found:
        logger.debug("ngspice found (PATH): %s", found)
        return found

    logger.info("ngspice not found on this system")
    return None


# =============================================================================
# LTspice Discovery
# =============================================================================

# Common LTspice install locations (Windows only)
_LTSPICE_SEARCH_PATHS = [
    # LTspice XVII (older)
    r'C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe',
    r'C:\Program Files (x86)\LTC\LTspiceXVII\XVIIx86.exe',
    # LTspice 24 (Analog Devices)
    r'C:\Program Files\ADI\LTspice\LTspice.exe',
    # User-local installs
    os.path.expandvars(r'%LOCALAPPDATA%\Programs\ADI\LTspice\LTspice.exe'),
    os.path.expandvars(r'%LOCALAPPDATA%\LTspice\LTspice.exe'),
    os.path.expandvars(r'%APPDATA%\LTspice\LTspice.exe'),
    os.path.expandvars(
        r'%USERPROFILE%\AppData\Local\Programs\ADI\LTspice\LTspice.exe'),
]


def find_ltspice() -> Optional[str]:
    """
    Find LTspice executable (Windows only).

    Searches common install paths and system PATH.

    Returns:
        Absolute path to LTspice, or None if not found.
    """
    if sys.platform != 'win32':
        logger.debug("LTspice discovery skipped (non-Windows platform)")
        return None

    for path_str in _LTSPICE_SEARCH_PATHS:
        p = Path(path_str)
        if p.exists():
            logger.debug("LTspice found: %s", p)
            return str(p)

    # Fallback: search PATH
    for name in ['LTspice.exe', 'XVIIx64.exe']:
        found = shutil.which(name)
        if found:
            logger.debug("LTspice found (PATH): %s", found)
            return found

    logger.info("LTspice not found on this system")
    return None


# =============================================================================
# Convenience Functions
# =============================================================================

def spice_available() -> bool:
    """Check if any SPICE engine is available."""
    return find_ngspice() is not None or find_ltspice() is not None


def spice_summary() -> str:
    """
    Return a human-readable summary of SPICE availability.

    Example:
        "ngspice: /usr/bin/ngspice | LTspice: not found"
    """
    ng = find_ngspice()
    lt = find_ltspice()
    parts = [
        f"ngspice: {ng or 'not found'}",
        f"LTspice: {lt or 'not found'}",
    ]
    return ' | '.join(parts)

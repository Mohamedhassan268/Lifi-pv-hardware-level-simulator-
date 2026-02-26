# systems/base.py
"""
Abstract Base Class for Paper System Definitions

Each paper-specific system module should implement this interface
to ensure consistent structure across papers.

Usage:
    from systems.base import BaseSystem

    class MyPaperSystem(BaseSystem):
        name = 'mypaper2025'
        paper_reference = 'Author et al., Journal 2025'
        ...
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseSystem(ABC):
    """
    Abstract base for paper-specific system definitions.

    Subclasses provide:
        - Paper parameters (locked from publication)
        - SPICE netlist generation (if applicable)
        - Schematic generation (if applicable)
        - Theoretical value computation
    """

    # ── Metadata (override in subclass) ──

    name: str = ''
    """Short key for this system (e.g., 'kadirvelu2021')."""

    paper_reference: str = ''
    """Full paper citation string."""

    simulation_engine: str = 'spice'
    """Default engine: 'spice' or 'python'."""

    # ── Abstract Interface ──

    @abstractmethod
    def get_params(self) -> Dict:
        """
        Return all locked paper parameters as a dict.

        Returns:
            Dict mapping parameter names to values.
        """
        ...

    @abstractmethod
    def get_validation_targets(self) -> Dict:
        """
        Return validation targets from the paper.

        Returns:
            Dict with keys like 'target_ber', 'target_harvested_power_uW', etc.
        """
        ...

    @abstractmethod
    def calculate_theoretical_values(self) -> Dict:
        """
        Compute expected values from paper equations.

        Returns:
            Dict of computed values (optical gain, photocurrent, efficiency, etc.)
        """
        ...

    # ── Optional Methods (override if applicable) ──

    def generate_netlist(self, **kwargs) -> Optional[str]:
        """
        Generate SPICE netlist for this system.

        Returns:
            Netlist string, or None if SPICE is not supported for this paper.
        """
        return None

    def generate_schematics(self, output_dir: str, fmt: str = 'svg') -> List[str]:
        """
        Generate circuit schematics for this system.

        Args:
            output_dir: Directory to save schematic files.
            fmt: Output format ('svg', 'png', 'pdf').

        Returns:
            List of generated file paths.
        """
        return []

    def supported_modulations(self) -> List[str]:
        """Return list of modulation schemes this system supports."""
        return []

"""KiCad schematic & BOM export for hardware-faithful LiFi-PV simulator.

Public API:
    export_kicad(preset_name, out_dir) -> ExportResult
"""

from .kicad_exporter import export_kicad, ExportResult

__all__ = ['export_kicad', 'ExportResult']

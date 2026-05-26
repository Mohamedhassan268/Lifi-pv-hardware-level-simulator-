"""KiCad metadata for simulator components.

Centralizes the mapping from simulator component registry names (e.g. "INA322")
to KiCad symbol/footprint/pin metadata. Kept here rather than on the component
classes so the simulation layer stays focused on physics.

To add a new part:
    1. Add an entry to PART_METADATA keyed by its COMPONENT_REGISTRY name.
    2. Verify the symbol_lib_id matches a symbol in your KiCad install (or
       provide a symbol library file alongside the output).
    3. Verify the footprint matches a KiCad footprint library name.

Pin mappings follow the physical part's datasheet pinout.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PartMetadata:
    symbol_lib_id: str
    footprint: str
    pins: dict[int, str]
    part_number: str
    manufacturer: str
    datasheet_url: str = ""


# Generic passive / power symbols (built into every KiCad install)
GENERIC_R = PartMetadata(
    symbol_lib_id="Device:R",
    footprint="Resistor_SMD:R_0603_1608Metric",
    pins={1: "~", 2: "~"},
    part_number="",
    manufacturer="",
)
GENERIC_C = PartMetadata(
    symbol_lib_id="Device:C",
    footprint="Capacitor_SMD:C_0603_1608Metric",
    pins={1: "~", 2: "~"},
    part_number="",
    manufacturer="",
)
GENERIC_C_POL = PartMetadata(
    symbol_lib_id="Device:C_Polarized",
    footprint="Capacitor_Tantalum_SMD:CP_Elec_6.3x5.4",
    pins={1: "+", 2: "-"},
    part_number="",
    manufacturer="",
)
GENERIC_L = PartMetadata(
    symbol_lib_id="Device:L",
    footprint="Inductor_SMD:L_Taiyo-Yuden_NR-30xx",
    pins={1: "~", 2: "~"},
    part_number="",
    manufacturer="",
)
GENERIC_D_SCHOTTKY = PartMetadata(
    symbol_lib_id="Device:D_Schottky",
    footprint="Diode_SMD:D_SOD-123",
    pins={1: "K", 2: "A"},
    part_number="",
    manufacturer="",
)
GENERIC_VCC = PartMetadata(
    symbol_lib_id="power:VCC",
    footprint="",
    pins={1: "VCC"},
    part_number="",
    manufacturer="",
)
GENERIC_GND = PartMetadata(
    symbol_lib_id="power:GND",
    footprint="",
    pins={1: "GND"},
    part_number="",
    manufacturer="",
)


# Simulator-registered parts.  Pin maps from the manufacturer datasheets.
PART_METADATA: dict[str, PartMetadata] = {
    # ---- Instrumentation / op-amps ----
    "INA322": PartMetadata(
        symbol_lib_id="Amplifier_Instrumentation:INA322EA",
        footprint="Package_SO:MSOP-8_3x3mm_P0.65mm",
        pins={1: "R1", 2: "R2", 3: "VIN+", 4: "V-", 5: "REF",
              6: "OUT", 7: "V+", 8: "VIN-"},
        part_number="INA322EA/250",
        manufacturer="Texas Instruments",
        datasheet_url="https://www.ti.com/lit/ds/symlink/ina322.pdf",
    ),
    "TLV2379": PartMetadata(
        symbol_lib_id="Amplifier_Operational:TLV2374IPW",
        footprint="Package_SO:TSSOP-14_4.4x5mm_P0.65mm",
        pins={1: "OUT1", 2: "IN1-", 3: "IN1+", 4: "V+",
              5: "IN2+", 6: "IN2-", 7: "OUT2",
              8: "OUT3", 9: "IN3-", 10: "IN3+", 11: "V-",
              12: "IN4+", 13: "IN4-", 14: "OUT4"},
        part_number="TLV2379IPWR",
        manufacturer="Texas Instruments",
        datasheet_url="https://www.ti.com/lit/ds/symlink/tlv2379.pdf",
    ),
    "ADA4891": PartMetadata(
        symbol_lib_id="Amplifier_Operational:ADA4891-1ARJZ",
        footprint="Package_TO_SOT_SMD:SOT-23-5",
        pins={1: "OUT", 2: "V-", 3: "IN+", 4: "IN-", 5: "V+"},
        part_number="ADA4891-1ARJZ",
        manufacturer="Analog Devices",
        datasheet_url="https://www.analog.com/media/en/technical-documentation/data-sheets/ADA4891-1_4891-2.pdf",
    ),
    # ---- Comparator ----
    "TLV7011": PartMetadata(
        symbol_lib_id="Comparator:TLV7011",
        footprint="Package_TO_SOT_SMD:SOT-553",
        pins={1: "OUT", 2: "V-", 3: "IN+", 4: "IN-", 5: "V+"},
        part_number="TLV7011DPWR",
        manufacturer="Texas Instruments",
        datasheet_url="https://www.ti.com/lit/ds/symlink/tlv7011.pdf",
    ),
    # ---- MOSFETs ----
    "BSD235N": PartMetadata(
        symbol_lib_id="Device:Q_NMOS_GSD",
        footprint="Package_TO_SOT_SMD:SOT-23",
        pins={1: "G", 2: "S", 3: "D"},
        part_number="BSD235N",
        manufacturer="Infineon",
        datasheet_url="https://www.infineon.com/dgdl/Infineon-BSD235N-DS-v02_02-EN.pdf",
    ),
    "NTS4409": PartMetadata(
        symbol_lib_id="Device:Q_NMOS_GSD",
        footprint="Package_TO_SOT_SMD:SOT-23",
        pins={1: "G", 2: "S", 3: "D"},
        part_number="NTS4409NT1G",
        manufacturer="ON Semiconductor",
        datasheet_url="https://www.onsemi.com/pdf/datasheet/nts4409-d.pdf",
    ),
    # ---- LED ----
    "LXM5_PD01": PartMetadata(
        symbol_lib_id="Device:LED",
        footprint="LED_SMD:LED_LUXEON_Rebel",
        pins={1: "K", 2: "A"},
        part_number="LXM5-PD01",
        manufacturer="Philips Lumileds",
        datasheet_url="https://lumileds.com/wp-content/uploads/files/DS68.pdf",
    ),
    # ---- Solar cell (GaAs) ----
    "KXOB25_04X3F": PartMetadata(
        symbol_lib_id="Device:Solar_Cell",
        footprint="OptoDevice:KXOB25-04X3F",
        pins={1: "+", 2: "-"},
        part_number="KXOB25-04X3F",
        manufacturer="IXYS / Anysolar",
        datasheet_url="https://ixapps.ixys.com/DataSheet/KXOB25-04X3F_Nov16.pdf",
    ),
    # ---- Photodiodes ----
    "BPW34": PartMetadata(
        symbol_lib_id="Device:Photodiode",
        footprint="OptoDevice:Vishay_DIP-2_LPT80A",
        pins={1: "K", 2: "A"},
        part_number="BPW34",
        manufacturer="Vishay",
        datasheet_url="https://www.vishay.com/docs/81521/bpw34.pdf",
    ),
    # ---- Power management ----
    "BQ25570": PartMetadata(
        symbol_lib_id="Battery_Management:BQ25570",
        footprint="Package_DFN_QFN:QFN-20-1EP_3.5x3.5mm_P0.5mm_EP2x2mm",
        pins={1: "VSS", 2: "VIN_DC", 3: "EN", 4: "VOC_SAMP",
              5: "VREF_SAMP", 6: "VRDIV", 7: "VBAT_OV",
              8: "OT_PROG", 9: "OK_PROG", 10: "OK_HYST",
              11: "VBAT_OK", 12: "VOUT_EN", 13: "VOUT_SET",
              14: "VOUT", 15: "LBUCK", 16: "VSTOR",
              17: "VBAT", 18: "VBAT_UV", 19: "LBOOST", 20: "VIN_DC"},
        part_number="BQ25570RGRR",
        manufacturer="Texas Instruments",
        datasheet_url="https://www.ti.com/lit/ds/symlink/bq25570.pdf",
    ),
    # ---- Generic passives (keyed by type, not part number) ----
    "R": GENERIC_R,
    "C": GENERIC_C,
    "C_POL": GENERIC_C_POL,
    "L": GENERIC_L,
    "D_SCHOTTKY": GENERIC_D_SCHOTTKY,
    "VCC_FLAG": GENERIC_VCC,
    "GND_FLAG": GENERIC_GND,
}


def get_part_metadata(component_type: str) -> PartMetadata:
    """Look up KiCad metadata for a simulator component type.

    Raises KeyError with the list of known types if missing.
    """
    if component_type not in PART_METADATA:
        known = ", ".join(sorted(PART_METADATA.keys()))
        raise KeyError(
            f"No KiCad metadata registered for '{component_type}'. "
            f"Add an entry to kicad/part_library.py. Known: {known}"
        )
    return PART_METADATA[component_type]

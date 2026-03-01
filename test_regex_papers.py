"""
Test regex extractor on all available paper PDFs.
Compares extracted parameters against preset JSON ground truth.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ai.pdf_extractor import extract_text
from ai.regex_extractor import regex_extract

# Map: paper_key -> (pdf_path, preset_path)
PDF_DIR = r"C:\Users\HP OMEN\lifi_pv_simulator\paperpdf"
PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets")

PAPERS = {
    "kadirvelu2021": {
        "pdf": os.path.join(PDF_DIR, "A_Circuit_for_Simultaneous_Reception_of_Data_and_Power_Using_a_Solar_Cell (2).pdf"),
        "preset": os.path.join(PRESET_DIR, "kadirvelu2021.json"),
    },
    "gonzalez2024": {
        "pdf": os.path.join(PDF_DIR, "Design_and_Implementation_of_a_Low-Cost_VLC_Photovoltaic_Panel-Based_Receiver_with_off-the-Shelf_Components.pdf"),
        "preset": os.path.join(PRESET_DIR, "gonzalez2024.json"),
    },
    "correa2025": {
        "pdf": os.path.join(PDF_DIR, "correa morales .pdf"),
        "preset": os.path.join(PRESET_DIR, "correa2025.json"),
    },
    "sarwar2017": {
        "pdf": os.path.join(PDF_DIR, "VisiblelightCommuniationUsingSolarpanel (1).pdf"),
        "preset": os.path.join(PRESET_DIR, "sarwar2017.json"),
    },
    "xu2024": {
        "pdf": os.path.join(PDF_DIR, "ewsn24-final43.pdf"),
        "preset": os.path.join(PRESET_DIR, "xu2024.json"),
    },
}

# Fields to compare (skip fields that are 0/empty in preset = not from paper)
COMPARE_FIELDS = [
    'distance_m', 'led_radiated_power_mW', 'led_half_angle_deg',
    'sc_area_cm2', 'sc_responsivity', 'sc_cj_nF', 'sc_rsh_kOhm',
    'sc_iph_uA', 'sc_vmpp_mV', 'sc_impp_uA', 'sc_pmpp_uW',
    'r_sense_ohm', 'ina_gain_dB', 'data_rate_bps', 'modulation',
    'modulation_depth', 'bias_current_A',
    'bpf_f_low_Hz', 'bpf_f_high_Hz',
    'target_ber', 'target_harvested_power_uW',
    'n_cells_series', 'n_cells_parallel', 'lens_transmittance',
    'led_part', 'driver_part', 'pv_part', 'ina_part', 'comparator_part',
    'r_load_ohm', 'dcdc_fsw_kHz',
    'bfsk_f0_hz', 'bfsk_f1_hz',
]


def compare(extracted, preset, paper_key):
    """Compare extracted params against preset ground truth."""
    correct = 0
    wrong = 0
    missed = 0
    extra = 0
    details = []

    for field in COMPARE_FIELDS:
        expected = preset.get(field)
        got = extracted.get(field)

        # Skip fields with 0/empty/N-A in preset (not meaningful to test)
        if expected is None:
            continue
        if isinstance(expected, (int, float)) and expected == 0.0:
            continue
        if isinstance(expected, str) and expected in ('N/A', 'Generic', ''):
            continue

        if got is None:
            missed += 1
            details.append(f"  MISS  {field}: expected={expected}, got=None")
            continue

        # Compare
        if isinstance(expected, str):
            # String comparison (case-insensitive, partial match ok)
            if expected.lower() in str(got).lower() or str(got).lower() in expected.lower():
                correct += 1
                details.append(f"  OK    {field}: {got}")
            else:
                wrong += 1
                details.append(f"  WRONG {field}: expected={expected}, got={got}")
        else:
            # Numeric comparison (within 20% or exact for small values)
            try:
                got_f = float(got)
                exp_f = float(expected)
                if exp_f == 0:
                    if got_f == 0:
                        correct += 1
                        details.append(f"  OK    {field}: {got}")
                    else:
                        wrong += 1
                        details.append(f"  WRONG {field}: expected={expected}, got={got}")
                elif abs(got_f - exp_f) / abs(exp_f) <= 0.20:
                    correct += 1
                    details.append(f"  OK    {field}: {got} (expected {expected})")
                else:
                    wrong += 1
                    details.append(f"  WRONG {field}: expected={expected}, got={got}")
            except (ValueError, TypeError):
                wrong += 1
                details.append(f"  WRONG {field}: expected={expected}, got={got}")

    # Check for extracted fields not in COMPARE_FIELDS
    for field in extracted:
        if field not in COMPARE_FIELDS and field in preset:
            extra += 1

    total = correct + wrong + missed
    pct = (correct / total * 100) if total > 0 else 0

    return {
        'correct': correct,
        'wrong': wrong,
        'missed': missed,
        'total': total,
        'pct': pct,
        'details': details,
    }


def main():
    print("=" * 70)
    print("REGEX EXTRACTOR -- MULTI-PAPER TEST")
    print("=" * 70)

    summary = {}

    for paper_key, paths in PAPERS.items():
        print(f"\n{'-' * 70}")
        print(f"Paper: {paper_key}")
        print(f"{'-' * 70}")

        # Check PDF exists
        if not os.path.isfile(paths["pdf"]):
            print(f"  PDF not found: {paths['pdf']}")
            continue

        # Load preset
        with open(paths["preset"], "r") as f:
            preset = json.load(f)

        # Extract text from PDF
        print(f"  Extracting text from PDF...")
        text = extract_text(paths["pdf"])
        print(f"  Text length: {len(text)} chars")

        # Run regex extraction
        result = regex_extract(text)
        params = result['parameters']
        print(f"  Regex found: {len(params)} parameters")

        # Compare
        comp = compare(params, preset, paper_key)
        summary[paper_key] = comp

        print(f"\n  Results: {comp['correct']}/{comp['total']} correct "
              f"({comp['pct']:.0f}%) | {comp['wrong']} wrong | {comp['missed']} missed")
        print()
        for line in comp['details']:
            print(line)

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Paper':<20} {'Correct':>8} {'Wrong':>8} {'Missed':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"{'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")

    total_correct = 0
    total_all = 0
    for paper_key, comp in summary.items():
        print(f"{paper_key:<20} {comp['correct']:>8} {comp['wrong']:>8} "
              f"{comp['missed']:>8} {comp['total']:>8} {comp['pct']:>9.0f}%")
        total_correct += comp['correct']
        total_all += comp['total']

    if total_all:
        overall = total_correct / total_all * 100
        print(f"{'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")
        print(f"{'OVERALL':<20} {total_correct:>8} {'':>8} {'':>8} {total_all:>8} {overall:>9.0f}%")


if __name__ == '__main__':
    main()

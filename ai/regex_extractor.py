# ai/regex_extractor.py
"""
Regex-Based Parameter Pre-Extractor for LiFi-PV Papers.

Uses pattern matching to extract hardware parameters directly from text
before sending to an LLM. This provides a reliable baseline that small
models can then refine rather than extracting from scratch.

Works best with IEEE-style papers about LiFi, VLC, or OWC with PV receivers.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def regex_extract(text: str) -> dict:
    """
    Extract LiFi-PV parameters from paper text using regex patterns.

    Args:
        text: Full extracted text from PDF.

    Returns:
        Dict with 'parameters' and 'confidence' keys.
    """
    params = {}
    confidence = {}

    # --- Part numbers (specific IC/component identifiers) ---
    _extract_part_numbers(text, params, confidence)

    # --- Distance ---
    _extract_pattern(text, params, confidence, 'distance_m',
        [
            r'(?:distance|separation|gap|spacing)\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*(?:cm|centimeter)',
            r'(?:distance|separation|gap|spacing)\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*(?:m\b|meter)',
            r'(?:r|d)\s*=\s*(\d+\.?\d*)\s*(?:cm|m\b)',
            r'(\d+\.?\d*)\s*(?:cm|m)\s*(?:distance|separation|away|apart)',
        ],
        converters={'cm': lambda v: v / 100.0, 'm': lambda v: v})

    # --- LED radiated power ---
    _extract_pattern(text, params, confidence, 'led_radiated_power_mW',
        [
            r'(?:radiated|optical|transmitted)\s*power\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*mW',
            r'P\s*[_e]*\s*=\s*(\d+\.?\d*)\s*mW',
            r'(\d+\.?\d*)\s*mW\s*(?:radiated|optical|transmitted)',
        ])

    # --- LED half angle ---
    _extract_pattern(text, params, confidence, 'led_half_angle_deg',
        [
            r'(?:half[\s-]*(?:power\s*)?angle|semi[\s-]*angle|α\s*1\s*/\s*2|α½)\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*°?',
            r'(\d+\.?\d*)\s*°?\s*(?:half[\s-]*angle|semi[\s-]*angle)',
            r'(?:Fraen|lens|collimator).*?(\d+\.?\d*)\s*°',
        ])

    # --- Solar cell area ---
    # Try W×H pattern first (e.g., "5 cm × 1.8 cm" = 9 cm²)
    _extract_area(text, params, confidence)

    # --- Responsivity ---
    _extract_pattern(text, params, confidence, 'sc_responsivity',
        [
            r'(?:R\s*[_λ]*|responsivity)\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*A\s*/\s*W',
            r'(\d+\.?\d*)\s*A\s*/\s*W',
        ])

    # --- Junction capacitance ---
    _extract_pattern(text, params, confidence, 'sc_cj_nF',
        [
            r'(?:junction\s*)?capacitance\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*nF',
            r'C\s*[_j]*\s*=\s*(\d+\.?\d*)\s*nF',
            r'(\d+\.?\d*)\s*nF\s*(?:capacitance|junction)',
            r'(?:junction\s*)?capacitance\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*[µu]F',
            r'C\s*[_j]*\s*=\s*(\d+\.?\d*)\s*[µu]F',
        ],
        converters={'nF': lambda v: v, 'µF': lambda v: v * 1000.0,
                     'uF': lambda v: v * 1000.0})

    # --- Shunt resistance ---
    _extract_pattern(text, params, confidence, 'sc_rsh_kOhm',
        [
            r'(?:shunt\s*)?resistance\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*k[Ωo]',
            r'R\s*[_sh]*\s*=\s*(\d+\.?\d*)\s*k[Ωo]',
            r'(?:shunt\s*)?resistance\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*k\s*(?:ohm|Ω)',
            r'R\s*[_sh]*\s*=\s*(\d+\.?\d*)\s*k\s*(?:ohm|Ω)',
        ])

    # --- INA gain (look specifically for INA / instrumentation amplifier gain) ---
    # Prefer INA-specific gain over generic gain
    _extract_ina_gain(text, params, confidence)

    # --- Data rate ---
    _extract_pattern(text, params, confidence, 'data_rate_bps',
        [
            r'(?:data|bit)\s*rate\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*kbps',
            r'(\d+\.?\d*)\s*kbps',
            r'(?:data|bit)\s*rate\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*Mbps',
            r'(\d+\.?\d*)\s*Mbps',
        ],
        converters={'kbps': lambda v: v * 1000, 'Mbps': lambda v: v * 1e6})

    # --- Modulation ---
    if re.search(r'\bOOK\b', text):
        params['modulation'] = 'OOK'
        confidence['modulation'] = 'extracted'
    elif re.search(r'\bOFDM\b', text):
        params['modulation'] = 'OFDM'
        confidence['modulation'] = 'extracted'
    elif re.search(r'\bManchester\b', text, re.IGNORECASE):
        params['modulation'] = 'OOK_Manchester'
        confidence['modulation'] = 'extracted'

    # --- Modulation depth ---
    _extract_pattern(text, params, confidence, 'modulation_depth',
        [
            r'modulation\s*(?:depth|index)\s*(?:of|is|=|:)?\s*(0\.\d+)',
            r'(?:modulation\s*(?:depth|index)|m)\s*=\s*(0\.\d+)',
        ])
    # If captured as percentage (>1), convert
    if 'modulation_depth' in params:
        v = params['modulation_depth']
        if isinstance(v, (int, float)) and v > 1:
            params['modulation_depth'] = v / 100.0

    # --- Photocurrent ---
    _extract_pattern(text, params, confidence, 'sc_iph_uA',
        [
            r'(?:photo)?current\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*[µu]A',
            r'I\s*[_ph]*\s*=\s*(\d+\.?\d*)\s*[µu]A',
            r'(?:photo)?current\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*mA',
            r'I\s*[_ph]*\s*=\s*(\d+\.?\d*)\s*mA',
        ],
        converters={'µA': lambda v: v, 'uA': lambda v: v,
                     'mA': lambda v: v * 1000.0})

    # --- Vmpp (support both mV and V) ---
    _extract_pattern(text, params, confidence, 'sc_vmpp_mV',
        [
            r'V\s*[_]?\s*mpp\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*mV',
            r'V\s*[_]?\s*mpp\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*V\b',
            r'(?:MPP|maximum\s*power\s*point)\s*voltage\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*mV',
            r'(?:MPP|maximum\s*power\s*point)\s*voltage\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*V\b',
        ],
        converters={'mV': lambda v: v, 'V': lambda v: v * 1000.0})

    # --- Impp ---
    _extract_pattern(text, params, confidence, 'sc_impp_uA',
        [
            r'I\s*[_]?\s*mpp\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*[µu]A',
            r'I\s*[_]?\s*mpp\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*mA',
        ],
        converters={'µA': lambda v: v, 'uA': lambda v: v,
                     'mA': lambda v: v * 1000.0})

    # --- Pmpp ---
    _extract_pattern(text, params, confidence, 'sc_pmpp_uW',
        [
            r'P\s*[_]?\s*mpp\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*[µu]W',
            r'(?:MPP|maximum)\s*power\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*[µu]W',
            r'(?:maximum\s*)?power\s*of\s*(\d+\.?\d*)\s*[µu]W',
        ])

    # --- BPF frequencies ---
    _extract_bpf_frequencies(text, params, confidence)

    # --- BER (must be < 1, avoid matching years like 2021) ---
    _extract_ber(text, params, confidence)

    # --- Harvested power ---
    _extract_harvested_power(text, params, confidence)

    # --- Sense resistor ---
    _extract_pattern(text, params, confidence, 'r_sense_ohm',
        [
            r'(?:sense|sensing)\s*resistor?\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*[Ωo]',
            r'R\s*[_]?\s*sense\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*[Ωo]',
            r'(?:sense|sensing)\s*resistor?\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*(?:ohm|Ohm)',
        ])

    # --- Bias current ---
    _extract_pattern(text, params, confidence, 'bias_current_A',
        [
            r'(?:bias|DC)\s*current\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*mA',
            r'I\s*[_]?\s*(?:bias|DC)\s*=\s*(\d+\.?\d*)\s*mA',
        ],
        converters={'mA': lambda v: v / 1000.0})

    # --- DC-DC switching frequency ---
    _extract_pattern(text, params, confidence, 'dcdc_fsw_kHz',
        [
            r'switching\s*frequency\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*kHz',
            r'f\s*[_]?\s*sw\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*kHz',
        ])

    # --- Load resistance ---
    _extract_pattern(text, params, confidence, 'r_load_ohm',
        [
            r'R\s*[_]?\s*load\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*k[Ωo]',
            r'(?:load)\s*(?:resistance|resistor)?\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*k[Ωo]',
            r'R\s*[_]?\s*load\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*k\s*(?:ohm|Ω)',
            r'R\s*[_]?\s*load\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*[Ωo]hm',
        ],
        converters={'kΩ': lambda v: v * 1000, 'ko': lambda v: v * 1000,
                     'kohm': lambda v: v * 1000, 'kΩ': lambda v: v * 1000,
                     'Ω': lambda v: v, 'ohm': lambda v: v})

    # --- N cells series ---
    _extract_pattern(text, params, confidence, 'n_cells_series',
        [
            r'(\d+)\s*(?:cells?|panels?)\s*(?:in\s*)?series',
            r'series\s*(?:connection|connected)\s*(?:of\s*)?(\d+)',
        ])
    if 'n_cells_series' in params:
        params['n_cells_series'] = int(params['n_cells_series'])

    # --- Lens transmittance ---
    _extract_pattern(text, params, confidence, 'lens_transmittance',
        [
            r'(?:lens|optical)\s*(?:transmittance|transmission|efficiency)\s*(?:of|is|=|:)?\s*(0\.\d+)',
            r'T\s*[_]?\s*(?:lens|opt)\s*=\s*(0\.\d+)',
        ])

    logger.info("Regex extraction found %d parameters", len(params))
    return {'parameters': params, 'confidence': confidence}


def _extract_area(text: str, params: dict, confidence: dict):
    """Extract solar cell area, including W×H patterns."""
    # Pattern 1: Direct area in cm²
    m = re.search(
        r'(?:active\s*)?area\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*cm\s*[²2]',
        text, re.IGNORECASE)
    if m:
        params['sc_area_cm2'] = float(m.group(1))
        confidence['sc_area_cm2'] = 'extracted'
        return

    # Pattern 2: W cm × H cm (e.g., "5 cm × 1.8 cm")
    m = re.search(
        r'(\d+\.?\d*)\s*cm\s*[×x×]\s*(\d+\.?\d*)\s*cm',
        text, re.IGNORECASE)
    if m:
        w, h = float(m.group(1)), float(m.group(2))
        params['sc_area_cm2'] = round(w * h, 2)
        confidence['sc_area_cm2'] = 'extracted'
        return

    # Pattern 3: A = X cm (without ²)
    m = re.search(
        r'A\s*(?:_\s*(?:cell|pv|sc))?\s*=\s*(\d+\.?\d*)\s*cm',
        text, re.IGNORECASE)
    if m:
        params['sc_area_cm2'] = float(m.group(1))
        confidence['sc_area_cm2'] = 'extracted'


def _extract_ina_gain(text: str, params: dict, confidence: dict):
    """Extract INA gain, preferring INA-specific context over generic gain."""
    # Pattern 1: INA gain specifically mentioned with dB
    # Look for patterns like "40dB amplifies" or "40 dB" near INA/instrumentation
    m = re.search(
        r'(?:INA|instrumentation)\s*[\w\s,]*?(\d+)\s*dB\s*(?:amplif|gain)',
        text, re.IGNORECASE)
    if m:
        params['ina_gain_dB'] = float(m.group(1))
        confidence['ina_gain_dB'] = 'extracted'
        return

    # Pattern 2: "gain of X dB (Y dB from INA" — take Y
    m = re.search(
        r'(\d+)\s*dB\s*\(\s*(\d+)\s*dB',
        text)
    if m:
        # The value in parentheses is typically the INA gain component
        params['ina_gain_dB'] = float(m.group(2))
        confidence['ina_gain_dB'] = 'extracted'
        return

    # Pattern 3: "X dB" near "amplif" or "INA" within 100 chars
    for m in re.finditer(r'(\d+)\s*dB', text):
        val = float(m.group(1))
        # Skip unreasonable values
        if val < 6 or val > 80:
            continue
        # Check surrounding context for INA/amplifier
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 80)
        context = text[start:end].lower()
        if any(kw in context for kw in ('ina', 'instrumentation', 'amplif',
                                         'r sense', 'r_sense', 'rsense')):
            params['ina_gain_dB'] = val
            confidence['ina_gain_dB'] = 'extracted'
            return


def _extract_bpf_frequencies(text: str, params: dict, confidence: dict):
    """Extract band-pass filter cutoff frequencies."""
    # High-pass (lower cutoff)
    m = re.search(
        r'(?:high[\s-]*pass|lower)\s*(?:cutoff|corner|cut-off)\s*'
        r'(?:frequency)?\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*(?:Hz|kHz)',
        text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        if 'kHz' in m.group(0):
            val *= 1000
        params['bpf_f_low_Hz'] = val
        confidence['bpf_f_low_Hz'] = 'extracted'

    # Low-pass (upper cutoff) — look for explicit "upper cutoff" or "low-pass cutoff"
    # Try multiple patterns and collect candidates
    candidates = []

    for m in re.finditer(
        r'(?:low[\s-]*pass|upper)\s*(?:cutoff|corner|cut[\s-]*off)\s*'
        r'(?:frequency)?\s*(?:of|is|=|:)?\s*(?:the\s*)?'
        r'(?:band[\s-]*pass\s*filter\s*)?'
        r'(?:to\s*)?(?:achiev\w+\s*)?'
        r'(\d+\.?\d*)\s*(Hz|kHz)',
        text, re.IGNORECASE):
        val = float(m.group(1))
        if m.group(2).lower() == 'khz':
            val *= 1000
        candidates.append(val)

    # Also look for "f_H" pattern
    for m in re.finditer(r'f\s*[_]?\s*H\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*(Hz|kHz)',
                          text, re.IGNORECASE):
        val = float(m.group(1))
        if m.group(2).lower() == 'khz':
            val *= 1000
        candidates.append(val)

    # Pick the most reasonable BPF upper frequency
    # Filter: must be > bpf_f_low if we have it, and between 1kHz and 1MHz
    f_low = params.get('bpf_f_low_Hz', 0)
    valid_candidates = [c for c in candidates
                        if c > f_low and 1000 <= c <= 1_000_000]
    if valid_candidates:
        # Take the smallest reasonable value (papers typically use tight BPF)
        params['bpf_f_high_Hz'] = min(valid_candidates)
        confidence['bpf_f_high_Hz'] = 'extracted'


def _extract_ber(text: str, params: dict, confidence: dict):
    """Extract BER, handling scientific notation and avoiding year numbers.

    Takes the LAST BER value found since papers typically report their own
    results after citing others' results.
    """
    candidates = []

    # Pattern 1: Scientific notation (e.g., "1.008×10−3", "BER falls to 1.008×10−3")
    for m in re.finditer(
        r'BER\s*(?:\w+\s+)*?(?:of|is|=|:|to)?\s*(\d+\.?\d*)\s*[×x]\s*10\s*[−\-^]\s*(\d+)',
        text, re.IGNORECASE):
        mantissa = float(m.group(1))
        exponent = int(m.group(2))
        val = mantissa * (10 ** (-exponent))
        if val < 0.5:
            candidates.append((m.start(), val))

    # Pattern 2: Direct small value (e.g., "BER of 0.001")
    for m in re.finditer(
        r'BER\s*(?:of|is|=|:)?\s*(0\.\d+)',
        text, re.IGNORECASE):
        val = float(m.group(1))
        if val < 0.5:
            candidates.append((m.start(), val))

    if candidates:
        # Take the last occurrence (typically the paper's own results)
        candidates.sort(key=lambda x: x[0])
        params['target_ber'] = candidates[-1][1]
        confidence['target_ber'] = 'extracted'


def _extract_harvested_power(text: str, params: dict, confidence: dict):
    """Extract harvested power, preferring result-specific mentions."""
    candidates = []

    # Collect all µW values near "harvest" context
    for m in re.finditer(r'(\d+\.?\d*)\s*[µu]W', text):
        val = float(m.group(1))
        if val < 10 or val > 10000:
            continue
        # Check if "harvest" or "result" is nearby
        start = max(0, m.start() - 100)
        end = min(len(text), m.end() + 50)
        context = text[start:end].lower()
        if 'harvest' in context or 'result' in context or 'show' in context:
            # Penalize "ambient" / "around" mentions (those are available, not harvested)
            if 'around' in context or 'ambient' in context:
                candidates.append((m.start(), val, 'ambient'))
            else:
                candidates.append((m.start(), val, 'result'))

    # Prefer 'result' type, then take the first one found
    results = [c for c in candidates if c[2] == 'result']
    if results:
        params['target_harvested_power_uW'] = results[0][1]
        confidence['target_harvested_power_uW'] = 'extracted'
        return

    # Fallback to any harvested power mention
    if candidates:
        params['target_harvested_power_uW'] = candidates[0][1]
        confidence['target_harvested_power_uW'] = 'extracted'


def _extract_part_numbers(text: str, params: dict, confidence: dict):
    """Extract known component part numbers."""
    # LED parts
    led_patterns = [
        r'\b(LXM\d[\w-]+)\b', r'\b(OSRAM[\w-]+)\b', r'\b(Lumileds[\w-]+)\b',
        r'\b(Cree[\w-]+)\b', r'\b(XPE\d[\w-]+)\b', r'\b(XPG\d[\w-]+)\b',
    ]
    for pat in led_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            params['led_part'] = m.group(1).strip()
            confidence['led_part'] = 'extracted'
            break

    # Driver parts
    driver_patterns = [
        r'\b(ADA\d{3,4}[\w]*)\b', r'\b(TPS\d{4,5}[\w]*)\b',
        r'\b(LM\d{3,4}[\w]*)\b', r'\b(MAX\d{3,4}[\w]*)\b',
    ]
    for pat in driver_patterns:
        m = re.search(pat, text)
        if m:
            # Take only the alphanumeric part number (no trailing words)
            val = re.match(r'[A-Z]{2,4}\d{3,5}[A-Z]?', m.group(1))
            params['driver_part'] = val.group(0) if val else m.group(1)
            confidence['driver_part'] = 'extracted'
            break

    # Solar cell / PV parts — prefer specific part numbers over brand names
    pv_patterns = [
        r'\b(KXOB\d[\w-]+)\b', r'\b(SM\d{3}[\w-]+)\b', r'\b(BPW\d{2}[\w]*)\b',
        r'\b(SFH\d{3,4}[\w]*)\b', r'\b(OPT\d{3,4}[\w]*)\b',
    ]
    for pat in pv_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            params['pv_part'] = m.group(1).strip()
            confidence['pv_part'] = 'extracted'
            break
    # Fallback: brand name only (no trailing context)
    if 'pv_part' not in params:
        m = re.search(r'\b(Alta\s*Devices)\b', text, re.IGNORECASE)
        if m:
            params['pv_part'] = m.group(1).strip()
            confidence['pv_part'] = 'extracted'

    # INA / amplifier parts — limit to just the part number
    ina_patterns = [
        r'\b(INA\d{3})\b', r'\b(AD\d{4})\b', r'\b(OPA\d{3})\b',
        r'\b(LTC\d{4})\b',
    ]
    for pat in ina_patterns:
        m = re.search(pat, text)
        if m:
            params['ina_part'] = m.group(1)
            confidence['ina_part'] = 'extracted'
            break

    # Comparator parts — look for part number near "comparator" context
    _extract_comparator(text, params, confidence)


def _extract_comparator(text: str, params: dict, confidence: dict):
    """Extract comparator part number using context-aware matching."""
    # First look for explicit "comparator TLVxxxx" or "TLVxxxx comparator"
    m = re.search(r'comparator\s+(\w*TLV\d{4})\b', text, re.IGNORECASE)
    if m:
        params['comparator_part'] = m.group(1)
        confidence['comparator_part'] = 'extracted'
        return
    m = re.search(r'\b(TLV\d{4})\b[\s\S]{0,20}comparator', text, re.IGNORECASE)
    if m:
        params['comparator_part'] = m.group(1)
        confidence['comparator_part'] = 'extracted'
        return

    # Look for LMV comparators
    m = re.search(r'\b(LMV\d{3,4})\b', text)
    if m:
        params['comparator_part'] = m.group(1)
        confidence['comparator_part'] = 'extracted'
        return

    # Generic: find any TLV part number that appears near "comparator"
    for m in re.finditer(r'\b(TLV\d{4})\b', text):
        start = max(0, m.start() - 100)
        end = min(len(text), m.end() + 100)
        context = text[start:end].lower()
        if 'comparator' in context:
            params['comparator_part'] = m.group(1)
            confidence['comparator_part'] = 'extracted'
            return


def _extract_pattern(text: str, params: dict, confidence: dict,
                     field: str, patterns: list,
                     converters: Optional[dict] = None):
    """Try multiple regex patterns to extract a numeric field."""
    if field in params:
        return  # Already found

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))

                # Apply unit conversion if needed
                if converters:
                    match_text = m.group(0)
                    for unit_key, conv_func in converters.items():
                        if unit_key.lower() in match_text.lower():
                            val = conv_func(val)
                            break

                params[field] = val
                confidence[field] = 'extracted'
                return
            except (ValueError, IndexError):
                continue

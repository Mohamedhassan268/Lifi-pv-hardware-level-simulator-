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
    _extract_distance(text, params, confidence)

    # --- LED radiated power ---
    _extract_led_power(text, params, confidence)

    # --- LED half angle ---
    _extract_pattern(text, params, confidence, 'led_half_angle_deg',
        [
            r'(?:half[\s-]*(?:power\s*)?angle|semi[\s-]*angle|[\u03b1]\s*1\s*/\s*2|[\u03b1]\xbd)\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*[\xb0]?',
            r'(\d+\.?\d*)\s*[\xb0]?\s*(?:half[\s-]*angle|semi[\s-]*angle)',
            r'(?:Fraen|lens|collimator).*?(\d+\.?\d*)\s*[\xb0]',
            r'(?:beam|viewing)\s*angle\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*[\xb0]',
            r'(?:half[\s-]*(?:power\s*)?angle|semi[\s-]*angle)\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*(?:degree)',
        ])

    # --- Solar cell area ---
    _extract_area(text, params, confidence)

    # --- Responsivity ---
    _extract_pattern(text, params, confidence, 'sc_responsivity',
        [
            r'(?:R\s*[_\u03bb]*|responsivity)\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*A\s*/\s*W',
            r'(\d+\.?\d*)\s*A\s*/\s*W',
        ])

    # --- Junction capacitance ---
    _extract_pattern(text, params, confidence, 'sc_cj_nF',
        [
            r'(?:junction\s*)?capacitance\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*nF',
            r'C\s*[_j]*\s*=\s*(\d+\.?\d*)\s*nF',
            r'(\d+\.?\d*)\s*nF\s*(?:capacitance|junction)',
            r'(?:junction\s*)?capacitance\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*[\xb5u]F',
            r'C\s*[_j]*\s*=\s*(\d+\.?\d*)\s*[\xb5u]F',
            r'(?:junction\s*)?capacitance\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*pF',
            r'C\s*[_j]*\s*=\s*(\d+\.?\d*)\s*pF',
        ],
        converters={'nF': lambda v: v, '\xb5F': lambda v: v * 1000.0,
                     'uF': lambda v: v * 1000.0, 'pF': lambda v: v / 1000.0})

    # --- Shunt resistance ---
    _extract_pattern(text, params, confidence, 'sc_rsh_kOhm',
        [
            r'(?:shunt\s*)resistance\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*k[\u03a9\u2126]',
            r'R\s*[_]?\s*sh(?:unt)?\s*(?:of|is|=|:)?\s*=?\s*(\d+\.?\d*)\s*k[\u03a9\u2126]',
            r'(?:shunt\s*)resistance\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*k\s*(?:ohm|[\u03a9\u2126])',
            r'R\s*[_]?\s*sh(?:unt)?\s*=\s*(\d+\.?\d*)\s*k\s*(?:ohm|[\u03a9\u2126])',
        ])

    # --- INA gain (look specifically for INA / instrumentation amplifier gain) ---
    _extract_ina_gain(text, params, confidence)

    # --- Data rate ---
    _extract_data_rate(text, params, confidence)

    # --- Modulation ---
    _extract_modulation(text, params, confidence)

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
            r'(?:photo)?current\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*[\xb5u]A',
            r'I\s*[_ph]*\s*=\s*(\d+\.?\d*)\s*[\xb5u]A',
            r'(?:photo)?current\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*mA',
            r'I\s*[_ph]*\s*=\s*(\d+\.?\d*)\s*mA',
        ],
        converters={'\xb5A': lambda v: v, 'uA': lambda v: v,
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
            r'I\s*[_]?\s*mpp\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*[\xb5u]A',
            r'I\s*[_]?\s*mpp\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*mA',
        ],
        converters={'\xb5A': lambda v: v, 'uA': lambda v: v,
                     'mA': lambda v: v * 1000.0})

    # --- Pmpp ---
    _extract_pattern(text, params, confidence, 'sc_pmpp_uW',
        [
            r'P\s*[_]?\s*mpp\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*[\xb5u]W',
            r'(?:MPP|maximum)\s*power\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*[\xb5u]W',
            r'(?:maximum\s*)?power\s*of\s*(\d+\.?\d*)\s*[\xb5u]W',
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
            r'(?:sense|sensing)\s*resistor?\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*[\u03a9\u2126]',
            r'R\s*[_]?\s*sense\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*[\u03a9\u2126]',
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
    _extract_load_resistance(text, params, confidence)

    # --- N cells series ---
    _extract_n_cells(text, params, confidence)

    # --- Lens transmittance ---
    _extract_pattern(text, params, confidence, 'lens_transmittance',
        [
            r'(?:lens|optical)\s*(?:transmittance|transmission|efficiency)\s*(?:of|is|=|:)?\s*(0\.\d+)',
            r'T\s*[_]?\s*(?:lens|opt)\s*=\s*(0\.\d+)',
        ])

    # --- BFSK frequencies ---
    _extract_bfsk_frequencies(text, params, confidence)

    logger.info("Regex extraction found %d parameters", len(params))
    return {'parameters': params, 'confidence': confidence}


def _extract_distance(text: str, params: dict, confidence: dict):
    """Extract distance with context-aware matching."""
    candidates = []

    # Pattern: "distance of/is X cm/m"
    for m in re.finditer(
        r'(?:distance|separation|gap|spacing)\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*(cm|m)\b',
        text, re.IGNORECASE):
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == 'cm':
            val /= 100.0
        # Check context: prefer "our"/"proposed"/"between transmitter"
        start = max(0, m.start() - 120)
        end = min(len(text), m.end() + 80)
        ctx = text[start:end].lower()
        priority = 1
        if any(kw in ctx for kw in ('our', 'proposed', 'implement', 'between', 'transmitter', 'receiver', 'apart')):
            priority = 3
        if val > 0 and val < 100:  # reasonable distance
            candidates.append((priority, m.start(), val))

    # Pattern: "X cm apart/between"
    for m in re.finditer(
        r'(\d+\.?\d*)\s*(cm|m)\s*(?:apart|between|away)',
        text, re.IGNORECASE):
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == 'cm':
            val /= 100.0
        if val > 0 and val < 100:
            candidates.append((3, m.start(), val))

    # Pattern: "over a X-m channel" or "via a X-m channel" (Sarwar style)
    for m in re.finditer(
        r'(?:over|via|through)\s+(?:a\s+)?(\d+\.?\d*)[\s-]*(m|cm)\s*(?:air\s*)?(?:channel|link|path)',
        text, re.IGNORECASE):
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == 'cm':
            val /= 100.0
        if val > 0 and val < 100:
            candidates.append((3, m.start(), val))

    # Pattern: "d = X m" or "r = X cm"
    for m in re.finditer(
        r'(?:r|d)\s*=\s*(\d+\.?\d*)\s*(cm|m)\b',
        text, re.IGNORECASE):
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == 'cm':
            val /= 100.0
        if val > 0 and val < 100:
            candidates.append((2, m.start(), val))

    if candidates:
        # Pick highest priority, then earliest occurrence
        candidates.sort(key=lambda x: (-x[0], x[1]))
        params['distance_m'] = candidates[0][2]
        confidence['distance_m'] = 'extracted'


def _extract_led_power(text: str, params: dict, confidence: dict):
    """Extract LED radiated/electrical power."""
    # Direct radiated power in mW
    m = re.search(
        r'(?:radiated|optical|transmitted)\s*power\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*mW',
        text, re.IGNORECASE)
    if m:
        params['led_radiated_power_mW'] = float(m.group(1))
        confidence['led_radiated_power_mW'] = 'extracted'
        return

    m = re.search(r'P\s*[_e]*\s*=\s*(\d+\.?\d*)\s*mW', text)
    if m:
        params['led_radiated_power_mW'] = float(m.group(1))
        confidence['led_radiated_power_mW'] = 'extracted'
        return

    # LED wattage: "X-W LED" or "X W LED" - context-aware
    for m in re.finditer(
        r'(\d+\.?\d*)[\s-]*W\s*(?:blue\s*|white\s*|red\s*|green\s*)?(?:LED|light)',
        text, re.IGNORECASE):
        val = float(m.group(1))
        if 1 <= val <= 100:
            # Check it's not a reference
            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 50)
            ctx = text[start:end].lower()
            if re.search(r'\[\d+\]', ctx) and not any(kw in ctx for kw in ('our', 'proposed', 'we use', 'transmitter')):
                continue
            # Skip if W/m2 (irradiance, not LED power)
            if re.search(r'W/m', m.group(0)):
                continue
            params['led_radiated_power_mW'] = val * 1000.0
            confidence['led_radiated_power_mW'] = 'extracted'
            return

    for m in re.finditer(
        r'(?:LED|light)\s*(?:of|rated?\s*at|with)?\s*(\d+\.?\d*)[\s-]*W\b',
        text, re.IGNORECASE):
        val = float(m.group(1))
        if 1 <= val <= 100:
            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 50)
            ctx = text[start:end].lower()
            if re.search(r'\[\d+\]', ctx) and not any(kw in ctx for kw in ('our', 'proposed', 'we use', 'transmitter')):
                continue
            params['led_radiated_power_mW'] = val * 1000.0
            confidence['led_radiated_power_mW'] = 'extracted'
            return


def _extract_data_rate(text: str, params: dict, confidence: dict):
    """Extract data rate with context-aware matching.

    Uses a priority system to distinguish the paper's own results from
    values cited from references. 'downlink'/'our'/'proposed' context
    gets highest priority; generic references get penalty.
    """
    candidates = []

    def _rate_context(m_obj):
        """Score context around a match: higher = more likely the paper's own rate."""
        start = max(0, m_obj.start() - 150)
        end = min(len(text), m_obj.end() + 80)
        ctx = text[start:end].lower()
        # Strong indicators the value is the paper's own
        if any(kw in ctx for kw in ('downlink', 'our system', 'our proposed',
                                     'we achieve', 'we successfully', 'we propose',
                                     'this work', 'this paper', 'sends')):
            return 4
        # Moderate indicators
        if any(kw in ctx for kw in ('proposed', 'implement', 'experiment')):
            # But penalize if it's clearly a reference (e.g., "[18]", "authors of")
            if re.search(r'\[\d+\]|authors\s*of', ctx):
                return 1
            return 3
        # Weak indicators
        if any(kw in ctx for kw in ('system', 'achieve', 'demonstrat', 'successfully')):
            if re.search(r'\[\d+\]|authors\s*of|in\s*\[', ctx):
                return 1
            return 2
        return 1

    # Pattern 1: "data/bit/baud rate of X unit"
    for m in re.finditer(
        r'(?:baud|data|bit)\s*rate\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*(kBd|kbps|bps|Mbps|Mbit/s|Mb/s|Gbit/s|Gb/s|Bd)',
        text, re.IGNORECASE):
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == 'kbd':
            val *= 1000
        elif unit == 'bd':
            pass  # already in baud = bps for OOK
        elif unit in ('kbps',):
            val *= 1000
        elif unit in ('mbps', 'mbit/s', 'mb/s'):
            val *= 1e6
        elif unit in ('gbit/s', 'gb/s'):
            val *= 1e9
        priority = _rate_context(m)
        candidates.append((priority, m.start(), val))

    # Pattern 2: "at X bps/kbps" (Xu uses "at 400 bps")
    for m in re.finditer(
        r'(?:at|of)\s+(\d+\.?\d*)\s*(bps|kbps|Mbps)\b',
        text, re.IGNORECASE):
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == 'kbps':
            val *= 1000
        elif unit == 'mbps':
            val *= 1e6
        priority = _rate_context(m)
        candidates.append((priority, m.start(), val))

    # Pattern 3: "X Mbit/s" standalone (Sarwar uses "15.03 Mbit/s")
    for m in re.finditer(
        r'(\d+\.?\d*)\s*(?:Mbit/s|Mb/s)\b',
        text):
        val = float(m.group(1)) * 1e6
        priority = _rate_context(m)
        candidates.append((priority, m.start(), val))

    if candidates:
        # Pick highest priority, then first match
        candidates.sort(key=lambda x: (-x[0], x[1]))
        params['data_rate_bps'] = candidates[0][2]
        confidence['data_rate_bps'] = 'extracted'


def _extract_modulation(text: str, params: dict, confidence: dict):
    """Extract modulation scheme with priority ordering.

    Priority: BFSK > PWM-ASK > OFDM > OOK_Manchester > OOK
    We detect the modulation the paper's system actually uses,
    not what it references from other work.
    """
    # Check for BFSK/FSK (these are rare enough to be definitive)
    if re.search(r'\b(?:BFSK|binary\s*frequency\s*shift\s*keying)\b', text, re.IGNORECASE):
        params['modulation'] = 'BFSK'
        confidence['modulation'] = 'extracted'
        return

    # Check for PWM-ASK (also rare/definitive)
    if re.search(r'\bPWM[\s\-]*ASK\b', text, re.IGNORECASE):
        params['modulation'] = 'PWM_ASK'
        confidence['modulation'] = 'extracted'
        return

    # For OOK vs OFDM vs Manchester, use stricter context matching
    # to avoid confusing references to other work with the paper's own modulation
    ofdm_matches = list(re.finditer(r'\bOFDM\b', text))
    ook_matches = list(re.finditer(r'\bOOK\b', text))
    manchester_matches = list(re.finditer(r'\bManchester\b', text, re.IGNORECASE))

    def _score_modulation_context(m_obj):
        """Score whether a modulation mention is the paper's own system."""
        start = max(0, m_obj.start() - 150)
        end = min(len(text), m_obj.end() + 100)
        ctx = text[start:end].lower()
        score = 0
        # Strong positive indicators
        if any(kw in ctx for kw in ('we use', 'we implement', 'we propose', 'this work',
                                     'this paper', 'our system', 'our proposed', 'chosen',
                                     'we adopt', 'was chosen', 'encoding', 'we employ')):
            score += 3
        elif any(kw in ctx for kw in ('our', 'proposed', 'implement')):
            score += 2
        elif any(kw in ctx for kw in ('system', 'using')):
            score += 1
        # Negative indicators (references to other work)
        if re.search(r'\[\d+\]', ctx):
            score -= 1
        if any(kw in ctx for kw in ('authors of', 'in the literature', 'have shown',
                                     'table i', 'table 1', 'were able')):
            score -= 2
        return max(score, 0)

    ofdm_score = sum(_score_modulation_context(m) for m in ofdm_matches)
    ook_score = sum(_score_modulation_context(m) for m in ook_matches)
    manchester_score = sum(_score_modulation_context(m) for m in manchester_matches)

    # Heuristic: if paper describes a thresholding/comparator receiver (OOK system
    # architecture) and mentions OOK anywhere, it's likely an OOK system even if
    # OFDM is discussed as comparison/reference
    has_threshold_receiver = bool(re.search(
        r'threshold(?:ing)?\s*(?:receiver|circuit|data\s*recei|detect)',
        text, re.IGNORECASE))
    has_comparator_component = bool(re.search(
        r'\b(?:TLV\d{4}|LMV\d{3,4}|comparator)\b', text, re.IGNORECASE))
    if (has_threshold_receiver or has_comparator_component) and len(ook_matches) > 0:
        ook_score += 8  # Strong boost: OOK infrastructure (comparator/thresholding) in system

    # Decide based on weighted contextual usage
    scores = [
        (ofdm_score, 'OFDM'),
        (ook_score, 'OOK'),
        (manchester_score, 'OOK_Manchester'),
    ]
    scores.sort(key=lambda x: -x[0])

    if scores[0][0] > 0:
        params['modulation'] = scores[0][1]
        confidence['modulation'] = 'extracted'
    elif len(ofdm_matches) > len(ook_matches) + len(manchester_matches):
        params['modulation'] = 'OFDM'
        confidence['modulation'] = 'extracted'
    elif len(manchester_matches) > 0:
        params['modulation'] = 'OOK_Manchester'
        confidence['modulation'] = 'extracted'
    elif len(ook_matches) > 0:
        params['modulation'] = 'OOK'
        confidence['modulation'] = 'extracted'
    elif len(ofdm_matches) > 0:
        params['modulation'] = 'OFDM'
        confidence['modulation'] = 'extracted'


def _extract_area(text: str, params: dict, confidence: dict):
    """Extract solar cell area, including W x H patterns in cm and mm.

    Context-aware to avoid matching non-panel dimensions (e.g., greenhouse).
    """
    # Pattern 1: Direct area in cm2
    m = re.search(
        r'(?:active\s*)?area\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*cm\s*[^a-zA-Z]',
        text, re.IGNORECASE)
    if m:
        params['sc_area_cm2'] = float(m.group(1))
        confidence['sc_area_cm2'] = 'extracted'
        return

    # Pattern 2: W cm x H cm (e.g., "5 cm x 1.8 cm") - context-aware
    for m in re.finditer(
        r'(\d+\.?\d*)\s*cm\s*[\xd7x\xd7]\s*(\d+\.?\d*)\s*cm',
        text, re.IGNORECASE):
        w, h = float(m.group(1)), float(m.group(2))
        area = round(w * h, 2)
        # Skip if near greenhouse/room/enclosure context
        start = max(0, m.start() - 100)
        end = min(len(text), m.end() + 50)
        ctx = text[start:end].lower()
        if any(kw in ctx for kw in ('greenhouse', 'room', 'enclosure', 'chamber', 'dimension')):
            continue
        # Skip unreasonably large areas (> 500 cm2)
        if area > 500:
            continue
        params['sc_area_cm2'] = area
        confidence['sc_area_cm2'] = 'extracted'
        return

    # Pattern 3: "W by H cm" (e.g., "11 by 6 cm") - context-aware
    for m in re.finditer(
        r'(\d+\.?\d*)\s*(?:by|x)\s*(\d+\.?\d*)\s*cm\b',
        text, re.IGNORECASE):
        w, h = float(m.group(1)), float(m.group(2))
        area = round(w * h, 2)
        start = max(0, m.start() - 100)
        end = min(len(text), m.end() + 50)
        ctx = text[start:end].lower()
        if any(kw in ctx for kw in ('greenhouse', 'room', 'enclosure', 'chamber', 'dimension')):
            continue
        if area > 500:
            continue
        # Prefer solar/PV/panel context
        if any(kw in ctx for kw in ('solar', 'pv', 'panel', 'cell', 'photovoltaic', 'silicon')):
            params['sc_area_cm2'] = area
            confidence['sc_area_cm2'] = 'extracted'
            return
        # Accept if no better match
        if 'sc_area_cm2' not in params:
            params['sc_area_cm2'] = area
            confidence['sc_area_cm2'] = 'extracted'

    if 'sc_area_cm2' in params:
        return

    # Pattern 4: W mm x H mm (e.g., "25 mm x 30 mm")
    for m in re.finditer(
        r'(\d+\.?\d*)\s*mm\s*[\xd7x\xd7]\s*(\d+\.?\d*)\s*mm',
        text, re.IGNORECASE):
        w, h = float(m.group(1)), float(m.group(2))
        area = round(w * h / 100.0, 2)  # mm2 to cm2
        start = max(0, m.start() - 100)
        end = min(len(text), m.end() + 50)
        ctx = text[start:end].lower()
        if any(kw in ctx for kw in ('greenhouse', 'room', 'enclosure', 'chamber', 'dimension')):
            continue
        if area > 500:
            continue
        params['sc_area_cm2'] = area
        confidence['sc_area_cm2'] = 'extracted'
        return

    # Pattern 5: A = X cm
    m = re.search(
        r'A\s*(?:_\s*(?:cell|pv|sc))?\s*=\s*(\d+\.?\d*)\s*cm',
        text, re.IGNORECASE)
    if m:
        params['sc_area_cm2'] = float(m.group(1))
        confidence['sc_area_cm2'] = 'extracted'


def _extract_ina_gain(text: str, params: dict, confidence: dict):
    """Extract INA gain, preferring INA-specific context over generic gain."""
    # Pattern 1: INA gain specifically mentioned with dB
    m = re.search(
        r'(?:INA|instrumentation)\s*[\w\s,]*?(\d+)\s*dB\s*(?:amplif|gain)',
        text, re.IGNORECASE)
    if m:
        params['ina_gain_dB'] = float(m.group(1))
        confidence['ina_gain_dB'] = 'extracted'
        return

    # Pattern 2: "gain of X dB (Y dB from INA" -- take Y
    m = re.search(
        r'(\d+)\s*dB\s*\(\s*(\d+)\s*dB',
        text)
    if m:
        params['ina_gain_dB'] = float(m.group(2))
        confidence['ina_gain_dB'] = 'extracted'
        return

    # Pattern 3: "gain of X dB" or "X dB gain"
    for m in re.finditer(r'(?:gain\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*dB|(\d+\.?\d*)\s*dB\s*gain)', text, re.IGNORECASE):
        val = float(m.group(1) or m.group(2))
        if val < 6 or val > 80:
            continue
        # Check surrounding context for amplifier
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 80)
        context = text[start:end].lower()
        if any(kw in context for kw in ('ina', 'instrumentation', 'amplif', 'tia',
                                         'r sense', 'r_sense', 'rsense', 'op-amp', 'opamp', 'op amp')):
            params['ina_gain_dB'] = val
            confidence['ina_gain_dB'] = 'extracted'
            return

    # Pattern 4: "X dB" near "amplif" or "INA" within 100 chars
    for m in re.finditer(r'(\d+)\s*dB', text):
        val = float(m.group(1))
        if val < 6 or val > 80:
            continue
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

    # Low-pass (upper cutoff)
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

    # Also "3 dB bandwidth of X kHz/Hz" or "bandwidth of X"
    for m in re.finditer(
        r'(?:3\s*dB\s*)?bandwidth\s*(?:of|is|=|:)?\s*(?:around\s*)?(\d+\.?\d*)\s*(Hz|kHz|MHz)',
        text, re.IGNORECASE):
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == 'khz':
            val *= 1000
        elif unit == 'mhz':
            val *= 1e6
        # Only consider as BPF if near PV/panel/filter context
        start = max(0, m.start() - 100)
        end = min(len(text), m.end() + 50)
        ctx = text[start:end].lower()
        if any(kw in ctx for kw in ('pv', 'panel', 'filter', 'solar', 'receiver', 'cell')):
            candidates.append(val)

    # "f_H" pattern
    for m in re.finditer(r'f\s*[_]?\s*H\s*(?:of|is|=|:)?\s*(\d+\.?\d*)\s*(Hz|kHz)',
                          text, re.IGNORECASE):
        val = float(m.group(1))
        if m.group(2).lower() == 'khz':
            val *= 1000
        candidates.append(val)

    f_low = params.get('bpf_f_low_Hz', 0)
    valid_candidates = [c for c in candidates
                        if c > f_low and 100 <= c <= 10_000_000]
    if valid_candidates:
        params['bpf_f_high_Hz'] = min(valid_candidates)
        confidence['bpf_f_high_Hz'] = 'extracted'


def _extract_ber(text: str, params: dict, confidence: dict):
    """Extract BER, handling scientific notation and avoiding year numbers.

    Takes the LAST BER value found since papers typically report their own
    results after citing others' results.
    """
    candidates = []

    # Pattern 1: Scientific notation (e.g., "1.008x10-3", "BER falls to 1.008x10-3")
    for m in re.finditer(
        r'BER\s*(?:\w+\s+)*?(?:of|is|=|:|to)?\s*(\d+\.?\d*)\s*[\xd7x]\s*10\s*[\u2212\-^]\s*(\d+)',
        text, re.IGNORECASE):
        mantissa = float(m.group(1))
        exponent = int(m.group(2))
        val = mantissa * (10 ** (-exponent))
        if val < 0.5:
            candidates.append((m.start(), val))

    # Pattern 1b: Scientific notation without "BER" prefix
    for m in re.finditer(
        r'(?:bit\s*error\s*rat[ei]o?|error\s*rate)\s*(?:\w+\s+)*?(?:of|is|=|:|to)?\s*(\d+\.?\d*)\s*[\xd7x]\s*10\s*[\u2212\-^]\s*(\d+)',
        text, re.IGNORECASE):
        mantissa = float(m.group(1))
        exponent = int(m.group(2))
        val = mantissa * (10 ** (-exponent))
        if val < 0.5:
            candidates.append((m.start(), val))

    # Pattern 1c: Standalone scientific notation near BER context (e.g., "1.6883(cid:XXXX)10-3")
    for m in re.finditer(
        r'(\d+\.?\d*)\s*(?:\(cid:\d+\)|[\xd7x\xb7])\s*10\s*[\u2212\-^]\s*(\d+)',
        text):
        mantissa = float(m.group(1))
        exponent = int(m.group(2))
        val = mantissa * (10 ** (-exponent))
        if val < 0.5:
            # Check if BER is nearby
            start = max(0, m.start() - 120)
            end = min(len(text), m.end() + 40)
            ctx = text[start:end].lower()
            if 'ber' in ctx or 'bit error' in ctx or 'error rate' in ctx:
                candidates.append((m.start(), val))

    # Pattern 2: "BER 10-X" or "BER 10^-X" (compact notation from PDF extraction)
    for m in re.finditer(
        r'BER\s*(?:of|is|=|:)?\s*10\s*[\u2212\-^]\s*(\d+)',
        text, re.IGNORECASE):
        exponent = int(m.group(1))
        val = 10 ** (-exponent)
        if val < 0.5:
            candidates.append((m.start(), val))

    # Pattern 3: Direct small value (e.g., "BER of 0.001")
    for m in re.finditer(
        r'BER\s*(?:of|is|=|:)?\s*(0\.\d+)',
        text, re.IGNORECASE):
        val = float(m.group(1))
        if val < 0.5:
            candidates.append((m.start(), val))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        params['target_ber'] = candidates[-1][1]
        confidence['target_ber'] = 'extracted'


def _extract_harvested_power(text: str, params: dict, confidence: dict):
    """Extract harvested power, preferring result-specific mentions."""
    candidates = []

    for m in re.finditer(r'(\d+\.?\d*)\s*[\xb5u]W', text):
        val = float(m.group(1))
        if val < 10 or val > 10000:
            continue
        start = max(0, m.start() - 100)
        end = min(len(text), m.end() + 50)
        context = text[start:end].lower()
        if 'harvest' in context or 'result' in context or 'show' in context:
            if 'around' in context or 'ambient' in context:
                candidates.append((m.start(), val, 'ambient'))
            else:
                candidates.append((m.start(), val, 'result'))

    results = [c for c in candidates if c[2] == 'result']
    if results:
        params['target_harvested_power_uW'] = results[0][1]
        confidence['target_harvested_power_uW'] = 'extracted'
        return

    if candidates:
        params['target_harvested_power_uW'] = candidates[0][1]
        confidence['target_harvested_power_uW'] = 'extracted'


def _extract_load_resistance(text: str, params: dict, confidence: dict):
    """Extract load resistance with context-aware matching."""
    candidates = []

    # "load of X Ohm/kOhm" or "R_load = X"
    for m in re.finditer(
        r'(?:load\s*(?:of|is|=|:)?\s*|R\s*[_]?\s*load\s*(?:of|is|=|:)?\s*)(\d+\.?\d*)\s*(k?[\u03a9\u2126]|k?\s*[Oo]hm)',
        text, re.IGNORECASE):
        val = float(m.group(1))
        unit = m.group(2).lower()
        if 'k' in unit:
            val *= 1000
        # Check context
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 80)
        ctx = text[start:end].lower()
        priority = 1
        if any(kw in ctx for kw in ('pv', 'panel', 'solar', 'implement', 'connected', 'decided', 'chosen', 'our')):
            priority = 3
        candidates.append((priority, m.start(), val))

    # "a load of X Ohm" or "load resistance of X"
    for m in re.finditer(
        r'(?:a\s+)?load\s*(?:resistance)?\s*(?:of|is|=|:)\s*(\d+\.?\d*)\s*(k?[\u03a9\u2126]|k?\s*[Oo]hm)',
        text, re.IGNORECASE):
        val = float(m.group(1))
        unit = m.group(2).lower()
        if 'k' in unit:
            val *= 1000
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 80)
        ctx = text[start:end].lower()
        priority = 2
        if any(kw in ctx for kw in ('pv', 'panel', 'solar', 'implement', 'connected', 'decided', 'chosen', 'our')):
            priority = 3
        candidates.append((priority, m.start(), val))

    # "BW-enhancer-220 Ohm load" style
    for m in re.finditer(
        r'(\d+\.?\d*)\s*(k?[\u03a9\u2126])\s*(?:load|resistive)',
        text, re.IGNORECASE):
        val = float(m.group(1))
        unit = m.group(2).lower()
        if 'k' in unit:
            val *= 1000
        candidates.append((2, m.start(), val))

    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        params['r_load_ohm'] = candidates[0][2]
        confidence['r_load_ohm'] = 'extracted'


def _extract_n_cells(text: str, params: dict, confidence: dict):
    """Extract number of cells in series/parallel."""
    # Series: "Xs-Yp" pattern (e.g., "2s-8p")
    m = re.search(r'(\d+)s[\s-]*(\d+)p', text)
    if m:
        params['n_cells_series'] = int(m.group(1))
        confidence['n_cells_series'] = 'extracted'
        params['n_cells_parallel'] = int(m.group(2))
        confidence['n_cells_parallel'] = 'extracted'
        return

    # "X cells in series" or "X panels in series"
    m = re.search(r'(\d+)\s*(?:cells?|panels?)\s*(?:in\s*)?series', text, re.IGNORECASE)
    if m:
        params['n_cells_series'] = int(m.group(1))
        confidence['n_cells_series'] = 'extracted'

    # "series connection of X"
    m = re.search(r'series\s*(?:connection|connected)\s*(?:of\s*)?(\d+)', text, re.IGNORECASE)
    if m and 'n_cells_series' not in params:
        params['n_cells_series'] = int(m.group(1))
        confidence['n_cells_series'] = 'extracted'

    # Parallel
    m = re.search(r'(\d+)\s*(?:cells?|panels?)\s*(?:in\s*)?parallel', text, re.IGNORECASE)
    if m and 'n_cells_parallel' not in params:
        params['n_cells_parallel'] = int(m.group(1))
        confidence['n_cells_parallel'] = 'extracted'

    m = re.search(r'parallel\s*(?:connection|connected|sets)\s*(?:of\s*)?(\d+)', text, re.IGNORECASE)
    if m and 'n_cells_parallel' not in params:
        params['n_cells_parallel'] = int(m.group(1))
        confidence['n_cells_parallel'] = 'extracted'


def _extract_bfsk_frequencies(text: str, params: dict, confidence: dict):
    """Extract BFSK tone frequencies."""
    # Normalize text for BFSK search: rejoin hyphenated words across lines
    norm = re.sub(r'-\s*\n\s*', '', text)
    # Also collapse whitespace for better matching
    norm_compact = re.sub(r'\s+', ' ', norm)

    # "1600Hz signal to represent a '0'" (may have no spaces in PDF text)
    m = re.search(
        r'(\d+)\s*Hz\s*(?:sig(?:nal)?|tone)?\s*(?:[\w\s]*?)(?:to\s*)?represent\w*\s*(?:a\s*)?[\'\"\u2018\u2019\u00ab\u00bb]?0[\'\"\u2018\u2019\u00ab\u00bb]?',
        norm_compact, re.IGNORECASE)
    if m:
        params['bfsk_f0_hz'] = float(m.group(1))
        confidence['bfsk_f0_hz'] = 'extracted'

    m = re.search(
        r'(\d+)\s*Hz\s*(?:sig(?:nal)?|tone)?\s*(?:[\w\s]*?)(?:to\s*)?represent\w*\s*(?:a\s*)?[\'\"\u2018\u2019\u00ab\u00bb]?1[\'\"\u2018\u2019\u00ab\u00bb]?',
        norm_compact, re.IGNORECASE)
    if m:
        params['bfsk_f1_hz'] = float(m.group(1))
        confidence['bfsk_f1_hz'] = 'extracted'

    # Alternative: "f0 = X Hz, f1 = Y Hz"
    if 'bfsk_f0_hz' not in params:
        m = re.search(r'f\s*[_]?\s*0\s*=\s*(\d+)\s*Hz', text, re.IGNORECASE)
        if m:
            params['bfsk_f0_hz'] = float(m.group(1))
            confidence['bfsk_f0_hz'] = 'extracted'

    if 'bfsk_f1_hz' not in params:
        m = re.search(r'f\s*[_]?\s*1\s*=\s*(\d+)\s*Hz', text, re.IGNORECASE)
        if m:
            params['bfsk_f1_hz'] = float(m.group(1))
            confidence['bfsk_f1_hz'] = 'extracted'


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

    # Driver parts -- context-aware to avoid false matches
    driver_patterns = [
        r'\b(ADA\d{3,4}[\w]*)\b', r'\b(TPS\d{4,5}[\w]*)\b',
        r'\b(LM\d{3,4}[\w]*)\b', r'\b(MAX\d{3,4}[\w]*)\b',
    ]
    for pat in driver_patterns:
        for m in re.finditer(pat, text):
            part = m.group(1)
            # Clean: take only the alphanumeric part number
            val = re.match(r'[A-Z]{2,4}\d{3,5}[A-Z]?', part)
            part_clean = val.group(0) if val else part
            # Check context to distinguish driver from other uses (e.g., TPS as power regulator)
            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 80)
            context = text[start:end].lower()
            if any(kw in context for kw in ('driver', 'led', 'transmit', 'amplif', 'buffer')):
                params['driver_part'] = part_clean
                confidence['driver_part'] = 'extracted'
                break
            elif 'driver_part' not in params:
                # Accept first match as fallback
                params['driver_part'] = part_clean
                confidence['driver_part'] = 'extracted'
        if 'driver_part' in params:
            break

    # Solar cell / PV parts -- context-aware
    pv_patterns = [
        r'\b(KXOB\d[\w-]+)\b', r'\b(SM\d{3}[\w-]+)\b', r'\b(BPW\d{2}[\w]*)\b',
        r'\b(SFH\d{3,4}[\w]*)\b', r'\b(OPT\d{3,4}[\w]*)\b',
    ]
    for pat in pv_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            part = m.group(1).strip()
            # Check context: prefer "our"/"use"/"receiver" over table references
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 80)
            ctx = text[start:end].lower()
            if any(kw in ctx for kw in ('our', 'use', 'using', 'receiver', 'detector', 'proposed')):
                params['pv_part'] = part
                confidence['pv_part'] = 'extracted'
                break
            elif 'pv_part' not in params:
                params['pv_part'] = part
                confidence['pv_part'] = 'extracted'
        if 'pv_part' in params:
            break
    # Fallback: brand name only
    if 'pv_part' not in params:
        m = re.search(r'\b(Alta\s*Devices)\b', text, re.IGNORECASE)
        if m:
            params['pv_part'] = m.group(1).strip()
            confidence['pv_part'] = 'extracted'

    # INA / amplifier parts
    ina_patterns = [
        r'\b(INA\d{3})\b', r'\b(AD\d{4})\b', r'\b(OPA\d{3})\b',
        r'\b(LTC\d{4})\b', r'\b(TL\d{3})\b', r'\b(TL0\d{2})\b',
    ]
    for pat in ina_patterns:
        m = re.search(pat, text)
        if m:
            params['ina_part'] = m.group(1)
            confidence['ina_part'] = 'extracted'
            break

    # Comparator parts
    _extract_comparator(text, params, confidence)


def _extract_comparator(text: str, params: dict, confidence: dict):
    """Extract comparator part number using context-aware matching."""
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

    m = re.search(r'\b(LMV\d{3,4})\b', text)
    if m:
        params['comparator_part'] = m.group(1)
        confidence['comparator_part'] = 'extracted'
        return

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

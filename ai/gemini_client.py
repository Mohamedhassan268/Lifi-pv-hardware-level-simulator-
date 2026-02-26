# ai/gemini_client.py
"""
Google Gemini API Client for LiFi-PV Parameter Extraction.

Sends paper text + tables to Gemini with a structured prompt,
and parses the JSON response into a parameter dict.
"""

import json
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# Complete prompt template listing all SystemConfig fields
EXTRACTION_PROMPT = '''You are an expert in optical wireless communication (LiFi) and photovoltaic energy harvesting circuits.

Given a research paper about a LiFi-PV (light fidelity with photovoltaic receiver) system, extract all hardware parameters into a structured JSON format.

IMPORTANT RULES:
1. Only extract values that are EXPLICITLY stated in the paper text or tables.
2. For values you can reasonably estimate from the context (e.g., standard component values), mark confidence as "estimated".
3. For values not found at all, use null and mark confidence as "missing".
4. NEVER invent or hallucinate values. If unsure, use null.
5. Pay careful attention to units — convert everything to the units specified below.

REQUIRED OUTPUT FORMAT (JSON):
{{
    "parameters": {{
        "preset_name": "<short_name_like_author2024>",
        "paper_reference": "<Author et al., Journal, Year - Title>",

        "led_part": "<LED part number or null>",
        "driver_part": "<driver IC part number or null>",
        "bias_current_A": <float, LED bias current in Amps>,
        "modulation_depth": <float, 0-1>,
        "led_radiated_power_mW": <float, radiated optical power in mW>,
        "led_half_angle_deg": <float, LED half-power angle in degrees>,
        "led_driver_re": <float, driver series resistance in Ohms>,
        "led_gled": <float, LED electro-optical conversion efficiency 0-1>,
        "lens_transmittance": <float, lens/optical filter transmittance 0-1>,

        "distance_m": <float, TX-RX distance in meters>,
        "tx_angle_deg": <float, TX emission angle in degrees>,
        "rx_tilt_deg": <float, RX tilt angle in degrees>,

        "pv_part": "<solar cell part number or null>",
        "sc_area_cm2": <float, active area in cm2>,
        "sc_responsivity": <float, responsivity in A/W>,
        "sc_cj_nF": <float, junction capacitance in nF>,
        "sc_rsh_kOhm": <float, shunt resistance in kOhm>,
        "sc_iph_uA": <float, photocurrent in uA>,
        "sc_vmpp_mV": <float, max power point voltage in mV>,
        "sc_impp_uA": <float, max power point current in uA>,
        "sc_pmpp_uW": <float, max power point power in uW>,

        "r_sense_ohm": <float, sense resistor in Ohms>,
        "ina_part": "<instrumentation amplifier part or null>",
        "ina_gain_dB": <float, INA gain in dB>,
        "ina_gbw_kHz": <float, gain-bandwidth product in kHz>,

        "comparator_part": "<comparator part number or null>",

        "bpf_stages": <int, number of bandpass filter stages>,
        "bpf_f_low_Hz": <float, lower cutoff in Hz>,
        "bpf_f_high_Hz": <float, upper cutoff in Hz>,

        "dcdc_fsw_kHz": <float, DC-DC switching frequency in kHz>,
        "dcdc_l_uH": <float, inductor in uH>,
        "dcdc_cp_uF": <float, input capacitor in uF>,
        "dcdc_cl_uF": <float, output capacitor in uF>,
        "r_load_ohm": <float, load resistance in Ohms>,

        "data_rate_bps": <float, data rate in bits per second>,
        "modulation": "<OOK|OOK_Manchester|OFDM|BFSK|PWM_ASK>",
        "n_bits": 100,

        "target_harvested_power_uW": <float or null>,
        "target_ber": <float or null>,
        "target_noise_rms_mV": <float or null>
    }},
    "confidence": {{
        "<field_name>": "extracted"|"estimated"|"missing"
    }},
    "notes": "<any relevant observations about the paper or extraction>"
}}

EXAMPLE (Kadirvelu 2021):
{{
    "parameters": {{
        "preset_name": "kadirvelu2021",
        "paper_reference": "Kadirvelu et al., IEEE TGCN 2021 - A Circuit for Simultaneous Reception of Data and Power Using a Solar Cell",
        "led_part": "LXM5-PD01",
        "driver_part": "ADA4891",
        "bias_current_A": 0.350,
        "modulation_depth": 0.33,
        "led_radiated_power_mW": 9.3,
        "led_half_angle_deg": 9.0,
        "distance_m": 0.325,
        "pv_part": "KXOB25-04X3F",
        "sc_area_cm2": 9.0,
        "sc_responsivity": 0.457,
        "sc_cj_nF": 798.0,
        "sc_rsh_kOhm": 138.8,
        "ina_part": "INA322",
        "ina_gain_dB": 40.0,
        "comparator_part": "TLV7011",
        "bpf_stages": 2,
        "bpf_f_low_Hz": 700.0,
        "bpf_f_high_Hz": 10000.0,
        "data_rate_bps": 5000.0,
        "modulation": "OOK",
        "target_ber": 0.001008
    }},
    "confidence": {{
        "led_part": "extracted",
        "distance_m": "extracted",
        "sc_cj_nF": "extracted",
        "modulation_depth": "estimated"
    }}
}}

Now extract parameters from this paper:

--- PAPER TEXT ---
{paper_text}

--- TABLES ---
{tables_text}

Return ONLY the JSON object, no markdown fences or explanations.'''


class GeminiClient:
    """Wrapper around Google Gemini API for parameter extraction."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize with API key.

        Args:
            api_key: Google AI API key. Falls back to GEMINI_API_KEY env var.

        Raises:
            ImportError: If google-generativeai is not installed.
            ValueError: If no API key is provided.
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai not installed.\n"
                "Run: pip install google-generativeai")

        self.api_key = api_key or os.environ.get('GEMINI_API_KEY', '')
        if not self.api_key:
            raise ValueError(
                "No Gemini API key provided.\n"
                "Set GEMINI_API_KEY environment variable or pass api_key parameter.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("GeminiClient initialized (model=gemini-2.0-flash)")

    def extract_parameters(self, paper_text: str, tables_text: str = '') -> dict:
        """
        Send paper text to Gemini and extract parameters.

        Args:
            paper_text: Full text of the paper.
            tables_text: Formatted table text from pdf_extractor.

        Returns:
            Dict with keys: 'parameters', 'confidence', 'notes'

        Raises:
            RuntimeError: If Gemini API call fails.
        """
        prompt = EXTRACTION_PROMPT.format(
            paper_text=paper_text[:80_000],
            tables_text=tables_text[:20_000],
        )

        logger.info("Sending extraction request to Gemini (%d chars)", len(prompt))

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                ),
            )
            raw_text = response.text.strip()
        except Exception as e:
            logger.error("Gemini API call failed: %s", e)
            raise RuntimeError(f"Gemini API error: {e}") from e

        # Parse JSON from response
        return self._parse_response(raw_text)

    def _parse_response(self, raw_text: str) -> dict:
        """Parse Gemini's JSON response, handling common formatting issues."""
        # Strip markdown code fences if present
        text = raw_text
        if text.startswith('```'):
            lines = text.split('\n')
            # Remove first line (```json) and last line (```)
            lines = [l for l in lines if not l.strip().startswith('```')]
            text = '\n'.join(lines)

        try:
            result = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse Gemini response as JSON: %s", e)
            logger.debug("Raw response:\n%s", raw_text[:2000])
            raise RuntimeError(
                f"Failed to parse Gemini response as JSON: {e}\n"
                f"First 500 chars: {raw_text[:500]}") from e

        # Ensure expected structure
        if 'parameters' not in result:
            result = {'parameters': result, 'confidence': {}, 'notes': ''}

        if 'confidence' not in result:
            result['confidence'] = {}
        if 'notes' not in result:
            result['notes'] = ''

        logger.info("Parsed %d parameters from Gemini response",
                     len(result['parameters']))
        return result

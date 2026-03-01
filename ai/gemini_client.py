# ai/gemini_client.py
"""
Google Gemini API Client for LiFi-PV Parameter Extraction.

Sends paper text + tables to Gemini with a structured prompt,
and parses the JSON response into a parameter dict.

Uses the new `google-genai` SDK (successor to `google-generativeai`).
"""

import json
import os
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Try new SDK first, fall back to deprecated one
try:
    from google import genai
    GEMINI_AVAILABLE = True
    _USE_NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as genai_legacy
        GEMINI_AVAILABLE = True
        _USE_NEW_SDK = False
    except ImportError:
        GEMINI_AVAILABLE = False
        _USE_NEW_SDK = False


# Complete prompt template listing all SystemConfig fields
EXTRACTION_PROMPT = '''You are an expert in optical wireless communication (LiFi) and photovoltaic energy harvesting circuits.

Given a research paper about a LiFi-PV (light fidelity with photovoltaic receiver) system, extract all hardware parameters into a structured JSON format.

IMPORTANT RULES:
1. Only extract values EXPLICITLY stated in the paper text or tables.
2. For values you can reasonably estimate from context, mark confidence as "estimated".
3. For values not found at all, use null and mark confidence as "missing".
4. NEVER invent or hallucinate values. If unsure, use null.
5. Pay careful attention to units — convert everything to the units specified below.

EXTRACTION HINTS (read carefully):
- led_half_angle_deg: This is the LED half-power angle (α½). If a lens is used (e.g. Fraen, collimator), use the LENS half-angle (often 5-15°), NOT the bare LED angle (often 60-120°). Look for "half-angle", "α1/2", "beam angle", "semi-angle", "Lambertian order m" in the text.
- n_cells_series: How many solar cells are wired in series. Look for "N-cell module", "series connection", "multi-cell", or infer from voltage (e.g., GaAs single cell ≈ 0.7-1V, Si single cell ≈ 0.5V). Default 1 if a single cell is used.
- r_load_ohm: The DC-DC converter OUTPUT load resistance (in Ohms). This is NOT the solar cell optimal load. Look for "R_L", "load resistor", "output load". Can be very high (e.g. 180 kΩ) for energy harvesting.
- modulation_depth: Also called "modulation index", often denoted "m" or "β". Range 0-1. If the paper sweeps multiple values, use the one for main BER results.
- sc_iph_uA, sc_vmpp_mV, sc_impp_uA, sc_pmpp_uW: These are the solar cell operating point values. Look in IV curves, tables, or text mentioning "photocurrent", "MPP", "maximum power point", "Pmpp", "Vmpp", "Impp".
- bpf_f_low_Hz, bpf_f_high_Hz: Bandpass filter cutoff frequencies. Can be computed from R and C values: f = 1/(2π×R×C). Look for "high-pass", "low-pass", "cutoff frequency", "passband".
- data_rate_bps: The primary data rate in bits per second. Convert from kbps (×1000) or Mbps (×1e6). If multiple rates are tested, use the PRIMARY or DEFAULT rate.
- ina_gain_dB: Gain in DECIBELS. If the paper gives linear gain (e.g., 100×), convert: dB = 20×log10(linear). INA322 has gain ≈ 100× = 40 dB.
- bias_current_A: LED DC bias current in AMPS. Convert from mA (÷1000).
- led_gled: LED electro-optical conversion efficiency (W/A). Sometimes called "slope efficiency" or "responsivity" of the LED.
- dcdc_fsw_kHz: DC-DC boost converter switching frequency in kHz.

REQUIRED OUTPUT FORMAT (JSON):
{{
    "parameters": {{
        "preset_name": "<short_name_like_author2024>",
        "paper_reference": "<Author et al., Journal, Year - Title>",

        "led_part": "<LED part number or null>",
        "driver_part": "<LED driver IC part number or null>",
        "bias_current_A": <float, LED bias current in Amps>,
        "modulation_depth": <float, 0-1>,
        "led_radiated_power_mW": <float, radiated optical power in mW>,
        "led_half_angle_deg": <float, LED/lens half-power angle in degrees — use lens angle if lens present>,
        "led_driver_re": <float, driver series resistance in Ohms>,
        "led_gled": <float, LED electro-optical conversion efficiency W/A>,
        "lens_transmittance": <float, lens/optical filter transmittance 0-1>,

        "distance_m": <float, TX-RX distance in meters>,
        "tx_angle_deg": <float, TX emission angle in degrees>,
        "rx_tilt_deg": <float, RX tilt angle in degrees>,

        "pv_part": "<solar cell part number or null>",
        "n_cells_series": <int, number of cells in series, default 1>,
        "n_cells_parallel": <int, number of cells in parallel, default 1>,
        "sc_area_cm2": <float, active area in cm2>,
        "sc_responsivity": <float, responsivity in A/W>,
        "sc_cj_nF": <float, junction capacitance in nF>,
        "sc_rsh_kOhm": <float, shunt resistance in kOhm>,
        "sc_iph_uA": <float, photocurrent in uA>,
        "sc_vmpp_mV": <float, max power point voltage in mV>,
        "sc_impp_uA": <float, max power point current in uA>,
        "sc_pmpp_uW": <float, max power point power in uW>,

        "r_sense_ohm": <float, sense/transimpedance resistor in Ohms>,
        "ina_part": "<instrumentation amplifier part or null>",
        "ina_gain_dB": <float, INA gain in dB (convert from linear if needed: dB=20*log10(linear))>,
        "ina_gbw_kHz": <float, gain-bandwidth product in kHz>,

        "comparator_part": "<comparator part number or null>",

        "bpf_stages": <int, number of bandpass filter stages>,
        "bpf_f_low_Hz": <float, lower cutoff frequency in Hz>,
        "bpf_f_high_Hz": <float, upper cutoff frequency in Hz>,

        "dcdc_fsw_kHz": <float, DC-DC switching frequency in kHz>,
        "dcdc_l_uH": <float, inductor value in uH>,
        "dcdc_cp_uF": <float, input capacitor in uF>,
        "dcdc_cl_uF": <float, output capacitor in uF>,
        "r_load_ohm": <float, DC-DC OUTPUT load resistance in Ohms>,

        "data_rate_bps": <float, data rate in bits per second>,
        "modulation": "<OOK|OOK_Manchester|OFDM|BFSK|PWM_ASK>",
        "n_bits": 100,

        "target_harvested_power_uW": <float or null, harvested electrical power in uW>,
        "target_ber": <float or null, bit error rate>,
        "target_noise_rms_mV": <float or null, RMS noise voltage in mV>
    }},
    "confidence": {{
        "<field_name>": "extracted"|"estimated"|"missing"
    }},
    "notes": "<any relevant observations about the paper or extraction>"
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
            ImportError: If neither google-genai nor google-generativeai is installed.
            ValueError: If no API key is provided.
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "No Gemini SDK installed. Install one:\n"
                "  pip install google-genai          (recommended)\n"
                "  pip install google-generativeai    (deprecated)")

        self.api_key = api_key or os.environ.get('GEMINI_API_KEY', '')
        if not self.api_key:
            raise ValueError(
                "No Gemini API key provided.\n"
                "Set GEMINI_API_KEY environment variable or pass api_key parameter.")

        # Models to try in order (flash-lite has higher free-tier limits)
        self._models = ['gemini-2.0-flash', 'gemini-2.0-flash-lite']
        self._max_retries = 2
        self._retry_delay = 65  # seconds (API says retry in ~58s)

        if _USE_NEW_SDK:
            self._client = genai.Client(api_key=self.api_key)
            logger.info("GeminiClient initialized (new SDK, models=%s)",
                         self._models)
        else:
            genai_legacy.configure(api_key=self.api_key)
            self._client = None  # legacy uses module-level config
            logger.info("GeminiClient initialized (legacy SDK, models=%s)",
                         self._models)

    def extract_parameters(self, paper_text: str, tables_text: str = '',
                           progress_callback=None) -> dict:
        """
        Send paper text to Gemini and extract parameters.

        Automatically retries on rate-limit (429) errors and falls back
        to alternative models if the primary model quota is exhausted.

        Args:
            paper_text: Full text of the paper.
            tables_text: Formatted table text from pdf_extractor.
            progress_callback: Optional callback(message) for status updates.

        Returns:
            Dict with keys: 'parameters', 'confidence', 'notes'

        Raises:
            RuntimeError: If all retry attempts and model fallbacks fail.
        """
        prompt = EXTRACTION_PROMPT.format(
            paper_text=paper_text[:80_000],
            tables_text=tables_text[:20_000],
        )

        logger.info("Sending extraction request to Gemini (%d chars)", len(prompt))

        last_error = None

        for model_name in self._models:
            for attempt in range(self._max_retries + 1):
                try:
                    if progress_callback and (attempt > 0 or model_name != self._models[0]):
                        progress_callback(
                            f"Trying {model_name} (attempt {attempt + 1})...")

                    raw_text = self._call_api(model_name, prompt)
                    return self._parse_response(raw_text)

                except Exception as e:
                    last_error = e
                    err_str = str(e)

                    # Check if it's a rate limit error (429)
                    if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                        if attempt < self._max_retries:
                            wait = self._retry_delay
                            logger.warning(
                                "Rate limited on %s (attempt %d/%d). "
                                "Waiting %ds before retry...",
                                model_name, attempt + 1,
                                self._max_retries + 1, wait)
                            if progress_callback:
                                progress_callback(
                                    f"Rate limited. Waiting {wait}s before retry...")
                            time.sleep(wait)
                            continue
                        else:
                            # Try next model
                            logger.warning(
                                "Rate limit exhausted for %s. "
                                "Trying next model...", model_name)
                            break
                    else:
                        # Non-rate-limit error — don't retry
                        logger.error("Gemini API error on %s: %s",
                                     model_name, e)
                        raise RuntimeError(
                            f"Gemini API error ({model_name}): {e}") from e

        # All models and retries exhausted
        raise RuntimeError(
            f"All Gemini models rate-limited. Please wait a minute and try again.\n"
            f"If this persists, check your quota at: "
            f"https://ai.google.dev/gemini-api/docs/rate-limits\n\n"
            f"Last error: {last_error}")

    def _call_api(self, model_name: str, prompt: str) -> str:
        """Make a single API call and return the response text."""
        if _USE_NEW_SDK:
            response = self._client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    'temperature': 0.1,
                    'max_output_tokens': 4096,
                },
            )
            return response.text.strip()
        else:
            model = genai_legacy.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai_legacy.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                ),
            )
            return response.text.strip()

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

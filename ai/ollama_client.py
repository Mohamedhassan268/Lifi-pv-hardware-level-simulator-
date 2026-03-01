# ai/ollama_client.py
"""
Ollama Local LLM Client for LiFi-PV Parameter Extraction.

Uses Ollama's REST API to run open-source models locally.
No API key needed, no rate limits, fully offline.

Requirements:
    1. Install Ollama: https://ollama.com/download
    2. Pull a model: ollama pull qwen2.5:3b
"""

import json
import logging
import subprocess
import requests
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = 'http://localhost:11434'

# Import the full prompt for large models
from ai.gemini_client import EXTRACTION_PROMPT

# Compact prompt for small models (<=3b) — short, structured, with example
COMPACT_EXTRACTION_PROMPT = '''You are a LiFi-PV hardware parameter extractor. Extract values from the paper into JSON.

Rules: Only use EXPLICIT values from the paper. Use null if not found. Convert units as specified.

Key hints:
- led_half_angle_deg: If a LENS is used, use the lens half-angle (5-15 deg), NOT bare LED (60-120 deg).
- ina_gain_dB: POSITIVE value in dB. 100x linear = 40 dB.
- r_load_ohm: DC-DC output load in Ohms (can be 180 kOhm = 180000).
- data_rate_bps: In bits/second. Convert: 5 kbps = 5000 bps.
- sc_area_cm2: Active area in cm2.
- n_cells_series: Number of cells in series (default 1 for single cell).

EXAMPLE output (different paper):
{{
  "parameters": {{
    "preset_name": "smith2023",
    "paper_reference": "Smith et al., IEEE Photonics 2023 - Indoor LiFi-PV System",
    "led_part": "OSRAM-LW5SM",
    "driver_part": "TPS61169",
    "bias_current_A": 0.5,
    "modulation_depth": 0.4,
    "led_radiated_power_mW": 15.0,
    "led_half_angle_deg": 12.0,
    "distance_m": 1.0,
    "pv_part": "SM141K04LV",
    "n_cells_series": 4,
    "sc_area_cm2": 6.0,
    "sc_responsivity": 0.55,
    "sc_cj_nF": 500.0,
    "sc_rsh_kOhm": 200.0,
    "sc_iph_uA": 300.0,
    "sc_vmpp_mV": 1800.0,
    "sc_impp_uA": 280.0,
    "sc_pmpp_uW": 504.0,
    "r_sense_ohm": 10.0,
    "ina_part": "AD8421",
    "ina_gain_dB": 20.0,
    "ina_gbw_kHz": 10000.0,
    "comparator_part": "LMV7271",
    "bpf_stages": 1,
    "bpf_f_low_Hz": 500.0,
    "bpf_f_high_Hz": 50000.0,
    "dcdc_fsw_kHz": 100.0,
    "dcdc_l_uH": 10.0,
    "dcdc_cp_uF": 4.7,
    "dcdc_cl_uF": 22.0,
    "r_load_ohm": 10000.0,
    "data_rate_bps": 100000.0,
    "modulation": "OOK",
    "n_bits": 100,
    "target_harvested_power_uW": 150.0,
    "target_ber": 0.001
  }},
  "confidence": {{
    "led_part": "extracted",
    "distance_m": "extracted",
    "modulation_depth": "estimated",
    "led_gled": "missing"
  }},
  "notes": "Parameters extracted from Tables I and II."
}}

Fields to extract (use null if not found):
preset_name, paper_reference, led_part, driver_part, bias_current_A, modulation_depth,
led_radiated_power_mW, led_half_angle_deg, led_driver_re, led_gled, lens_transmittance,
distance_m, tx_angle_deg, rx_tilt_deg, pv_part, n_cells_series, sc_area_cm2, sc_responsivity,
sc_cj_nF, sc_rsh_kOhm, sc_iph_uA, sc_vmpp_mV, sc_impp_uA, sc_pmpp_uW,
r_sense_ohm, ina_part, ina_gain_dB, ina_gbw_kHz, comparator_part,
bpf_stages, bpf_f_low_Hz, bpf_f_high_Hz,
dcdc_fsw_kHz, dcdc_l_uH, dcdc_cp_uF, dcdc_cl_uF, r_load_ohm,
data_rate_bps, modulation, n_bits,
target_harvested_power_uW, target_ber, target_noise_rms_mV

Paper text:
{paper_text}

Tables:
{tables_text}

Return ONLY the JSON object.'''

# Model recommendations based on VRAM
# (min_vram_mb, model_name, description)
MODEL_RECOMMENDATIONS = [
    (8000, 'qwen2.5:14b', 'Best quality, needs 8+ GB VRAM'),
    (5000, 'qwen2.5:7b',  'Good quality, needs 5+ GB VRAM'),
    (2500, 'qwen2.5:3b',  'Decent quality, fits 3-4 GB VRAM'),
    (1500, 'qwen2.5:1.5b', 'Basic quality, fits 2 GB VRAM'),
]


def get_gpu_vram_mb() -> Optional[int]:
    """Detect GPU VRAM in MB using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return int(result.stdout.strip().split('\n')[0])
    except Exception:
        pass
    return None


def recommend_model() -> str:
    """Recommend the best model for the available GPU."""
    vram = get_gpu_vram_mb()
    if vram is None:
        # No GPU detected, use smallest model (CPU mode)
        return 'qwen2.5:3b'
    for min_vram, model, _ in MODEL_RECOMMENDATIONS:
        if vram >= min_vram:
            return model
    return 'qwen2.5:1.5b'


def is_ollama_running() -> bool:
    """Check if Ollama server is running."""
    try:
        r = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def list_models() -> list:
    """List available Ollama models."""
    try:
        r = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=5)
        if r.status_code == 200:
            data = r.json()
            return [m['name'] for m in data.get('models', [])]
    except Exception as e:
        logger.warning("Could not list Ollama models: %s", e)
    return []


class OllamaClient:
    """Local LLM client using Ollama for parameter extraction."""

    def __init__(self, model: str = 'qwen2.5:3b',
                 base_url: str = OLLAMA_BASE_URL):
        """
        Initialize Ollama client.

        Args:
            model: Model name (must be pulled first via 'ollama pull <model>').
            base_url: Ollama API base URL (default: http://localhost:11434).

        Raises:
            RuntimeError: If Ollama is not running or model not available.
        """
        self.model = model
        self.base_url = base_url
        self._api_url = f'{base_url}/api/generate'
        self._force_cpu = False  # Set by _check_gpu_fit()

        # Check Ollama is running
        if not is_ollama_running():
            raise RuntimeError(
                "Ollama is not running.\n"
                "Start it by opening the Ollama app, or run: ollama serve")

        # Check model is available
        available = list_models()
        # Match by base name (e.g., 'qwen2.5:7b' matches 'qwen2.5:7b')
        model_found = any(
            m == model or m.startswith(model.split(':')[0])
            for m in available
        )
        if not model_found:
            raise RuntimeError(
                f"Model '{model}' not found in Ollama.\n"
                f"Available models: {available or '(none)'}\n"
                f"Pull it with: ollama pull {model}")

        # Check if model fits in GPU, auto-enable CPU fallback
        self._check_gpu_fit()

        logger.info("OllamaClient initialized (model=%s, url=%s, cpu=%s)",
                     model, base_url, self._force_cpu)

    def _check_gpu_fit(self):
        """Check if model fits in GPU VRAM, set CPU fallback if needed."""
        vram = get_gpu_vram_mb()
        if vram is None:
            return  # No GPU info available, let Ollama decide

        # Estimate VRAM needed from model size tag
        size_tag = self.model.split(':')[-1] if ':' in self.model else ''
        # Rough VRAM estimates: 1.5b~1.5GB, 3b~2.5GB, 7b~5.5GB, 14b~10GB
        vram_needed = {
            '0.5b': 800, '1.5b': 1500, '3b': 2500,
            '7b': 5500, '8b': 6000, '13b': 9000, '14b': 10000,
        }
        needed = vram_needed.get(size_tag, 0)
        if needed > 0 and needed > vram:
            self._force_cpu = True
            logger.info("Model %s needs ~%d MB VRAM but GPU has %d MB. "
                        "Using CPU mode (slower).", self.model, needed, vram)

    def extract_parameters(self, paper_text: str, tables_text: str = '',
                           progress_callback=None) -> dict:
        """
        Send paper text to local Ollama model and extract parameters.

        Args:
            paper_text: Full text of the paper.
            tables_text: Formatted table text from pdf_extractor.
            progress_callback: Optional callback(message) for status updates.

        Returns:
            Dict with keys: 'parameters', 'confidence', 'notes'
        """
        # Choose prompt and text strategy based on model size
        model_size = self.model.split(':')[-1] if ':' in self.model else ''
        is_small_model = any(s in model_size for s in ('1.5b', '3b', '1b', '0.5b'))

        if is_small_model:
            # Small models: use compact prompt + filtered key text
            from ai.pdf_extractor import extract_key_text
            key_text = extract_key_text(paper_text, max_chars=12_000)
            prompt = COMPACT_EXTRACTION_PROMPT.format(
                paper_text=key_text,
                tables_text=tables_text[:3_000] if tables_text.strip() != '(No tables extracted from PDF)' else '',
            )
        else:
            # Larger models: use full prompt
            prompt = EXTRACTION_PROMPT.format(
                paper_text=paper_text[:20_000],
                tables_text=tables_text[:5_000],
            )

        cpu_mode = self._force_cpu
        mode_str = "CPU" if cpu_mode else "GPU"
        logger.info("Sending to Ollama %s (%d chars, %s mode)",
                     self.model, len(prompt), mode_str)
        if progress_callback:
            progress_callback(
                f"Running {self.model} locally ({mode_str} mode)..."
                + (" This may take several minutes." if cpu_mode else ""))

        raw_text = self._call_ollama(prompt, cpu_mode, progress_callback)

        if not raw_text:
            raise RuntimeError("Ollama returned an empty response.")

        return self._parse_response(raw_text)

    def _call_ollama(self, prompt: str, cpu_mode: bool,
                     progress_callback=None) -> str:
        """Make the actual Ollama API call with auto-retry on CUDA errors."""
        # num_ctx: 4096 fits in 4GB VRAM with 3b model
        # Larger contexts (8192+) cause Ollama to silently fall back to CPU
        vram = get_gpu_vram_mb()
        ctx_size = 4096 if (vram and vram < 6000) else 8192
        options = {
            'temperature': 0.1,
            'num_predict': 4096,
            'num_ctx': ctx_size,
        }
        if cpu_mode:
            options['num_gpu'] = 0
        else:
            options['num_gpu'] = 999  # Force all layers to GPU

        try:
            response = requests.post(
                self._api_url,
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': options,
                },
                timeout=900,  # 15 min max for CPU inference
            )

            # Auto-retry on CUDA error with CPU fallback
            if response.status_code == 500 and not cpu_mode:
                error_body = response.text
                if 'CUDA' in error_body or 'gpu' in error_body.lower():
                    logger.warning(
                        "GPU error on %s, retrying in CPU mode...", self.model)
                    if progress_callback:
                        progress_callback(
                            f"GPU error — retrying {self.model} in CPU mode "
                            f"(slower but works)...")
                    self._force_cpu = True
                    options['num_gpu'] = 0
                    response = requests.post(
                        self._api_url,
                        json={
                            'model': self.model,
                            'prompt': prompt,
                            'stream': False,
                            'options': options,
                        },
                        timeout=900,
                    )

            if response.status_code == 500:
                raise RuntimeError(
                    f"Ollama server error: {response.text[:300]}")

            response.raise_for_status()
            data = response.json()
            return data.get('response', '').strip()

        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama timed out after 15 minutes.\n"
                f"The model may be too slow for this paper length.\n"
                f"Try a smaller model or shorter paper.")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Lost connection to Ollama.\n"
                "Make sure Ollama is still running.")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}") from e

    def _parse_response(self, raw_text: str) -> dict:
        """Parse the model's JSON response with robust error recovery."""
        import re as _re

        # Strip markdown code fences
        text = raw_text
        if '```' in text:
            lines = text.split('\n')
            lines = [l for l in lines if not l.strip().startswith('```')]
            text = '\n'.join(lines)

        # Try to find JSON object in the response
        json_start = text.find('{')
        json_end = text.rfind('}')
        if json_start >= 0 and json_end > json_start:
            text = text[json_start:json_end + 1]

        # Try parsing as-is first
        result = None
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Common fixes for small model JSON issues
            fixed = text
            # Fix trailing commas before } or ]
            fixed = _re.sub(r',\s*([}\]])', r'\1', fixed)
            # Fix unquoted NaN/Infinity
            fixed = fixed.replace(': NaN', ': null')
            fixed = fixed.replace(': Infinity', ': null')
            # Fix single quotes to double quotes (careful with apostrophes)
            if "'" in fixed and '"' not in fixed:
                fixed = fixed.replace("'", '"')
            try:
                result = json.loads(fixed)
            except json.JSONDecodeError:
                pass

        if result is None:
            logger.error("Failed to parse Ollama response as JSON")
            logger.debug("Raw response:\n%s", raw_text[:2000])
            raise RuntimeError(
                f"Model returned invalid JSON. This can happen with smaller models.\n"
                f"Try again or use a larger model.\n\n"
                f"First 300 chars: {raw_text[:300]}")

        # Ensure expected structure
        if 'parameters' not in result:
            result = {'parameters': result, 'confidence': {}, 'notes': ''}
        if 'confidence' not in result:
            result['confidence'] = {}
        if 'notes' not in result:
            result['notes'] = ''

        logger.info("Parsed %d parameters from Ollama response",
                     len(result['parameters']))
        return result

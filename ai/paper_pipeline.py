# ai/paper_pipeline.py
"""
AI Paper Reader Pipeline — Full orchestration from PDF to SystemConfig.

Supports two backends:
    - 'gemini': Google Gemini API (needs API key, fast)
    - 'ollama': Local Ollama model (no API key, runs offline)

Pipeline steps:
    1. Extract text and tables from PDF (pdf_extractor)
    2. Regex pre-extraction of known parameter patterns
    3. Send to LLM for structured parameter extraction
    4. Merge regex + LLM results (regex wins on conflicts for numeric values)
    5. Validate extracted parameters (parameter_validator)
    6. Convert to SystemConfig and optionally save as preset
"""

import json
import logging
import os
from dataclasses import fields
from pathlib import Path
from typing import Optional, Callable

from ai.pdf_extractor import extract_text, extract_tables, format_tables_as_text
from ai.parameter_validator import validate_parameters, format_validation_report

logger = logging.getLogger(__name__)


class PaperReaderPipeline:
    """Orchestrates the full paper-to-config extraction pipeline."""

    def __init__(self, backend: str = 'ollama',
                 api_key: str = '',
                 ollama_model: str = 'qwen2.5:3b'):
        """
        Initialize pipeline.

        Args:
            backend: 'ollama' for local model, 'gemini' for Google API.
            api_key: Google AI API key (only needed for 'gemini' backend).
            ollama_model: Ollama model name (only for 'ollama' backend).
        """
        self.backend = backend

        if backend == 'gemini':
            from ai.gemini_client import GeminiClient
            self.client = GeminiClient(api_key=api_key)
        elif backend == 'ollama':
            from ai.ollama_client import OllamaClient
            self.client = OllamaClient(model=ollama_model)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'ollama' or 'gemini'.")

        self._progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable):
        """Set a callback function for progress updates: callback(step, message)."""
        self._progress_callback = callback

    def _report(self, step: int, message: str):
        """Report progress to callback if set."""
        logger.info("Pipeline step %d: %s", step, message)
        if self._progress_callback:
            self._progress_callback(step, message)

    def run(self, pdf_path: str) -> dict:
        """
        Run the full extraction pipeline on a PDF.

        Uses a hybrid approach:
        1. Regex extracts reliable values (part numbers, numeric params)
        2. LLM fills in gaps and handles complex extraction
        3. Merge: regex values take priority for numeric fields

        Args:
            pdf_path: Path to the research paper PDF.

        Returns:
            Dict with keys:
                'parameters': dict of extracted parameter values
                'confidence': dict of field -> confidence level
                'validation': validation result dict
                'notes': str with AI notes
                'pdf_path': original PDF path

        Raises:
            FileNotFoundError: If PDF doesn't exist.
            RuntimeError: If extraction or API call fails.
        """
        pdf_path = str(pdf_path)
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Step 1: Extract text
        self._report(1, "Extracting text from PDF...")
        paper_text = extract_text(pdf_path)
        if len(paper_text.strip()) < 100:
            raise RuntimeError(
                "Very little text extracted from PDF. "
                "The file may be image-based (scanned). "
                "Try a text-based PDF or OCR the document first.")

        # Step 2: Extract tables + regex pre-extraction
        self._report(2, "Extracting tables and scanning for parameters...")
        tables = extract_tables(pdf_path)
        tables_text = format_tables_as_text(tables)

        # Regex pre-extraction
        from ai.regex_extractor import regex_extract
        regex_result = regex_extract(paper_text)
        regex_params = regex_result.get('parameters', {})
        regex_confidence = regex_result.get('confidence', {})
        logger.info("Regex pre-extraction found %d parameters", len(regex_params))

        # Step 3: Send to LLM
        backend_label = 'Ollama (local)' if self.backend == 'ollama' else 'Gemini API'
        self._report(3, f"Sending to {backend_label} for extraction...")
        try:
            llm_result = self.client.extract_parameters(
                paper_text, tables_text,
                progress_callback=lambda msg: self._report(3, msg))
            llm_params = llm_result.get('parameters', {})
            llm_confidence = llm_result.get('confidence', {})
            notes = llm_result.get('notes', '')
        except Exception as e:
            logger.warning("LLM extraction failed: %s. Using regex-only results.", e)
            llm_params = {}
            llm_confidence = {}
            notes = f"LLM extraction failed ({e}). Using regex-only results."

        # Step 4: Merge regex + LLM results
        self._report(4, "Merging and validating results...")
        params, confidence = _merge_results(
            regex_params, regex_confidence,
            llm_params, llm_confidence)

        # Step 5: Validate
        validation = validate_parameters(params, confidence)

        # Step 6: Clean up parameters
        self._report(5, "Finalizing results...")
        clean_params = _clean_parameters(params)

        return {
            'parameters': clean_params,
            'confidence': confidence,
            'validation': validation,
            'notes': notes,
            'pdf_path': pdf_path,
            'raw_text_length': len(paper_text),
            'tables_found': len(tables),
            'backend': self.backend,
            'regex_params_count': len(regex_params),
            'llm_params_count': len(llm_params),
        }


def _merge_results(regex_params: dict, regex_confidence: dict,
                   llm_params: dict, llm_confidence: dict) -> tuple:
    """
    Merge regex and LLM extraction results.

    Strategy:
    - Regex values always win (high reliability)
    - LLM fills gaps only for fields the regex didn't extract
    - LLM values that match the example prompt are rejected (likely copied)
    - LLM string fields (part numbers, references) are accepted more freely

    Returns:
        Tuple of (merged_params, merged_confidence)
    """
    merged = {}
    confidence = {}

    # Known example values from COMPACT_EXTRACTION_PROMPT that small models copy
    _EXAMPLE_VALUES = {
        'bias_current_A': 0.5, 'modulation_depth': 0.4,
        'led_radiated_power_mW': 15.0, 'led_half_angle_deg': 12.0,
        'distance_m': 1.0, 'n_cells_series': 4, 'sc_area_cm2': 6.0,
        'sc_responsivity': 0.55, 'sc_cj_nF': 500.0, 'sc_rsh_kOhm': 200.0,
        'sc_iph_uA': 300.0, 'sc_vmpp_mV': 1800.0, 'sc_impp_uA': 280.0,
        'sc_pmpp_uW': 504.0, 'r_sense_ohm': 10.0, 'ina_gain_dB': 20.0,
        'ina_gbw_kHz': 10000.0, 'bpf_stages': 1, 'bpf_f_low_Hz': 500.0,
        'bpf_f_high_Hz': 50000.0, 'dcdc_fsw_kHz': 100.0, 'dcdc_l_uH': 10.0,
        'dcdc_cp_uF': 4.7, 'dcdc_cl_uF': 22.0, 'r_load_ohm': 10000.0,
        'data_rate_bps': 100000.0, 'target_harvested_power_uW': 150.0,
        'target_ber': 0.001, 'n_bits': 100,
    }

    # String fields where LLM values are generally trustworthy
    _STRING_FIELDS = {
        'preset_name', 'paper_reference', 'led_part', 'driver_part',
        'pv_part', 'ina_part', 'comparator_part', 'modulation',
    }

    # Sanity bounds for LLM values (reject obviously wrong values)
    _SANITY_BOUNDS = {
        'n_cells_series': (1, 50),
        'n_cells_parallel': (1, 50),
        'bpf_stages': (1, 6),
        'dcdc_fsw_kHz': (10, 5000),
        'bias_current_A': (0.001, 5.0),
        'modulation_depth': (0.01, 1.0),
        'ina_gbw_kHz': (10, 100000),
    }

    # Accept LLM results selectively
    for key, value in llm_params.items():
        if value is None:
            continue
        # Always accept string fields from LLM
        if key in _STRING_FIELDS:
            merged[key] = value
            confidence[key] = llm_confidence.get(key, 'estimated')
            continue
        # For numeric fields, reject if value matches the example exactly
        example_val = _EXAMPLE_VALUES.get(key)
        if example_val is not None and isinstance(value, (int, float)):
            try:
                if abs(float(value) - example_val) < 0.001:
                    logger.debug("Rejecting LLM value %s=%s (matches example)", key, value)
                    continue
            except (ValueError, TypeError):
                pass
        # Sanity check bounds
        bounds = _SANITY_BOUNDS.get(key)
        if bounds and isinstance(value, (int, float)):
            lo, hi = bounds
            if not (lo <= float(value) <= hi):
                logger.debug("Rejecting LLM value %s=%s (out of bounds %s)", key, value, bounds)
                continue
        # Accept LLM value with 'estimated' confidence
        merged[key] = value
        confidence[key] = llm_confidence.get(key, 'estimated')

    # Override with regex results (more reliable for what it finds)
    for key, value in regex_params.items():
        if value is not None:
            merged[key] = value
            confidence[key] = 'extracted'

    logger.info("Merged: %d params (regex=%d, llm=%d, after filtering)",
                len(merged), len(regex_params), len(llm_params))
    return merged, confidence


def _clean_parameters(params: dict) -> dict:
    """
    Clean extracted parameters: remove nulls, convert types.

    Args:
        params: Raw parameter dict from LLM.

    Returns:
        Cleaned dict with proper types and no null values.
    """
    from cosim.system_config import SystemConfig

    valid_fields = {f.name: f for f in fields(SystemConfig)}
    cleaned = {}

    for key, value in params.items():
        if value is None:
            continue
        if key not in valid_fields:
            continue

        field_info = valid_fields[key]
        field_type = field_info.type

        try:
            # Handle Optional types
            if 'Optional' in str(field_type):
                if 'float' in str(field_type):
                    cleaned[key] = float(value)
                elif 'int' in str(field_type):
                    cleaned[key] = int(value)
                else:
                    cleaned[key] = value
            elif field_type == 'float' or field_type is float:
                cleaned[key] = float(value)
            elif field_type == 'int' or field_type is int:
                cleaned[key] = int(float(value))
            elif field_type == 'bool' or field_type is bool:
                cleaned[key] = bool(value)
            elif field_type == 'str' or field_type is str:
                cleaned[key] = str(value)
            else:
                cleaned[key] = value
        except (TypeError, ValueError) as e:
            logger.warning("Could not convert %s=%r to %s: %s",
                           key, value, field_type, e)

    return cleaned


def params_to_config(params: dict):
    """
    Convert cleaned parameter dict to a SystemConfig instance.

    Args:
        params: Cleaned parameter dict (from pipeline result['parameters']).

    Returns:
        SystemConfig instance with extracted values.
    """
    from cosim.system_config import SystemConfig

    valid_fields = {f.name for f in fields(SystemConfig)}
    filtered = {k: v for k, v in params.items() if k in valid_fields}
    return SystemConfig(**filtered)


def save_as_preset(params: dict, presets_dir: Optional[str] = None) -> str:
    """
    Save extracted parameters as a new preset JSON file.

    Args:
        params: Cleaned parameter dict.
        presets_dir: Directory to save preset in. Defaults to presets/.

    Returns:
        Path to the saved preset file.
    """
    from cosim.system_config import SystemConfig

    if presets_dir is None:
        presets_dir = str(Path(__file__).parent.parent / 'presets')

    os.makedirs(presets_dir, exist_ok=True)

    name = params.get('preset_name', 'extracted_paper')
    # Sanitize filename
    safe_name = "".join(c for c in name if c.isalnum() or c in ('_', '-')).lower()
    if not safe_name:
        safe_name = 'extracted_paper'

    filepath = os.path.join(presets_dir, f'{safe_name}.json')

    # Build full config to get defaults for missing fields
    config = params_to_config(params)
    full_dict = config.to_dict()

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(full_dict, f, indent=2)

    logger.info("Saved preset to: %s", filepath)
    return filepath

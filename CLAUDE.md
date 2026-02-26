# CLAUDE.md — Project Guide

## Quick Reference

```bash
python cli.py test              # Run 133 module self-tests
python cli.py gui               # Launch PyQt6 desktop GUI
python cli.py validate          # Generate all 23 paper figures
python cli.py validate kadirvelu2021  # Single paper
python cli.py pipeline --preset kadirvelu2021  # Full co-simulation
python cli.py components        # List all 14 components
```

## Architecture

```
CLI (cli.py) / GUI (gui/)
        |
   cosim/pipeline.py          # TX -> Channel -> RX orchestrator
        |
   +----+----+
   |         |
 SPICE    Python               # Dual engine: spice_finder.py auto-detects
 engine   engine               # Falls back to Python if no SPICE available
   |         |
 cosim/   cosim/
 ltspice  python_engine.py
 _runner
```

- **components/** — 14 hardware components with datasheet params + SPICE models
- **physics/** — PN junction, photodetection, LED emission, 6-source noise
- **materials/** — GaAs, Si, InGaP semiconductor properties
- **systems/** — Paper-specific system definitions (params, netlists, schematics)
- **cosim/** — Simulation infrastructure (config, pipeline, SPICE runners, PWL)
- **papers/** — 6 paper validation scripts, each with `run_validation(output_dir) -> bool`
- **presets/** — 7 JSON configuration files
- **gui/** — PyQt6 app with 7 tabs

## How to Add a New Component

1. Create the class in the appropriate file under `components/` (e.g., `solar_cells.py`):
   - Inherit from `PhotodetectorBase`, `LEDBase`, `AmplifierBase`, etc. (see `components/base.py`)
   - Define all electrical parameters as class attributes
   - Implement `.spice_model()` returning a SPICE subcircuit string
   - Set `.datasheet_url`

2. Register it in `components/__init__.py`:
   ```python
   COMPONENT_REGISTRY['MY_NEW_PART'] = MyNewPart
   ```

3. Add a test in `tests/test_suite.py`:
   ```python
   def test_my_new_part():
       comp = MyNewPart()
       assert comp.spice_model()
       assert comp.responsivity > 0  # or relevant parameter
   ```

4. Run tests: `python cli.py test`

## How to Add a New Paper

1. **Create a preset** in `presets/newpaper2025.json`:
   - Copy an existing preset and modify fields
   - Set `simulation_engine` to `"python"` or `"spice"`
   - Set `modulation` to the paper's scheme
   - Add `target_*` fields for validation

2. **Create a validation script** in `papers/newpaper_2025.py`:
   ```python
   PARAMS = { ... }    # Locked paper parameters
   TARGETS = { ... }   # Expected results

   def run_validation(output_dir=None) -> bool:
       # Generate figures, compare against TARGETS
       # Return True if validation passes
   ```

3. **Register it** in `papers/__init__.py`:
   ```python
   from . import newpaper_2025
   PAPERS['newpaper2025'] = {
       'label': 'Author et al. (2025)',
       'reference': 'Journal Name 2025',
       'module': newpaper_2025,
       'run': newpaper_2025.run_validation,
   }
   ```

4. Run: `python cli.py validate newpaper2025`

## How to Add a New Modulation Scheme

1. Add modulator in `cosim/python_engine.py` Transmitter class:
   ```python
   def modulate_newmod(self, bits, t, config): ...
   ```

2. Add demodulator in the Demodulator class:
   ```python
   def demodulate_newmod(self, signal, t, config): ...
   ```

3. Add the modulation name to `SystemConfig.modulation` choices
4. Update `run_python_simulation()` dispatch to handle the new scheme

## SPICE Engine

- Auto-detection via `cosim/spice_finder.py` (centralized, cross-platform)
- ngspice bundled for Windows in `ngspice-45.2_64/`
- LTspice detected from common Windows install paths
- If no SPICE engine found, pipeline auto-falls back to Python engine

## Logging

- All logs go to `workspace/simulator.log`
- CLI: add `-v` for verbose console output
- GUI: logs silently to file; check log for debugging

## Key Conventions

- Paper parameters are **locked** — never modify values from the original publication
- Component values come from **datasheets**, not approximations
- All figures should be **publication-quality** (matplotlib, 150+ DPI)
- Validation threshold: **20% error** for PASS
- Use `SystemConfig.from_preset(name)` to load paper configurations

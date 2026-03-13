# Hardware-Faithful LiFi-PV Simulator

A component-level, hardware-faithful simulator for **simultaneous light communication and energy harvesting** using photovoltaic (solar) cells. This is a generalized framework that supports multiple receiver architectures, modulation schemes, and paper configurations through a unified dual-engine pipeline.

The simulator faithfully models every component in the signal chain — from LED driver through optical channel to solar cell, transimpedance amplifier, bandpass filters, and comparator — using real device parameters, SPICE-level circuit simulation, and physics-based Python models.

---

## Features

- **Dual simulation engine** — SPICE (ngspice/LTspice) for wired circuit simulation + Python engine for wireless channel and signal processing
- **Hardware-faithful components** — 14 real components modelled with datasheet parameters (KXOB25 GaAs solar cell, INA322 INA, TLV7011 comparator, etc.)
- **6-source noise model** — shot, thermal, ambient light, amplifier, ADC quantization, and processing/threshold noise
- **5 modulation schemes** — OOK, Manchester, DCO-OFDM, BFSK, PWM-ASK
- **Lambertian optical channel** — with optional Beer-Lambert attenuation, multipath reflections, and MIMO support
- **Paper-agnostic architecture** — `SystemConfig` + JSON presets allow different paper implementations without code changes
- **Full component registry** — solar cells, LEDs, photodiodes, amplifiers, comparators, and MOSFETs with `COMPONENT_REGISTRY` lookup
- **7-tab PyQt6 desktop GUI** — system setup, component browser, channel analysis, simulation engine, results viewer, schematics, and validation
- **13-command CLI** — full control from the terminal for scripting and automation
- **Signal analysis** — BER computation, eye diagrams, frequency response, SNR analysis, and energy harvest metrics
- **Circuit schematics** — 8 schemdraw diagrams of the receiver signal chain with SVG/PNG/PDF export
- **Validation framework** — automated comparison against published paper targets (6 papers, 23 figures)
- **Session management** — each simulation run is tracked with config, netlists, raw data, and parsed results

---

## Installation

### Prerequisites

- Python 3.10 or newer
- (Optional) ngspice or LTspice for SPICE circuit simulation — the simulator auto-detects available engines and falls back to the Python engine if none is found

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd hardware_faithful_simulator

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# (Optional) Install extras for SPICE, AI paper reader, secure key storage
pip install -r requirements-optional.txt
```

### Dependencies

**Core** (`requirements.txt`):

| Package      | Purpose                          |
|--------------|----------------------------------|
| `numpy`      | Numerical computation            |
| `scipy`      | Signal processing, filters       |
| `matplotlib` | Plotting (GUI and analysis)      |
| `PyQt6`      | Desktop GUI framework            |
| `schemdraw`  | Circuit schematic generation     |
| `requests`   | HTTP requests                    |
| `PyYAML`     | YAML configuration               |

**Optional** (`requirements-optional.txt`):

| Package        | Purpose                            |
|----------------|-------------------------------------|
| `PySpice`      | SPICE integration (fallback: Python engine) |
| `google-genai` | AI paper reader (Gemini API)       |
| `pdfplumber`   | PDF text extraction                |
| `keyring`      | Secure API key storage             |

### Verify SPICE Engine

```bash
python cli.py ltspice    # Check LTspice
python cli.py simulate   # Check ngspice
```

If no SPICE engine is found, the simulator automatically uses the Python engine.

---

## Quick Start

### Launch the GUI

```bash
python cli.py gui
```

### Run a Full Simulation Pipeline

```bash
# SPICE engine (requires ngspice or LTspice)
python cli.py pipeline --preset kadirvelu2021 --bits 127 --tstop 0.03

# Python engine (no SPICE needed)
python cli.py pipeline --preset kadirvelu2021 --engine python
```

### Run Self-Tests

```bash
python cli.py test       # 133 module self-tests
pytest tests/            # pytest suite
```

### Validate Against Papers

```bash
python cli.py validate                   # All 6 papers, 23 figures
python cli.py validate kadirvelu2021     # Single paper
```

---

## Architecture

```
CLI (cli.py) / GUI (gui/)
        |
   cosim/pipeline.py              # TX -> Channel -> RX orchestrator
        |
   +----+------+
   |            |
 SPICE       Python               # Dual engine (auto-detected)
 engine      engine
   |            |
 ngspice/    cosim/python_engine.py
 LTspice         |
   |         +---+---+
   |         |       |
   +---------+  cosim/channel.py       # Lambertian + Beer-Lambert + MIMO
              |  cosim/noise.py         # 6-source physical noise model
              |  cosim/modulation.py    # 5 modulation schemes + BER
              |
         Shared modules
         (used by both engines)
```

### Directory Layout

```
hardware_faithful_simulator/
|
|-- cli.py                  # 13-command CLI entry point
|-- gui.py                  # GUI launcher
|
|-- cosim/                  # Co-simulation infrastructure
|   |-- pipeline.py         #   TX->Channel->RX orchestrator
|   |-- system_config.py    #   SystemConfig dataclass + JSON preset loader
|   |-- channel.py          #   Optical channel (Lambertian, Beer-Lambert, MIMO)
|   |-- noise.py            #   6-source noise model
|   |-- modulation.py       #   Modulation/demodulation dispatch
|   |-- python_engine.py    #   Pure-Python RX simulation
|   |-- ltspice_runner.py   #   LTspice/ngspice process launcher
|   |-- spice_finder.py     #   Cross-platform SPICE auto-detection
|   |-- raw_parser.py       #   SPICE .raw binary file parser
|   |-- pwl_writer.py       #   Piecewise-linear source file writer
|   |-- session.py          #   Session directory management
|
|-- components/             # 14 hardware components with COMPONENT_REGISTRY
|   |-- components_pv_cells.py    #   Solar cells (KXOB25-04X3F GaAs)
|   |-- leds.py                   #   LEDs (Luxeon LXM5-PD01)
|   |-- components_photodiodes.py #   Photodiodes
|   |-- amplifiers.py             #   Amplifiers (INA322, TLV2379, ADA4891)
|   |-- comparators.py            #   Comparators (TLV7011)
|   |-- mosfets.py                #   MOSFETs (BSD235N, NTS4409)
|   |-- components_base.py        #   Base classes
|
|-- systems/                # Paper-specific implementations
|   |-- kadirvelu2021.py          #   Kadirvelu 2021 parameters
|   |-- kadirvelu2021_netlist.py  #   SPICE netlist generator
|   |-- kadirvelu2021_channel.py  #   Optical channel model
|   |-- kadirvelu2021_schematic.py#   schemdraw circuit diagrams
|
|-- physics/                # Physics models
|   |-- pn_junction.py      #   PN junction I-V characteristics
|   |-- led_emission.py     #   LED electro-optical model
|   |-- photodetection.py   #   Photodetection physics
|   |-- physics_noise.py    #   Noise spectral densities
|   |-- physics_tia.py      #   Transimpedance amplifier model
|
|-- materials/              # Semiconductor material properties
|-- simulation/             # Signal generation and analysis
|-- papers/                 # 6 paper validation scripts
|-- presets/                # 7 JSON configuration files
|-- gui/                    # PyQt6 desktop GUI (7 tabs)
|-- spice_models/           # SPICE device model files
|-- ngspice-45.2_64/        # Bundled ngspice for Windows
|-- workspace/              # Simulation session storage
|-- tests/                  # Test suite
```

### Simulation Engines

| Engine | Best for | How it works |
|--------|----------|--------------|
| **SPICE** | Wired circuit simulation (RX chain, DC-DC) | Generates netlist → runs ngspice/LTspice → parses `.raw` output |
| **Python** | Wireless channel, all modulation schemes | Physics-based models: channel gain, noise injection, signal processing |

Both engines share the same channel model (`cosim/channel.py`), noise model (`cosim/noise.py`), and modulation dispatch (`cosim/modulation.py`).

### Signal Chain (Receiver)

```
Optical Power (from channel model)
    |
    v
SOLAR CELL (GaAs KXOB25-04X3F)
    |
    v
R_sense (current-to-voltage conversion)
    |
    v
INA322 Instrumentation Amplifier (40 dB gain, Vref offset)
    |
    v
Bandpass Filter Stage 1 (active, TLV2379, 700 Hz - 10.6 kHz)
    |
    v
Bandpass Filter Stage 2 (active, TLV2379, 700 Hz - 10.6 kHz)
    |
    v
TLV7011 Comparator (threshold slicer, 260 ns prop delay)
    |
    v
Digital Output (recovered data)
```

---

## Noise Model (6 Sources)

| # | Source | Formula | Physical Origin |
|---|--------|---------|-----------------|
| 1 | Shot noise | `2q(I_ph + I_dark)Bn` | Photocurrent quantum statistics |
| 2 | Thermal noise | `4kTBn / R_load` | Johnson-Nyquist |
| 3 | Ambient light | `2q·I_ambient·Bn` | Background illumination photocurrent |
| 4 | Amplifier | `(e_n² + (i_n·Z_in)²)Bn` | INA322 input-referred noise |
| 5 | ADC quantization | `V_LSB²/12` | Quantization step size |
| 6 | Processing | `σ_offset² + σ_jitter²` | Comparator offset + timing jitter |

---

## Supported Modulation Schemes

| Scheme | Config key | Description | Validated against |
|--------|-----------|-------------|-------------------|
| OOK | `OOK` | On-Off Keying | Kadirvelu 2021 |
| Manchester OOK | `OOK_Manchester` | Manchester-encoded OOK | Gonzalez 2024 |
| DCO-OFDM | `OFDM` | DC-biased optical OFDM, Gray-coded M-QAM | Sarwar 2017, Oliveira 2024 |
| BFSK | `BFSK` | Binary FSK via LC shutter | Xu 2024 |
| PWM-ASK | `PWM_ASK` | Pulse-width modulated ASK | Correa 2025 |

---

## CLI Reference

```
Usage: python cli.py <command> [options]
```

| Command       | Description                                      |
|---------------|--------------------------------------------------|
| `test`        | Run all module self-tests                        |
| `components`  | List all registered components or inspect one    |
| `channel`     | Show optical channel link budget (`--sweep`)     |
| `netlist`     | Generate SPICE netlist (`--type full/rx/ook`)    |
| `params`      | Show all paper parameters for a preset           |
| `simulate`    | Run ngspice transient simulation                 |
| `schematic`   | Generate circuit schematics (`--format svg/png`) |
| `ber`         | Display theoretical BER curves                   |
| `prbs`        | Generate and preview PRBS sequence               |
| `gui`         | Launch the PyQt6 desktop GUI                     |
| `pipeline`    | Run the full TX->Channel->RX co-simulation       |
| `presets`     | List or inspect available presets                 |
| `ltspice`     | Check SPICE engine installation status           |
| `validate`    | Run paper validation (all or specific)           |

### Examples

```bash
# Full pipeline with SPICE engine
python cli.py pipeline --preset kadirvelu2021 --bits 127 --tstop 0.03

# Full pipeline with Python engine
python cli.py pipeline --preset kadirvelu2021 --engine python

# List all components
python cli.py components

# Channel link budget with distance sweep
python cli.py channel --sweep

# Generate schematics
python cli.py schematic --format png -o ./schematics

# Validate against all papers
python cli.py validate
```

---

## GUI Overview

Launch with `python cli.py gui`. The GUI provides 7 tabs:

| Tab | Name | Description |
|-----|------|-------------|
| 1 | System Setup | TX/Channel/RX parameter configuration with live QuickInfo panel |
| 2 | Components | Browse COMPONENT_REGISTRY, view datasheet params and plots |
| 3 | Channel | Interactive link budget, distance sweep visualization |
| 4 | Simulation | Execute pipeline with live progress, browse sessions |
| 5 | Results | Waveforms, eye diagrams, BER/SNR, energy harvest, signal explorer |
| 6 | Schematics | 8 schemdraw circuit diagrams with SVG/PNG/PDF export |
| 7 | Validation | Automated PASS/FAIL comparison against paper targets |

---

## Presets

JSON files in `presets/` that define all parameters for a paper or configuration.

| Preset | Paper | Modulation | Key feature |
|--------|-------|------------|-------------|
| `kadirvelu2021` | Kadirvelu et al. 2021, IEEE TGCN | OOK | Simultaneous comms + harvesting |
| `gonzalez2024` | Gonzalez et al. 2024 | Manchester | Mains rejection with notch filter |
| `sarwar2017` | Sarwar et al. 2017 | OFDM | DCO-OFDM with M-QAM |
| `xu2024` | Xu et al. 2024 | BFSK | LC-shutter passive modulation |
| `correa2025` | Correa et al. 2025 | PWM-ASK | Greenhouse with humidity |
| `oliveira2024` | Oliveira et al. 2024 | OFDM | Spectral classification |

---

## Key Results

Using the Kadirvelu 2021 preset:

| Metric | Value |
|--------|-------|
| Distance | 32.5 cm |
| Data rate | 5 kbps OOK |
| PRBS order | 7 (127-bit sequence) |
| BER | 0 (error-free at 32.5 cm) |
| Validation | Matches paper targets within 20% |

---

## References

### Papers

1. **Kadirvelu, S. et al.** (2021). "Simultaneous Light Communication and Energy Harvesting Using Solar Cells." *IEEE Trans. Green Comms. Netw.*, vol. 5, no. 3.
2. **Gonzalez, O. et al.** (2024). Manchester-coded VLC with solar cell receivers.
3. **Sarwar, R. et al.** (2017). DCO-OFDM for simultaneous VLC and energy harvesting.
4. **Xu, Y. et al.** (2024). BFSK modulation with LC-shutter and reconfigurable PV array.
5. **Correa, C. et al.** (2025). PWM-ASK for greenhouse optical wireless with humidity effects.
6. **Oliveira, L. et al.** (2024). "Reconfigurable MIMO-based self-powered battery-less light communication system." *Light: Science & Applications*.

### Component Datasheets

- **KXOB25-04X3F** — IXYS/Anysolar GaAs photovoltaic cell
- **INA322** — Texas Instruments precision instrumentation amplifier
- **TLV7011** — Texas Instruments nanopower comparator
- **TLV2379** — Texas Instruments low-power op-amp
- **ADA4891** — Analog Devices high-speed op-amp
- **LXM5-PD01** — Lumileds Luxeon high-power LED
- **BSD235N** / **NTS4409** — Infineon/ON Semi MOSFETs for DC-DC converter

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

# Hardware-Faithful LiFi-PV Simulator

A component-level, hardware-faithful simulator for **simultaneous light communication and energy harvesting** using photovoltaic (solar) cells. Built around the circuit design from Kadirvelu et al. 2021 (IEEE TGCN), this tool implements a full 3-step co-simulation pipeline: **TX (OOK modulation) -> Channel (Lambertian optical) -> RX (SPICE circuit simulation via LTspice)**.

The simulator faithfully models every component in the signal chain --- from LED driver through optical channel to solar cell, transimpedance amplifier, bandpass filters, and comparator --- using real device parameters and SPICE-level circuit simulation.

---

## Features

- **Hardware-faithful simulation** -- every component modelled with real datasheet parameters (KXOB25 GaAs solar cell, INA322 INA, TLV7011 comparator, etc.)
- **3-step co-simulation pipeline** -- TX signal generation, Lambertian optical channel model, and SPICE-based RX circuit simulation via LTspice
- **Paper-agnostic architecture** -- `SystemConfig` + JSON presets allow different paper implementations without code changes
- **Full component registry** -- solar cells, LEDs, photodiodes, amplifiers, comparators, and MOSFETs with `COMPONENT_REGISTRY` lookup
- **7-tab PyQt6 desktop GUI** -- system setup, component browser, channel analysis, simulation engine, results viewer, schematics, and validation
- **13-command CLI** -- full control from the terminal for scripting and automation
- **Signal analysis** -- BER computation, eye diagrams, frequency response, SNR analysis, and energy harvest metrics
- **Circuit schematics** -- 8 schemdraw diagrams of the receiver signal chain with SVG/PNG/PDF export
- **Validation framework** -- automated comparison of simulation results against published paper targets (PASS/FAIL)
- **Session management** -- each simulation run is tracked with config, netlists, raw data, and parsed results

---

## Installation

### Prerequisites

- Python 3.10 or newer
- LTspice XVII (for SPICE circuit simulation) -- [download from Analog Devices](https://www.analog.com/en/resources/design-tools-and-calculators/ltspice-simulator.html)

### Install Dependencies

```bash
pip install numpy scipy matplotlib PyQt6 schemdraw
```

| Package      | Purpose                          | Required |
|--------------|----------------------------------|----------|
| `numpy`      | Numerical computation            | Yes      |
| `scipy`      | Signal processing, filters       | Yes      |
| `matplotlib` | Plotting (GUI and analysis)      | Yes      |
| `PyQt6`      | Desktop GUI framework            | Yes      |
| `schemdraw`  | Circuit schematic generation     | Optional |

### Verify LTspice

```bash
python cli.py ltspice
```

This will report whether LTspice is found on your system and its installation path.

---

## Quick Start

### Launch the GUI

```bash
python cli.py gui
```

### Run a Full Simulation Pipeline

```bash
python cli.py pipeline --preset kadirvelu2021 --bits 127 --tstop 0.03
```

This executes the complete TX -> Channel -> RX co-simulation using the Kadirvelu 2021 preset with a full PRBS-7 sequence (127 bits) and a 30 ms transient stop time.

### Run Self-Tests

```bash
python cli.py test
```

### Run the Test Suite with pytest

```bash
pytest tests/
```

---

## Architecture

```
hardware_faithful_simulator/
|
|-- cli.py                  # 13-command CLI entry point
|-- gui.py                  # GUI launcher (alternative entry)
|
|-- cosim/                  # Co-simulation infrastructure
|   |-- pipeline.py         #   3-step TX->Channel->RX pipeline
|   |-- session.py          #   Session manager (directories, config, artifacts)
|   |-- system_config.py    #   SystemConfig dataclass + JSON preset loader
|   |-- ltspice_runner.py   #   LTspice process launcher and monitor
|   |-- raw_parser.py       #   LTspice .raw binary file parser
|   |-- pwl_writer.py       #   Piecewise-linear (PWL) source file writer
|
|-- components/             # Component library with COMPONENT_REGISTRY
|   |-- components_pv_cells.py    #   Solar cells (KXOB25-04X3F GaAs)
|   |-- leds.py                   #   LEDs (Luxeon LXM5-PD01)
|   |-- components_photodiodes.py #   Photodiodes
|   |-- amplifiers.py             #   Amplifiers (INA322, TLV2379, ADA4891)
|   |-- comparators.py            #   Comparators (TLV7011)
|   |-- mosfets.py                #   MOSFETs (BSD235N, NTS4409)
|   |-- components_base.py        #   Base classes
|
|-- systems/                # Paper-specific implementations
|   |-- kadirvelu2021.py          #   Kadirvelu 2021 parameters and simulation
|   |-- kadirvelu2021_netlist.py  #   SPICE netlist generator
|   |-- kadirvelu2021_channel.py  #   Lambertian optical channel model
|   |-- kadirvelu2021_schematic.py#   schemdraw circuit diagrams
|
|-- simulation/             # Signal generation and analysis
|   |-- prbs_generator.py   #   PRBS-N sequence and OOK waveform generation
|   |-- analysis.py         #   BER, eye diagram, frequency response, SNR
|
|-- physics/                # Physics models
|   |                       #   PN junction, LED emission, photodetection,
|   |                       #   noise models, transimpedance amplifier
|
|-- materials/              # Material properties
|   |-- properties.py       #   Semiconductor material data (GaAs, Si, etc.)
|   |-- reference_data.py   #   Published reference data for validation
|
|-- gui/                    # PyQt6 desktop GUI (7 tabs)
|   |-- main_window.py      #   Main window with tab container
|   |-- tab_system_setup.py #   Tab 1: TX/Channel/RX parameter configuration
|   |-- tab_component_library.py  #   Tab 2: Component browser
|   |-- tab_channel_config.py     #   Tab 3: Channel analysis
|   |-- tab_simulation_engine.py  #   Tab 4: Pipeline execution
|   |-- tab_results.py            #   Tab 5: Results viewer (5 sub-tabs)
|   |-- tab_schematics.py         #   Tab 6: Circuit schematics
|   |-- tab_validation.py         #   Tab 7: Paper target validation
|   |-- widgets.py                #   Shared custom widgets
|
|-- presets/                # JSON parameter presets
|   |-- kadirvelu2021.json  #   Kadirvelu et al. 2021 (IEEE TGCN)
|   |-- fakidis2020.json    #   Fakidis et al. 2020
|
|-- schematics/             # Generated schematic outputs
|-- spice_models/           # SPICE device model files
|-- spice_libs/             # SPICE library files
|-- workspace/              # Simulation session storage
|-- tests/                  # pytest test suite
|-- integration/            # Integration tests
```

### Signal Chain (Receiver)

The receiver signal chain faithfully models the Kadirvelu 2021 circuit:

```
Optical Power
    |
    v
SOLAR CELL (GaAs KXOB25-04X3F)
    |
    v
R_sense (current-to-voltage)
    |
    v
INA322 Instrumentation Amplifier (Gain = 100, Vref offset)
    |
    v
Bandpass Filter Stage 1 (active, 37 Hz - 10.6 kHz)
    |
    v
Bandpass Filter Stage 2 (active, 37 Hz - 10.6 kHz)
    |
    v
TLV7011 Comparator (threshold slicer)
    |
    v
Digital Output (recovered data)
```

### Co-Simulation Pipeline

1. **TX Stage** -- Generates a PRBS-N bit sequence, applies OOK modulation with configurable modulation depth, and writes a PWL source file for SPICE.
2. **Channel Stage** -- Computes the Lambertian optical channel gain based on distance, angles, LED radiation pattern, and receiver area. Produces a link budget.
3. **RX Stage** -- Generates a full SPICE netlist of the receiver circuit, runs LTspice, parses the `.raw` output, and extracts waveforms at every node in the signal chain.

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
| `params`      | Show all Kadirvelu 2021 paper parameters         |
| `simulate`    | Run ngspice transient simulation                 |
| `schematic`   | Generate circuit schematics (`--format svg/png`) |
| `ber`         | Display theoretical BER curve for OOK            |
| `prbs`        | Generate and preview PRBS sequence               |
| `gui`         | Launch the PyQt6 desktop GUI                     |
| `pipeline`    | Run the full TX->Channel->RX co-simulation       |
| `presets`     | List or inspect available presets                |
| `ltspice`     | Check LTspice installation status                |

### Examples

```bash
# List all registered components
python cli.py components

# Inspect a specific component
python cli.py components KXOB25-04X3F

# Show channel link budget with distance sweep
python cli.py channel --sweep

# Generate OOK netlist for 127 bits at 5 kbps
python cli.py netlist --type ook --bits 127 --bitrate 5000 -o receiver.cir

# Show paper parameters
python cli.py params

# Generate PRBS-7 preview
python cli.py prbs --order 7 --bits 127

# Run full pipeline with custom parameters
python cli.py pipeline --preset kadirvelu2021 --bits 127 --tstop 0.03

# List available presets
python cli.py presets

# Inspect a preset
python cli.py presets kadirvelu2021

# Generate schematics as PNG
python cli.py schematic --format png -o ./schematics

# Show theoretical BER vs SNR
python cli.py ber
```

---

## GUI Overview

Launch with `python cli.py gui`. The GUI provides 7 tabs:

### Tab 1: System Setup
Configure all TX, Channel, and RX parameters in one place. A live QuickInfo panel displays derived values (channel gain, photocurrent, SNR estimate) as you adjust parameters. Load and save presets.

### Tab 2: Components
Browse all registered components in the `COMPONENT_REGISTRY`. View datasheet parameters, spectral response plots, and I-V characteristics.

### Tab 3: Channel
Interactive link budget table showing power at each stage of the optical channel. Distance sweep plot for visualizing received power and photocurrent vs. distance.

### Tab 4: Simulation
Execute the 3-step co-simulation pipeline with live progress status for each stage. Browse previous sessions and preview waveform outputs.

### Tab 5: Results
Five sub-tabs for comprehensive analysis:
- **Waveforms** -- Time-domain plots at every node in the signal chain
- **Eye Diagram** -- Eye opening analysis for signal quality assessment
- **BER/SNR** -- Bit error rate computation and BER vs. Distance curves
- **Energy Harvest** -- Power extraction and harvesting efficiency metrics
- **Signal Explorer** -- Interactive waveform viewer with measurement tools and CSV/PNG export

### Tab 6: Schematics
Displays 8 schemdraw circuit diagrams covering the full receiver architecture. Export individual diagrams or all schematics to SVG, PNG, or PDF.

### Tab 7: Validation
Automated comparison of simulation results against published paper targets. Each metric is evaluated as PASS or FAIL with tolerance bounds.

---

## Presets

Presets are JSON files in the `presets/` directory that define all parameters for a specific paper or configuration. The simulator loads presets through `SystemConfig.from_preset()`.

### Available Presets

| Preset            | Reference                                   |
|-------------------|---------------------------------------------|
| `kadirvelu2021`   | Kadirvelu et al. 2021, IEEE TGCN            |
| `fakidis2020`     | Fakidis et al. 2020                          |

### Creating a New Preset

1. Copy an existing preset JSON file in `presets/`.
2. Modify the parameters for your target paper or configuration.
3. The preset is automatically discovered by `SystemConfig.list_presets()`.

Preset files define TX parameters (LED power, modulation depth, bit rate), channel parameters (distance, angles, Lambertian order), RX circuit parameters (component values, gains, filter cutoffs), and validation targets.

---

## Key Results

Using the Kadirvelu 2021 preset with default parameters:

| Metric                  | Value                        |
|-------------------------|------------------------------|
| Distance                | 32.5 cm                     |
| Data rate               | 5 kbps OOK                  |
| PRBS order              | 7 (127-bit sequence)         |
| BER                     | 0 (error-free at 32.5 cm)   |
| Simulation time         | ~45 s (full PRBS-7 via LTspice) |
| Validation              | Matches Kadirvelu 2021 paper targets |

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Make your changes and ensure all tests pass:
   ```bash
   python cli.py test
   pytest tests/
   ```
4. Commit with a descriptive message.
5. Open a pull request with a clear description of the changes.

### Adding a New Paper Implementation

1. Create a new preset JSON in `presets/`.
2. If the paper uses a different circuit topology, add a new system module in `systems/`.
3. Register any new components in the `COMPONENT_REGISTRY` via `components/__init__.py`.
4. Add validation targets to the preset for the Validation tab.

### Adding a New Component

1. Create or extend a module in `components/` (e.g., `components/amplifiers.py`).
2. Inherit from the appropriate base class in `components/components_base.py`.
3. Register the component in `components/__init__.py` so it appears in `COMPONENT_REGISTRY`.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## References

1. **Kadirvelu, S. et al.** (2021). "Simultaneous Light Communication and Energy Harvesting Using Solar Cells." *IEEE Transactions on Green Communications and Networking*, vol. 5, no. 3. DOI: [10.1109/TGCN.2021.3079138](https://doi.org/10.1109/TGCN.2021.3079138)

2. **Fakidis, J. et al.** (2020). Indoor Visible Light Communication with Solar Cell Receivers. *IEEE/OSA Journal of Lightwave Technology*.

### Component Datasheets

- **KXOB25-04X3F** -- IXYS/Anysolar GaAs photovoltaic cell
- **INA322** -- Texas Instruments precision instrumentation amplifier
- **TLV7011** -- Texas Instruments nanopower comparator
- **TLV2379** -- Texas Instruments low-power op-amp
- **ADA4891** -- Analog Devices high-speed op-amp
- **LXM5-PD01** -- Lumileds Luxeon high-power LED
- **BSD235N** / **NTS4409** -- Infineon/ON Semi MOSFETs for DC-DC converter

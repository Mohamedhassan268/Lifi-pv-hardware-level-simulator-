# Hardware-Faithful Simulator: Strategic Foundation Analysis

## The Core Question

**What's the best base to build on, given that:**
1. Papers vary widely in detail (some have schematics, some are system-level)
2. We're not rushing - quality over speed
3. The goal is to demonstrate a novel idea clearly

---

## Landscape Analysis: How Papers Specify Receivers

### Category A: Component-Level Specs (Full Schematic)
**Example:** Kadirvelu 2021, Your existing validation

| What They Give | What We Need |
|----------------|--------------|
| R_sh = 138.8 kΩ | ✓ Direct use |
| C_j = 798 pF | ✓ Direct use |
| R_sense = 1 Ω | ✓ Direct use |
| INA322 op-amp | ✓ Can model |
| BPF circuit | ✓ Can derive f_3dB |

**Our approach works perfectly here.**

---

### Category B: System-Level with Part Numbers
**Example:** Correa 2025, Oliveira 2024

| What They Give | What We Need to Derive |
|----------------|----------------------|
| "KXOB25-04X3F" | R, C, I_dark from datasheet |
| "9 cm² GaAs" | Capacitance from physics |
| "f_3dB ≈ 14 kHz" | Validate against our model |
| "OOK/OFDM" | Modulation is separate |

**Our approach can work - component models fill the gaps.**

---

### Category C: Pure System-Level (No Components)
**Example:** Many theoretical papers

| What They Give | The Problem |
|----------------|-------------|
| "Solar panel receiver" | Which one? What area? |
| "Responsivity = 0.5 A/W" | Hardcoded, not derived |
| "Bandwidth = 10 kHz" | How? RC? TIA? Filter? |
| No circuit details | Can't validate components |

**These papers are actually LESS useful because they force you to guess.**

---

## The Key Insight

**Papers that specify components (Categories A & B) are actually easier to work with** than system-level papers because:

1. Component specs are physics-constrained (can't just make up values)
2. Datasheets exist - ground truth available
3. Derived parameters can be validated
4. Changing one component shows realistic impact

**System-level papers (Category C) are actually harder** because:
1. No way to know if parameters are self-consistent
2. Can't trace where values come from
3. "Just use these numbers" doesn't teach anything

---

## What Solcore/pvlib Actually Provide

### Solcore (Solar Cell Physics)
```
GOOD FOR:
✓ Material properties (bandgap, mobility, absorption)
✓ Multi-junction cell physics
✓ Quantum efficiency calculations
✓ I-V curve generation

NOT FOR:
✗ Communication system modeling
✗ Circuit-level simulation
✗ AC response / bandwidth
✗ Noise analysis
```

### pvlib (Photovoltaic System)
```
GOOD FOR:
✓ Solar irradiance modeling
✓ Sun position calculations
✓ Temperature corrections
✓ System-level energy yield

NOT FOR:
✗ Communication systems
✗ High-frequency response
✗ Small-signal analysis
✗ Individual cell physics
```

**Verdict:** These are tools for *solar energy* applications, not *optical communication*. 
They help with material properties but don't solve our unique problem.

---

## What Makes Our Problem Unique

### Traditional Solar Cell Simulation (What Solcore Does)
```
Goal: Maximize energy harvest over hours/days
Timescale: Seconds to hours
Signals: DC or very slowly varying
Metrics: kWh, efficiency, MPPT
```

### LiFi-PV Simulation (What We Need)
```
Goal: Communicate data while harvesting
Timescale: Microseconds to milliseconds
Signals: AC modulated on DC bias
Metrics: BER, SNR, bandwidth, AND harvested power
```

**The junction capacitance that Solcore ignores is our PRIMARY concern.**

---

## The Real Foundation We Need

### Layer 1: Material Physics (DONE ✓)
```python
# What we built
from materials import GAAS
print(GAAS.bandgap(300))  # 1.42 eV
print(GAAS.intrinsic_concentration(300))  # 2.1e6 cm⁻³
```
- Temperature-dependent properties
- Varshni equation for bandgap
- Mobility models
- **Solcore could enhance this, but isn't required**

### Layer 2: Component Models (DONE ✓)
```python
# What we built
from components import KXOB25_04X3F
pv = KXOB25_04X3F()
print(pv.responsivity)  # 0.457 A/W
print(pv.capacitance)   # 798 pF
print(pv.bandwidth(220))  # Derived from physics
```
- Datasheet parameters locked in
- Physics-based derivation where possible
- Validation against papers

### Layer 3: Small-Signal AC Model (NEEDED)
```python
# What's missing
class SmallSignalModel:
    """
    The key insight: Solar cells under modulated light
    behave as a current source in parallel with C_j and R_sh
    
         I_ph(t)      C_j       R_sh
           |          ||         |
        ───┴────┬─────||────┬────┴───
                |           |
               ─┴─         ─┴─
                           GND
    """
    def impedance(self, freq):
        """Z(f) = R_sh || (1/jωC_j)"""
        pass
    
    def transfer_function(self, R_load):
        """H(f) = V_out / I_ph"""
        pass
```

### Layer 4: System Integration (FUTURE)
```python
# Future vision
system = LiFiPVSystem(
    tx=LED("OSRAM_LRW5SN"),
    channel=OpticalChannel(distance=0.5),
    rx=SolarCell("KXOB25-04X3F"),
    amplifier=TIA("OPA380", R_f=100e3),
)
results = system.simulate(modulation="OOK", data_rate=10e3)
```

---

## Recommended Foundation Strategy

### Option A: Solcore-First (NOT Recommended)
```
+ Rigorous physics
+ Professionally maintained
- Overkill for our needs
- Learning curve
- Dependency risk
- Doesn't solve AC problem
```

### Option B: Custom Materials + SPICE Parser (Current Approach) ✓
```
+ We control everything
+ Tailored to LiFi-PV
+ Already working
+ No dependencies
+ Easy to extend
- More work upfront (but done!)
```

### Option C: Hybrid (Future Enhancement)
```
Use Solcore for:
  - Complex multi-junction cells
  - Accurate absorption spectra
  - Validation reference

Keep custom for:
  - AC small-signal models
  - Bandwidth calculations
  - System simulation
```

---

## What to Build Next (Priority Order)

### 1. Small-Signal AC Model (HIGH VALUE)
**Why:** This is the unique contribution - nobody has this for LiFi-PV

```python
class SolarCellSmallSignal:
    """Frequency-domain response of PV under modulation."""
    
    def __init__(self, component):
        self.R_sh = component.shunt_resistance
        self.C_j = component.capacitance
        self.R_s = component.series_resistance
        
    def impedance(self, f):
        """Complex impedance Z(jω)."""
        omega = 2 * np.pi * f
        Z_C = 1 / (1j * omega * self.C_j)
        Z_parallel = (self.R_sh * Z_C) / (self.R_sh + Z_C)
        return Z_parallel + self.R_s
    
    def transfer_function(self, f, R_load):
        """Voltage transfer H(f) = V_out/V_in."""
        Z_pv = self.impedance(f)
        return R_load / (R_load + Z_pv)
    
    def bandwidth(self, R_load):
        """Find -3dB frequency."""
        # ... search for |H(f)|² = 0.5
```

### 2. LED Emission Model (MEDIUM VALUE)
**Why:** Completes the TX side

```python
class LED:
    def __init__(self, part_number):
        # Load from database
        pass
    
    def emission_spectrum(self, I_drive):
        """Spectral power distribution P(λ)."""
        pass
    
    def modulation_response(self, f):
        """H_LED(f) - carrier recombination limit."""
        pass
```

### 3. Spectral Overlap Calculator (MEDIUM VALUE)
**Why:** Realistic responsivity requires LED×PV spectral match

```python
def effective_responsivity(led, pv):
    """
    R_eff = ∫ P_LED(λ) × R_PV(λ) dλ / ∫ P_LED(λ) dλ
    """
    pass
```

### 4. Amplifier/TIA Model (LOWER PRIORITY)
**Why:** Important but well-understood; can use datasheet values

---

## Summary: Best Foundation

| Approach | Verdict |
|----------|---------|
| Start with Solcore | ❌ Overkill, wrong focus |
| Start with pvlib | ❌ Wrong domain entirely |
| Start with SPICE only | ⚠️ Too circuit-focused |
| **Our current approach** | ✅ **Right balance** |

**What we built IS the right foundation:**
1. Materials physics (temperature-dependent)
2. Component models (datasheet + physics)
3. SPICE parser (for expanding library)

**What to add:**
1. Small-signal AC model (the unique contribution)
2. Transfer function analysis
3. Spectral overlap calculation

**What NOT to worry about yet:**
- Full Solcore integration (can add later if needed)
- SPICE circuit solving (overkill)
- Complex multi-physics (diminishing returns)

---

## Your Idea in One Sentence

> "Select a component by part number, and the simulator derives all electrical 
> parameters from physics - just like how you'd pick a chip in Proteus and it 
> knows its characteristics."

**This is novel for LiFi-PV.** Most papers either:
- Hardcode values with no traceability
- Use pure theory with no real components
- Do experiments with no simulation

**Your approach bridges theory and practice.**

---

## Next Concrete Step

Want me to build the **Small-Signal AC Model** module? This would:

1. Take any component from our library
2. Generate frequency response H(f)
3. Calculate bandwidth rigorously  
4. Show Bode plots
5. Demonstrate the "parameters emerge" philosophy

This is the piece that makes the demo compelling.

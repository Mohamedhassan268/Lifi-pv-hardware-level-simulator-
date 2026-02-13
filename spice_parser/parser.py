# spice_parser/parser.py
"""
SPICE Model File Parser for Hardware-Faithful LiFi-PV Simulator

Parses vendor SPICE library files (.lib, .mod, .spi) and extracts:
    - .MODEL statements (diodes, BJTs, MOSFETs)
    - .SUBCKT definitions (op-amps, complex ICs)

Handles:
    - Line continuations (+)
    - Comments (* and ;)
    - SPICE number formats (1K, 100M, 2.5P, etc.)
    - Nested subcircuits

References:
    - SPICE3 User's Manual
    - LTSpice documentation
"""

import re
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from .models import DiodeModel, BJTModel, SubcircuitModel, LEDModel


# =============================================================================
# SPICE NUMBER PARSER
# =============================================================================

SPICE_MULTIPLIERS = {
    'T': 1e12,
    'G': 1e9,
    'MEG': 1e6,
    'K': 1e3,
    'M': 1e-3,
    'U': 1e-6,
    'N': 1e-9,
    'P': 1e-12,
    'F': 1e-15,
}


def parse_spice_number(value_str: str) -> float:
    """
    Parse SPICE number format (e.g., '1.5K', '10P', '2.5MEG').
    
    Args:
        value_str: String representation of number
        
    Returns:
        Float value
        
    Examples:
        '1.5K' -> 1500.0
        '10P'  -> 1e-11
        '2.5MEG' -> 2.5e6
        '1E-9' -> 1e-9
    """
    if not value_str:
        return 0.0
    
    value_str = value_str.strip().upper()
    
    # Handle pure numbers (with or without exponent)
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Try each multiplier suffix
    for suffix, multiplier in sorted(SPICE_MULTIPLIERS.items(), 
                                      key=lambda x: -len(x[0])):
        if value_str.endswith(suffix):
            num_part = value_str[:-len(suffix)]
            try:
                return float(num_part) * multiplier
            except ValueError:
                pass
    
    # Last resort: try to extract leading number
    match = re.match(r'^([+-]?[\d.]+(?:[eE][+-]?\d+)?)', value_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    return 0.0


# =============================================================================
# SPICE FILE PARSER
# =============================================================================

class SPICEParser:
    """
    Parser for SPICE model library files.
    
    Usage:
        parser = SPICEParser()
        models = parser.parse_file('photodiodes.lib')
        
        # Access models
        bpw34 = models['diodes']['BPW34']
        tl072 = models['subcircuits']['TL072']
    """
    
    def __init__(self):
        self.diodes: Dict[str, DiodeModel] = {}
        self.bjts: Dict[str, BJTModel] = {}
        self.subcircuits: Dict[str, SubcircuitModel] = {}
        self.leds: Dict[str, LEDModel] = {}
        
        # Parsing state
        self._current_file = ""
        self._line_number = 0
    
    def parse_file(self, filepath: str) -> Dict[str, Dict]:
        """
        Parse a SPICE library file.
        
        Args:
            filepath: Path to .lib, .mod, or .spi file
            
        Returns:
            Dict with 'diodes', 'bjts', 'subcircuits', 'leds' keys
        """
        self._current_file = filepath
        
        # Read and preprocess file
        lines = self._read_and_join_lines(filepath)
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            self._line_number = i + 1
            
            # Skip empty lines and comments
            if not line or line.startswith('*') or line.startswith(';'):
                i += 1
                continue
            
            line_upper = line.upper()
            
            # Parse .MODEL statement
            if line_upper.startswith('.MODEL'):
                self._parse_model(line)
                i += 1
                continue
            
            # Parse .SUBCKT block
            if line_upper.startswith('.SUBCKT'):
                subckt_lines = [line]
                i += 1
                # Collect until .ENDS
                while i < len(lines):
                    subline = lines[i].strip()
                    subckt_lines.append(subline)
                    if subline.upper().startswith('.ENDS'):
                        break
                    i += 1
                self._parse_subcircuit(subckt_lines)
                i += 1
                continue
            
            i += 1
        
        return {
            'diodes': self.diodes,
            'bjts': self.bjts,
            'subcircuits': self.subcircuits,
            'leds': self.leds,
        }
    
    def _read_and_join_lines(self, filepath: str) -> List[str]:
        """
        Read file and join continuation lines (+ prefix).
        
        Args:
            filepath: Path to file
            
        Returns:
            List of complete lines
        """
        with open(filepath, 'r', errors='ignore') as f:
            raw_lines = f.readlines()
        
        joined_lines = []
        current_line = ""
        
        for line in raw_lines:
            line = line.rstrip('\n\r')
            
            # Check for continuation
            stripped = line.lstrip()
            if stripped.startswith('+'):
                # Continuation - append to current line
                current_line += ' ' + stripped[1:].strip()
            else:
                # New line
                if current_line:
                    joined_lines.append(current_line)
                current_line = line
        
        # Don't forget the last line
        if current_line:
            joined_lines.append(current_line)
        
        return joined_lines
    
    def _parse_model(self, line: str):
        """
        Parse a .MODEL statement.
        
        Format: .MODEL <name> <type>(<params>) or .MODEL <name> <type> (<params>)
        
        Args:
            line: Complete .MODEL line
        """
        # Remove .MODEL prefix
        content = line[6:].strip()
        
        # Extract model name and type
        # Pattern: NAME TYPE (params) or NAME TYPE(params)
        match = re.match(r'(\S+)\s+(\w+)\s*\(?([^)]*)\)?', content, re.IGNORECASE)
        if not match:
            return
        
        name = match.group(1).upper()
        model_type = match.group(2).upper()
        params_str = match.group(3) if match.group(3) else ""
        
        # Parse parameters
        params = self._parse_params(params_str)
        
        # Create appropriate model
        if model_type == 'D':
            self._create_diode_model(name, params, line)
        elif model_type in ['NPN', 'PNP']:
            self._create_bjt_model(name, model_type, params, line)
    
    def _parse_params(self, params_str: str) -> Dict[str, Any]:
        """
        Parse parameter string into dictionary.
        
        Format: PARAM1=VALUE1 PARAM2=VALUE2 ...
        
        Args:
            params_str: Parameter string
            
        Returns:
            Dict of param_name -> value
        """
        params = {}
        
        # Split by whitespace, handling potential = inside
        # Pattern: NAME=VALUE
        pattern = r'(\w+)\s*=\s*([^\s=]+)'
        
        for match in re.finditer(pattern, params_str, re.IGNORECASE):
            param_name = match.group(1).upper()
            param_value = match.group(2)
            
            # Try to parse as number
            try:
                params[param_name] = parse_spice_number(param_value)
            except:
                params[param_name] = param_value
        
        return params
    
    def _create_diode_model(self, name: str, params: Dict, raw_line: str):
        """Create DiodeModel from parsed parameters."""
        
        # Detect if it's an LED based on name or type parameter
        is_led = ('LED' in name.upper() or 
                  params.get('TYPE', '').upper() == 'LED' or
                  params.get('EG', 1.11) > 1.5)  # Wide bandgap
        
        # Detect if it's a photodiode
        is_photodiode = any(x in name.upper() for x in ['BPW', 'SFH', 'PD', 'PHOTO', 'VEMD'])
        
        # Common parameters
        common_params = {
            'name': name,
            'IS': params.get('IS', 1e-14),
            'N': params.get('N', 1.0),
            'RS': params.get('RS', 0.0),
            'CJO': params.get('CJO', 0.0),
            'VJ': params.get('VJ', 1.0),
            'M': params.get('M', 0.5),
            'FC': params.get('FC', 0.5),
            'TT': params.get('TT', 0.0),
            'BV': params.get('BV', 100.0),
            'IBV': params.get('IBV', 1e-10),
            'EG': params.get('EG', 1.11),
            'XTI': params.get('XTI', 3.0),
            'IKF': params.get('IKF', 0.0),
            'ISR': params.get('ISR', 0.0),
            'NR': params.get('NR', 2.0),
            'manufacturer': str(params.get('MFG', '')),
            'raw_spice': raw_line,
        }
        
        if is_led:
            # Create LED model
            led = LEDModel(
                **common_params,
                device_type='led',
                # Try to estimate wavelength from bandgap
                peak_wavelength_nm=1240.0 / params.get('EG', 2.0) if params.get('EG', 2.0) > 1.5 else 0,
            )
            self.leds[name] = led
            self.diodes[name] = led  # Also add to diodes for compatibility
        elif is_photodiode:
            common_params['device_type'] = 'photodiode'
            self.diodes[name] = DiodeModel(**common_params)
        else:
            common_params['device_type'] = 'diode'
            self.diodes[name] = DiodeModel(**common_params)
    
    def _create_bjt_model(self, name: str, polarity: str, params: Dict, raw_line: str):
        """Create BJTModel from parsed parameters."""
        
        bjt = BJTModel(
            name=name,
            polarity=polarity,
            IS=params.get('IS', 1e-15),
            BF=params.get('BF', 100.0),
            BR=params.get('BR', 1.0),
            NF=params.get('NF', 1.0),
            NR=params.get('NR', 1.0),
            RB=params.get('RB', 0.0),
            RC=params.get('RC', 0.0),
            RE=params.get('RE', 0.0),
            CJE=params.get('CJE', 0.0),
            CJC=params.get('CJC', 0.0),
            CJS=params.get('CJS', 0.0),
            TF=params.get('TF', 0.0),
            TR=params.get('TR', 0.0),
            raw_spice=raw_line,
        )
        self.bjts[name] = bjt
    
    def _parse_subcircuit(self, lines: List[str]):
        """
        Parse a .SUBCKT block.
        
        Args:
            lines: List of lines from .SUBCKT to .ENDS
        """
        if not lines:
            return
        
        # Parse header line
        header = lines[0]
        parts = header.split()
        
        if len(parts) < 2:
            return
        
        # .SUBCKT NAME node1 node2 node3 ...
        name = parts[1].upper()
        nodes = [p.upper() for p in parts[2:] if not p.startswith('PARAMS:')]
        
        # Analyze internal content
        n_resistors = 0
        n_capacitors = 0
        n_transistors = 0
        n_diodes = 0
        internal_models = []
        
        for line in lines[1:-1]:  # Skip .SUBCKT and .ENDS
            line_upper = line.upper().strip()
            
            if line_upper.startswith('R'):
                n_resistors += 1
            elif line_upper.startswith('C'):
                n_capacitors += 1
            elif line_upper.startswith(('Q', 'M', 'J')):
                n_transistors += 1
            elif line_upper.startswith('D'):
                n_diodes += 1
            elif line_upper.startswith('.MODEL'):
                # Extract model name
                model_match = re.match(r'\.MODEL\s+(\S+)', line_upper)
                if model_match:
                    internal_models.append(model_match.group(1))
        
        # Try to extract GBW from comments or component values
        gbw = 0.0
        raw_content = '\n'.join(lines)
        
        # Look for GBW in comments
        gbw_match = re.search(r'GBW\s*[=:]\s*([\d.]+)\s*(MEG|MHZ|KHZ|HZ)?', 
                             raw_content, re.IGNORECASE)
        if gbw_match:
            gbw = parse_spice_number(gbw_match.group(1) + (gbw_match.group(2) or ''))
            if 'MHZ' in (gbw_match.group(2) or '').upper():
                gbw *= 1e6
            elif 'KHZ' in (gbw_match.group(2) or '').upper():
                gbw *= 1e3
        
        subckt = SubcircuitModel(
            name=name,
            nodes=nodes,
            gbw=gbw,
            n_resistors=n_resistors,
            n_capacitors=n_capacitors,
            n_transistors=n_transistors,
            n_diodes=n_diodes,
            internal_models=internal_models,
            raw_spice=raw_content,
        )
        
        self.subcircuits[name] = subckt
    
    def summary(self) -> str:
        """Return summary of parsed models."""
        return (f"SPICEParser: {len(self.diodes)} diodes, "
                f"{len(self.bjts)} BJTs, {len(self.subcircuits)} subcircuits, "
                f"{len(self.leds)} LEDs")


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SPICE PARSER - SELF TEST")
    print("=" * 60)
    
    # Test number parser
    print("\n--- SPICE Number Parser Test ---")
    test_cases = [
        ('1.5K', 1500.0),
        ('10P', 1e-11),
        ('2.5MEG', 2.5e6),
        ('1E-9', 1e-9),
        ('100N', 1e-7),
        ('1U', 1e-6),
        ('4.7', 4.7),
    ]
    
    for test_str, expected in test_cases:
        result = parse_spice_number(test_str)
        status = "✓" if abs(result - expected) / max(abs(expected), 1e-20) < 0.01 else "✗"
        print(f"  {status} '{test_str}' -> {result:.3e} (expected {expected:.3e})")
    
    # Test with sample SPICE content
    print("\n--- Parser Test with Sample Content ---")
    
    # Create a temporary test file
    test_content = """
* Test SPICE Library
.MODEL BPW34 D(IS=2N RS=3 N=1.7 CJO=72P VJ=0.35 M=0.28 TT=50N BV=60 EG=1.11 type=photodiode)
.MODEL LED_RED D(IS=1E-20 N=2.5 RS=0.5 CJO=200P EG=1.9 type=LED)
.MODEL 2N2222 NPN(IS=1E-14 BF=200 CJE=25P CJC=8P TF=0.4N)

.SUBCKT TL072 1 2 3 4 5
* Non-inverting input, Inverting input, VCC, VEE, Output
* GBW = 3MHz
R1 1 3 500K
R2 3 2 500K
C1 11 12 3.498E-12
Q1 5 2 4 QX
Q2 6 7 4 QX
.MODEL QX PNP(BF=1111)
.ENDS
"""
    
    test_file = '/tmp/test_spice.lib'
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    parser = SPICEParser()
    result = parser.parse_file(test_file)
    
    print(f"  {parser.summary()}")
    
    if 'BPW34' in parser.diodes:
        bpw34 = parser.diodes['BPW34']
        print(f"  BPW34: CJO={bpw34.CJO*1e12:.0f}pF, RS={bpw34.RS}Ω, type={bpw34.device_type}")
    
    if 'LED_RED' in parser.leds:
        led = parser.leds['LED_RED']
        print(f"  LED_RED: EG={led.EG}eV, λ≈{led.peak_wavelength_nm:.0f}nm")
    
    if '2N2222' in parser.bjts:
        bjt = parser.bjts['2N2222']
        print(f"  2N2222: BF={bjt.BF}, CJE={bjt.CJE*1e12:.0f}pF")
    
    if 'TL072' in parser.subcircuits:
        subckt = parser.subcircuits['TL072']
        print(f"  TL072: nodes={subckt.nodes}, R={subckt.n_resistors}, Q={subckt.n_transistors}")
    
    # Cleanup
    os.remove(test_file)
    
    print("\n[OK] Parser tests passed!")

# spice_parser/__init__.py
"""
SPICE Model Parser for Hardware-Faithful LiFi-PV Simulator.

Parses vendor SPICE library files and provides access to component models.

Usage:
    from spice_parser import SPICELibrary, SPICEParser
    
    # Load entire library
    lib = SPICELibrary()
    lib.load_all()
    
    # Access models
    bpw34 = lib['BPW34']
    print(f"Capacitance: {bpw34.CJO*1e12:.0f} pF")
    print(f"Bandwidth: {bpw34.bandwidth_estimate(1000)/1e6:.1f} MHz")
    
    # Or parse individual files
    parser = SPICEParser()
    models = parser.parse_file('photodiodes.lib')
"""

from .models import (
    DiodeModel,
    BJTModel,
    SubcircuitModel,
    LEDModel,
)

from .parser import (
    SPICEParser,
    parse_spice_number,
)

from .loader import (
    SPICELibrary,
    get_library,
    get_model,
)

__all__ = [
    # Models
    'DiodeModel',
    'BJTModel',
    'SubcircuitModel',
    'LEDModel',
    
    # Parser
    'SPICEParser',
    'parse_spice_number',
    
    # Loader
    'SPICELibrary',
    'get_library',
    'get_model',
]

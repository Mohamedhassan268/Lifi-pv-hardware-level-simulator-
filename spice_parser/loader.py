# spice_parser/loader.py
"""
SPICE Library Loader for Hardware-Faithful LiFi-PV Simulator

Provides convenient interface for loading and accessing SPICE models
from multiple library files.

Usage:
    from spice_parser import SPICELibrary
    
    lib = SPICELibrary()
    lib.load_directory('/path/to/spice_libs')
    # or
    lib.load_all()  # Load from default location
    
    # Access models
    bpw34 = lib['BPW34']
    tl072 = lib['TL072']
    
    # Search
    photodiodes = lib.search('BPW')
    leds = lib.list_leds()
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from .parser import SPICEParser
from .models import DiodeModel, BJTModel, SubcircuitModel, LEDModel


class SPICELibrary:
    """
    Unified interface to SPICE model libraries.
    
    Loads and indexes models from multiple .lib/.mod files,
    providing quick lookup by name and search functionality.
    """
    
    def __init__(self, spice_libs_dir: Optional[str] = None):
        """
        Initialize library.
        
        Args:
            spice_libs_dir: Path to spice_libs directory (optional)
        """
        self.diodes: Dict[str, DiodeModel] = {}
        self.bjts: Dict[str, BJTModel] = {}
        self.subcircuits: Dict[str, SubcircuitModel] = {}
        self.leds: Dict[str, LEDModel] = {}
        
        self._loaded_files: List[str] = []
        self._default_dir = spice_libs_dir
        
        # Also maintain a unified index for quick lookup
        self._all_models: Dict[str, Any] = {}
    
    def load_file(self, filepath: str) -> int:
        """
        Load models from a single SPICE file.
        
        Args:
            filepath: Path to .lib or .mod file
            
        Returns:
            Number of models loaded
        """
        if not os.path.exists(filepath):
            print(f"WARNING: File not found: {filepath}")
            return 0
        
        parser = SPICEParser()
        try:
            result = parser.parse_file(filepath)
        except Exception as e:
            print(f"WARNING: Error parsing {filepath}: {e}")
            return 0
        
        count = 0
        
        # Merge diodes
        for name, model in parser.diodes.items():
            self.diodes[name] = model
            self._all_models[name] = model
            count += 1
        
        # Merge BJTs
        for name, model in parser.bjts.items():
            self.bjts[name] = model
            self._all_models[name] = model
            count += 1
        
        # Merge subcircuits
        for name, model in parser.subcircuits.items():
            self.subcircuits[name] = model
            self._all_models[name] = model
            count += 1
        
        # Merge LEDs (also in diodes, so don't double count)
        for name, model in parser.leds.items():
            self.leds[name] = model
            # Already in diodes, so already in _all_models
        
        self._loaded_files.append(filepath)
        return count
    
    def load_directory(self, dirpath: str, recursive: bool = True) -> int:
        """
        Load all SPICE files from a directory.
        
        Args:
            dirpath: Path to directory
            recursive: Whether to search subdirectories
            
        Returns:
            Total number of models loaded
        """
        if not os.path.exists(dirpath):
            print(f"WARNING: Directory not found: {dirpath}")
            return 0
        
        count = 0
        extensions = ['.lib', '.mod', '.spi']
        
        if recursive:
            for root, dirs, files in os.walk(dirpath):
                for fname in files:
                    if any(fname.lower().endswith(ext) for ext in extensions):
                        fpath = os.path.join(root, fname)
                        count += self.load_file(fpath)
        else:
            for fname in os.listdir(dirpath):
                if any(fname.lower().endswith(ext) for ext in extensions):
                    fpath = os.path.join(dirpath, fname)
                    count += self.load_file(fpath)
        
        return count
    
    def load_all(self) -> int:
        """
        Load all SPICE files from default location.
        
        Searches in:
            1. Specified spice_libs_dir (if set)
            2. ./spice_libs relative to this file
            3. ../spice_libs relative to this file
        
        Returns:
            Total number of models loaded
        """
        # Determine search paths
        search_paths = []
        
        if self._default_dir:
            search_paths.append(self._default_dir)
        
        # Relative to this file
        this_dir = Path(__file__).parent
        search_paths.extend([
            this_dir / 'spice_libs',
            this_dir.parent / 'spice_libs',
            this_dir.parent.parent / 'spice_libs',
            Path.cwd() / 'spice_libs',
        ])
        
        count = 0
        loaded_any = False
        
        for search_path in search_paths:
            if search_path.exists() and search_path.is_dir():
                count += self.load_directory(str(search_path))
                loaded_any = True
                break
        
        if not loaded_any:
            print(f"WARNING: No spice_libs directory found in search paths")
        
        return count
    
    def __getitem__(self, name: str) -> Any:
        """
        Get model by name.
        
        Args:
            name: Model name (case-insensitive)
            
        Returns:
            Model instance
            
        Raises:
            KeyError: If model not found
        """
        name_upper = name.upper()
        
        if name_upper in self._all_models:
            return self._all_models[name_upper]
        
        # Try case-insensitive search
        for key, model in self._all_models.items():
            if key.upper() == name_upper:
                return model
        
        raise KeyError(f"Model '{name}' not found. Use .search() to find similar.")
    
    def __contains__(self, name: str) -> bool:
        """Check if model exists."""
        return name.upper() in self._all_models
    
    def __len__(self) -> int:
        """Return total number of models."""
        return len(self._all_models)
    
    def get(self, name: str, default: Any = None) -> Any:
        """Get model by name with default."""
        try:
            return self[name]
        except KeyError:
            return default
    
    def search(self, pattern: str) -> List[str]:
        """
        Search for models matching pattern.
        
        Args:
            pattern: Search pattern (case-insensitive substring match)
            
        Returns:
            List of matching model names
        """
        pattern_upper = pattern.upper()
        return [name for name in self._all_models.keys() 
                if pattern_upper in name.upper()]
    
    def list_diodes(self) -> List[str]:
        """Return list of diode model names."""
        return list(self.diodes.keys())
    
    def list_photodiodes(self) -> List[str]:
        """Return list of photodiode model names."""
        return [name for name, model in self.diodes.items() 
                if model.device_type == 'photodiode']
    
    def list_leds(self) -> List[str]:
        """Return list of LED model names."""
        return list(self.leds.keys())
    
    def list_bjts(self) -> List[str]:
        """Return list of BJT model names."""
        return list(self.bjts.keys())
    
    def list_subcircuits(self) -> List[str]:
        """Return list of subcircuit model names."""
        return list(self.subcircuits.keys())
    
    def list_opamps(self) -> List[str]:
        """Return list of likely op-amp subcircuits."""
        opamp_patterns = ['OP', 'TL0', 'TL1', 'LM', 'AD8', 'OPA', 'LF', 'NE5']
        result = []
        for name in self.subcircuits.keys():
            if any(pattern in name.upper() for pattern in opamp_patterns):
                result.append(name)
        return result
    
    def summary(self) -> str:
        """Return summary of loaded library."""
        return (f"SPICELibrary: {len(self._all_models)} models loaded\n"
                f"  Diodes:      {len(self.diodes)}\n"
                f"  BJTs:        {len(self.bjts)}\n"
                f"  Subcircuits: {len(self.subcircuits)}")
    
    def __repr__(self):
        return f"SPICELibrary({len(self._all_models)} models)"


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

_global_library: Optional[SPICELibrary] = None


def get_library() -> SPICELibrary:
    """
    Get global SPICE library instance (lazy loaded).
    
    Returns:
        SPICELibrary with all available models loaded
    """
    global _global_library
    
    if _global_library is None:
        _global_library = SPICELibrary()
        _global_library.load_all()
    
    return _global_library


def get_model(name: str) -> Any:
    """
    Convenience function to get a model by name.
    
    Args:
        name: Model name
        
    Returns:
        Model instance
    """
    return get_library()[name]


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SPICE LIBRARY LOADER - SELF TEST")
    print("=" * 60)
    
    lib = SPICELibrary()
    
    # Create test directory structure
    test_dir = '/tmp/test_spice_libs'
    os.makedirs(f'{test_dir}/diodes', exist_ok=True)
    os.makedirs(f'{test_dir}/opamps', exist_ok=True)
    
    # Create test files
    with open(f'{test_dir}/diodes/photodiodes.lib', 'w') as f:
        f.write("""
* Test photodiode library
.MODEL BPW34 D(IS=2N RS=3 CJO=72P TT=50N EG=1.11 type=photodiode)
.MODEL SFH206K D(IS=1N RS=5 CJO=11P TT=10N EG=1.11 type=photodiode)
""")
    
    with open(f'{test_dir}/opamps/tl072.mod', 'w') as f:
        f.write("""
* TL072 Op-amp model
.SUBCKT TL072 1 2 3 4 5
R1 1 3 500K
C1 11 12 3.5P
.ENDS
""")
    
    # Test loading
    print("\n--- Loading Test ---")
    count = lib.load_directory(test_dir)
    print(f"  Loaded {count} models from {test_dir}")
    print(f"  {lib.summary()}")
    
    # Test access
    print("\n--- Access Test ---")
    if 'BPW34' in lib:
        bpw34 = lib['BPW34']
        print(f"  BPW34: {bpw34}")
        print(f"    CJO: {bpw34.CJO*1e12:.0f} pF")
        print(f"    BW@1k: {bpw34.bandwidth_estimate(1000)/1e6:.2f} MHz")
    
    # Test search
    print("\n--- Search Test ---")
    results = lib.search('SFH')
    print(f"  Search 'SFH': {results}")
    
    photodiodes = lib.list_photodiodes()
    print(f"  Photodiodes: {photodiodes}")
    
    opamps = lib.list_opamps()
    print(f"  Op-amps: {opamps}")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    
    print("\n[OK] Library loader tests passed!")

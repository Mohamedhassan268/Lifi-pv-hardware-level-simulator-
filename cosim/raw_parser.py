# cosim/raw_parser.py
"""
LTspice .raw file parser.

Reads binary .raw files produced by LTspice and extracts
time vectors and voltage/current traces.

LTspice .raw format:
    - ASCII header (ends with "Binary:" or "Values:")
    - Binary block: float64 values (transient) or complex128 (AC)

Usage:
    from cosim.raw_parser import LTSpiceRawParser

    parser = LTSpiceRawParser('output.raw')
    print(parser.list_traces())
    time = parser.get_time()
    v_out = parser.get_trace('V(out)')
"""

import struct
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict


class LTSpiceRawParser:
    """Parse LTspice binary .raw files."""

    def __init__(self, raw_path=None):
        """
        Initialize parser and optionally load a file.

        Args:
            raw_path: Path to .raw file (loaded immediately if given)
        """
        self._header = {}
        self._variables = []   # list of (index, name, type)
        self._data = None      # numpy array (n_points x n_vars)
        self._is_complex = False
        self._n_points = 0
        self._n_vars = 0

        if raw_path is not None:
            self.load(raw_path)

    def load(self, raw_path) -> None:
        """
        Load and parse a .raw file.

        Args:
            raw_path: Path to .raw file
        """
        raw_path = Path(raw_path)
        if not raw_path.exists():
            raise FileNotFoundError(f'Raw file not found: {raw_path}')

        with open(raw_path, 'rb') as f:
            raw_bytes = f.read()

        self._parse(raw_bytes)

    def _parse(self, raw_bytes: bytes) -> None:
        """Parse the raw file bytes (header + binary data)."""
        # LTspice can write UTF-16 LE headers
        # Detect UTF-16: explicit BOM, or null bytes in second position
        # (LTspice 24 often writes UTF-16 LE without BOM)
        if raw_bytes[:2] == b'\xff\xfe':
            self._parse_utf16(raw_bytes)
            return

        # Check for UTF-16 LE without BOM: 'T\x00' (start of "Title:")
        if len(raw_bytes) > 3 and raw_bytes[1:2] == b'\x00' and raw_bytes[3:4] == b'\x00':
            self._parse_utf16(raw_bytes, has_bom=False)
            return

        # ASCII header
        self._parse_ascii(raw_bytes)

    def _parse_ascii(self, raw_bytes: bytes) -> None:
        """Parse ASCII-header .raw file."""
        # Find end of header
        binary_marker = b'Binary:\n'
        values_marker = b'Values:\n'

        bin_pos = raw_bytes.find(binary_marker)
        val_pos = raw_bytes.find(values_marker)

        if bin_pos >= 0:
            header_bytes = raw_bytes[:bin_pos]
            data_bytes = raw_bytes[bin_pos + len(binary_marker):]
            is_binary = True
        elif val_pos >= 0:
            header_bytes = raw_bytes[:val_pos]
            data_bytes = raw_bytes[val_pos + len(values_marker):]
            is_binary = False
        else:
            raise ValueError('Cannot find Binary: or Values: marker in .raw file')

        header_text = header_bytes.decode('ascii', errors='replace')
        self._parse_header(header_text)

        if is_binary:
            self._parse_binary_data(data_bytes)
        else:
            self._parse_ascii_data(data_bytes.decode('ascii', errors='replace'))

    def _parse_utf16(self, raw_bytes: bytes, has_bom: bool = True) -> None:
        """Parse UTF-16 LE header .raw file (common for LTspice 24)."""
        start = 2 if has_bom else 0  # skip BOM if present

        # Find Binary: marker in UTF-16
        binary_marker_utf16 = 'Binary:\n'.encode('utf-16-le')

        pos = raw_bytes.find(binary_marker_utf16, start)
        if pos < 0:
            # Try Values: marker
            values_marker_utf16 = 'Values:\n'.encode('utf-16-le')
            pos = raw_bytes.find(values_marker_utf16, start)
            if pos < 0:
                raise ValueError('Cannot find Binary:/Values: in UTF-16 .raw file')
            header_text = raw_bytes[start:pos].decode('utf-16-le', errors='replace')
            data_bytes = raw_bytes[pos + len(values_marker_utf16):]
            self._parse_header(header_text)
            self._parse_ascii_data(data_bytes.decode('utf-16-le', errors='replace'))
            return

        header_text = raw_bytes[start:pos].decode('utf-16-le', errors='replace')
        data_bytes = raw_bytes[pos + len(binary_marker_utf16):]

        self._parse_header(header_text)
        self._parse_binary_data(data_bytes)

    def _parse_header(self, header: str) -> None:
        """Extract metadata and variable list from header text."""
        self._variables = []
        self._header = {}
        in_variables = False

        for line in header.splitlines():
            line = line.strip()
            if not line:
                continue

            if line.startswith('Variables:'):
                in_variables = True
                continue

            if in_variables:
                # Format: index \t name \t type
                parts = line.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    idx = int(parts[0])
                    name = parts[1]
                    var_type = parts[2]
                    self._variables.append((idx, name, var_type))
                continue

            # Header key: value
            if ':' in line:
                key, _, val = line.partition(':')
                val = val.strip()
                self._header[key.strip()] = val

                if key.strip() == 'No. Points':
                    self._n_points = int(val)
                elif key.strip() == 'No. Variables':
                    self._n_vars = int(val)
                elif key.strip() == 'Flags' and 'complex' in val.lower():
                    self._is_complex = True

    def _parse_binary_data(self, data_bytes: bytes) -> None:
        """Parse binary data block into numpy array."""
        n_pts = self._n_points
        n_vars = self._n_vars

        if n_pts == 0 or n_vars == 0:
            self._data = np.array([])
            return

        if self._is_complex:
            # AC analysis: all variables are complex128 (16 bytes each)
            expected = n_pts * n_vars * 16
            if len(data_bytes) < expected:
                # Sometimes time/freq is float64 while others are complex128
                # Try: first var = float64 (8 bytes), rest = complex128 (16 bytes)
                row_size = 8 + (n_vars - 1) * 16
                expected_alt = n_pts * row_size
                if len(data_bytes) >= expected_alt:
                    self._parse_binary_mixed(data_bytes, n_pts, n_vars)
                    return
                n_pts = len(data_bytes) // (n_vars * 16)

            data = np.frombuffer(data_bytes[:n_pts * n_vars * 16],
                                 dtype=np.complex128)
            self._data = data.reshape(n_pts, n_vars)
        else:
            # Transient: first variable (time) is float64 (8 bytes),
            # remaining are float32 (4 bytes) in some versions,
            # or all float64 (8 bytes)
            row_size_f64 = n_vars * 8
            row_size_mixed = 8 + (n_vars - 1) * 4

            if len(data_bytes) >= n_pts * row_size_f64:
                # All float64
                data = np.frombuffer(data_bytes[:n_pts * row_size_f64],
                                     dtype=np.float64)
                self._data = data.reshape(n_pts, n_vars)
            elif len(data_bytes) >= n_pts * row_size_mixed:
                # Mixed: time=float64, rest=float32
                self._parse_binary_mixed_real(data_bytes, n_pts, n_vars)
            else:
                # Try best guess with float64
                total_vals = len(data_bytes) // 8
                if total_vals >= n_vars:
                    n_pts = total_vals // n_vars
                    data = np.frombuffer(data_bytes[:n_pts * n_vars * 8],
                                         dtype=np.float64)
                    self._data = data.reshape(n_pts, n_vars)
                else:
                    self._data = np.array([])

        self._n_points = self._data.shape[0] if self._data.size > 0 else 0

    def _parse_binary_mixed(self, data_bytes, n_pts, n_vars):
        """Parse AC data: freq=float64, others=complex128."""
        row_size = 8 + (n_vars - 1) * 16
        result = np.zeros((n_pts, n_vars), dtype=np.complex128)

        for i in range(n_pts):
            offset = i * row_size
            freq = struct.unpack_from('d', data_bytes, offset)[0]
            result[i, 0] = complex(freq, 0)
            for j in range(1, n_vars):
                real, imag = struct.unpack_from('dd', data_bytes,
                                                 offset + 8 + (j - 1) * 16)
                result[i, j] = complex(real, imag)

        self._data = result

    def _parse_binary_mixed_real(self, data_bytes, n_pts, n_vars):
        """Parse transient data: time=float64, others=float32."""
        row_size = 8 + (n_vars - 1) * 4
        result = np.zeros((n_pts, n_vars), dtype=np.float64)

        for i in range(n_pts):
            offset = i * row_size
            result[i, 0] = struct.unpack_from('d', data_bytes, offset)[0]
            for j in range(1, n_vars):
                val = struct.unpack_from('f', data_bytes,
                                          offset + 8 + (j - 1) * 4)[0]
                result[i, j] = float(val)

        self._data = result

    def _parse_ascii_data(self, text: str) -> None:
        """Parse ASCII Values: data block."""
        lines = text.strip().splitlines()
        n_vars = self._n_vars
        if n_vars == 0:
            self._data = np.array([])
            return

        rows = []
        current_row = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Each point starts with its index
            if parts[0].isdigit() and len(current_row) == n_vars:
                rows.append(current_row)
                current_row = []
            for p in parts:
                try:
                    current_row.append(float(p))
                except ValueError:
                    pass
            if len(current_row) >= n_vars + 1:
                # index + n_vars values
                current_row = current_row[1:n_vars + 1]

        if len(current_row) == n_vars:
            rows.append(current_row)

        self._data = np.array(rows) if rows else np.array([])
        self._n_points = self._data.shape[0] if self._data.size > 0 else 0

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def list_traces(self) -> List[str]:
        """Return list of trace names (e.g. 'time', 'V(out)', 'I(R1)')."""
        return [name for _, name, _ in self._variables]

    def get_trace(self, name: str) -> np.ndarray:
        """
        Get trace data by name.

        Args:
            name: Trace name (case-insensitive, e.g. 'V(out)')

        Returns:
            numpy array of values
        """
        if self._data is None or self._data.size == 0:
            return np.array([])

        name_lower = name.lower()
        for idx, var_name, _ in self._variables:
            if var_name.lower() == name_lower:
                col = self._data[:, idx]
                if self._is_complex:
                    return col  # return complex
                return np.real(col)

        available = self.list_traces()
        raise KeyError(f"Trace '{name}' not found. Available: {available}")

    def get_time(self) -> np.ndarray:
        """Get the time (or frequency for AC) vector."""
        if self._data is None or self._data.size == 0:
            return np.array([])
        col = self._data[:, 0]
        return np.real(col)

    @property
    def n_points(self) -> int:
        return self._n_points

    @property
    def n_variables(self) -> int:
        return self._n_vars

    @property
    def is_complex(self) -> bool:
        return self._is_complex

    @property
    def header(self) -> Dict[str, str]:
        return dict(self._header)

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Return all traces as {name: array} dict."""
        result = {}
        for _, name, _ in self._variables:
            result[name] = self.get_trace(name)
        return result

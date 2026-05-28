# PyInstaller spec for the LiFi/PV backend sidecar.
#
# Produces a single-file executable that Tauri spawns as a child process.
# Bundles every Python module the simulator needs — cosim, components,
# physics, materials, systems, papers — plus the presets/ JSON files and
# the spice_libs/ directory (read at runtime by some pipeline paths).
#
# Build:
#   ./venv/Scripts/pyinstaller.exe backend/lifi-backend.spec --clean
#
# Output:
#   dist/lifi-backend.exe  (Windows)
#   dist/lifi-backend      (Linux/macOS)

# ruff: noqa
# pyright: reportUndefinedVariable=false
# (Analysis/PYZ/EXE are PyInstaller injected names; the spec runs in PyInstaller's exec()).

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = Path.cwd()  # PyInstaller invokes from repo root

# Collect every submodule of the simulator packages so dynamic imports
# (e.g. `from cosim.pipeline import ...`) don't fail in the frozen build.
hidden_imports: list[str] = []
for pkg in ("cosim", "components", "physics", "materials", "systems", "papers"):
    if (ROOT / pkg / "__init__.py").exists():
        hidden_imports += collect_submodules(pkg)

# Phase A: PySpice + libngspice (in-process SPICE engine, dual-run capable).
# Pulled in lazily by cosim.pyspice_runner; without these PyInstaller
# misses the cffi-built _NgSpiceShared bindings.
hidden_imports += collect_submodules("PySpice")
hidden_imports += ["cffi", "_cffi_backend"]

# Phase 8/13: pyldpc + its numba/llvmlite backends for LDPC FEC on the
# ieee_802_11bb preset. cosim.fec lazy-imports `from pyldpc import ...`,
# so without explicit hidden-imports the bundled sidecar reports
# "No module named 'pyldpc'" at decode time. numba/llvmlite ship native
# .pyd / .dll files that PyInstaller needs collect_dynamic_libs for.
binaries: list[tuple[str, str]] = []
for _opt_pkg in ("pyldpc", "numba", "llvmlite"):
    try:
        hidden_imports += collect_submodules(_opt_pkg)
        binaries += collect_dynamic_libs(_opt_pkg)
    except Exception:
        # pyldpc / numba aren't strictly required for the rest of the
        # simulator to boot, so don't break the build if they're absent.
        pass

# Phase 10: schemdraw drives `/api/schematic/{preset}/{idx}` (Inspector
# Schematics tab). PyInstaller's static analysis catches `import schemdraw`
# but can miss the backend module loaded dynamically by Drawing(canvas='svg')
# at first-use time, causing the bundled sidecar to return 500 with
# ModuleNotFoundError on /api/schematic/<preset>/<idx>. Collect everything
# under schemdraw so all backends + element libraries ship.
try:
    hidden_imports += collect_submodules("schemdraw")
except Exception:
    pass

# Always include uvicorn's lifespan-on / standard websocket implementations.
hidden_imports += [
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "websockets",
    "websockets.legacy",
    "websockets.legacy.server",
]

# Bundle the preset JSON files + spice libs.
datas: list[tuple[str, str]] = []
if (ROOT / "presets").is_dir():
    datas.append((str(ROOT / "presets"), "presets"))
if (ROOT / "spice_libs").is_dir():
    datas.append((str(ROOT / "spice_libs"), "spice_libs"))
if (ROOT / "spice_models").is_dir():
    datas.append((str(ROOT / "spice_models"), "spice_models"))

# Phase A: bundle PySpice's NgSpice runtime (ngspice.dll + lib + share/spinit).
# The DLL is the actual SPICE solver libngspice; PySpice loads it via cffi
# at NgSpiceShared.new_instance() time. The patched spinit (codemodel
# directives commented out — see scripts/install_pyspice.py) lives under
# share/ngspice/scripts/ and is read at DLL init.
import PySpice as _pyspice
_pyspice_root = Path(_pyspice.__file__).parent
_ngs_root = _pyspice_root / "Spice" / "NgSpice" / "Spice64_dll"
if _ngs_root.exists():
    datas.append((str(_ngs_root), "PySpice/Spice/NgSpice/Spice64_dll"))
    # Also include PySpice's api.h which cffi parses on import.
    _ngs_api_h = _pyspice_root / "Spice" / "NgSpice" / "api.h"
    if _ngs_api_h.exists():
        datas.append((str(_ngs_api_h), "PySpice/Spice/NgSpice"))

a = Analysis(
    [str(ROOT / "backend" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # GUI deps not needed in the sidecar — PyQt6, matplotlib qt backends, etc.
        "PyQt6",
        "PyQt5",
        "PySide6",
        "tkinter",
        "test",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="lifi-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,           # UPX is flaky on Windows + antivirus; leave off
    runtime_tmpdir=None,
    console=True,        # keep stdout/stderr so Tauri can log it
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

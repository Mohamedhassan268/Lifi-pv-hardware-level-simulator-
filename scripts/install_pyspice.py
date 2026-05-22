"""Idempotent installer for PySpice + ngspice shared library.

Run once after `pip install -r requirements.txt`. Does two things:

1. If `ngspice.dll` is missing from the PySpice install, downloads it via
   `PySpice.Scripts.pyspice_post_installation --install-ngspice-dll`.
2. Patches the bundled `spinit` to comment out `codemodel` directives.
   The .cm files shipped with PySpice 1.5's ngspice 34 zip don't load against
   the Windows shared-lib build and spam stderr on every simulation run.
   We don't need any codemodels for the RX chain (B-source tanh/limit are
   native to ngspice core), so disabling them is the cleanest fix.

Safe to re-run.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _patch_spinit(spinit: Path) -> bool:
    """Comment out any active `codemodel` directives. Returns True if changed."""
    text = spinit.read_text(encoding='utf-8')
    lines = text.splitlines()
    changed = False
    out = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('codemodel ') and not stripped.startswith('* '):
            out.append('* ' + line.lstrip() + '   ;; disabled by install_pyspice.py')
            changed = True
        else:
            out.append(line)
    if changed:
        spinit.write_text('\n'.join(out) + '\n', encoding='utf-8')
    return changed


def main() -> int:
    try:
        import PySpice
    except ImportError:
        print("ERROR: PySpice not installed. Run: pip install PySpice", file=sys.stderr)
        return 1

    pyspice_root = Path(PySpice.__file__).parent
    ngspice_root = pyspice_root / 'Spice' / 'NgSpice' / 'Spice64_dll'
    dll = ngspice_root / 'dll-vs' / 'ngspice.dll'
    spinit = ngspice_root / 'share' / 'ngspice' / 'scripts' / 'spinit'

    if dll.exists():
        print(f"OK ngspice.dll present at {dll}")
    else:
        print("ngspice.dll missing - invoking PySpice post-install...")
        from PySpice.Scripts.pyspice_post_installation import main as post
        saved_argv = sys.argv[:]
        try:
            sys.argv = ['pyspice-post-installation', '--install-ngspice-dll']
            post()
        except SystemExit:
            pass
        finally:
            sys.argv = saved_argv
        if not dll.exists():
            print(f"ERROR: post-install ran but DLL still missing at {dll}", file=sys.stderr)
            return 2
        print(f"OK ngspice.dll installed at {dll}")

    if spinit.exists():
        if _patch_spinit(spinit):
            print(f"OK patched spinit (disabled codemodel directives): {spinit}")
        else:
            print(f"OK spinit already clean: {spinit}")
    else:
        print(f"WARN spinit not found at {spinit}", file=sys.stderr)

    try:
        from PySpice.Spice.NgSpice.Shared import NgSpiceShared
        ngs = NgSpiceShared.new_instance()
        print(f"OK NgSpiceShared loads: ngspice version {ngs.ngspice_version}")
    except Exception as e:
        print(f"ERROR: NgSpiceShared failed to load: {e}", file=sys.stderr)
        return 3

    print("\nInstall complete.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

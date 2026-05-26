"""
Entrypoint for the bundled backend sidecar.

Invocation forms:
  python -m backend --port 8000              (dev)
  lifi-backend.exe --port 12345              (Tauri-spawned, PyInstaller)

The Tauri shell discovers a free port, passes it via --port, then polls
GET /health until the server is ready before injecting the port into the
webview.
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from backend.app import app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lifi-backend")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1 — local only)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind (default: 8000)")
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warning", "error"])
    args = parser.parse_args(argv)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        # Tauri kills the process group on window close; disable signal
        # handlers so uvicorn doesn't fight that.
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

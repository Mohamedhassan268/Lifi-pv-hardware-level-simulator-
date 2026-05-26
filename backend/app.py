"""
FastAPI entry point for the hardware-faithful LiFi/PV simulator.

Wraps the existing cosim/* simulation core without modifying it.
The PyQt6 GUI remains the legacy reference; this is the new React backend.

Dev:  uvicorn backend.app:app --reload --port 8000
Prod: launched as a Tauri sidecar (see src-tauri/).
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Make the repo root importable so `cosim`, `components`, etc. resolve.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.routers import ac_ws as ac_ws_router
from backend.routers import components as components_router
from backend.routers import config as config_router
from backend.routers import equations as equations_router
from backend.routers import link_budget as link_budget_router
from backend.routers import pipeline_ws as pipeline_ws_router
from backend.routers import presets as presets_router
from backend.routers import standards as standards_router
from backend.routers import sweep_ws as sweep_ws_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="LiFi/PV Simulator API",
        version="0.1.0",
        description="REST + WebSocket surface over the cosim pipeline.",
    )

    # Tauri loads from tauri://localhost in prod; Vite from localhost:5173 in dev.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "tauri://localhost",
            "https://tauri.localhost",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(presets_router.router, prefix="/api/presets", tags=["presets"])
    app.include_router(config_router.router, prefix="/api/config", tags=["config"])
    app.include_router(components_router.router, prefix="/api/components", tags=["components"])
    app.include_router(link_budget_router.router, prefix="/api/link-budget", tags=["link-budget"])
    app.include_router(standards_router.router, prefix="/api/standards", tags=["standards"])
    app.include_router(equations_router.router, prefix="/api", tags=["equations"])
    app.include_router(pipeline_ws_router.router, tags=["pipeline"])
    app.include_router(sweep_ws_router.router, tags=["sweep"])
    app.include_router(ac_ws_router.router, tags=["ac"])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()

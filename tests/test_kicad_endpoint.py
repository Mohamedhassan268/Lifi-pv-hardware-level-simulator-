"""HTTP tests for the KiCad export endpoints.

Skipped when FastAPI / httpx aren't installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_list_kicad_presets_includes_kadirvelu(client):
    r = client.get("/api/kicad/presets")
    assert r.status_code == 200
    body = r.json()
    assert "presets" in body
    assert "kadirvelu2021" in body["presets"]


def test_export_round_trip(client):
    r = client.post("/api/kicad/export", json={"preset": "kadirvelu2021"})
    assert r.status_code == 200
    body = r.json()
    assert body["preset"] == "kadirvelu2021"
    assert body["schematic_path"].endswith(".kicad_sch")
    assert body["bom_path"].endswith("_bom.csv")
    assert body["component_count"] > 0
    assert body["net_count"] > 0
    # warnings list is always present (may be empty)
    assert isinstance(body["warnings"], list)

    # Files must actually exist on disk after export.
    assert Path(body["schematic_path"]).exists()
    assert Path(body["bom_path"]).exists()


def test_download_bom(client):
    # Ensure the export ran in this session.
    client.post("/api/kicad/export", json={"preset": "kadirvelu2021"})

    r = client.get("/api/kicad/download/kadirvelu2021/bom")
    assert r.status_code == 200
    assert len(r.content) > 0
    assert r.headers["content-type"].startswith("text/csv")


def test_download_unknown_kind_400(client):
    r = client.get("/api/kicad/download/kadirvelu2021/pcb")
    assert r.status_code == 400


def test_export_unknown_preset_404(client):
    r = client.post("/api/kicad/export", json={"preset": "nonexistent"})
    assert r.status_code == 404


def test_download_before_export_404(client, tmp_path, monkeypatch):
    # Point the export dir somewhere empty so the download must miss.
    import backend.routers.kicad as kicad_router

    monkeypatch.setattr(kicad_router, "_OUT_DIR", tmp_path)
    r = client.get("/api/kicad/download/kadirvelu2021/sch")
    assert r.status_code == 404

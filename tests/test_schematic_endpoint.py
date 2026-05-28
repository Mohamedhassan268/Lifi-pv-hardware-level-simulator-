"""HTTP tests for the schematic SVG rendering endpoints.

Skipped when FastAPI / httpx / schemdraw aren't installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("schemdraw")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_index_lists_known_presets(client):
    r = client.get("/api/schematic/index")
    assert r.status_code == 200
    body = r.json()
    keys = {p["key"] for p in body["presets"]}
    # All six papers should have at least one drawing.
    for k in ("kadirvelu2021", "sarwar2017", "gonzalez2024", "oliveira2024",
              "xu2024", "correa2025"):
        assert k in keys, f"missing {k} in {keys}"
    # Drawings are indexed 0..N-1.
    for entry in body["presets"]:
        for i, d in enumerate(entry["drawings"]):
            assert d["index"] == i
            assert isinstance(d["name"], str) and d["name"]


def test_render_returns_svg(client):
    r = client.get("/api/schematic/kadirvelu2021/0")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg+xml")
    assert r.content.startswith(b"<?xml")
    assert b"<svg" in r.content
    # Sanity bound: the simplest drawing should be at least a few KB.
    assert len(r.content) > 1_000


def test_render_unknown_preset_404(client):
    r = client.get("/api/schematic/nonexistent/0")
    assert r.status_code == 404


def test_render_out_of_range_404(client):
    r = client.get("/api/schematic/kadirvelu2021/99")
    assert r.status_code == 404


@pytest.mark.parametrize(
    "preset",
    ["kadirvelu2021", "sarwar2017", "gonzalez2024", "oliveira2024",
     "xu2024", "correa2025"],
)
def test_first_drawing_per_paper_renders(client, preset):
    """Smoke-test the 'Full System' drawing for every paper."""
    r = client.get(f"/api/schematic/{preset}/0")
    assert r.status_code == 200, r.text
    assert r.content.startswith(b"<?xml")

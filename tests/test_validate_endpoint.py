"""
Tests for POST /api/config/validate.

Skipped when FastAPI is not installed (it's an optional dependency — the
Python-only / CLI workflow doesn't need it).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")  # required by starlette TestClient

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402
from cosim.system_config import SystemConfig  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_valid_preset_returns_no_errors(client):
    cfg = SystemConfig.from_preset("kadirvelu2021").to_dict()
    r = client.post("/api/config/validate", json=cfg)
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["errors"] == []
    # Structured issues always present (may include WARNING/INFO)
    assert isinstance(body["issues"], list)
    for issue in body["issues"]:
        assert issue["level"] in ("warning", "info")
        assert {"level", "field", "message"}.issubset(issue.keys())


def test_error_level_makes_response_invalid(client):
    cfg = SystemConfig.from_preset("kadirvelu2021").to_dict()
    cfg["distance_m"] = -1.0
    r = client.post("/api/config/validate", json=cfg)
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False
    assert any("distance" in e for e in body["errors"])
    rule_ids = {i["rule_id"] for i in body["issues"]}
    assert "physical.distance_nonpositive" in rule_ids


def test_datasheet_warning_surfaces(client):
    cfg = SystemConfig.from_preset("kadirvelu2021").to_dict()
    cfg["bias_current_A"] = 2.5  # LXM5-PD01 max is 0.7 A
    r = client.post("/api/config/validate", json=cfg)
    assert r.status_code == 200
    body = r.json()
    # Valid (no ERROR-level), but datasheet warning must be present.
    assert body["valid"] is True
    rule_ids = {i["rule_id"] for i in body["issues"]}
    assert "datasheet.led_overcurrent" in rule_ids


def test_issue_schema_fields(client):
    cfg = SystemConfig.from_preset("xu2024").to_dict()
    r = client.post("/api/config/validate", json=cfg)
    assert r.status_code == 200
    body = r.json()
    assert body["issues"], "xu2024 should surface at least one issue"
    issue = body["issues"][0]
    # Backend contract: these keys exist on every issue.
    for key in ("level", "field", "message", "suggestion", "rule_id"):
        assert key in issue

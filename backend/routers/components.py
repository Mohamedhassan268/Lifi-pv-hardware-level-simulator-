"""GET /api/components — component registry browser."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from components import COMPONENT_REGISTRY, get_component, list_components

router = APIRouter()


# Map each component's `type` (from get_parameters()) to a human category.
# Falls back to the type string itself for anything not listed.
_TYPE_TO_CATEGORY = {
    "solar_cell": "Solar Cell",
    "photodiode": "Photodiode",
    "LED": "LED",
    "instrumentation_amplifier": "Amplifier",
    "operational_amplifier": "Amplifier",
    "transimpedance_amplifier": "Amplifier",
    "comparator": "Comparator",
    "N-MOSFET": "MOSFET",
    "P-MOSFET": "MOSFET",
    "mppt_boost_charger": "Power",
    "integrated_adc": "ADC",
}


def _resolve_category(comp: object) -> str:
    if hasattr(comp, "get_parameters"):
        try:
            t = comp.get_parameters().get("type")
            if isinstance(t, str):
                return _TYPE_TO_CATEGORY.get(t, t.replace("_", " ").title())
        except Exception:  # noqa: BLE001
            pass
    return "Other"


# Which SystemConfig field does each category populate when the user picks
# a component in the Workbench. The frontend uses this to wire dropdowns
# back into configStore.patch().
_CATEGORY_TO_CONFIG_FIELD = {
    "Solar Cell": "pv_part",
    "Photodiode": "pv_part",
    "LED": "led_part",
    "Amplifier": "ina_part",
    "Comparator": "comparator_part",
    "MOSFET": "driver_part",
    "Power": "dcdc_part",
    "ADC": "adc_part",
}


@router.get("")
def list_all() -> list[dict[str, str]]:
    seen: dict[str, dict[str, str]] = {}
    for part in list_components():
        cls = COMPONENT_REGISTRY[part]
        # The registry has alias keys (KXOB25-04X3F and KXOB25_04X3F); dedupe by class.
        if cls.__name__ in seen:
            continue
        try:
            comp = cls()
        except Exception:  # noqa: BLE001
            continue
        category = _resolve_category(comp)
        seen[cls.__name__] = {
            "part": part,
            "category": category,
            "class": cls.__name__,
            "config_field": _CATEGORY_TO_CONFIG_FIELD.get(category, ""),
        }
    return list(seen.values())


@router.get("/categories")
def categories() -> list[str]:
    """Distinct category names in alphabetical order."""
    cats = {row["category"] for row in list_all()}
    return sorted(cats)


@router.get("/{part}")
def get_part(part: str) -> dict[str, Any]:
    try:
        comp = get_component(part)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    params: dict[str, Any] = {}
    if hasattr(comp, "get_parameters"):
        try:
            params = comp.get_parameters()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"get_parameters failed: {e}") from e

    category = _resolve_category(comp)
    return {
        "part": part,
        "class": comp.__class__.__name__,
        "category": category,
        "config_field": _CATEGORY_TO_CONFIG_FIELD.get(category, ""),
        "parameters": params,
    }

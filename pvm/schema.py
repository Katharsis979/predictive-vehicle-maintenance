"""Load config/vehicle_schema.yaml and expose small helper functions.

Keeping parameter definitions in a data file (instead of hard-coding them in
the app) is a common engineering pattern called "configuration over code":
you can add or change parameters without rewriting logic.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "config" / "vehicle_schema.yaml"


@lru_cache(maxsize=4)
def load_schema(path: str | Path = SCHEMA_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    for feat in schema["features"]:
        feat.setdefault("powertrains", list(schema["powertrains"]))
        feat.setdefault("help", "")
        feat.setdefault("unit", "")
    return schema


def component_names(schema: dict | None = None) -> list[str]:
    schema = schema or load_schema()
    return list(schema["components"].keys())


def target_name(component: str) -> str:
    """Name of the label column for a component's failure."""
    return f"fail_{component}"


def applicability(schema: dict | None = None) -> dict[str, tuple[str, set[str]]]:
    """Map target column -> ("powertrain", {powertrains it applies to}).

    The model uses this to output "not applicable" (NaN) instead of a risk
    score, e.g. an EV has no engine oil to fail.
    """
    schema = schema or load_schema()
    return {
        target_name(c): ("powertrain", set(info["powertrains"]))
        for c, info in schema["components"].items()
    }


def features_for(powertrain: str, group: str | None = None, schema: dict | None = None) -> list[dict]:
    schema = schema or load_schema()
    return [
        f for f in schema["features"]
        if powertrain in f["powertrains"] and (group is None or f["group"] == group)
    ]

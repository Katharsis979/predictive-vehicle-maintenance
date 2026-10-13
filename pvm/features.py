"""Feature engineering: turn raw readings into physically meaningful signals.

Tree models can learn interactions on their own, but handing them a ratio a
mechanic would compute anyway ("how far over its oil interval is this car?",
"how hot is the coolant compared to the outside air?") makes learning
faster and more reliable, especially with limited data.

Every derived feature is only added when its inputs exist, so this works on
any dataset — columns it doesn't recognise are simply passed through.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DERIVED_FEATURES = {
    # name: (required input columns, function, explanation)
    "coolant_above_ambient_c": (
        ["coolant_temp_c", "ambient_temp_c"],
        lambda d: d["coolant_temp_c"] - d["ambient_temp_c"],
        "How hard the cooling system is working against outside air.",
    ),
    "oil_minus_coolant_c": (
        ["oil_temp_c", "coolant_temp_c"],
        lambda d: d["oil_temp_c"] - d["coolant_temp_c"],
        "Oil running much hotter than coolant suggests degraded oil/friction.",
    ),
    "oil_pressure_per_krpm": (
        ["oil_pressure_kpa", "engine_rpm"],
        lambda d: d["oil_pressure_kpa"] / (d["engine_rpm"].clip(lower=500) / 1000),
        "Oil pressure normalised by engine speed (pressure naturally rises with rpm).",
    ),
    "km_per_year": (
        ["odometer_km", "vehicle_age_years"],
        lambda d: d["odometer_km"] / d["vehicle_age_years"].clip(lower=0.25),
        "Usage intensity.",
    ),
    "brake_wear_mm_per_1000km": (
        ["brake_pad_mm", "km_since_brake_service"],
        lambda d: (12 - d["brake_pad_mm"]) / (d["km_since_brake_service"].clip(lower=1000) / 1000),
        "Observed pad wear rate since the last brake job.",
    ),
    "tread_wear_mm_per_1000km": (
        ["tire_tread_mm", "km_on_tires"],
        lambda d: (9 - d["tire_tread_mm"]) / (d["km_on_tires"].clip(lower=1000) / 1000),
        "Observed tread wear rate on the current set.",
    ),
    "hv_temp_above_ambient_c": (
        ["hv_battery_temp_c", "ambient_temp_c"],
        lambda d: d["hv_battery_temp_c"] - d["ambient_temp_c"],
        "Pack heating relative to outside air (weak cells run hot).",
    ),
    "mech_power_proxy": (
        ["engine_rpm", "engine_load_pct"],
        lambda d: d["engine_rpm"] * d["engine_load_pct"] / 100,
        "Rough proxy for mechanical power output (speed × load).",
    ),
}


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name, (needs, fn, _) in DERIVED_FEATURES.items():
        if all(c in out.columns for c in needs):
            with np.errstate(divide="ignore", invalid="ignore"):
                out[name] = fn(out).replace([np.inf, -np.inf], np.nan)
    return out

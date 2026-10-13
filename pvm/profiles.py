"""Make/model profiles and VIN decoding — the "case by case" layer.

The ML model was trained on generic vehicle *types* (sedan, pickup, EV…).
A profile for a specific car (e.g. a 2018 Kia Optima) adapts the inputs and
adds model-specific knowledge on top:

1. Picks the right vehicle_type / powertrain for the model.
2. Rescales "km since service" by the manufacturer's interval. If Kia says
   change the oil every 12,070 km and the model was trained on cars with a
   9,000 km interval, then 6,000 km on a Kia is like 6,000 × 9,000/12,070
   ≈ 4,470 km on the training cars. This is a simple form of *domain
   adaptation*: translating your data into the units the model learned in.
3. Watches for that model's known problems (matched by trouble code).
4. Checks live readings against normal ranges.
"""
from __future__ import annotations

import json
import re
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import load_schema

MODELS_PATH = Path(__file__).resolve().parent.parent / "config" / "vehicle_models.yaml"


@lru_cache(maxsize=2)
def load_models(path: str | Path = MODELS_PATH) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def profile_label(key: str) -> str:
    p = load_models()["models"][key]
    return f"{p['make']} {p['model']} {p['years'][0]}–{p['years'][1]} ({p['generation']})"


def match_profile(make: str | None, model: str | None, year: int | None) -> str | None:
    """Find the profile for a make/model/year, or None if we don't have one."""
    if not make or not model:
        return None
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())  # noqa: E731
    for key, p in load_models()["models"].items():
        if norm(p["make"]) == norm(make) and norm(p["model"]) == norm(model):
            if year is None or p["years"][0] <= int(year) <= p["years"][1]:
                return key
    return None


# ---------------------------------------------------------------------------
# VIN decoding
# ---------------------------------------------------------------------------
_TRANSLIT = {**{str(d): d for d in range(10)},
             **dict(zip("ABCDEFGH", range(1, 9))), **dict(zip("JKLMN", range(1, 6))), "P": 7, "R": 9,
             **dict(zip("STUVWXYZ", range(2, 10)))}
_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]
_YEAR_CODES = "ABCDEFGHJKLMNPRSTVWXY123456789"  # 30-year cycle: A=1980/2010 … 9=2009/2039

# World Manufacturer Identifier (first 3 characters) -> make. A small sample;
# online decoding covers everything else.
WMI = {"KNA": "Kia", "KND": "Kia", "5XX": "Kia", "5XY": "Kia", "KMH": "Hyundai", "5NP": "Hyundai",
       "5NM": "Hyundai", "1HG": "Honda", "2HG": "Honda", "JHM": "Honda", "4T1": "Toyota", "JTD": "Toyota",
       "5YJ": "Tesla", "1FT": "Ford", "1FA": "Ford", "1G1": "Chevrolet", "3VW": "Volkswagen", "WVW": "Volkswagen"}


def vin_check_digit_ok(vin: str) -> bool:
    """North-American VINs carry a check digit in position 9.

    Each character becomes a number, is multiplied by a position weight, the
    products are summed, and the sum mod 11 must equal the check digit
    (10 is written as 'X'). It catches almost every single-character typo.
    """
    total = sum(_TRANSLIT[c] * w for c, w in zip(vin, _WEIGHTS))
    r = total % 11
    return vin[8] == ("X" if r == 10 else str(r))


def decode_vin_offline(vin: str) -> dict:
    vin = vin.strip().upper()
    if len(vin) != 17 or any(c in "IOQ" for c in vin) or not vin.isalnum():
        raise ValueError("A VIN is 17 letters/digits and never contains I, O or Q.")
    code = vin[9]
    year = None
    if code in _YEAR_CODES:
        base = 2010 if vin[6].isalpha() else 1980  # NHTSA rule for cars & light trucks
        year = base + _YEAR_CODES.index(code)
    return {"vin": vin, "make": WMI.get(vin[:3]), "model": None, "year": year,
            "check_digit_ok": vin_check_digit_ok(vin), "source": "offline"}


def decode_vin(vin: str, online: bool = True, timeout: float = 6.0) -> dict:
    """Decode a VIN. Tries NHTSA's free vPIC service, falls back to offline rules."""
    info = decode_vin_offline(vin)
    if not online:
        return info
    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{info['vin']}?format=json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            r = json.load(resp)["Results"][0]
    except Exception:
        return info
    title = lambda s: s.title() if s and s.isupper() and len(s) > 3 else (s or None)  # noqa: E731
    info.update({
        "make": title(r.get("Make")) or info["make"],
        "model": r.get("Model") or None,
        "year": int(r["ModelYear"]) if r.get("ModelYear") else info["year"],
        "body": r.get("BodyClass") or None,
        "fuel": r.get("FuelTypePrimary") or None,
        "electrification": r.get("ElectrificationLevel") or None,
        "displacement_l": r.get("DisplacementL") or None,
        "turbo": (r.get("Turbo") or "").lower() == "yes",
        "engine": r.get("EngineModel") or None,
        "source": "NHTSA vPIC",
    })
    return info


def guess_vehicle_type(info: dict) -> str:
    """For cars without a profile: pick the closest generic vehicle type."""
    elec = (info.get("electrification") or "").upper()
    body = (info.get("body") or "").lower()
    fuel = (info.get("fuel") or "").lower()
    if "BEV" in elec or fuel == "electric":
        return "ev"
    if "HEV" in elec:
        return "hybrid"
    if "pickup" in body:
        return "pickup"
    if "van" in body:
        return "delivery_van" if "diesel" in fuel else "suv"
    if "sport utility" in body or "crossover" in body:
        return "suv"
    if "diesel" in fuel:
        return "delivery_van"
    try:
        if float(info.get("displacement_l") or 0) and float(info["displacement_l"]) < 1.8:
            return "compact"
    except ValueError:
        pass
    return "sedan"


# ---------------------------------------------------------------------------
# Adapting model inputs to a specific car
# ---------------------------------------------------------------------------
def adapt_row_to_profile(row: pd.DataFrame, profile_key: str | None, severe: bool = False,
                         turbo: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Rescale service-history inputs to the manufacturer's intervals.

    Returns the adapted row and human-readable notes describing each change.
    """
    notes: list[str] = []
    if not profile_key:
        return row, notes
    cfg = load_models()
    p = cfg["models"][profile_key]
    ref = cfg["reference_intervals_km"]
    pt = p["powertrain"]
    iv = p.get("service_intervals_km") or {}
    out = row.copy()

    def scale(col, ref_km, mfr_km, what):
        if mfr_km and ref_km and col in out and pd.notna(out[col].iloc[0]):
            f = ref_km / mfr_km
            out[col] = out[col] * f
            notes.append(f"{what}: {p['make']} interval {mfr_km:,.0f} km vs model's reference {ref_km:,.0f} km "
                         f"→ km since service scaled ×{f:.2f}")

    oil = iv.get("oil_severe") if severe and iv.get("oil_severe") else iv.get("oil")
    scale("km_since_oil_change", ref["oil"].get(pt), oil, "Oil")
    plugs = iv.get("spark_plugs_turbo") if turbo and iv.get("spark_plugs_turbo") else iv.get("spark_plugs")
    scale("km_since_tuneup", ref["spark_plugs"].get(pt), plugs, "Spark plugs")
    scale("km_since_coolant_service", ref["coolant"].get(pt), iv.get("coolant_first"), "Coolant")
    return out, notes


def known_issue_alerts(profile_key: str | None, year: int | None, dtcs: list[str]) -> list[dict]:
    """Model-specific issues that apply to this car, flagged if a matching code is present."""
    if not profile_key:
        return []
    out = []
    codes = {c.upper() for c in dtcs}
    for issue in load_models()["models"][profile_key].get("known_issues") or []:
        y0, y1 = issue.get("years", [0, 9999])
        if year is not None and not (y0 <= year <= y1):
            continue
        hit = sorted(codes & {c.upper() for c in issue.get("dtcs", [])})
        out.append({**issue, "triggered": bool(hit), "matched_codes": hit})
    return out


def spec_checks(readings: dict, profile_key: str | None = None, engine_running: bool = True,
                warm: bool = True) -> list[dict]:
    """Compare readings against normal ranges (generic, overridden per profile)."""
    cfg = load_models()
    checks = dict(cfg.get("generic_checks") or {})
    if profile_key:
        checks.update(cfg["models"][profile_key].get("checks") or {})
    results = []
    for col, rule in checks.items():
        v = readings.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        when = rule.get("when", "any")
        if (when == "warm" and not warm) or (when == "running" and not engine_running):
            continue
        ok = rule["min"] <= v <= rule["max"]
        results.append({"column": col, "value": v, "min": rule["min"], "max": rule["max"], "ok": ok,
                        "note": rule.get("note", "")})
    return results


def profile_vehicle_type(profile_key: str) -> tuple[str, str]:
    p = load_models()["models"][profile_key]
    return p["vehicle_type"], p["powertrain"]


def vehicle_types() -> dict:
    return load_schema()["vehicle_types"]

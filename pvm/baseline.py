"""Learning what's normal for ONE specific car (a "digital twin").

The fleet model knows what healthy cars look like in general. But your car
has its own personality: its thermostat might hold 93 °C where another runs
at 88 °C, and its fuel trims might always sit at +3 %. Those are normal *for
it*. What matters is when it changes.

So for each signal (coolant temp, fuel trims, voltage, …) we train a small
regression model on your own logged drives that predicts the signal from
the driving conditions (rpm, speed, load, outside temperature). Then, on a
new drive:

    residual = actual − what your car normally shows in these conditions
    z-score  = residual / typical size of residuals

A z-score of 0 is perfectly normal for your car; beyond ±3 (sustained) means
"this is unusual for this car" — often long before a warning light.

VehicleStore keeps each car's logs, service records and baseline on disk.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

CONTEXT = ["engine_rpm", "speed_kph", "engine_load_pct", "throttle_pct", "ambient_temp_c", "intake_temp_c"]
MONITORED = ["coolant_temp_c", "oil_temp_c", "fuel_trim_ltft_pct", "fuel_trim_stft_pct", "module_voltage_v",
             "timing_advance_deg", "maf_gps", "intake_pressure_kpa"]
MIN_ROWS = 120


def warm_running(df: pd.DataFrame) -> pd.DataFrame:
    d = df
    if "engine_rpm" in d:
        d = d[d["engine_rpm"] > 400]
    if "coolant_temp_c" in d:
        d = d[d["coolant_temp_c"] >= 75]
    return d


class PersonalBaseline:
    def __init__(self):
        self.models: dict[str, HistGradientBoostingRegressor] = {}
        self.context: dict[str, list[str]] = {}
        self.scale: dict[str, float] = {}
        self.n_rows = 0

    def fit(self, history: pd.DataFrame) -> "PersonalBaseline":
        d = warm_running(history)
        self.n_rows = len(d)
        for target in MONITORED:
            if target not in d:
                continue
            ctx = [c for c in CONTEXT if c in d and c != target and d[c].notna().any()]
            sub = d[ctx + [target]].dropna(subset=[target])
            if len(sub) < MIN_ROWS or not ctx or sub[target].std() == 0:
                continue
            model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
                                                  min_samples_leaf=20)
            # Out-of-fold predictions give an honest estimate of normal scatter.
            oof = cross_val_predict(model, sub[ctx], sub[target], cv=3)
            resid = sub[target].to_numpy() - oof
            mad = np.median(np.abs(resid - np.median(resid))) * 1.4826  # robust standard deviation
            self.scale[target] = float(max(mad, 1e-3))
            self.models[target] = model.fit(sub[ctx], sub[target])
            self.context[target] = ctx
        return self

    @property
    def ready(self) -> bool:
        return bool(self.models)

    def score(self, recent: pd.DataFrame) -> pd.DataFrame:
        """Per-signal deviation of `recent` readings from this car's normal."""
        d = warm_running(recent)
        rows = []
        for target, model in self.models.items():
            if target not in d:
                continue
            ctx = self.context[target]
            sub = d.dropna(subset=[target])
            if sub.empty or not all(c in sub for c in ctx):
                continue
            expected = model.predict(sub[ctx])
            resid = sub[target].to_numpy() - expected
            z = resid / self.scale[target]
            rows.append({"signal": target, "actual": float(np.median(sub[target])),
                         "expected": float(np.median(expected)), "difference": float(np.median(resid)),
                         "z": float(np.median(z)), "samples": int(len(sub))})
        return pd.DataFrame(rows)


class VehicleStore:
    """One folder per vehicle under data/vehicles/: info.json, log.csv, baseline.joblib."""

    def __init__(self, root: str | Path = Path(__file__).resolve().parent.parent / "data" / "vehicles"):
        self.root = Path(root)

    def _dir(self, vehicle_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", vehicle_id)[:40] or "vehicle"
        p = self.root / safe
        p.mkdir(parents=True, exist_ok=True)
        return p

    def list(self) -> list[str]:
        return sorted(p.name for p in self.root.glob("*") if (p / "info.json").exists()) if self.root.exists() else []

    def load_info(self, vehicle_id: str) -> dict:
        f = self._dir(vehicle_id) / "info.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def save_info(self, vehicle_id: str, info: dict) -> None:
        (self._dir(vehicle_id) / "info.json").write_text(json.dumps(info, indent=2, default=str))

    def append_log(self, vehicle_id: str, df: pd.DataFrame, session: str) -> None:
        f = self._dir(vehicle_id) / "log.csv"
        df = df.assign(session=session)
        if f.exists():
            old = pd.read_csv(f)
            df = pd.concat([old[old["session"] != session], df], ignore_index=True)
        df.to_csv(f, index=False)

    def load_log(self, vehicle_id: str) -> pd.DataFrame:
        f = self._dir(vehicle_id) / "log.csv"
        return pd.read_csv(f) if f.exists() else pd.DataFrame()

    def save_baseline(self, vehicle_id: str, baseline: PersonalBaseline) -> None:
        joblib.dump(baseline, self._dir(vehicle_id) / "baseline.joblib")

    def load_baseline(self, vehicle_id: str) -> PersonalBaseline | None:
        f = self._dir(vehicle_id) / "baseline.joblib"
        return joblib.load(f) if f.exists() else None


def km_since_services(info: dict, odometer_km: float | None) -> dict:
    """Turn stored service records ("oil changed at 88,400 km") into km-since values."""
    out = {}
    if odometer_km is None:
        return out
    mapping = {"oil": "km_since_oil_change", "coolant": "km_since_coolant_service", "brakes": "km_since_brake_service",
               "transmission": "km_since_trans_service", "spark_plugs": "km_since_tuneup", "tires": "km_on_tires"}
    for svc, col in mapping.items():
        at = (info.get("service_odometer_km") or {}).get(svc)
        if at is not None and odometer_km >= at:
            out[col] = odometer_km - at
    return out

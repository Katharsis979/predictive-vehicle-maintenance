"""Run with:  pytest -q

These tests use a small simulated fleet so they finish in seconds.
"""
import numpy as np
import pandas as pd
import pytest

from pvm.model import MaintenanceModel
from pvm.schema import applicability, component_names, load_schema, target_name
from pvm.simulate import generate_fleet, project_forward

TARGETS = [target_name(c) for c in component_names()]


@pytest.fixture(scope="module")
def fleet():
    return generate_fleet(6000, seed=1)


@pytest.fixture(scope="module")
def model(fleet):
    m = MaintenanceModel(TARGETS, "rul_km", applicability(), group_col="powertrain", drop_cols=["vehicle_id"])
    m.fit(fleet, compute_importance=False)
    return m


def test_simulator_shapes_and_applicability(fleet):
    schema = load_schema()
    ev = fleet["powertrain"] == "electric"
    assert fleet.loc[ev, "fail_engine_oil"].isna().all()           # EVs have no engine oil
    assert fleet.loc[~ev, "fail_engine_oil"].notna().all()
    assert fleet.loc[ev, "coolant_temp_c"].isna().all()
    for f in schema["features"]:
        assert f["name"] in fleet.columns, f"schema feature {f['name']} missing from simulator"
    for t in TARGETS:
        rate = fleet[t].mean()
        assert 0.005 < rate < 0.35, f"{t} prevalence {rate:.3f} is unrealistic"


def test_hidden_truth_is_not_a_feature(model):
    assert not any(c.startswith("_") for c in model.feature_names_)
    assert not set(TARGETS) & set(model.feature_names_)
    assert "rul_km" not in model.feature_names_


def test_model_beats_chance_on_unseen_fleet(model):
    from sklearn.metrics import roc_auc_score
    new = generate_fleet(3000, seed=77)
    pred = model.predict(new)
    for t in TARGETS:
        ok = new[t].notna()
        assert roc_auc_score(new.loc[ok, t], pred.loc[ok, f"risk_{t}"]) > 0.85, t


def test_not_applicable_is_nan(model, fleet):
    pred = model.predict(fleet[fleet["powertrain"] == "electric"].head(20))
    assert pred["risk_fail_engine_oil"].isna().all()
    assert pred["risk_fail_hv_battery"].notna().all()


def test_handles_missing_and_extra_columns(model, fleet):
    sparse = fleet.head(50).drop(columns=["brake_pad_mm", "oil_pressure_kpa", "vibration_rms_g"])
    sparse["some_new_sensor"] = 1.0                       # unknown column: ignored
    sparse.loc[:10, "vehicle_type"] = "hovercraft"        # unknown category: treated as missing
    pred = model.predict(sparse)
    vals = pred.filter(like="risk_").to_numpy().ravel()
    vals = vals[~np.isnan(vals)]
    assert len(vals) and ((vals >= 0) & (vals <= 1)).all()


def test_risk_rises_when_oil_change_is_overdue(model, fleet):
    car = fleet[(fleet["powertrain"] == "gasoline") & (fleet["km_since_oil_change"] < 2000)].head(1)
    proj = project_forward(car.iloc[0], np.array([0, 20000]))
    risk = model.predict(proj)["risk_fail_engine_oil"].to_numpy()
    assert risk[1] > risk[0] + 0.3


def test_explanation_points_at_the_cause(model, fleet):
    # Age a car 25,000 km past its oil change (sensors drift consistently), then ask why.
    car = fleet[(fleet["powertrain"] == "gasoline") & (fleet["km_since_oil_change"] < 2000)].head(1)
    aged = project_forward(car.iloc[0], np.array([25000.0]))
    ex = model.explain(aged, "fail_engine_oil")
    assert ex.iloc[0]["feature"] in {"km_since_oil_change", "oil_pressure_kpa", "oil_temp_c", "engine_rpm"}
    assert ex.iloc[0]["contribution"] > 0


def test_generic_dataset_any_columns():
    """The model must work on a dataset with columns it has never heard of."""
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({
        "pump_hours": rng.uniform(0, 10000, n),
        "bearing_temp": rng.normal(60, 10, n),
        "site": rng.choice(["north", "south", "east"], n),
        "serial": [f"S{i}" for i in range(n)],              # ID column: should be skipped
    })
    risk = 1 / (1 + np.exp(-(df["pump_hours"] / 1500 - 5 + (df["bearing_temp"] - 60) / 5)))
    df["failed"] = (rng.random(n) < risk).astype(int)
    m = MaintenanceModel(["failed"])
    rep = m.fit(df, compute_importance=False)
    assert "serial" not in m.feature_names_
    assert rep["targets"]["failed"]["roc_auc"] > 0.85

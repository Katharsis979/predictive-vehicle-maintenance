"""Tests for the OBD layer, vehicle profiles and the personal baseline (no hardware needed)."""
import numpy as np
import pandas as pd
import pytest

from pvm.baseline import PersonalBaseline, VehicleStore, km_since_services
from pvm.obd_live import (DEMO_VIN, SimulatedCarSource, battery_metrics, dtc_component, load_log, snapshot)
from pvm.profiles import (adapt_row_to_profile, decode_vin_offline, known_issue_alerts, match_profile,
                          spec_checks, vin_check_digit_ok)


def drive(fault="none", n=300, seed=0, ambient=22.0):
    car = SimulatedCarSource(fault=fault, seed=seed, ambient_c=ambient)
    rows = []
    for i in range(n):
        r = car.read()
        r["t"] = i
        r["dtc_count"] = len(car.read_dtcs())
        rows.append(r)
    return pd.DataFrame(rows), car


# ---- VIN & profiles -------------------------------------------------------
def test_vin_check_digit():
    assert vin_check_digit_ok("1HGCM82633A004352")       # textbook example
    assert not vin_check_digit_ok("1HGCM82643A004352")   # one typo
    assert vin_check_digit_ok(DEMO_VIN)


def test_vin_offline_decode_year_and_make():
    info = decode_vin_offline(DEMO_VIN)
    assert info["make"] == "Kia" and info["year"] == 2018
    with pytest.raises(ValueError):
        decode_vin_offline("1HGCM82633A00435O")           # letter O is never allowed


def test_profile_matching_by_generation():
    assert match_profile("KIA", "Optima", 2013) == "kia_optima_tf"
    assert match_profile("Kia", "Optima", 2018) == "kia_optima_jf"
    assert match_profile("Kia", "K5", 2023) == "kia_k5_dl3"
    assert match_profile("Kia", "Optima", 2030) is None


def test_interval_rescaling_uses_manufacturer_interval():
    row = pd.DataFrame([{"km_since_oil_change": 12070.0}])
    out, notes = adapt_row_to_profile(row, "kia_optima_jf")
    assert out["km_since_oil_change"].iloc[0] == pytest.approx(9000, rel=1e-3)  # one full Kia interval = one reference interval
    severe, _ = adapt_row_to_profile(row, "kia_optima_jf", severe=True)
    assert severe["km_since_oil_change"].iloc[0] > out["km_since_oil_change"].iloc[0]
    assert notes


def test_known_issue_triggers_on_code_and_year():
    hits = known_issue_alerts("kia_optima_tf", 2015, ["P1326"])
    assert any(i["triggered"] for i in hits)
    assert known_issue_alerts("kia_optima_tf", 2018, ["P1326"]) == []   # outside covered years


# ---- Simulated car, snapshots, logs --------------------------------------
def test_battery_metrics_detect_weak_battery():
    good, _ = drive("none", 20)
    weak, _ = drive("weak_battery", 20)
    g, w = battery_metrics(good), battery_metrics(weak)
    assert g["battery_12v_resting_v"] > 12.4 > w["battery_12v_resting_v"]
    assert w["cranking_voltage_v"] < 9.6 < g["cranking_voltage_v"]


def test_battery_metrics_ignore_running_engine():
    h = pd.DataFrame({"engine_rpm": [1300, 0, 640], "module_voltage_v": [14.7, 14.78, 14.7]})
    assert battery_metrics(h) == {}     # 14.8 V is alternator output, not a resting battery


def test_faults_show_up_in_checks_and_codes():
    hot, car = drive("overheating", 200)
    snap = snapshot(hot)
    assert any(not c["ok"] and c["column"] == "coolant_temp_c" for c in spec_checks(snap))
    assert [c for c, _ in car.read_dtcs()] == ["P0217"]
    assert dtc_component("P0217") == "cooling_system"
    lean, _ = drive("lean", 200)
    assert snapshot(lean)["fuel_trim_ltft_pct"] > 10


def test_phone_log_import_units_and_aliases():
    wide = pd.DataFrame({
        "Device Time": [0, 1], "Engine RPM(rpm)": [800, 900], "Speed (OBD)(mph)": [0, 10],
        "GPS Speed (Meters/second)": [0, 4], "Engine Coolant Temperature(°F)": [212, 194],
        "O2 Sensor1 Voltage(V)": [0.4, 0.5], "Voltage (Control Module)(V)": [14.1, 14.2],
        "Fuel Trim Bank 1 Long Term(%)": [2, 3]})
    d = load_log(wide)
    assert d["coolant_temp_c"].tolist() == pytest.approx([100, 90])
    assert d["speed_kph"].iloc[1] == pytest.approx(16.09, rel=1e-3)
    assert d["module_voltage_v"].iloc[0] == 14.1          # not the O2 sensor voltage
    assert "fuel_trim_ltft_pct" in d

    long = pd.DataFrame({"SECONDS": [0, 0, 1, 1], "PID": ["Engine RPM", "Coolant temperature"] * 2,
                         "VALUE": [800, 85, 820, 86], "UNITS": ["rpm", "°C"] * 2})
    d2 = load_log(long)
    assert d2["engine_rpm"].tolist() == [800, 820] and d2["coolant_temp_c"].tolist() == [85, 86]


# ---- Personal baseline ----------------------------------------------------
def test_baseline_flags_change_for_this_car():
    history = pd.concat([drive("none", 500, seed=s, ambient=a)[0] for s, a in [(1, 12), (2, 22), (3, 30)]],
                        ignore_index=True)
    base = PersonalBaseline().fit(history)
    assert "coolant_temp_c" in base.models
    normal = base.score(drive("none", 300, seed=9)[0].tail(150)).set_index("signal")
    hot = base.score(drive("overheating", 300, seed=9)[0].tail(150)).set_index("signal")
    assert abs(normal.loc["coolant_temp_c", "z"]) < 2
    assert hot.loc["coolant_temp_c", "z"] > 3


def test_vehicle_store_roundtrip(tmp_path):
    store = VehicleStore(tmp_path)
    store.save_info("ABC123", {"service_odometer_km": {"oil": 90000}})
    h, _ = drive("none", 30)
    store.append_log("ABC123", h, session="s1")
    store.append_log("ABC123", h, session="s1")        # re-saving a session replaces it
    assert len(store.load_log("ABC123")) == len(h)
    assert km_since_services(store.load_info("ABC123"), 96000) == {"km_since_oil_change": 6000}


# ---- Real python-OBD path against an emulator (optional) -------------------
# Start an emulator in another terminal:   python -m elm -n 35000 -s car
# then run:   PVM_ELM_EMULATOR=socket://localhost:35000 pytest tests/test_obd.py -k emulator
import os  # noqa: E402


@pytest.mark.skipif(not os.environ.get("PVM_ELM_EMULATOR"), reason="set PVM_ELM_EMULATOR to test against ELM327-emulator")
def test_elm327_emulator_end_to_end():
    from pvm.obd_live import ELM327Source
    src = ELM327Source(os.environ["PVM_ELM_EMULATOR"], timeout=3)
    src.connect()
    try:
        assert {"engine_rpm", "coolant_temp_c", "speed_kph"} <= set(src.supported())
        r = src.read()
        assert 0 <= r["engine_rpm"] < 10000 and -40 <= r["coolant_temp_c"] < 215
        assert isinstance(src.read_dtcs(), list)
    finally:
        src.close()

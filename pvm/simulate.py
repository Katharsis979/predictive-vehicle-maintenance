"""Physics-inspired simulator for a fleet of vehicles.

WHY SIMULATE?
Real fleet maintenance data (sensor logs + what later broke) is valuable and
usually private. So we build a *generative model* of how vehicles wear out,
use it to create labeled training data, and then train the ML model on it.
The key idea that makes this a fair test of the ML model:

    hidden component health  -->  noisy sensor readings  -->  ML model
            (the "truth")             (what a car reports)     (must infer health)

The simulator knows each component's true health (columns starting with
"_health_"), but those columns are NEVER given to the model. The model only
sees what a real car would report: mileage, service history, and sensors.
It has to learn the relationships (e.g. "low oil pressure at high oil temp
means degraded oil") from data, just like it would with real logs.

The pipeline is split into steps so the app can reuse them:
    sample_configs()          -> who the vehicles are and how they're used
    compute_health()          -> hidden true health of each component (0..1)
    generate_sensors()        -> what the sensors would read
    generate_labels()         -> what fails in the next 5,000 km
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import component_names, load_schema, target_name

FAILURE_HORIZON_KM = 5_000

# Columns that describe the vehicle and its history (everything else is a sensor reading or a label).
CONFIG_COLS = [
    "vehicle_type", "powertrain", "climate", "vehicle_age_years", "odometer_km", "driving_aggression",
    "payload_pct", "ambient_temp_c", "dc_fast_charge_pct", "km_since_oil_change", "km_since_coolant_service",
    "km_since_brake_service", "km_since_trans_service", "km_since_tuneup", "km_on_tires", "tire_age_months",
    "battery_12v_age_months", "speed_kph", "engine_rpm", "engine_load_pct",
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# 1. Vehicle configurations
# ---------------------------------------------------------------------------
def sample_configs(n: int, rng: np.random.Generator, vehicle_type: str | None = None) -> pd.DataFrame:
    schema = load_schema()
    vtypes = schema["vehicle_types"]
    names = list(vtypes)
    # Fleet mix: more passenger cars than heavy trucks.
    weights = np.array([0.16, 0.18, 0.16, 0.12, 0.10, 0.06, 0.10, 0.12])
    vt = np.full(n, vehicle_type) if vehicle_type else rng.choice(names, size=n, p=weights / weights.sum())

    powertrain = np.array([vtypes[v]["powertrain"] for v in vt])
    annual_nominal = np.array([vtypes[v]["annual_km"] for v in vt], dtype=float)

    age = rng.gamma(shape=2.0, scale=3.0, size=n).clip(0.1, 25)
    annual = annual_nominal * rng.lognormal(0, 0.35, size=n)
    odometer = (age * annual).clip(100, 1_200_000)

    climate = rng.choice(["hot", "temperate", "cold"], size=n, p=[0.3, 0.45, 0.25])
    ambient = np.select(
        [climate == "hot", climate == "temperate"],
        [rng.normal(31, 5, n), rng.normal(17, 7, n)],
        rng.normal(-3, 8, n),
    ).clip(-30, 50)

    aggression = rng.beta(2, 3, size=n)
    heavy = np.isin(vt, ["delivery_van", "heavy_truck"])
    payload = np.where(heavy, rng.beta(3, 2, n), np.where(vt == "pickup", rng.beta(1.5, 3, n), rng.beta(1, 5, n))) * 100

    def since(scale):
        """km since a service: most owners keep up, some fall far behind."""
        return np.minimum(rng.exponential(scale, n), odometer)

    oil_interval = np.select([powertrain == "diesel", powertrain == "hybrid"], [15000, 10000], 9000)
    ice = powertrain != "electric"

    # Fleets service heavy vehicles' brakes and tires more often (they wear faster).
    wear_service = np.select([vt == "heavy_truck", vt == "delivery_van"], [0.4, 0.7], 1.0)
    km_on_tires = np.minimum(since(18000) * wear_service, odometer)
    tire_age = np.minimum(km_on_tires / annual * 12 + rng.uniform(0, 14, n), age * 12 + 1)

    df = pd.DataFrame({
        "vehicle_type": vt,
        "powertrain": powertrain,
        "climate": climate,
        "vehicle_age_years": age.round(1),
        "odometer_km": odometer.round(-1),
        "driving_aggression": aggression.round(2),
        "payload_pct": payload.round(0),
        "ambient_temp_c": ambient.round(1),
        "dc_fast_charge_pct": np.where(powertrain == "electric", rng.beta(2, 4, n) * 100, np.nan).round(0),
        "km_since_oil_change": np.where(ice, np.minimum(rng.exponential(oil_interval * 0.5), odometer), np.nan).round(-1),
        "km_since_coolant_service": np.where(ice, since(75000), np.nan).round(-1),
        "km_since_brake_service": (since(20000) * wear_service).round(-1),
        "km_since_trans_service": np.where(ice, since(70000), np.nan).round(-1),
        "km_since_tuneup": np.where(ice, since(40000), np.nan).round(-1),
        "km_on_tires": km_on_tires.round(-1),
        "tire_age_months": tire_age.round(0),
        "battery_12v_age_months": np.minimum(rng.exponential(22, n), age * 12).round(0),
    })
    df = add_operating_point(df, rng)
    return df


def add_operating_point(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Speed / rpm / load at the moment the snapshot was taken."""
    n = len(df)
    pt = df["powertrain"].to_numpy()
    aggr = df["driving_aggression"].to_numpy()
    payload = df["payload_pct"].to_numpy()

    speed = np.where(rng.random(n) < 0.15, rng.uniform(0, 5, n), rng.gamma(4, 18, n)).clip(0, 160)
    rpm_ice = 750 + speed * 19 * (1 + 0.35 * aggr) + rng.normal(0, 150, n)
    rpm_ice = np.where(pt == "diesel", rpm_ice * 0.7, rpm_ice).clip(600, 6000)
    rpm_ev = speed * 95 + rng.normal(0, 60, n)
    rpm = np.where(pt == "electric", rpm_ev.clip(0, 16000), rpm_ice)

    load = 18 + 0.28 * speed + 0.25 * payload + 18 * aggr + rng.normal(0, 7, n)
    df = df.copy()
    df["speed_kph"] = speed.round(0)
    df["engine_rpm"] = rpm.round(-1)
    df["engine_load_pct"] = load.clip(3, 100).round(0)
    return df


# ---------------------------------------------------------------------------
# 2. Hidden component health (1 = like new, 0 = failing)
# ---------------------------------------------------------------------------
def _f(df, col, fill=0.0):
    return df[col].fillna(fill).to_numpy(dtype=float) if col in df else np.full(len(df), fill)


def compute_health(df: pd.DataFrame, rng: np.random.Generator | None = None, defects: bool = True) -> pd.DataFrame:
    """Return hidden health 0..1 for each component, plus remaining life in km.

    `defects=False` turns off random sudden defects, which gives smooth,
    deterministic curves for the what-if forecast in the app.
    """
    rng = rng or np.random.default_rng(0)
    schema = load_schema()
    n = len(df)
    vt = df["vehicle_type"].to_numpy()
    pt = df["powertrain"].to_numpy()
    mass = np.array([schema["vehicle_types"].get(v, {"mass_kg": 1600})["mass_kg"] for v in vt], dtype=float)
    annual = np.array([schema["vehicle_types"].get(v, {"annual_km": 16000})["annual_km"] for v in vt], dtype=float)

    age = _f(df, "vehicle_age_years")
    odo = _f(df, "odometer_km")
    aggr = _f(df, "driving_aggression", 0.4)
    payload = _f(df, "payload_pct") / 100
    ambient = _f(df, "ambient_temp_c", 18)
    hot = np.clip((ambient - 22) / 15, 0, 1)        # 0 at 22 °C, 1 at 37 °C+

    stress = 1 + 0.6 * aggr + 0.5 * payload
    # Commercial vehicles use heavier-duty parts rated for far more km.
    duty = np.select([vt == "heavy_truck", vt == "delivery_van"], [3.0, 1.6], 1.0)

    def defect(base_p, size_lo=0.3, size_hi=0.8):
        if not defects:
            return np.zeros(n)
        hit = rng.random(n) < base_p
        return hit * rng.uniform(size_lo, size_hi, n)

    h, life = {}, {}  # life = km from "new" to "failed" under current usage

    # Engine oil: degrades with km since change; heat + hard driving accelerate it.
    oil_interval = np.select([pt == "diesel", pt == "hybrid"], [15000, 10000], 9000)
    life["engine_oil"] = 2.4 * oil_interval / (stress * (1 + 0.3 * hot))
    # Random defects (leaks, worn pump, coolant contamination) are only visible in the
    # sensors — this is what forces the model to read oil pressure/temp, not just the odometer.
    h["engine_oil"] = 1 - _f(df, "km_since_oil_change") / life["engine_oil"] - defect(0.02 + 0.002 * age, 0.4, 0.9)

    # Cooling: old coolant + aging pump/hoses; heat and towing stress it; random leaks.
    life["cooling_system"] = duty * 260000 / (stress * (1 + 0.5 * hot))
    wear = 0.55 * _f(df, "km_since_coolant_service") + 0.22 * odo
    h["cooling_system"] = 1 - wear / life["cooling_system"] - defect(0.015 + 0.003 * age)

    # 12V battery: calendar aging, much faster in heat.
    life_months = 66 * (1 - 0.35 * hot)
    life["battery_12v"] = life_months / 12 * annual
    h["battery_12v"] = 1 - _f(df, "battery_12v_age_months") / life_months - defect(0.015, 0.2, 0.6)

    # High-voltage battery: state of health falls with age, km, heat, fast charging.
    fast = _f(df, "dc_fast_charge_pct") / 100
    km_factor = np.where(pt == "hybrid", 0.25, 0.4)
    soh = 100 - 1.1 * age * (1 + 0.6 * hot) - odo / 10000 * km_factor * (1 + 1.2 * fast)
    soh = soh - defect(0.01, 5, 15) * 1.0
    h["hv_battery"] = (soh - 68) / 32
    soh_loss_per_km = 1.1 * (1 + 0.6 * hot) / annual + km_factor * (1 + 1.2 * fast) / 10000
    life["hv_battery"] = 32 / soh_loss_per_km

    # Brakes: pads wear by km; heavier, aggressive, loaded vehicles wear faster;
    # regenerative braking (EV/hybrid) saves pads.
    regen = np.select([pt == "electric", pt == "hybrid"], [0.4, 0.55], 1.0)
    pad_wear = 0.11 * (mass / 1500) ** 0.6 * (1 + 0.9 * aggr) * (1 + 0.4 * payload) * regen  # mm per 1000 km
    years_since_brake = _f(df, "km_since_brake_service") / annual
    pad = 12 - _f(df, "km_since_brake_service") / 1000 * pad_wear
    moisture = 0.4 + 0.75 * years_since_brake + (rng.normal(0, 0.25, n) if defects else 0)
    h["brakes"] = (pad - 2) / 10 - 0.10 * np.clip(moisture - 3, 0, None)
    stuck = defect(0.015, 0.4, 0.8)  # e.g. a sticking caliper eating one pad
    pad = pad - stuck * 10
    h["brakes"] = h["brakes"] - stuck
    life["brakes"] = 10 / pad_wear * 1000

    # Tires: tread wears with km; rubber also ages; slow leaks happen.
    tread_wear = 0.12 * (mass / 1500) ** 0.5 * (1 + 0.8 * aggr) * (1 + 0.25 * hot) * np.where(pt == "electric", 1.2, 1)
    tread = 9 - _f(df, "km_on_tires") / 1000 * tread_wear
    h_age = 1 - _f(df, "tire_age_months") / 130
    h["tires"] = np.minimum((tread - 1.6) / 7.4, h_age) - defect(0.01, 0.3, 0.6)
    life["tires"] = 7.4 / tread_wear * 1000

    # Transmission: fluid age + total km; towing/payload is the big stressor.
    life["transmission"] = duty * 300000 / (stress * (1 + 0.8 * payload))
    wear = 0.6 * _f(df, "km_since_trans_service") + 0.25 * odo
    h["transmission"] = 1 - wear / life["transmission"] - defect(0.005 + 0.002 * age)

    # Fuel & ignition: spark plugs (gas) / injectors (diesel).
    ign_life = np.where(pt == "diesel", 260000, 150000)
    life["fuel_ignition"] = duty * ign_life / stress
    wear = _f(df, "km_since_tuneup") + 0.08 * odo
    h["fuel_ignition"] = 1 - wear / life["fuel_ignition"] - defect(0.015)

    out = pd.DataFrame(index=df.index)
    for comp in component_names():
        out[f"_health_{comp}"] = np.clip(h[comp], 0, 1)
        out[f"_life_{comp}"] = life[comp]
    # Extra hidden quantities sensors are derived from
    out["_pad_mm"] = np.clip(pad, 0, 12)
    out["_brake_moisture"] = np.clip(moisture, 0, 8)
    out["_tread_mm"] = np.clip(tread, 0, 9)
    out["_soh"] = np.clip(soh, 40, 100)
    return out


# ---------------------------------------------------------------------------
# 3. Sensor readings (what the car actually reports, with noise + gaps)
# ---------------------------------------------------------------------------
# Probability that a vehicle simply doesn't report a sensor (no sensor fitted,
# telematics gap). The model must cope with these blanks.
OPTIONAL_SENSORS = {
    "brake_pad_mm": 0.45, "tire_tread_mm": 0.30, "brake_fluid_moisture_pct": 0.35,
    "vibration_rms_g": 0.20, "hv_cell_spread_mv": 0.10, "fuel_trim_ltft_pct": 0.10,
    "exhaust_gas_temp_c": 0.15, "cranking_voltage_v": 0.15,
}


def generate_sensors(df: pd.DataFrame, health: pd.DataFrame, rng: np.random.Generator, missing: bool = True,
                     noise: float = 1.0) -> pd.DataFrame:
    """noise=0 gives the 'expected' reading with no random scatter (used for smooth forecasts)."""
    n = len(df)
    pt = df["powertrain"].to_numpy()
    ice = pt != "electric"
    hv = np.isin(pt, ["hybrid", "electric"])
    amb = _f(df, "ambient_temp_c", 18)
    load = _f(df, "engine_load_pct", 40)
    rpm = _f(df, "engine_rpm", 1500)
    payload = _f(df, "payload_pct")
    d = {c: 1 - health[f"_health_{c}"].to_numpy() for c in component_names()}  # degradation 0..1
    N = lambda s: rng.normal(0, s * noise, n) if noise > 0 else np.zeros(n)  # noqa: E731
    pois = lambda lam: rng.poisson(lam) if noise > 0 else np.round(lam)  # noqa: E731
    rand = lambda: rng.random(n) if noise > 0 else np.ones(n)  # noqa: E731

    coolant = 88 + 0.06 * load + 0.12 * (amb - 20) + d["cooling_system"] ** 2 * 45 + N(2)
    oil_temp = coolant + 8 + 0.08 * load + d["engine_oil"] * 14 + N(3)
    oil_press = 110 + 0.065 * rpm - d["engine_oil"] ** 1.5 * 120 - 0.8 * np.clip(oil_temp - 105, 0, None) + N(15)

    resting_v = 12.72 - d["battery_12v"] * 1.15 - 0.006 * np.clip(10 - amb, 0, None) + N(0.05)
    crank_v = 10.6 - d["battery_12v"] ** 1.3 * 2.2 - 0.03 * np.clip(5 - amb, 0, None) + N(0.15)

    soh = health["_soh"].to_numpy() + N(1.5)
    cell_spread = 8 + d["hv_battery"] ** 2 * 90 + N(3)
    fast = _f(df, "dc_fast_charge_pct") / 100
    hv_temp = 0.5 * amb + 16 + 0.1 * load + d["hv_battery"] * 10 + 8 * fast + N(2)

    leak = (rand() < 0.03 + 0.25 * d["tires"] ** 3) * rng.uniform(25, 90, n)
    p_min = 236 - 0.7 * (20 - amb) - leak + N(6)
    p_spread = np.abs(N(5)) + leak * 0.9

    trans_temp = 74 + 0.25 * load + 0.2 * payload + 0.1 * amb + d["transmission"] ** 2 * 45 + N(3)
    slips = pois(d["transmission"] ** 3 * 25)
    misfires = pois(d["fuel_ignition"] ** 3 * 40 + 0.2)
    ltft = N(2.5) + d["fuel_ignition"] ** 2 * 12 * (rng.choice([-1, 1], n) if noise > 0 else 1)
    egt = 330 + 3.2 * load + d["fuel_ignition"] * 140 + N(20)
    vib = 0.04 + d["fuel_ignition"] ** 2 * 0.22 * ice + d["tires"] ** 2 * 0.12 + d["transmission"] ** 2 * 0.10 * ice + np.abs(N(0.01))
    dtc = pois(0.1 + sum((d[c] ** 4) * 1.6 for c in d))

    s = pd.DataFrame(index=df.index)
    s["coolant_temp_c"] = np.where(ice, coolant, np.nan)
    s["oil_temp_c"] = np.where(ice, oil_temp, np.nan)
    s["oil_pressure_kpa"] = np.where(ice, np.clip(oil_press, 20, 600), np.nan)
    s["battery_12v_resting_v"] = resting_v.clip(11, 13.2)
    s["cranking_voltage_v"] = crank_v.clip(7, 12)
    s["hv_battery_soh_pct"] = np.where(hv, soh.clip(40, 100), np.nan)
    s["hv_cell_spread_mv"] = np.where(hv, cell_spread.clip(0, 150), np.nan)
    s["hv_battery_temp_c"] = np.where(hv, hv_temp, np.nan)
    s["brake_pad_mm"] = (health["_pad_mm"].to_numpy() + N(0.4)).clip(0, 13)
    s["brake_fluid_moisture_pct"] = (health["_brake_moisture"].to_numpy() + N(0.2)).clip(0, 8)
    s["tire_tread_mm"] = (health["_tread_mm"].to_numpy() + N(0.3)).clip(0, 10)
    s["tire_pressure_min_kpa"] = p_min.clip(100, 300)
    s["tire_pressure_spread_kpa"] = p_spread.clip(0, 120)
    s["transmission_temp_c"] = np.where(ice, trans_temp, np.nan)
    s["trans_slip_events"] = np.where(ice, slips, np.nan)
    s["misfire_count"] = np.where(ice, misfires, np.nan)
    s["fuel_trim_ltft_pct"] = np.where(ice, ltft.clip(-25, 25), np.nan)
    s["exhaust_gas_temp_c"] = np.where(pt == "diesel", egt, np.nan)
    s["vibration_rms_g"] = vib.clip(0, 0.6)
    s["dtc_count"] = dtc.clip(0, 15)

    if missing:
        for col, p in OPTIONAL_SENSORS.items():
            s.loc[rng.random(n) < p, col] = np.nan
    return s.round(3)


# ---------------------------------------------------------------------------
# 4. Labels: what fails in the next FAILURE_HORIZON_KM, and remaining life
# ---------------------------------------------------------------------------
def generate_labels(df: pd.DataFrame, health: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    schema = load_schema()
    pt = df["powertrain"].to_numpy()
    n = len(df)
    out = pd.DataFrame(index=df.index)
    ruls = []
    for comp, info in schema["components"].items():
        applies = np.isin(pt, info["powertrains"])
        hc = health[f"_health_{comp}"].to_numpy()
        # Failure becomes likely as health drops below ~0.2, but it's random —
        # a worn part doesn't fail on an exact schedule.
        p = _sigmoid(14 * (0.18 - hc)) + 0.003
        fail = (rng.random(n) < p).astype(float)
        out[target_name(comp)] = np.where(applies, fail, np.nan)

        rul = hc * health[f"_life_{comp}"].to_numpy()
        ruls.append(np.where(applies, rul, np.inf))
    rul_km = np.min(np.vstack(ruls), axis=0) * rng.lognormal(0, 0.15, n)
    out["rul_km"] = np.clip(rul_km, 0, 80000).round(-1)
    return out


def simulate_from_config(cfg: pd.DataFrame, rng: np.random.Generator, missing: bool = True,
                         defects: bool = True, labels: bool = True, noise: float = 1.0) -> pd.DataFrame:
    health = compute_health(cfg, rng, defects=defects)
    sensors = generate_sensors(cfg, health, rng, missing=missing, noise=noise)
    parts = [cfg.reset_index(drop=True), sensors.reset_index(drop=True)]
    if labels:
        parts.append(generate_labels(cfg, health, rng).reset_index(drop=True))
    hidden = health[[c for c in health.columns if c.startswith("_health_")]]
    parts.append(hidden.reset_index(drop=True))
    return pd.concat(parts, axis=1)


def generate_fleet(n: int = 30000, seed: int = 42, vehicle_type: str | None = None) -> pd.DataFrame:
    """One row = one vehicle snapshot, with labels for what happens next."""
    rng = np.random.default_rng(seed)
    cfg = sample_configs(n, rng, vehicle_type=vehicle_type)
    df = simulate_from_config(cfg, rng)
    df.insert(0, "vehicle_id", [f"V{seed % 1000:03d}-{i:05d}" for i in range(n)])
    return df


def project_forward(cfg_row: pd.Series, km_steps: np.ndarray, annual_km: float | None = None) -> pd.DataFrame:
    """What-if: drive this vehicle `km` further with NO maintenance.

    Advances every odometer-style counter and the calendar clock, then
    re-simulates sensors (without noise surprises) so the model can predict
    how risk grows.
    """
    schema = load_schema()
    annual = annual_km or schema["vehicle_types"][cfg_row["vehicle_type"]]["annual_km"]
    cfg_row = cfg_row[[c for c in CONFIG_COLS if c in cfg_row.index]]
    rows = []
    for km in km_steps:
        r = cfg_row.copy()
        years = km / annual
        for col in ["odometer_km", "km_since_oil_change", "km_since_coolant_service", "km_since_brake_service",
                    "km_since_trans_service", "km_since_tuneup", "km_on_tires"]:
            if col in r and pd.notna(r[col]):
                r[col] = r[col] + km
        r["vehicle_age_years"] = r["vehicle_age_years"] + years
        r["tire_age_months"] = r["tire_age_months"] + years * 12
        r["battery_12v_age_months"] = r["battery_12v_age_months"] + years * 12
        rows.append(r)
    cfg = pd.DataFrame(rows).reset_index(drop=True)
    out = simulate_from_config(cfg, np.random.default_rng(7), missing=False, defects=False,
                               labels=False, noise=0.0)
    out.insert(0, "km_ahead", km_steps)
    return out

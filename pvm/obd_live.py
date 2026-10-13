"""Reading live data from a car's OBD-II port.

HOW OBD-II WORKS (the short version)
------------------------------------
Every car sold in the US since 1996 has an OBD-II port under the dashboard.
Inside the car, the computers (ECUs) talk over a network called the CAN bus.
A cheap "ELM327" adapter (USB, Bluetooth or Wi-Fi) plugs into that port and
translates simple text commands from your computer into CAN messages.

You ask for a value by sending a *mode* and a *PID* (parameter ID) in hex:

    you send:   01 0C          (mode 01 = live data, PID 0C = engine rpm)
    car sends:  41 0C 1A F8    (41 = "reply to mode 01", then data bytes A=0x1A, B=0xF8)
    rpm = (256*A + B) / 4 = (256*26 + 248) / 4 = 1726 rpm

Mode 01 PIDs are standardised (SAE J1979), so they work on every car — but
they only cover emissions-related sensors. Oil pressure, transmission
temperature, tire pressures, etc. use manufacturer-specific "mode 22" PIDs
(see config/vehicle_models.yaml → custom_pids).

The python-OBD library handles the bytes and unit conversions for us.

This module provides interchangeable data *sources* with the same interface:
    ELM327Source      a real adapter (USB / Bluetooth serial port / Wi-Fi)
    SimulatedCarSource  a virtual car that drives a trip (no hardware needed)
    load_log()          a CSV recorded by a phone app (Torque, Car Scanner, OBD Fusion…)
plus LivePoller, which reads a source in the background, and helpers that
turn a stream of readings into the inputs the ML model expects.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Standard (mode 01) PIDs we read, mapped to our column names.
# Columns that match a feature in vehicle_schema.yaml feed the ML model
# directly; the others are shown live and used by the personal baseline.
# ---------------------------------------------------------------------------
STANDARD_PIDS = [
    # column               python-OBD command             label                       unit
    ("engine_rpm",          "RPM",                       "Engine speed",              "rpm"),
    ("speed_kph",           "SPEED",                     "Vehicle speed",             "km/h"),
    ("engine_load_pct",     "ENGINE_LOAD",               "Engine load",               "%"),
    ("coolant_temp_c",      "COOLANT_TEMP",              "Coolant temp",              "°C"),
    ("oil_temp_c",          "OIL_TEMP",                  "Oil temp",                  "°C"),
    ("intake_temp_c",       "INTAKE_TEMP",               "Intake air temp",           "°C"),
    ("ambient_temp_c",      "AMBIANT_AIR_TEMP",          "Ambient temp",              "°C"),
    ("module_voltage_v",    "CONTROL_MODULE_VOLTAGE",    "System voltage",            "V"),
    ("fuel_trim_stft_pct",  "SHORT_FUEL_TRIM_1",         "Short-term fuel trim",      "%"),
    ("fuel_trim_ltft_pct",  "LONG_FUEL_TRIM_1",          "Long-term fuel trim",       "%"),
    ("throttle_pct",        "THROTTLE_POS",              "Throttle",                  "%"),
    ("maf_gps",             "MAF",                       "Mass air flow",             "g/s"),
    ("intake_pressure_kpa", "INTAKE_PRESSURE",           "Intake manifold pressure",  "kPa"),
    ("timing_advance_deg",  "TIMING_ADVANCE",            "Timing advance",            "°"),
    ("fuel_level_pct",      "FUEL_LEVEL",                "Fuel level",                "%"),
    ("run_time_s",          "RUN_TIME",                  "Engine run time",           "s"),
    ("km_since_dtc_clear",  "DISTANCE_SINCE_DTC_CLEAR",  "Distance since codes cleared", "km"),
    ("hv_soc_pct",          "HYBRID_BATTERY_REMAINING",  "Hybrid battery charge",     "%"),
]
SIGNAL_META = {c: {"label": lbl, "unit": u} for c, _, lbl, u in STANDARD_PIDS}
SIGNAL_META.update({
    "odometer_km": {"label": "Odometer", "unit": "km"},
    "dtc_count": {"label": "Trouble codes", "unit": ""},
    "battery_12v_resting_v": {"label": "12V resting voltage", "unit": "V"},
    "cranking_voltage_v": {"label": "Cranking voltage", "unit": "V"},
})

MISFIRE_CODES = re.compile(r"^P030\d$|^P031[0-2]$")


class ObdSource:
    """Common interface for every data source."""
    name = "source"

    def connect(self) -> None: ...
    def read(self) -> dict: ...                      # one sample: {column: value}
    def read_dtcs(self) -> list[tuple[str, str]]: return []
    def read_vin(self) -> str | None: return None
    def supported(self) -> list[str]: return []
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Real hardware
# ---------------------------------------------------------------------------
class ELM327Source(ObdSource):
    """A real ELM327 adapter.

    port examples:
        None                        auto-detect (USB / paired Bluetooth)
        "COM4"                      Windows
        "/dev/ttyUSB0", "/dev/rfcomm0", "/dev/tty.OBDII-SPPDev"   Linux / macOS
        "socket://192.168.0.10:35000"   most Wi-Fi adapters
    """
    name = "ELM327 adapter"

    def __init__(self, port: str | None = None, baudrate: int | None = None, timeout: float = 2.0,
                 custom_pids: list[dict] | None = None):
        self.port, self.baudrate, self.timeout = port or None, baudrate, timeout
        self.custom_pids = custom_pids or []
        self.conn = None
        self._cmds: dict[str, object] = {}

    @staticmethod
    def list_ports() -> list[str]:
        import obd
        return obd.scan_serial()

    def connect(self) -> None:
        import obd
        obd.logger.setLevel(obd.logging.WARNING)
        baud = self.baudrate
        if baud is None and self.port and self.port.startswith("socket://"):
            baud = 38400  # Wi-Fi adapters ignore baud rate, but setting it skips a slow auto-detect that fails over TCP
        self.conn = obd.OBD(self.port, baudrate=baud, fast=False, timeout=self.timeout)
        if not self.conn.is_connected():
            status = self.conn.status()
            self.conn.close()
            raise ConnectionError(f"Could not talk to the car ({status}). Check the adapter is plugged in, "
                                  "the ignition is ON, and the port is right.")
        for col, cmd_name, *_ in STANDARD_PIDS:
            cmd = obd.commands[cmd_name]
            if self.conn.supports(cmd):
                self._cmds[col] = cmd
        # Odometer (PID A6) is newer than python-OBD's table, so define it ourselves:
        # 4 data bytes, value = (A·2²⁴ + B·2¹⁶ + C·2⁸ + D) / 10 km.
        odo = obd.OBDCommand("ODOMETER", "Odometer", b"01A6", 6,
                             lambda msgs: int.from_bytes(bytes(msgs[0].data[2:6]), "big") / 10)
        if not self.conn.query(odo, force=True).is_null():
            self._cmds["odometer_km"] = (odo, True)
        for spec in self.custom_pids:
            self._cmds[spec["column"]] = (self._custom_command(spec), True)

    @staticmethod
    def _custom_command(spec: dict):
        """Build a manufacturer-specific (mode 22-style) command from a YAML spec."""
        import obd
        formula = compile(spec["formula"], spec["column"], "eval")

        def decoder(msgs):
            data = list(msgs[0].data)
            return float(eval(formula, {"__builtins__": {}}, {"B": data}))  # formula comes from your own YAML

        return obd.OBDCommand(spec["column"].upper(), spec["column"], spec["command"].encode(),
                              int(spec.get("bytes", 0)), decoder, header=spec.get("header", "7E0").encode())

    def supported(self) -> list[str]:
        return list(self._cmds)

    def read(self) -> dict:
        out = {}
        for col, cmd in self._cmds.items():
            force = False
            if isinstance(cmd, tuple):
                cmd, force = cmd
            r = self.conn.query(cmd, force=force)
            if r.is_null():
                continue
            v = r.value
            out[col] = float(v.magnitude) if hasattr(v, "magnitude") else float(v)
        return out

    def read_dtcs(self) -> list[tuple[str, str]]:
        import obd
        r = self.conn.query(obd.commands.GET_DTC)
        return [] if r.is_null() else [(c, d or "") for c, d in r.value]

    def read_vin(self) -> str | None:
        import obd
        r = self.conn.query(obd.commands.VIN, force=True)
        if r.is_null():
            return None
        v = r.value.decode("ascii", "ignore") if isinstance(r.value, (bytes, bytearray)) else str(r.value)
        v = re.sub(r"[^A-Z0-9]", "", v.upper())
        return v[-17:] if len(v) >= 17 else None

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()


# ---------------------------------------------------------------------------
# Virtual car for demos and testing
# ---------------------------------------------------------------------------
FAULTS = {
    "none": "No fault",
    "overheating": "Cooling system: thermostat/water pump failing (runs hot)",
    "weak_battery": "Weak 12V battery",
    "charging": "Alternator undercharging",
    "lean": "Vacuum leak (lean, high fuel trims)",
    "misfire": "Cylinder 2 misfire",
}
DEMO_VIN = "5XXGT4L37JG000001"  # valid-format 2018 Kia Optima VIN (serial is fictional)


class SimulatedCarSource(ObdSource):
    """A virtual car that warms up and drives a trip, reporting the same PIDs a real car would.

    It starts with the key on and engine off (so you can see resting battery
    voltage), cranks, then drives a varied cycle. Optional faults change the
    readings the way the real failure would.
    """
    name = "Simulated car"

    def __init__(self, fault: str = "none", ambient_c: float = 22.0, odometer_km: float = 96000,
                 seed: int = 0, vin: str = DEMO_VIN):
        self.fault, self.ambient, self.odo, self.vin = fault, ambient_c, odometer_km, vin
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.coolant = ambient_c
        self.speed = 0.0
        self.ltft = {"lean": 14.0, "misfire": 6.0}.get(fault, 1.5)

    def supported(self) -> list[str]:
        return [c for c, *_ in STANDARD_PIDS if c not in ("hv_soc_pct", "oil_temp_c")] + ["odometer_km"]

    def read(self) -> dict:
        t, rng, f = self.t, self.rng, self.fault
        self.t += 1
        weak = f == "weak_battery"
        resting_v = 12.05 if weak else 12.62
        if t < 4:                                   # key on, engine off
            return {"engine_rpm": 0.0, "speed_kph": 0.0, "module_voltage_v": resting_v + rng.normal(0, 0.01),
                    "coolant_temp_c": self.coolant, "ambient_temp_c": self.ambient, "intake_temp_c": self.ambient,
                    "run_time_s": 0.0, "fuel_level_pct": 58.0, "odometer_km": self.odo,
                    "km_since_dtc_clear": 1240.0, "fuel_trim_ltft_pct": self.ltft}
        if t == 4:                                  # cranking
            return {"engine_rpm": 220.0, "speed_kph": 0.0, "module_voltage_v": (8.7 if weak else 10.4),
                    "coolant_temp_c": self.coolant, "ambient_temp_c": self.ambient, "run_time_s": 0.0,
                    "odometer_km": self.odo}

        run = t - 4
        # Drive cycle: idle 20 s, then city/highway mix built from a few sine waves.
        target = 0 if run < 20 else max(0.0, 55 + 35 * np.sin(run / 40) + 20 * np.sin(run / 13))
        self.speed += np.clip(target - self.speed, -6, 4) + rng.normal(0, 0.8)
        self.speed = max(self.speed, 0.0)
        accel = target - self.speed
        gear_ratio = 110 if self.speed < 20 else 52 if self.speed < 45 else 33 if self.speed < 75 else 26
        rpm = max(720 + rng.normal(0, 15), self.speed * gear_ratio + rng.normal(0, 40))
        load = float(np.clip(22 + 0.25 * self.speed + 6 * max(accel, 0) + rng.normal(0, 3), 12, 95))
        thermostat = 112 if f == "overheating" else 91
        # Warm-up is sped up ~10x so a demo reaches operating temperature in about a minute.
        self.coolant += (thermostat + 0.04 * load - self.coolant) * 0.07 + rng.normal(0, 0.15)
        self.odo += self.speed / 3600
        if f == "misfire":
            rpm += rng.normal(0, 60)
        ltft = self.ltft + rng.normal(0, 0.4)
        stft = rng.normal(0, 3) + (5 if f == "lean" else 0)
        volts = (13.25 if f == "charging" else 14.25) + rng.normal(0, 0.05)
        return {
            "engine_rpm": round(rpm), "speed_kph": round(self.speed), "engine_load_pct": round(load, 1),
            "coolant_temp_c": round(self.coolant, 1), "intake_temp_c": round(self.ambient + 8 + 0.03 * load, 1),
            "ambient_temp_c": self.ambient, "module_voltage_v": round(volts, 2),
            "fuel_trim_stft_pct": round(stft, 1), "fuel_trim_ltft_pct": round(ltft, 1),
            "throttle_pct": round(load * 0.6, 1), "maf_gps": round(rpm * load / 4000, 2),
            "intake_pressure_kpa": round(25 + 0.7 * load, 1), "timing_advance_deg": round(18 - 0.08 * load, 1),
            "fuel_level_pct": 58.0, "run_time_s": float(run), "km_since_dtc_clear": 1240.0 + (self.odo % 1),
            "odometer_km": round(self.odo, 1),
        }

    def read_dtcs(self) -> list[tuple[str, str]]:
        warm = self.coolant > 80
        codes = {
            "overheating": [("P0217", "Engine Coolant Over Temperature Condition")] if self.coolant > 108 else [],
            "lean": [("P0171", "System Too Lean (Bank 1)")] if warm else [],
            "misfire": [("P0300", "Random/Multiple Cylinder Misfire Detected"),
                        ("P0302", "Cylinder 2 Misfire Detected")] if self.t > 30 else [],
            "charging": [("P0562", "System Voltage Low")] if self.t > 10 else [],
        }
        return codes.get(self.fault, [])

    def read_vin(self) -> str | None:
        return self.vin


# ---------------------------------------------------------------------------
# Logs recorded by phone apps
# ---------------------------------------------------------------------------
# Header patterns -> our column. Checked in order; first match wins.
LOG_ALIASES = [
    ("engine_rpm",          r"\brpm\b|engine\s*speed"),
    ("speed_kph",           r"(vehicle\s*)?speed|\bvss\b"),
    ("engine_load_pct",     r"(calculated\s*)?engine\s*load|\bload\b"),
    ("coolant_temp_c",      r"coolant"),
    ("oil_temp_c",          r"oil\s*temp"),
    ("intake_temp_c",       r"intake\s*air\s*temp|\biat\b"),
    ("ambient_temp_c",      r"ambient"),
    ("fuel_trim_ltft_pct",  r"long\s*term.*trim|trim.*long\s*term|\bltft"),
    ("fuel_trim_stft_pct",  r"short\s*term.*trim|trim.*short\s*term|\bstft"),
    ("module_voltage_v",    r"(control\s*)?module\s*voltage|battery\s*voltage|\bvoltage\b"),
    ("throttle_pct",        r"throttle"),
    ("maf_gps",             r"mass\s*air|\bmaf\b"),
    ("intake_pressure_kpa", r"manifold.*pressure|\bmap\b"),
    ("timing_advance_deg",  r"timing"),
    ("fuel_level_pct",      r"fuel\s*level"),
    ("odometer_km",         r"odometer"),
    ("run_time_s",          r"run\s*time"),
]
TIME_PATTERN = r"^(time|seconds?|timestamp|device\s*time|elapsed)"


def _to_metric(series: pd.Series, header: str) -> pd.Series:
    h = header.lower()
    if re.search(r"°f|\(f\)|degf|fahrenheit", h):
        return (series - 32) * 5 / 9
    if re.search(r"mph|mile", h):
        return series * 1.609344
    if re.search(r"psi", h):
        return series * 6.894757
    return series


def load_log(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise a phone-app OBD log into our column names (metric units, 't' in seconds)."""
    cols_lower = {c: c.lower().strip() for c in df.columns}
    # "Long" format (one row per PID reading), e.g. Car Scanner: SECONDS, PID, VALUE, UNITS
    if {"pid", "value"} <= set(cols_lower.values()):
        rename = {c: v for c, v in cols_lower.items()}
        d = df.rename(columns=rename)
        tcol = next((c for c in d.columns if re.match(TIME_PATTERN, c)), None)
        d["t"] = pd.to_numeric(d[tcol], errors="coerce").round() if tcol else np.arange(len(d))
        if "units" in d:
            d["pid"] = d["pid"].astype(str) + " (" + d["units"].astype(str) + ")"
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        df = d.pivot_table(index="t", columns="pid", values="value", aggfunc="mean").reset_index()
        cols_lower = {c: str(c).lower().strip() for c in df.columns}

    out = pd.DataFrame(index=df.index)
    tcol = next((c for c, l in cols_lower.items() if re.match(TIME_PATTERN, l) or l == "t"), None)
    if tcol is not None:
        t = pd.to_numeric(df[tcol], errors="coerce")
        if t.isna().all():
            t = pd.to_datetime(df[tcol], errors="coerce").astype("int64") / 1e9
        out["t"] = t - t.min()
    else:
        out["t"] = np.arange(len(df), dtype=float)

    used = {tcol}
    for col, pattern in LOG_ALIASES:
        for c, l in cols_lower.items():
            if c in used:
                continue
            if col == "speed_kph" and re.search(r"engine\s*speed|rpm|gps", l):
                continue
            if col == "module_voltage_v" and re.search(r"o2|oxygen|lambda|sensor", l):
                continue
            if re.search(pattern, l):
                out[col] = _to_metric(pd.to_numeric(df[c], errors="coerce"), l)
                used.add(c)
                break
    return out.dropna(axis=1, how="all")


# ---------------------------------------------------------------------------
# Background polling
# ---------------------------------------------------------------------------
@dataclass
class PollerStatus:
    running: bool = False
    error: str | None = None
    samples: int = 0
    seconds_per_sample: float = 0.0


class LivePoller:
    """Reads a source on a background thread and keeps the recent history.

    A real ELM327 answers one PID at a time (~50–150 ms each), so reading 15
    PIDs takes about a second — that sets our sampling rate.
    """

    def __init__(self, source: ObdSource, interval_s: float = 1.0, history: int = 3600,
                 dtc_every_s: float = 15.0):
        self.source, self.interval, self.dtc_every = source, interval_s, dtc_every_s
        self.buffer: deque[dict] = deque(maxlen=history)
        self.dtcs: list[tuple[str, str]] = []
        self.status = PollerStatus()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self.status.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        t0, last_dtc = time.time(), -1e9
        while not self._stop.is_set():
            start = time.time()
            try:
                row = self.source.read()
                row["t"] = round(start - t0, 2)
                if start - last_dtc >= self.dtc_every:
                    self.dtcs = self.source.read_dtcs()
                    last_dtc = start
                row["dtc_count"] = float(len(self.dtcs))
                with self._lock:
                    self.buffer.append(row)
                self.status.samples += 1
                self.status.seconds_per_sample = time.time() - start
            except Exception as e:  # adapter unplugged, ignition off, …
                self.status.error = f"{type(e).__name__}: {e}"
                break
            self._stop.wait(max(0.0, self.interval - (time.time() - start)))
        self.status.running = False

    def history(self) -> pd.DataFrame:
        with self._lock:
            return pd.DataFrame(list(self.buffer))

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.source.close()


# ---------------------------------------------------------------------------
# From a stream of readings to model inputs
# ---------------------------------------------------------------------------
def battery_metrics(hist: pd.DataFrame) -> dict:
    """Derive 12V battery health indicators from system voltage.

    * Engine off, key on  -> the voltage is the battery's resting voltage.
    * The instant of cranking -> the starter motor draws ~150 A and voltage
      dips; how far it dips shows battery strength.
    Note: cranking lasts ~1 s. At one sample per second we may catch the
    bottom of the dip or miss it — a classic sampling-rate limitation.
    """
    out = {}
    if hist.empty or "module_voltage_v" not in hist or "engine_rpm" not in hist:
        return out
    v, rpm = hist["module_voltage_v"], hist["engine_rpm"].fillna(-1)
    # Engine off AND voltage below charging level (an alternator would push it above ~13.2 V).
    off = v[(rpm == 0) & (v < 13.0)]
    if len(off):
        out["battery_12v_resting_v"] = float(off.median())
        running = (rpm > 400).to_numpy()
        if running.any():
            first_run = int(running.argmax())
            # Look just before the engine caught: that's the cranking dip.
            window = v.iloc[max(0, first_run - 3): first_run + 1]
            if first_run > 0 and len(window) and window.min() < out["battery_12v_resting_v"] - 0.3:
                out["cranking_voltage_v"] = float(window.min())
    return out


def engine_state(hist: pd.DataFrame) -> dict:
    last = hist.iloc[-1] if len(hist) else pd.Series(dtype=float)
    running = bool(last.get("engine_rpm", 0) and last.get("engine_rpm", 0) > 400)
    warm = bool(last.get("coolant_temp_c", 0) and last.get("coolant_temp_c", 0) >= 75)
    return {"running": running, "warm": warm}


def snapshot(hist: pd.DataFrame, window: int = 10) -> dict:
    """Summarise recent readings into one value per signal.

    Uses the median of the last `window` samples taken with the engine
    running and warm (falls back to whatever is available). The median
    ignores one-off glitches better than the mean.
    """
    if hist.empty:
        return {}
    h = hist
    if "engine_rpm" in h:
        running = h[h["engine_rpm"] > 400]
        if "coolant_temp_c" in h:
            warm = running[running["coolant_temp_c"] >= 75]
            running = warm if len(warm) >= 5 else running
        h = running if len(running) else h
    recent = h.tail(window)
    snap = {c: float(recent[c].median()) for c in recent.columns if c != "t" and recent[c].notna().any()}
    if "dtc_count" in hist:
        snap["dtc_count"] = float(hist["dtc_count"].iloc[-1])
    if "odometer_km" in hist and hist["odometer_km"].notna().any():
        snap["odometer_km"] = float(hist["odometer_km"].dropna().iloc[-1])
    snap.update(battery_metrics(hist))
    return snap


# ---------------------------------------------------------------------------
# Trouble codes -> components
# ---------------------------------------------------------------------------
# A DTC means the car's own computer has already detected a fault, so it is
# stronger evidence than any wear prediction. Generic SAE code ranges:
DTC_COMPONENTS = [
    (r"^P0(11[5-9]|12[58]|21[78]|48[0-3])", "cooling_system"),   # coolant temp sensor/thermostat/overheat/fan
    (r"^P05(2[0-4])|^P1326", "engine_oil"),                      # oil pressure sensor; Kia/Hyundai bearing-wear (KSDS)
    (r"^P0(30\d|31[0-2]|35\d|36[0-2]|17[1-5]|20[0-8]|26[1-9]|27\d|28\d)", "fuel_ignition"),  # misfire, coils, trims, injectors
    (r"^P0(56[0-3]|62[0-2])", "battery_12v"),                    # system voltage / generator (charging)
    (r"^P0(7\d\d)|^P08[0-9]\d", "transmission"),
    (r"^C0", "brakes"),                                          # chassis codes are mostly ABS/brakes
    (r"^P0A[0-9A-F]{2}|^P0B[0-9A-F]{2}|^P0C[0-9A-F]{2}", "hv_battery"),  # hybrid/EV system
]


def dtc_component(code: str) -> str | None:
    code = code.upper().strip()
    for pattern, comp in DTC_COMPONENTS:
        if re.match(pattern, code):
            return comp
    return None

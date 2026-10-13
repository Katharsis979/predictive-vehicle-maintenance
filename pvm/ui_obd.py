"""The "Live OBD" tab: connect to a car, watch live data, and diagnose it."""
from __future__ import annotations

import datetime as dt

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .baseline import MIN_ROWS, PersonalBaseline, VehicleStore, km_since_services, warm_running
from .obd_live import (FAULTS, MISFIRE_CODES, SIGNAL_META, ELM327Source, LivePoller, SimulatedCarSource,
                       dtc_component, engine_state, load_log, snapshot)
from .profiles import (adapt_row_to_profile, decode_vin, guess_vehicle_type, known_issue_alerts, load_models,
                       match_profile, profile_label, profile_vehicle_type, spec_checks)
from .schema import load_schema
from .simulate import CONFIG_COLS
from .ui import COMPONENTS, feat_meta, fmt, render_diagnosis

MI = 1.609344
SCHEMA = load_schema()
VTYPES = SCHEMA["vehicle_types"]
TILES = ["engine_rpm", "speed_kph", "coolant_temp_c", "engine_load_pct", "module_voltage_v",
         "fuel_trim_ltft_pct", "fuel_trim_stft_pct", "intake_temp_c", "fuel_level_pct", "odometer_km"]
# OBD readings that the fleet model uses directly
MODEL_SIGNALS = ["engine_rpm", "speed_kph", "engine_load_pct", "ambient_temp_c", "coolant_temp_c", "oil_temp_c",
                 "fuel_trim_ltft_pct", "battery_12v_resting_v", "cranking_voltage_v", "dtc_count"]
# (service, label, default "km ago" for a reasonably maintained car)
SERVICES = [("oil", "Oil change", 3000), ("spark_plugs", "Spark plugs", 40000), ("coolant", "Coolant service", 50000),
            ("brakes", "Brake service", 15000), ("transmission", "Transmission service", 50000),
            ("tires", "Tires installed", 20000)]


def _ss():
    return st.session_state


def _disconnect():
    p = _ss().pop("obd_poller", None)
    if p:
        p.stop()


def _on_connected(vin: str | None):
    _ss()["obd_vin"] = vin or ""
    _ss()["obd_session"] = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if vin:
        try:
            _ss()["obd_vin_info"] = decode_vin(vin, online=_ss().get("obd_online_vin", True))
        except ValueError:
            _ss()["obd_vin_info"] = None


def _current_history() -> pd.DataFrame:
    p = _ss().get("obd_poller")
    if p is not None:
        return p.history()
    return _ss().get("obd_log_hist", pd.DataFrame())


def _connection_panel():
    st.markdown("#### 1. Connect")
    src = st.segmented_control("Data source", ["Simulated car", "OBD-II adapter", "Phone app log"],
                               default="Simulated car", key="obd_src")
    connected = _ss().get("obd_poller") is not None

    if connected:
        p = _ss()["obd_poller"]
        st.success(f"Connected: **{p.source.name}** — reading {len(p.source.supported())} signals")
        if st.button("Disconnect", key="obd_disc"):
            _disconnect()
            st.rerun()
        return

    if src == "Simulated car":
        st.caption("A virtual 2018 Kia Optima that starts with the key on, cranks, warms up and drives. "
                   "Inject a fault to see how the readings and diagnosis change.")
        fault = st.selectbox("Fault to simulate", list(FAULTS), format_func=FAULTS.get, key="obd_fault")
        amb = st.slider("Outside temperature (°C)", -15, 40, 22, key="obd_amb")
        if st.button("▶️ Start the virtual car", type="primary", key="obd_go_sim"):
            source = SimulatedCarSource(fault=fault, ambient_c=amb, seed=int(np.random.default_rng().integers(1e6)))
            source.connect()
            _ss()["obd_poller"] = LivePoller(source, interval_s=1.0)
            _ss()["obd_poller"].start()
            _on_connected(source.read_vin())
            st.rerun()

    elif src == "OBD-II adapter":
        st.caption("Plug an ELM327 adapter into the port under the dashboard and turn the ignition ON. "
                   "**Run the app on your own laptop** (`streamlit run app.py`): a cloud-hosted copy can't reach "
                   "hardware in your car.")
        kind = st.radio("Adapter type", ["USB / Bluetooth", "Wi-Fi"], horizontal=True, key="obd_kind")
        if kind == "Wi-Fi":
            st.caption("Join the adapter's Wi-Fi network first. Most use 192.168.0.10, port 35000.")
            host = st.text_input("Adapter address", "192.168.0.10:35000", key="obd_host")
            port = f"socket://{host.strip()}"
        else:
            if st.button("Scan for ports", key="obd_scan"):
                try:
                    _ss()["obd_ports"] = ELM327Source.list_ports()
                except Exception as e:
                    st.error(f"Port scan failed: {e}")
            ports = _ss().get("obd_ports", [])
            choice = st.selectbox("Port", ["Auto-detect"] + ports, key="obd_port_pick")
            manual = st.text_input("…or type a port (COM4, /dev/rfcomm0, /dev/tty.OBDII…)", key="obd_port_manual")
            port = manual.strip() or (None if choice == "Auto-detect" else choice)
            st.caption("Bluetooth adapters must be paired in your computer's settings first (PIN is often 1234 or 0000).")
        profile_key = _ss().get("obd_profile_key")
        custom = (load_models()["models"].get(profile_key, {}).get("custom_pids") or []) if profile_key else []
        if st.button("🔌 Connect", type="primary", key="obd_go_real"):
            with st.spinner("Talking to the adapter (this can take ~10 s)…"):
                try:
                    source = ELM327Source(port, custom_pids=custom)
                    source.connect()
                    vin = source.read_vin()
                except Exception as e:
                    st.error(str(e))
                    return
            _ss()["obd_poller"] = LivePoller(source, interval_s=1.0)
            _ss()["obd_poller"].start()
            _on_connected(vin)
            st.rerun()

    else:
        st.caption("Record a drive with a phone app (Torque Pro, Car Scanner, OBD Fusion…), export it as CSV, "
                   "and upload it here. Column names and °F/mph units are recognised automatically.")
        up = st.file_uploader("OBD log (.csv)", type="csv", key="obd_log_up")
        if up is not None:
            try:
                raw = pd.read_csv(up, sep=None, engine="python")
            except Exception:
                up.seek(0)
                raw = pd.read_csv(up)
            hist = load_log(raw)
            _ss()["obd_log_hist"] = hist
            found = [c for c in hist.columns if c != "t"]
            st.success(f"Read {len(hist):,} samples with {len(found)} recognised signals: "
                       + ", ".join(SIGNAL_META.get(c, {"label": c})["label"] for c in found))
            if "obd_session" not in _ss():
                _ss()["obd_session"] = f"log-{up.name}"


def _vehicle_panel():
    st.markdown("#### 2. Identify the vehicle")
    vin = st.text_input("VIN (17 characters — read from the car when supported)", key="obd_vin",
                        help="On the driver-side dashboard corner, door jamb sticker, or registration.")
    st.checkbox("Look up VIN details online (NHTSA)", value=True, key="obd_online_vin")
    if st.button("Decode VIN", key="obd_decode") and vin:
        try:
            _ss()["obd_vin_info"] = decode_vin(vin, online=_ss()["obd_online_vin"])
        except ValueError as e:
            st.error(str(e))
    info = _ss().get("obd_vin_info")
    if info:
        bits = [str(x) for x in (info.get("year"), info.get("make"), info.get("model")) if x]
        extra = ", ".join(str(x) for x in (info.get("displacement_l") and f"{info['displacement_l']} L",
                                          info.get("engine"), info.get("fuel")) if x)
        st.write(f"**{' '.join(bits) or 'Unknown vehicle'}** {('· ' + extra) if extra else ''} "
                 f"({info['source']})")
        if not info.get("check_digit_ok"):
            st.caption("⚠️ The VIN check digit doesn't match — double-check for typos "
                       "(outside North America the check digit isn't required).")

    models = load_models()["models"]
    matched = match_profile(info.get("make"), info.get("model"), info.get("year")) if info else None
    options = list(models) + [f"generic:{v}" for v in VTYPES]
    default = matched or (f"generic:{guess_vehicle_type(info)}" if info and info.get("model") else "kia_optima_jf")
    if _ss().get("obd_profile_auto") != default:  # re-select when a new VIN is decoded
        _ss()["obd_profile"] = default
        _ss()["obd_profile_auto"] = default
    pick = st.selectbox("Vehicle profile", options, key="obd_profile",
                        format_func=lambda k: profile_label(k) if k in models
                        else f"Generic {VTYPES[k.split(':')[1]]['label'].lower()}")
    if pick in models:
        _ss()["obd_profile_key"] = pick
        vtype, powertrain = profile_vehicle_type(pick)
        p = models[pick]
        if "spark_plugs_turbo" in (p.get("service_intervals_km") or {}):
            st.checkbox("Turbo engine", value=bool(info and info.get("turbo")), key="obd_turbo")
        st.checkbox("Severe driving (short trips, towing, extreme heat/cold, dusty roads)", key="obd_severe")
    else:
        _ss()["obd_profile_key"] = None
        vtype = pick.split(":")[1]
        powertrain = VTYPES[vtype]["powertrain"]
        st.caption("No specific profile for this car yet — using generic values. Add one in "
                   "`config/vehicle_models.yaml`.")
    vehicle_id = (vin or "").strip().upper() or ("demo-optima" if _ss().get("obd_src") == "Simulated car" else "my-car")
    return pick if pick in models else None, vtype, powertrain, vehicle_id, info


@st.fragment(run_every=1.0)
def _live_fragment():
    _live_view(live=True)


def _live_view(live: bool):
    hist = _current_history()
    p = _ss().get("obd_poller")
    if p is not None:
        s = p.status
        line = f"{s.samples:,} samples · {s.seconds_per_sample:.2f} s per sample"
        if s.error:
            st.error(f"Connection lost — {s.error}")
        else:
            st.caption(("🟢 live · " if live else "") + line)
    if hist.empty:
        st.info("Waiting for data…" if p is not None else "Connect a source above to see live data.")
        return
    last = hist.iloc[-1]
    state = engine_state(hist)
    st.caption(f"Engine: **{'running' if state['running'] else 'off'}**"
               + (f" · {'warm' if state['warm'] else 'warming up'}" if state["running"] else ""))
    shown = [c for c in TILES if c in hist and pd.notna(last.get(c))]
    cols = st.columns(5)
    for i, c in enumerate(shown):
        prev = hist[c].iloc[-6] if len(hist) > 6 else np.nan
        delta = None if pd.isna(prev) else f"{last[c] - prev:+.1f}"
        v = last[c] / MI if c in ("odometer_km",) and _ss().get("obd_units", "mi") == "mi" else last[c]
        label = SIGNAL_META.get(c, {"label": c})["label"]
        unit = "mi" if c == "odometer_km" and _ss().get("obd_units", "mi") == "mi" else SIGNAL_META.get(c, {}).get("unit", "")
        cols[i % 5].metric(label, f"{v:,.0f} {unit}" if abs(v) >= 100 else f"{v:,.1f} {unit}", delta)

    plot_opts = [c for c in hist.columns if c not in ("t", "dtc_count") and hist[c].notna().any()]
    default = [c for c in ["coolant_temp_c", "engine_rpm", "fuel_trim_ltft_pct", "module_voltage_v"] if c in plot_opts]
    chosen = st.multiselect("Signals to chart", plot_opts, default=default, key="obd_plot",
                            format_func=lambda c: SIGNAL_META.get(c, {"label": c})["label"])
    recent = hist.tail(600)
    gcols = st.columns(2)
    for i, c in enumerate(chosen):
        ch = alt.Chart(recent).mark_line(strokeWidth=2).encode(
            x=alt.X("t:Q", title="seconds"), y=alt.Y(f"{c}:Q", title=SIGNAL_META.get(c, {}).get("unit", ""),
                                                      scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("t:Q"), alt.Tooltip(f"{c}:Q", format=".2f")],
        ).properties(title=SIGNAL_META.get(c, {"label": c})["label"], height=170)
        gcols[i % 2].altair_chart(ch, width="stretch")

    dtcs = p.dtcs if p is not None else []
    if dtcs:
        st.error("**Trouble codes stored:** " + " · ".join(f"`{c}` {d}" for c, d in dtcs))
    elif p is not None:
        st.caption("No trouble codes stored.")


def _service_inputs(info_store: dict, odometer_km: float | None, units: str) -> tuple[dict, float | None]:
    """Collect maintenance history; remembered per vehicle."""
    f = MI if units == "mi" else 1.0
    st.caption("OBD can't tell when parts were last serviced, so enter the odometer reading at each service. "
               "They're saved for this vehicle, and 'distance since service' updates automatically as the "
               "odometer climbs.")
    c = st.columns(3)
    odo_default = odometer_km if odometer_km is not None else info_store.get("last_odometer_km", 96000)
    odo = c[0].number_input(f"Current odometer ({units})", 0.0, 2_000_000.0, float(round(odo_default / f)), 100.0,
                            key="obd_odo", disabled=odometer_km is not None,
                            help="Read from the car when it supports PID A6.") * f
    rec = dict(info_store.get("service_odometer_km") or {})
    for i, (svc, label, ago) in enumerate(SERVICES):
        cur = rec.get(svc)
        val = c[(i + 1) % 3].number_input(f"{label} at ({units})", 0.0, 2_000_000.0,
                                          float(round((cur if cur is not None else max(odo - ago, 0)) / f)),
                                          100.0, key=f"obd_svc_{svc}")
        rec[svc] = val * f
    return rec, odo


def render_obd_tab(model) -> None:
    st.subheader("🔌 Live OBD-II")
    store = VehicleStore()
    left, right = st.columns([1.1, 1])
    with left:
        _connection_panel()
    with right:
        profile_key, vtype, powertrain, vehicle_id, vin_info = _vehicle_panel()

    st.markdown("#### 3. Live data")
    st.segmented_control("Distance units", ["mi", "km"], default="mi", key="obd_units")
    if _ss().get("obd_poller") is not None:
        _live_fragment()
    else:
        _live_view(live=False)

    hist = _current_history()
    if hist.empty:
        return

    st.markdown("#### 4. AI diagnosis")
    units = _ss().get("obd_units") or "mi"
    info_store = store.load_info(vehicle_id)
    snap = snapshot(hist)
    with st.expander("Maintenance history & usage", expanded=not info_store):
        rec, odo = _service_inputs(info_store, snap.get("odometer_km"), units)
        c = st.columns(3)
        climate = c[0].selectbox("Climate", SCHEMA["climates"], index=1, key="obd_climate")
        aggr = c[1].slider("Driving aggression", 0.0, 1.0, float(info_store.get("driving_aggression", 0.4)), 0.05,
                           key="obd_aggr")
        payload = c[2].slider("Typical load / towing (%)", 0, 100, int(info_store.get("payload_pct", 10)), 5,
                              key="obd_payload")
        year = (vin_info or {}).get("year") or info_store.get("year")
        b12 = c[0].number_input("12V battery age (months)", 0, 200, int(info_store.get("battery_12v_age_months", 30)),
                                key="obd_b12")
        tire_age = c[1].number_input("Tire age (months)", 0, 200, int(info_store.get("tire_age_months", 24)),
                                     key="obd_tire_age")
        if st.button("💾 Save for this vehicle", key="obd_save_info"):
            info_store.update({"service_odometer_km": rec, "last_odometer_km": odo, "driving_aggression": aggr,
                               "payload_pct": payload, "battery_12v_age_months": b12, "tire_age_months": tire_age,
                               "profile": profile_key, "vehicle_type": vtype, "year": year, "vin": _ss().get("obd_vin")})
            store.save_info(vehicle_id, info_store)
            st.toast(f"Saved for {vehicle_id}")

    # ---- Build the model's input row from OBD readings + history ----
    age = max(dt.date.today().year - int(year), 0.5) if year else 6.0
    service = km_since_services({"service_odometer_km": rec}, odo)
    readings = {k: snap[k] for k in MODEL_SIGNALS if k in snap}
    if not len(warm_running(hist)):
        # Cold-engine temperatures and fuel trims say nothing about health — leave them out.
        for k in ("coolant_temp_c", "oil_temp_c", "fuel_trim_ltft_pct"):
            readings.pop(k, None)
        st.info("🌡️ The engine isn't warm yet, so temperature and fuel-trim readings are left out of the "
                "diagnosis. Keep it running until the coolant passes 75 °C for a full assessment.")
    if powertrain != "electric" and "misfire_count" not in readings and _ss().get("obd_poller") is not None:
        codes = [c for c, _ in _ss()["obd_poller"].dtcs]
        if any(MISFIRE_CODES.match(c) for c in codes):
            st.caption("Misfire codes present — the model sees them through the trouble-code count.")
    row = pd.DataFrame([{**{c: np.nan for c in CONFIG_COLS}, "vehicle_type": vtype, "powertrain": powertrain,
                         "climate": climate, "vehicle_age_years": age, "odometer_km": odo,
                         "driving_aggression": aggr, "payload_pct": payload,
                         "battery_12v_age_months": b12, "tire_age_months": tire_age, **service, **readings}])
    adapted, notes = adapt_row_to_profile(row, profile_key, severe=_ss().get("obd_severe", False),
                                          turbo=_ss().get("obd_turbo", False))
    sensor_feats = [f["name"] for f in SCHEMA["features"] if f["group"] == "sensor" and powertrain in f["powertrains"]]
    have = [s for s in sensor_feats if s in adapted and pd.notna(adapted[s].iloc[0])]
    st.caption(f"Your car supplied **{len(have)} of {len(sensor_feats)}** sensor inputs the model can use "
               f"({', '.join(feat_meta(s)['label'] for s in have) or 'none yet'}). The rest are treated as unknown — "
               "brake pads, tire tread, oil pressure and transmission temperature need manufacturer PIDs or a "
               "manual check.")
    if notes:
        with st.expander(f"Adjusted for {profile_label(profile_key)}"):
            for n in notes:
                st.markdown(f"- {n}")

    # ---- Gather every kind of evidence first ----
    dtcs = _ss()["obd_poller"].dtcs if _ss().get("obd_poller") else []
    codes = [c for c, _ in dtcs]
    issues = known_issue_alerts(profile_key, int(year) if year else None, codes)
    warm = len(warm_running(hist)) > 0
    checks = spec_checks(snap, profile_key, engine_running=engine_state(hist)["running"] or warm, warm=warm)
    stored = store.load_log(vehicle_id)
    baseline = store.load_baseline(vehicle_id)
    base_scores = baseline.score(hist.tail(300)) if baseline is not None and baseline.ready else pd.DataFrame()

    # ---- Findings: hard evidence (the car's own fault codes, out-of-range readings, drift) ----
    st.caption("Findings = things already wrong right now. Trouble codes mean the car's own computer detected a fault — "
               "that outranks any prediction.")
    n_bad_checks = sum(not c["ok"] for c in checks)
    n_drift = int((base_scores["z"].abs() >= 3).sum()) if not base_scores.empty else 0
    if dtcs or any(i["triggered"] for i in issues):
        st.markdown("### ⛔ Fault detected")
    elif n_bad_checks or n_drift:
        st.markdown("### ⚠️ Needs attention")
    else:
        st.markdown("### ✅ No current faults")
    found = False
    for code, desc in dtcs:
        comp = dtc_component(code)
        where = f" → **{COMPONENTS[comp]['label']}**" if comp else ""
        st.error(f"Trouble code `{code}` {desc}{where}" + (f"\n\n{COMPONENTS[comp]['action']}" if comp else ""))
        found = True
    for i in issues:
        if i["triggered"]:
            st.error(f"**{i['title']}** — code {', '.join(i['matched_codes'])} matches a known problem for this "
                     "model. Stop driving and contact a dealer.")
            found = True
    for c in checks:
        if not c["ok"]:
            st.warning(f"**{feat_meta(c['column'])['label']}** is {fmt(c['value'], c['column'])} "
                       f"(normal {c['min']}–{c['max']}). {c['note']}")
            found = True
    if not base_scores.empty:
        for _, r in base_scores[base_scores["z"].abs() >= 3].iterrows():
            st.warning(f"**{SIGNAL_META.get(r['signal'], {'label': r['signal']})['label']}** is "
                       f"{r['difference']:+.1f} {SIGNAL_META.get(r['signal'], {}).get('unit', '')} away from what this "
                       "car normally shows in the same conditions.")
            found = True
    if not found:
        st.success("No fault codes, no readings out of range" + (", nothing unusual for this car." if not
                                                                   base_scores.empty else "."))

    st.markdown("##### Predicted wear (AI model)")
    st.caption("Separate from current faults: how likely each component is to fail in the next 5,000 km, "
               "from mileage, service history and sensor patterns learned across a 40,000-vehicle fleet.")
    render_diagnosis(model, adapted, powertrain, key="obd", wear_wording=True, show_anomaly=False)

    # ---- Model-specific known issues ----
    if issues:
        st.markdown("#### Known issues for this model")
        for i in issues:
            box = st.error if i["triggered"] else st.info
            box(f"**{i['title']}**" + (f" — matching code {', '.join(i['matched_codes'])} is stored!" if i["triggered"]
                                       else "") + f"\n\n{i['detail']}\n\n*Source: {i['source']}*")
        st.caption("Check open recalls for your exact VIN at nhtsa.gov/recalls.")

    # ---- Range checks ----
    st.markdown("#### Reading checks")
    if checks:
        df = pd.DataFrame([{"Reading": feat_meta(c["column"])["label"], "Value": fmt(c["value"], c["column"]),
                            "Normal range": f"{c['min']} – {c['max']}", "Status": "✅" if c["ok"] else "⚠️",
                            "Why it matters": c["note"]} for c in checks])
        st.dataframe(df, hide_index=True, width="stretch")

    # ---- Personal baseline ----
    st.markdown("#### What's normal for *this* car")
    n_warm = len(warm_running(stored)) if not stored.empty else 0
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save this drive to the vehicle's history", key="obd_save_log"):
            store.append_log(vehicle_id, hist, session=_ss().get("obd_session", "session"))
            st.toast("Drive saved")
            st.rerun()
        st.caption(f"Saved history: {len(stored):,} samples ({n_warm:,} warm-engine). "
                   f"Needs {MIN_ROWS}+ warm samples to learn a baseline.")
    with c2:
        if st.button("🧠 Learn this car's normal", key="obd_fit_base", disabled=n_warm < MIN_ROWS):
            with st.spinner("Training one small model per signal…"):
                baseline = PersonalBaseline().fit(stored)
            store.save_baseline(vehicle_id, baseline)
            st.toast(f"Learned {len(baseline.models)} signals from {baseline.n_rows:,} samples")
            st.rerun()
    if not base_scores.empty:
        sc = base_scores.copy()
        sc["Signal"] = sc["signal"].map(lambda c: SIGNAL_META.get(c, {"label": c})["label"])
        sc["Status"] = np.where(sc["z"].abs() >= 3, "⚠️ unusual for this car",
                                np.where(sc["z"].abs() >= 2, "watch", "normal"))
        st.dataframe(sc[["Signal", "actual", "expected", "difference", "z", "Status"]]
                     .style.format({"actual": "{:.2f}", "expected": "{:.2f}", "difference": "{:+.2f}", "z": "{:+.1f}"}),
                     hide_index=True, width="stretch")
        st.caption("*Expected* = what this car normally shows at the same rpm, speed, load and temperature, "
                   "learned from its own saved drives. *z* = how many 'normal wobbles' away it is.")
    elif baseline is not None and baseline.ready:
        st.caption("Drive with a warm engine for a minute to compare against the baseline.")
    else:
        st.caption("Once trained, this compares every drive to the car's own history and flags drift "
                   "(e.g. running 6 °C hotter than usual) before it becomes a warning light.")

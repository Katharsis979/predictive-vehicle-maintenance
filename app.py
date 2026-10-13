"""Predictive Vehicle Maintenance — Streamlit dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from pvm.model import MaintenanceModel
from pvm.schema import component_names, features_for, load_schema, target_name
from pvm.simulate import (CONFIG_COLS, FAILURE_HORIZON_KM, compute_health, generate_fleet, project_forward,
                          sample_configs, simulate_from_config)
from pvm.train import load_or_train, print_report  # noqa: F401  (print_report handy in a REPL)

st.set_page_config(page_title="Predictive Vehicle Maintenance", page_icon="🚗", layout="wide")
SCHEMA = load_schema()
COMPONENTS = SCHEMA["components"]
VTYPES = SCHEMA["vehicle_types"]


@st.cache_resource(show_spinner=False)
def get_model() -> MaintenanceModel:
    bar = st.progress(0.0, text="First run: simulating 40,000 vehicles and training the models (~1 min)…")
    model = load_or_train(progress=lambda f, msg: bar.progress(min(f, 1.0), text=msg))
    bar.empty()
    return model


model = get_model()


def comp_label(target: str) -> str:
    return COMPONENTS.get(target.removeprefix("fail_"), {}).get("label", target)


def feat_meta(name: str) -> dict:
    return next((f for f in SCHEMA["features"] if f["name"] == name), {"label": name, "unit": ""})


def fmt(v, name: str) -> str:
    if pd.isna(v):
        return "—"
    unit = feat_meta(name).get("unit", "")
    v = float(v)
    s = f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"{s} {unit}".strip()


# =====================================================================
# Sidebar: describe the vehicle
# =====================================================================
def load_profile(vtype: str, seed: int) -> None:
    """Fill every input with a realistic random vehicle of this type."""
    cfg = sample_configs(1, np.random.default_rng(seed), vehicle_type=vtype)
    pt = cfg["powertrain"].iloc[0]
    st.session_state["climate"] = cfg["climate"].iloc[0]
    for f in features_for(pt):
        if f["name"] in cfg:
            st.session_state[f"in_{f['name']}"] = float(np.clip(cfg[f["name"]].iloc[0], f["min"], f["max"]))
    # Clear manual sensor values so they re-seed from this vehicle's simulated readings.
    for f in SCHEMA["features"]:
        if f["group"] == "sensor":
            st.session_state.pop(f"in_{f['name']}", None)
    st.session_state["missing_sensors"] = []


if "vtype" not in st.session_state:
    st.session_state["vtype"] = "sedan"
    st.session_state["seed"] = 5
    load_profile("sedan", 5)

with st.sidebar:
    st.header("🚙 Vehicle")
    vtype = st.selectbox("Vehicle type", list(VTYPES), format_func=lambda v: VTYPES[v]["label"], key="vtype",
                         on_change=lambda: load_profile(st.session_state["vtype"], st.session_state["seed"]))
    powertrain = VTYPES[vtype]["powertrain"]
    st.caption(f"Powertrain: **{powertrain}**")
    if st.button("🎲 Random vehicle of this type", width="stretch"):
        st.session_state["seed"] += 1
        load_profile(vtype, st.session_state["seed"])
        st.rerun()
    climate = st.selectbox("Climate", SCHEMA["climates"], key="climate")

    def slider(f):
        return st.slider(f"{f['label']} ({f['unit']})" if f["unit"] else f["label"], float(f["min"]), float(f["max"]),
                         step=float(f["step"]), key=f"in_{f['name']}", help=f.get("help") or None)

    st.subheader("Usage profile")
    inputs = {f["name"]: slider(f) for f in features_for(powertrain, "vehicle")}
    st.subheader("Maintenance history")
    inputs |= {f["name"]: slider(f) for f in features_for(powertrain, "service")}
    with st.expander("Operating point (when the reading was taken)"):
        inputs |= {f["name"]: slider(f) for f in features_for(powertrain, "operating")}

config = pd.DataFrame([{"vehicle_type": vtype, "powertrain": powertrain, "climate": climate, **inputs}])
for c in CONFIG_COLS:  # columns this powertrain doesn't have -> blank
    if c not in config:
        config[c] = np.nan


# =====================================================================
st.title("🚗 Predictive Vehicle Maintenance")
st.caption(f"Machine-learning models predict which components are likely to fail in the next "
           f"{FAILURE_HORIZON_KM:,} km, estimate remaining life, and flag unusual readings.")
tab_live, tab_forecast, tab_fleet, tab_train, tab_how = st.tabs(
    ["🩺 Diagnostics", "📈 Forecast", "🚚 Fleet analysis", "🧪 Train on your data", "🧠 How the AI works"])

# =====================================================================
# TAB 1 — Live diagnostics
# =====================================================================
with tab_live:
    sensor_feats = features_for(powertrain, "sensor")
    mode = st.segmented_control("Sensor readings", ["Simulated from this vehicle", "Enter manually"],
                                default="Simulated from this vehicle", key="sensor_mode")
    # "Expected" sensor readings for this exact vehicle (no random noise or sudden defects).
    sim = simulate_from_config(config, np.random.default_rng(0), missing=False, defects=False,
                               labels=False, noise=0.0)

    if mode == "Enter manually":
        st.caption("Drag the sliders like you're reading an OBD-II scanner. Mark sensors this vehicle doesn't "
                   "have as unavailable — the model handles missing data.")
        missing = st.multiselect("Sensors not available", [f["name"] for f in sensor_feats],
                                 format_func=lambda n: feat_meta(n)["label"], key="missing_sensors")
        cols = st.columns(3)
        sensors = {}
        for f in sensor_feats:  # Streamlit forgets hidden widgets, so re-seed from the simulator if needed
            if f"in_{f['name']}" not in st.session_state:
                st.session_state[f"in_{f['name']}"] = float(np.clip(sim[f["name"]].iloc[0], f["min"], f["max"]))
        for i, f in enumerate(sensor_feats):
            with cols[i % 3]:
                v = slider(f) if f["name"] not in missing else None
                sensors[f["name"]] = np.nan if v is None else v
        row = config.assign(**sensors)
        hidden = None
    else:
        row = sim[[c for c in sim.columns if not c.startswith("_")]]
        hidden = sim[[c for c in sim.columns if c.startswith("_health_")]]
        with st.expander("Simulated sensor readings", expanded=False):
            cols = st.columns(4)
            for i, f in enumerate(sensor_feats):
                cols[i % 4].metric(f["label"], fmt(row[f["name"]].iloc[0], f["name"]))

    pred = model.predict(row).iloc[0]
    risk_cols = [c for c in pred.index if c.startswith("risk_") and pd.notna(pred[c])]

    c1, c2, c3 = st.columns(3)
    overall = pred["overall_risk"]
    c1.metric("Chance something fails (next 5,000 km)", f"{overall:.0%}")
    if "rul_p50" in pred:
        c2.metric("Est. km until next repair needed", f"{pred['rul_p50']:,.0f} km",
                  help=f"80% range: {pred['rul_p10']:,.0f} – {pred['rul_p90']:,.0f} km")
    c3.metric("Anomaly score", f"{pred['anomaly_pct']:.0f} / 100",
              help="How unusual these readings are compared with healthy vehicles. "
                   "Above ~95 means 'unlike anything healthy' even if no known failure matches.")
    if overall < 0.2:
        st.success("**Status: Healthy.** No component is near its failure threshold.")
    elif overall < 0.5:
        st.warning("**Status: Watch.** At least one component is wearing — plan service soon.")
    else:
        st.error("**Status: Service needed.** One or more components are likely to fail soon.")
    if pred["anomaly_pct"] > 97:
        st.info("🔎 These readings are unusual compared with healthy vehicles. Worth a manual inspection even "
                "if no specific component is flagged.")

    # ---- risk by component ----
    rdf = pd.DataFrame({
        "component": [comp_label(c[5:]) for c in risk_cols],
        "target": [c[5:] for c in risk_cols],
        "risk": [pred[c] for c in risk_cols],
        "threshold": [model.thresholds_[c[5:]] for c in risk_cols],
    }).sort_values("risk", ascending=False)
    rdf["level"] = np.where(rdf["risk"] >= rdf["threshold"], "Alert", np.where(rdf["risk"] >= 0.1, "Watch", "OK"))
    chart = alt.Chart(rdf).mark_bar().encode(
        x=alt.X("risk:Q", title="Probability of failure in next 5,000 km", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%")),
        y=alt.Y("component:N", sort="-x", title=None),
        color=alt.Color("level:N", scale=alt.Scale(domain=["OK", "Watch", "Alert"],
                                                   range=["#3a9d5d", "#e0a526", "#d1495b"]), legend=None),
        tooltip=["component", alt.Tooltip("risk:Q", format=".1%"), alt.Tooltip("threshold:Q", format=".0%")],
    ).properties(height=34 * len(rdf) + 20)
    st.subheader("Risk by component")
    st.altair_chart(chart, width="stretch")

    # ---- explanations ----
    st.subheader("Why? — root-cause explanation")
    st.caption("For each input, the model asks: *if this vehicle had the value a typical healthy "
               f"{powertrain} vehicle has, how would the risk change?* Big changes = main causes.")
    flagged = rdf[rdf["level"] != "OK"]
    show = flagged if not flagged.empty else rdf.head(1)
    for _, r in show.iterrows():
        comp = r["target"].removeprefix("fail_")
        with st.expander(f"{r['component']} — {r['risk']:.0%} risk", expanded=r["level"] == "Alert"):
            ex = model.explain(row, r["target"], top_k=6)
            ex = ex[ex["contribution"].abs() >= 0.005]
            if ex.empty:
                st.write("No single input stands out — the risk comes from many small factors.")
            for _, e in ex.iterrows():
                arrow = "⬆️ raises" if e["contribution"] > 0 else "⬇️ lowers"
                st.markdown(f"- **{feat_meta(e['feature'])['label']}** = {fmt(e['value'], e['feature'])} "
                            f"(healthy typical: {fmt(e['typical_healthy'], e['feature'])}) — {arrow} risk by "
                            f"**{abs(e['contribution']) * 100:.0f} pts**")
            st.markdown(f"**Recommended action:** {COMPONENTS[comp]['action']}")

    if hidden is not None:
        with st.expander("🎓 Peek behind the curtain: the simulator's hidden 'true' health"):
            st.caption("The model never sees these numbers — it has to infer them from sensors and history. "
                       "Compare them with the risks above to judge how well it does.")
            truth = pd.DataFrame({"component": [comp_label(c.removeprefix("_health_")) for c in hidden.columns],
                                  "true health": hidden.iloc[0].values})
            truth = truth[truth["component"].isin(rdf["component"])]
            st.dataframe(truth.style.format({"true health": "{:.0%}"}).background_gradient(
                cmap="RdYlGn", vmin=0, vmax=1, subset=["true health"]), hide_index=True)

# =====================================================================
# TAB 2 — Forecast
# =====================================================================
with tab_forecast:
    st.subheader("What if you keep driving without maintenance?")
    st.caption("The simulator ages this vehicle forward (more km, older parts, sensors drifting) and the "
               "model re-predicts at each step. This turns risk scores into a service schedule.")
    horizon = st.slider("Look ahead (km)", 5000, 100000, 40000, 5000)
    steps = np.linspace(0, horizon, 41)
    proj = project_forward(config.iloc[0], steps)
    fp = model.predict(proj)
    long = (fp.filter(like="risk_").assign(km=steps)
            .melt("km", var_name="target", value_name="risk").dropna())
    long["component"] = long["target"].str.removeprefix("risk_fail_").map(lambda c: COMPONENTS[c]["label"])
    line = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=alt.X("km:Q", title="Km driven from today"),
        y=alt.Y("risk:Q", title="Failure risk (next 5,000 km)", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("component:N", title=None),
        tooltip=["component", alt.Tooltip("km:Q", format=",.0f"), alt.Tooltip("risk:Q", format=".0%")],
    ).properties(height=380)
    st.altair_chart(line, width="stretch")

    sched = []
    for t in model.classifiers_:
        col = f"risk_{t}"
        if col not in fp or fp[col].isna().all():
            continue
        over = np.where(fp[col].to_numpy() >= model.thresholds_[t])[0]
        sched.append({"Component": comp_label(t[5:]),
                      "Service due in": "now" if len(over) and over[0] == 0 else
                      (f"~{steps[over[0]]:,.0f} km" if len(over) else f"> {horizon:,} km"),
                      "_sort": steps[over[0]] if len(over) else 1e9,
                      "What to do": COMPONENTS[t[5:]]["action"]})
    st.markdown("#### Suggested service schedule")
    st.dataframe(pd.DataFrame(sched).sort_values("_sort").drop(columns="_sort"), hide_index=True, width="stretch")

# =====================================================================
# TAB 3 — Fleet analysis
# =====================================================================
with tab_fleet:
    st.subheader("Analyse many vehicles at once")
    src = st.radio("Data source", ["Generate a sample fleet", "Upload a CSV"], horizontal=True)
    fleet = None
    if src == "Generate a sample fleet":
        n = st.slider("Number of vehicles", 50, 5000, 500, 50)
        if st.button("Generate fleet"):
            st.session_state["fleet"] = generate_fleet(n, seed=int(np.random.default_rng().integers(1e6)))
        fleet = st.session_state.get("fleet")
    else:
        template = pd.DataFrame(columns=["vehicle_id"] + CONFIG_COLS + [f["name"] for f in SCHEMA["features"]
                                                                       if f["group"] == "sensor"])
        st.download_button("Download CSV template", template.to_csv(index=False), "fleet_template.csv")
        st.caption("Any subset of these columns works — blanks and missing columns are treated as unknown. "
                   "Include `vehicle_type` and `powertrain` for best results.")
        up = st.file_uploader("Fleet CSV", type="csv")
        if up is not None:
            fleet = pd.read_csv(up)

    if fleet is not None and len(fleet):
        out = model.predict(fleet)
        risk = out.filter(like="risk_")
        top = risk.idxmax(axis=1, skipna=True).str.removeprefix("risk_fail_")
        table = pd.DataFrame({
            "vehicle": fleet["vehicle_id"] if "vehicle_id" in fleet else fleet.index.astype(str),
            "type": fleet.get("vehicle_type", pd.Series("?", index=fleet.index)),
            "overall risk": out["overall_risk"],
            "most likely issue": top.map(lambda c: COMPONENTS.get(c, {}).get("label", c)),
            "est. km to repair": out.get("rul_p50"),
            "anomaly": out["anomaly_pct"],
        }).sort_values("overall risk", ascending=False)

        flagged = {comp_label(t[5:]): int((risk[f"risk_{t}"] >= th).sum()) for t, th in model.thresholds_.items()}
        a, b, c = st.columns(3)
        a.metric("Vehicles", f"{len(fleet):,}")
        b.metric("Need service soon", f"{(table['overall risk'] >= 0.5).sum():,}")
        c.metric("Unusual readings (anomaly > 97)", f"{(table['anomaly'] > 97).sum():,}")
        st.bar_chart(pd.Series(flagged, name="vehicles flagged").sort_values(ascending=False), horizontal=True)

        st.markdown("#### Priority list")
        st.dataframe(table.style.format({"overall risk": "{:.0%}", "est. km to repair": "{:,.0f}",
                                         "anomaly": "{:.0f}"})
                     .background_gradient(cmap="Reds", subset=["overall risk"], vmin=0, vmax=1),
                     hide_index=True, width="stretch", height=420)
        st.download_button("Download predictions", pd.concat([fleet, out], axis=1).to_csv(index=False),
                           "fleet_predictions.csv")

        # If the CSV has real outcomes, grade the model on it.
        labelled = [t for t in model.classifiers_ if t in fleet and fleet[t].notna().any()]
        if labelled:
            from sklearn.metrics import roc_auc_score
            st.markdown("#### Accuracy on this data (it contains real outcomes)")
            rows = []
            for t in labelled:
                ok = fleet[t].notna() & out[f"risk_{t}"].notna()
                y, p = fleet.loc[ok, t].astype(int), out.loc[ok, f"risk_{t}"]
                if y.nunique() == 2:
                    caught = ((p >= model.thresholds_[t]) & (y == 1)).sum()
                    rows.append({"component": comp_label(t[5:]), "failures": int(y.sum()),
                                 "caught by model": int(caught), "ROC-AUC": roc_auc_score(y, p)})
            st.dataframe(pd.DataFrame(rows).style.format({"ROC-AUC": "{:.3f}"}), hide_index=True)

# =====================================================================
# TAB 4 — Train on your own data
# =====================================================================
with tab_train:
    st.subheader("Train a model on any table of data")
    st.write("The model isn't hard-wired to these sensors. Give it **any CSV** — one row per vehicle (or machine, "
             "or anything) — tell it which columns are the outcomes, and it learns from every other column.")
    src = st.radio("Data", ["Upload a CSV", "Demo: simulated fleet with a random subset of sensors"], horizontal=True)
    data = None
    if src.startswith("Upload"):
        up = st.file_uploader("Training CSV", type="csv", key="train_csv")
        if up is not None:
            data = pd.read_csv(up)
    else:
        k = st.slider("How many input columns to keep", 3, 35, 10)
        if st.button("Build demo dataset"):
            full = generate_fleet(8000, seed=int(np.random.default_rng().integers(1e6)))
            inputs_all = [c for c in full.columns if not c.startswith(("fail_", "_")) and c not in ("rul_km", "vehicle_id")]
            keep = list(np.random.default_rng().choice(inputs_all, size=min(k, len(inputs_all)), replace=False))
            st.session_state["demo_data"] = full[keep + ["fail_brakes", "fail_tires", "fail_battery_12v", "rul_km"]]
        data = st.session_state.get("demo_data")

    if data is not None:
        st.dataframe(data.head(8), width="stretch")
        binary = [c for c in data.columns if set(pd.to_numeric(data[c], errors="coerce").dropna().unique()) <= {0, 1}
                  and data[c].nunique() == 2]
        numeric = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
        targets = st.multiselect("Outcome columns to predict (0/1)", binary,
                                 default=[c for c in binary if c.startswith("fail")] or binary[:1])
        rul = st.selectbox("Remaining-life column (optional, numeric)", ["(none)"] + [c for c in numeric if c not in targets],
                           index=(1 + [c for c in numeric if c not in targets].index("rul_km"))
                           if "rul_km" in data and "rul_km" not in targets else 0)
        drop = st.multiselect("Columns to ignore (IDs, leaky columns)", [c for c in data.columns if c not in targets])
        st.caption("⚠️ *Leakage:* never feed the model a column that's only known **after** the failure "
                   "(e.g. 'repair cost'). It will look brilliant in testing and useless in real life.")
        if targets and st.button("Train", type="primary"):
            custom = MaintenanceModel(classification_targets=targets,
                                      regression_target=None if rul == "(none)" else rul, drop_cols=drop)
            bar = st.progress(0.0)
            rep = custom.fit(data, progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
            bar.empty()
            st.session_state["custom_model"] = custom
            st.success(f"Trained on {rep['n_rows']:,} rows using {rep['n_features']} features "
                       f"(including any physics features it could derive).")
        custom = st.session_state.get("custom_model")
        if custom is not None:
            m = pd.DataFrame(custom.report_["targets"]).T
            st.dataframe(m, width="stretch")
            if "regression" in custom.report_:
                r = custom.report_["regression"]
                st.write(f"Remaining life: MAE **{r['mae']:,.0f}**, R² **{r['r2']:.3f}**, "
                         f"80% interval covers **{r['interval_coverage']:.0%}** of cases.")
            t = st.selectbox("Feature importance for", list(custom.importances_))
            if t:
                st.bar_chart(custom.importances_[t].head(12), horizontal=True)
            buf = io.BytesIO()
            import joblib
            joblib.dump(custom, buf, compress=3)
            st.download_button("Download trained model (.joblib)", buf.getvalue(), "custom_model.joblib")

# =====================================================================
# TAB 5 — How it works
# =====================================================================
with tab_how:
    st.subheader("Pipeline")
    st.markdown("""
1. **Simulate a fleet** (`pvm/simulate.py`). Each vehicle gets hidden *true* component health computed from
   physics-style wear rules (km since service, heat, load, driving style, regen braking…). Sensors are noisy
   functions of that health, and some sensors are randomly missing — just like real cars.
2. **Engineer features** (`pvm/features.py`): ratios a mechanic would compute, e.g. coolant temperature above
   ambient, oil pressure per 1,000 rpm, observed pad wear per 1,000 km.
3. **Train** (`pvm/model.py`) one gradient-boosted tree model per failure mode, quantile models for remaining
   life, and an Isolation Forest anomaly detector trained only on healthy vehicles.
4. **Evaluate on held-out data** the model never saw, and pick each alert threshold on a separate validation set.
5. **Explain** each prediction by swapping inputs for typical healthy values and measuring the change.
""")
    rep = model.report_
    st.subheader("Test-set performance")
    mt = pd.DataFrame(rep["targets"]).T
    mt.index = [comp_label(i[5:]) for i in mt.index]
    st.dataframe(mt[["prevalence", "roc_auc", "pr_auc", "recall", "precision", "threshold"]]
                 .style.format({"prevalence": "{:.1%}", "roc_auc": "{:.3f}", "pr_auc": "{:.3f}",
                                "recall": "{:.0%}", "precision": "{:.0%}", "threshold": "{:.2f}"}), width="stretch")
    with st.expander("What do these numbers mean?"):
        st.markdown("""
- **Prevalence** — how often this failure happens. If it's 7%, a model that always says "no failure" is 93% "accurate",
  which is why accuracy is a bad metric here.
- **ROC-AUC** — pick one vehicle that failed and one that didn't; this is the probability the model gave the failing
  one the higher risk. 0.5 = coin flip, 1.0 = perfect.
- **PR-AUC** — like ROC-AUC but focused on the rare positive class. Compare it to the prevalence (a random guess scores ≈ prevalence).
- **Recall** — of the vehicles that really failed, what fraction got an alert.
- **Precision** — of the alerts raised, what fraction were real.
- **Threshold** — risk level that triggers an alert, chosen to balance precision and recall (best F1).
""")
    if "regression" in rep:
        r = rep["regression"]
        st.write(f"**Remaining-life model:** typical error (MAE) {r['mae']:,.0f} km, R² {r['r2']:.3f}, "
                 f"and the 80% prediction interval contained the true value {r['interval_coverage']:.0%} of the time.")

    @st.cache_data(show_spinner="Scoring a fresh fleet…")
    def oracle_check() -> pd.DataFrame:
        from sklearn.metrics import roc_auc_score
        df = generate_fleet(6000, seed=2024)
        p = model.predict(df)
        rows = []
        for comp in component_names():
            t = target_name(comp)
            ok = df[t].notna()
            rows.append({"component": COMPONENTS[comp]["label"],
                         "model ROC-AUC": roc_auc_score(df.loc[ok, t], p.loc[ok, f"risk_{t}"]),
                         "oracle ROC-AUC (knows true health)": roc_auc_score(df.loc[ok, t], -df.loc[ok, f"_health_{comp}"])})
        return pd.DataFrame(rows)

    st.subheader("How close to perfect can it get?")
    st.caption("Failures are random even for a worn part, so no model can be 100% right. The *oracle* gets to "
               "see the simulator's hidden true health — it's the best score physically possible. "
               "The closer the model is to the oracle, the more of the available signal it has extracted.")
    st.dataframe(oracle_check().style.format({c: "{:.3f}" for c in ["model ROC-AUC", "oracle ROC-AUC (knows true health)"]}),
                 hide_index=True, width="stretch")

    st.subheader("What the model relies on")
    t = st.selectbox("Component", list(model.importances_), format_func=lambda t: comp_label(t[5:]))
    imp = model.importances_[t].head(12).rename("drop in ROC-AUC when shuffled")
    imp.index = [feat_meta(i)["label"] if feat_meta(i)["label"] != i else i for i in imp.index]
    st.bar_chart(imp, horizontal=True)
    st.caption("Permutation importance: shuffle one column so it carries no information, and measure how much "
               "worse the model gets.")

"""Streamlit pieces shared by several tabs."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .obd_live import SIGNAL_META
from .schema import load_schema

SCHEMA = load_schema()
COMPONENTS = SCHEMA["components"]


def comp_label(target: str) -> str:
    return COMPONENTS.get(target.removeprefix("fail_"), {}).get("label", target)


def feat_meta(name: str) -> dict:
    f = next((f for f in SCHEMA["features"] if f["name"] == name), None)
    if f:
        return f
    return SIGNAL_META.get(name, {"label": name, "unit": ""})


def fmt(v, name: str) -> str:
    if v is None or pd.isna(v):
        return "—"
    unit = feat_meta(name).get("unit", "")
    v = float(v)
    s = f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.1f}".rstrip("0").rstrip(".") if abs(v) >= 100 \
        else f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"{s} {unit}".strip()


def render_diagnosis(model, row: pd.DataFrame, powertrain: str, hidden: pd.DataFrame | None = None,
                     key: str = "", wear_wording: bool = False, show_anomaly: bool = True) -> pd.Series:
    """`wear_wording` phrases the status as a wear outlook (used next to live fault codes, where
    "Healthy" would contradict a code that is already stored)."""
    """Risk summary, per-component bars and explanations for one vehicle row."""
    pred = model.predict(row).iloc[0]
    risk_cols = [c for c in pred.index if c.startswith("risk_") and pd.notna(pred[c])]

    c1, c2, c3 = st.columns(3) if show_anomaly else (*st.columns(2), None)
    overall = pred["overall_risk"]
    c1.metric("Chance something fails (next 5,000 km)", f"{overall:.0%}")
    if "rul_p50" in pred:
        c2.metric("Est. km until next repair needed", f"{pred['rul_p50']:,.0f} km",
                  help=f"80% range: {pred['rul_p10']:,.0f} – {pred['rul_p90']:,.0f} km")
    if c3 is not None:
        c3.metric("Anomaly score", f"{pred['anomaly_pct']:.0f} / 100",
                  help="How unusual these readings are compared with healthy vehicles. "
                       "Above ~95 means 'unlike anything healthy' even if no known failure matches.")
    head = "Wear outlook" if wear_wording else "Status"
    if overall < 0.2:
        st.success(f"**{head}: {'Low' if wear_wording else 'Healthy'}.** No component is near its failure threshold.")
    elif overall < 0.5:
        st.warning(f"**{head}: Watch.** At least one component is wearing — plan service soon.")
    else:
        st.error(f"**{head}: {'High' if wear_wording else 'Service needed'}.** One or more components are likely "
                 "to fail soon.")
    if show_anomaly and pred["anomaly_pct"] > 97:
        st.info("🔎 These readings are unusual compared with healthy vehicles. Worth a manual inspection even "
                "if no specific component is flagged.")

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
    st.markdown("#### Risk by component")
    st.altair_chart(chart, width="stretch")

    st.markdown("#### Why? — root-cause explanation")
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
    return pred

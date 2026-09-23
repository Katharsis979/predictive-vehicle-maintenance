import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

st.set_page_config(page_title="AI Vehicle Maintenance", page_icon="🚗", layout="wide")

@st.cache_resource
def load_or_train_model():
    model_filename = "rf_model.pkl"
    feature_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
    else:
        df = pd.read_csv("ai4i2020.csv")
        X = df[feature_cols]
        y = df['Machine failure']

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_res, y_res)
        joblib.dump(model, model_filename)

    return model, feature_cols

model, feature_cols = load_or_train_model()

# Header
st.title("🚗 Real-Time Predictive Vehicle Maintenance System")
st.markdown("Simulate live OBD-II engine telemetry inputs to evaluate component breakdown risk.")

# Sidebar Presets
st.sidebar.header("🎯 Preset Test Scenarios")
scenario = st.sidebar.radio("Quick Presets:", ["Custom Sliders", "Normal Highway Driving", "Heavy Towing / Torque Strain", "High RPM / Over-revving"])

# Default slider values based on preset
if scenario == "Normal Highway Driving":
    defaults = (298.0, 308.0, 1500, 35.0, 50)
elif scenario == "Heavy Towing / Torque Strain":
    defaults = (302.0, 312.0, 1350, 68.0, 210)
elif scenario == "High RPM / Over-revving":
    defaults = (301.0, 310.0, 2750, 12.0, 180)
else:
    defaults = (300.0, 310.0, 1500, 40.0, 100)

st.sidebar.header("🔧 Manual Telemetry Controls")
air_temp = st.sidebar.slider("Air Temperature [K]", 295.0, 305.0, defaults[0], 0.1)
proc_temp = st.sidebar.slider("Process Temperature [K]", 305.0, 315.0, defaults[1], 0.1)
rpm = st.sidebar.slider("Engine Speed [RPM]", 1100, 2900, defaults[2], 10)
torque = st.sidebar.slider("Engine Torque [Nm]", 3.0, 80.0, defaults[3], 0.5)
wear = st.sidebar.slider("Component Strain / Wear [min]", 0, 260, defaults[4], 1)

input_data = pd.DataFrame([[air_temp, proc_temp, rpm, torque, wear]], columns=feature_cols)

# --- RUN PREDICTION ---
probabilities = model.predict_proba(input_data)[0]
failure_risk = probabilities[1] * 100
health_score = 100 - failure_risk

# --- 1. DYNAMIC METRICS WITH DELTAS ---
st.markdown("---")
st.subheader("📊 Live Telemetry vs. Safe Baseline")
col1, col2, col3, col4, col5 = st.columns(5)

# Safe baselines for comparison
col1.metric("Air Temp", f"{air_temp} K", f"{air_temp - 298.0:.1f} K", delta_color="inverse")
col2.metric("Proc Temp", f"{proc_temp} K", f"{proc_temp - 308.0:.1f} K", delta_color="inverse")
col3.metric("Engine Speed", f"{rpm} RPM", f"{rpm - 1500} RPM", delta_color="inverse")
col4.metric("Torque", f"{torque} Nm", f"{torque - 40.0:.1f} Nm", delta_color="inverse")
col5.metric("Tool Wear", f"{wear} min", f"{wear - 0} min", delta_color="inverse")

# --- 2. ENGINE HEALTH & DIAGNOSTICS ---
st.markdown("---")
col_health, col_diag = st.columns([1, 1.5])

with col_health:
    st.subheader("Engine Health Score")
    st.metric(label="System Vitality", value=f"{health_score:.1f} / 100")
    
    if health_score > 80:
        st.success("✅ **Status: Optimal.** Engine operating safely.")
    elif health_score > 50:
        st.warning("⚡ **Status: Moderate Strain.** Inspect components soon.")
    else:
        st.error("⚠️ **Status: Critical Risk.** Immediate maintenance required.")

with col_diag:
    st.subheader("📋 Mechanic Recommendations")
    report_text = f"DIAGNOSTIC REPORT\nHealth Score: {health_score:.1f}/100\n\nAlerts:\n"
    
    if health_score > 80:
        st.write("- All physical tolerances are within normal operating parameters.")
        report_text += "- System operating normally.\n"
    if torque > 60.0:
        st.write("- 🔧 **Drivetrain:** Excessive torque. Inspect transmission fluid and differential gears.")
        report_text += "- Drivetrain warning (High Torque)\n"
    if rpm > 2400:
        st.write("- 🔧 **Over-revving:** Speed exceeds safe limits. Check timing belt and valve springs.")
        report_text += "- Engine speed warning (High RPM)\n"
    if proc_temp - air_temp > 12.0:
        st.write("- 🔧 **Thermal:** Cooling system failing to dissipate heat. Inspect radiator.")
        report_text += "- Thermal warning (High Temp Delta)\n"
    if wear > 200:
        st.write("- 🔧 **Lifecycle:** Component strain nearing maximum limit. Replace primary wear parts.")
        report_text += "- Component wear warning (Lifecycle limit)\n"

# --- 3. EXPORTABLE REPORT ---
st.markdown("---")
st.download_button(
    label="📥 Download Diagnostic Report",
    data=report_text,
    file_name="engine_diagnostic_report.txt",
    mime="text/plain"
)
     

st.subheader("Live Telemetry Data Feed")
st.dataframe(input_data)

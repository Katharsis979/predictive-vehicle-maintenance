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

# Prediction
probabilities = model.predict_proba(input_data)[0]
failure_risk = probabilities[1] * 100

# Main Dashboard View
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Engine Risk Assessment")
    st.metric(label="Calculated Breakdown Risk", value=f"{failure_risk:.1f}%")
    st.progress(int(failure_risk))

    if failure_risk > 50:
        st.error("⚠️ CRITICAL RISK: Structural mechanical failure imminent! Immediate maintenance required.")
    elif failure_risk > 20:
        st.warning("⚡ MODERATE WARNING: Sensor anomalies detected. Inspect engine strain factors.")
    else:
        st.success("✅ SYSTEM HEALTHY: Operating within normal parameters.")

with col2:
    st.subheader("Model Decision Drivers (Global Feature Importance)")
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x=importances, y=feature_cols, palette="viridis", ax=ax)
    ax.set_xlabel("Importance Weight")
    st.pyplot(fig)

st.subheader("Live Telemetry Data Feed")
st.dataframe(input_data)
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

# --- VEHICLE PROFILES ---
# Scaled to fit within the model's trained parameters (Max RPM ~2900, Max Torque ~80)
VEHICLE_PROFILES = {
    "Compact Commuter (Economy)": {"base_rpm": 1800, "base_torque": 20.0, "base_temp": 305.0, "strain_mult": 1.0},
    "Commercial Delivery Van": {"base_rpm": 1500, "base_torque": 40.0, "base_temp": 308.0, "strain_mult": 1.2},
    "Heavy-Duty Tow Truck": {"base_rpm": 1300, "base_torque": 65.0, "base_temp": 312.0, "strain_mult": 1.5}
}

st.sidebar.header("🚙 Vehicle Configuration")
vehicle_type = st.sidebar.selectbox("Select Vehicle Profile", list(VEHICLE_PROFILES.keys()))
profile = VEHICLE_PROFILES[vehicle_type]

st.title("🚗 Predictive Vehicle Maintenance System")
tab1, tab2 = st.tabs(["🎛️ Live Telemetry Simulator", "📁 Batch Trip Analysis"])

# ==========================================
# TAB 1: LIVE SIMULATOR
# ==========================================
with tab1:
    st.sidebar.header("🔧 Manual Controls")
    air_temp = st.sidebar.slider("Air Temp [K]", 295.0, 305.0, 298.0, 0.1)
    proc_temp = st.sidebar.slider("Proc Temp [K]", 305.0, 315.0, profile["base_temp"], 0.1)
    rpm = st.sidebar.slider("Engine Speed [RPM]", 1100, 2900, profile["base_rpm"], 10)
    torque = st.sidebar.slider("Torque [Nm]", 3.0, 80.0, profile["base_torque"], 0.5)
    wear = st.sidebar.slider("Tool Wear [min]", 0, 260, 50, 1)

    input_data = pd.DataFrame([[air_temp, proc_temp, rpm, torque, wear]], columns=feature_cols)
    failure_risk = model.predict_proba(input_data)[0][1] * 100
    health_score = 100 - failure_risk

    st.subheader(f"Live Diagnostics: {vehicle_type}")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Air Temp", f"{air_temp} K")
    col2.metric("Proc Temp", f"{proc_temp} K", f"{proc_temp - profile['base_temp']:.1f} K", delta_color="inverse")
    col3.metric("RPM", f"{rpm}", f"{rpm - profile['base_rpm']}", delta_color="inverse")
    col4.metric("Torque", f"{torque} Nm", f"{torque - profile['base_torque']:.1f} Nm", delta_color="inverse")
    col5.metric("Wear", f"{wear} min")

    st.markdown("---")
    if health_score > 80:
        st.success(f"✅ **System Vitality: {health_score:.1f}/100** - Operating normally for a {vehicle_type}.")
    elif health_score > 50:
        st.warning(f"⚡ **System Vitality: {health_score:.1f}/100** - Moderate strain detected.")
    else:
        st.error(f"⚠️ **System Vitality: {health_score:.1f}/100** - Critical risk of mechanical failure.")

# ==========================================
# TAB 2: BATCH PROCESSING & TRIP GENERATOR
# ==========================================
with tab2:
    st.subheader(f"Trip Analytics for {vehicle_type}")
    
    # Generate Synthetic Drive Data
    if st.button("🛠️ Generate Sample 60-Minute Drive Log"):
        time_steps = np.arange(0, 60)
        
        # Simulate a drive: 10 min warmup, 40 min cruise, 10 min steep uphill stress
        sim_rpm = np.concatenate([
            np.linspace(1000, profile["base_rpm"], 10), 
            np.random.normal(profile["base_rpm"], 50, 40),
            np.random.normal(profile["base_rpm"] + 500, 100, 10) # Over-revving
        ])
        sim_torque = np.concatenate([
            np.linspace(10, profile["base_torque"], 10),
            np.random.normal(profile["base_torque"], 2, 40),
            np.random.normal(profile["base_torque"] + (15 * profile["strain_mult"]), 5, 10) # Uphill strain
        ])
        sim_proc_temp = np.linspace(295.0, profile["base_temp"] + (3 * profile["strain_mult"]), 60)
        
        sim_data = pd.DataFrame({
            'Air temperature [K]': np.full(60, 298.0),
            'Process temperature [K]': np.clip(sim_proc_temp, 295.0, 315.0),
            'Rotational speed [rpm]': np.clip(sim_rpm, 1100, 2900),
            'Torque [Nm]': np.clip(sim_torque, 3.0, 80.0),
            'Tool wear [min]': np.linspace(100, 101, 60) # Wear barely changes in 1 hour
        })
        
        st.session_state['sim_data'] = sim_data
        st.success("Sample trip generated! Click download, then upload it below.")
        
        st.download_button("📥 Download simulated_trip.csv", 
                           data=sim_data.to_csv(index=False).encode('utf-8'), 
                           file_name="simulated_trip.csv", mime="text/csv")

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV Trip Log", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        with st.spinner("Analyzing telemetry data..."):
            batch_df['Failure Risk (%)'] = model.predict_proba(batch_df[feature_cols])[:, 1] * 100
            
            st.markdown("### Engine Stress Timeline (Risk % over Trip Duration)")
            st.line_chart(batch_df['Failure Risk (%)'])
            
            critical_df = batch_df[batch_df['Failure Risk (%)'] > 50].copy()
            if not critical_df.empty:
                st.error(f"⚠️ Detected {len(critical_df)} critical strain events during this trip.")
                st.dataframe(critical_df.style.highlight_max(axis=0, color='darkred'))
            else:
                st.success("✅ No critical anomalies detected during this trip.")

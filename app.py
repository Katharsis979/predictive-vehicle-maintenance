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

# --- MAIN DASHBOARD INTERFACE ---
st.title("🚗 Predictive Vehicle Maintenance System")
st.markdown("Analyze live OBD-II engine telemetry or process batch historical logs.")

# Create Dual Workspaces
tab1, tab2 = st.tabs(["🎛️ Live Telemetry Simulator", "📁 Batch Log Processing"])

# ==========================================
# TAB 1: LIVE SIMULATOR (Your existing tool)
# ==========================================
with tab1:
    st.sidebar.header("🎯 Preset Scenarios (Live Tab)")
    scenario = st.sidebar.radio("Quick Presets:", ["Normal Highway", "Heavy Towing", "High RPM", "Custom"])

    if scenario == "Normal Highway":
        defaults = (298.0, 308.0, 1500, 35.0, 50)
    elif scenario == "Heavy Towing":
        defaults = (302.0, 312.0, 1350, 68.0, 210)
    elif scenario == "High RPM":
        defaults = (301.0, 310.0, 2750, 12.0, 180)
    else:
        defaults = (300.0, 310.0, 1500, 40.0, 100)

    st.sidebar.header("🔧 Manual Controls")
    air_temp = st.sidebar.slider("Air Temp [K]", 295.0, 305.0, defaults[0], 0.1)
    proc_temp = st.sidebar.slider("Proc Temp [K]", 305.0, 315.0, defaults[1], 0.1)
    rpm = st.sidebar.slider("Engine Speed [RPM]", 1100, 2900, defaults[2], 10)
    torque = st.sidebar.slider("Torque [Nm]", 3.0, 80.0, defaults[3], 0.5)
    wear = st.sidebar.slider("Tool Wear [min]", 0, 260, defaults[4], 1)

    input_data = pd.DataFrame([[air_temp, proc_temp, rpm, torque, wear]], columns=feature_cols)
    
    failure_risk = model.predict_proba(input_data)[0][1] * 100
    health_score = 100 - failure_risk

    st.subheader("Engine Health & Diagnostics")
    col_health, col_diag = st.columns([1, 1.5])

    with col_health:
        st.metric(label="System Vitality", value=f"{health_score:.1f} / 100")
        if health_score > 80:
            st.success("✅ **Status: Optimal.** Engine operating safely.")
        elif health_score > 50:
            st.warning("⚡ **Status: Moderate Strain.** Inspect components soon.")
        else:
            st.error("⚠️ **Status: Critical Risk.** Immediate maintenance required.")

    with col_diag:
        if torque > 60.0:
            st.write("- 🔧 **Drivetrain:** Excessive torque. Inspect transmission.")
        if rpm > 2400:
            st.write("- 🔧 **Over-revving:** Speed exceeds safe limits. Check timing belt.")
        if proc_temp - air_temp > 12.0:
            st.write("- 🔧 **Thermal:** Cooling system failing. Inspect radiator.")
        if wear > 200:
            st.write("- 🔧 **Lifecycle:** Component strain nearing maximum limit.")
        if health_score > 80:
            st.write("- All physical tolerances are within normal operating parameters.")

# ==========================================
# TAB 2: BATCH PROCESSING (The New Feature)
# ==========================================
with tab2:
    st.subheader("Upload Historical Telemetry Logs")
    st.markdown("Upload a CSV file containing engine sensor data to scan an entire trip or fleet for mechanical anomalies.")
    
    uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        # Check if the uploaded CSV has the required columns
        missing_cols = [col for col in feature_cols if col not in batch_df.columns]
        
        if missing_cols:
            st.error(f"⚠️ Uploaded CSV is missing required columns: {missing_cols}")
        else:
            with st.spinner("Analyzing telemetry data..."):
                # Run the model on the entire dataset instantly
                batch_df['Failure Risk (%)'] = model.predict_proba(batch_df[feature_cols])[:, 1] * 100
                batch_df['Risk Status'] = np.where(batch_df['Failure Risk (%)'] > 50, "CRITICAL", 
                                          np.where(batch_df['Failure Risk (%)'] > 20, "WARNING", "SAFE"))
                
                # Display high-level metrics
                critical_count = len(batch_df[batch_df['Risk Status'] == 'CRITICAL'])
                
                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.metric("Total Records Scanned", len(batch_df))
                col_b2.metric("Critical Strain Events", critical_count, delta_color="inverse")
                col_b3.metric("Max Peak Risk", f"{batch_df['Failure Risk (%)'].max():.1f}%")
                
                # Visualize the risk over time
                st.markdown("### Risk Timeline (Entire Trip)")
                st.line_chart(batch_df['Failure Risk (%)'])
                
                # Filter and isolate only the dangerous moments for the mechanic
                st.markdown("### ⚠️ Critical Strain Log")
                if critical_count > 0:
                    critical_df = batch_df[batch_df['Risk Status'] == 'CRITICAL'].copy()
                    st.dataframe(critical_df.style.highlight_max(axis=0, color='darkred'))
                    
                    # Allow user to download just the failure report
                    st.download_button(
                        label="📥 Download Filtered Critical Incident Log",
                        data=critical_df.to_csv(index=False).encode('utf-8'),
                        file_name="critical_incident_report.csv",
                        mime="text/csv"
                    )
                else:
                    st.success("No critical strain events detected in this log.")

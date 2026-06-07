import streamlit as st
import joblib
import numpy as np

model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Heart Failure Predictor", page_icon="❤️")

# Header
st.title("❤️ Heart Failure Predictor")
st.markdown("*AI-powered clinical decision support tool*")

# Create two tabs
tab1, tab2 = st.tabs(["📋 Basic Screening", "🏥 Clinical Assessment"])

# ========== TAB 1: Basic ==========
with tab1:
    st.markdown("### Quick Screening")
    st.info("ℹ️ For general users without ECG reports. Uses medical averages for complex features.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 100, 50, key="age1")
        sex = st.radio("Sex", ["Male", "Female"], key="sex1")
        bp = st.number_input("Blood Pressure", 90, 200, 120, key="bp1")
        cholesterol = st.number_input("Cholesterol", 100, 400, 200, key="chol1")
    
    with col2:
        heart_rate = st.slider("Max Heart Rate", 60, 220, 150, key="hr1")
        angina = st.radio("Chest pain during exercise?", ["No", "Yes"], key="angina1")
        blood_sugar = st.radio("High blood sugar (>120)?", ["No", "Yes"], key="bs1")
    
    sex_num = 1 if sex == "Male" else 0
    angina_num = 1 if angina == "Yes" else 0
    bs_num = 1 if blood_sugar == "Yes" else 0
    
    features = np.array([[
        age, sex_num, 1, bp, cholesterol, bs_num,
        0, heart_rate, angina_num, 0, 1
    ]])
    
    if st.button("🔍 Predict Risk", key="btn1", type="primary"):
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1]
        
        if pred == 1:
            st.error(f"⚠️ High Risk ({prob*100:.0f}%) - Please consult a doctor")
        else:
            st.success(f"✅ Low Risk ({100-prob*100:.0f}%) - Keep healthy habits")
        
        st.caption("⚠️ Note: Based on basic questions. For more accurate results, use Clinical Assessment tab.")

# ========== TAB 2: Clinical ==========
with tab2:
    st.markdown("### Complete Clinical Assessment")
    st.info("🏥 For patients with ECG reports and complete medical history available.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 100, 50, key="age2")
        sex = st.radio("Sex", ["Male", "Female"], key="sex2")
        bp = st.number_input("Blood Pressure", 90, 200, 120, key="bp2")
        cholesterol = st.number_input("Cholesterol", 100, 400, 200, key="chol2")
        
        cp_help = "ATA (Atypical Angina) | NAP (Non-anginal Pain) | ASY (Asymptomatic) | TA (Typical Angina)"
        cp = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"], help=cp_help, key="cp")
        
        fbs_help = "Fasting blood sugar > 120 mg/dl indicates diabetes risk"
        fbs = st.radio("Fasting Blood Sugar > 120", ["No", "Yes"], help=fbs_help, key="fbs")
    
    with col2:
        heart_rate = st.slider("Max Heart Rate", 60, 220, 150, key="hr2")
        angina = st.radio("Exercise Angina", ["No", "Yes"], key="angina2")
        
        ecg_help = "Normal | ST (ST-T abnormality) | LVH (Left ventricular hypertrophy)"
        resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"], help=ecg_help, key="ecg")
        
        oldpeak_help = "ST depression during exercise (0 = none, higher = worse)"
        oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.5, help=oldpeak_help, key="oldpeak")
        
        slope_help = "Up (Normal) | Flat (Borderline) | Down (High risk)"
        slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], help=slope_help, key="slope")
        
        blood_sugar = st.radio("High blood sugar (>120)?", ["No", "Yes"], key="bs2")
    
    sex_num = 1 if sex == "Male" else 0
    angina_num = 1 if angina == "Yes" else 0
    bs_num = 1 if blood_sugar == "Yes" else 0
    cp_num = {"ATA":0, "NAP":1, "ASY":2, "TA":3}[cp]
    fbs_num = 1 if fbs == "Yes" else 0
    ecg_num = {"Normal":0, "ST":1, "LVH":2}[resting_ecg]
    slope_num = {"Up":0, "Flat":1, "Down":2}[slope]
    
    features = np.array([[
        age, sex_num, cp_num, bp, cholesterol, fbs_num,
        ecg_num, heart_rate, angina_num, oldpeak, slope_num
    ]])
    
    if st.button("🔍 Predict Risk", key="btn2", type="primary"):
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1]
        
        if pred == 1:
            st.error(f"⚠️ High Risk of Heart Failure ({prob*100:.0f}%)")
            st.write("📋 Please consult a cardiologist for further evaluation.")
        else:
            st.success(f"✅ Low Risk ({100-prob*100:.0f}% healthy)")
            st.write("👍 Maintain healthy habits and regular checkups.")
        
        st.caption(f"✅ Model accuracy: 88.59% (using complete clinical data)")

# Footer
st.markdown("---")
st.caption("⚠️ Disclaimer: For educational/screening purposes only. Not a substitute for professional medical diagnosis.")

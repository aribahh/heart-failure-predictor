import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

st.set_page_config(page_title="Heart Failure Predictor", layout="centered")
st.title("❤️ Heart Failure Prediction System")
st.write("Enter patient clinical data below to predict heart failure risk")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Convert categorical to numerical
sex_num = 1 if sex == "M" else 0
chest_pain_num = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}[chest_pain]
resting_ecg_num = {"Normal": 0, "ST": 1, "LVH": 2}[resting_ecg]
exercise_angina_num = 1 if exercise_angina == "Y" else 0
st_slope_num = {"Up": 0, "Flat": 1, "Down": 2}[st_slope]

# Create input array
input_data = np.array([[age, sex_num, chest_pain_num, resting_bp, cholesterol, 
                        fasting_bs, resting_ecg_num, max_hr, exercise_angina_num, 
                        oldpeak, st_slope_num]])

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction button
if st.button("🔍 Predict Heart Failure Risk", type="primary"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK: Heart Failure Detected")
        st.metric("Confidence", f"{probability*100:.1f}%")
    else:
        st.success(f"✅ LOW RISK: No Heart Failure Detected")
        st.metric("Confidence", f"{(1-probability)*100:.1f}%")

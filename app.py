import streamlit as st
import joblib
import numpy as np

# Load your actual saved model and scaler
model = joblib.load('heart_disease_model.pkl')  # This is your Random Forest
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Heart Failure Predictor", layout="centered")
st.title("❤️ Heart Failure Prediction System")
st.write(f"**Model:** Random Forest | **Accuracy:** 88.59%")

# Input fields (same as your training features)
age = st.number_input("Age", 18, 100, 50)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting BP", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Convert to numbers (matching your training encoding)
sex_num = 1 if sex == "M" else 0
cp_num = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}[chest_pain]
resting_ecg_num = {"Normal": 0, "ST": 1, "LVH": 2}[resting_ecg]
exang_num = 1 if exercise_angina == "Y" else 0
slope_num = {"Up": 0, "Flat": 1, "Down": 2}[st_slope]

# Create input array (11 features, same order as your X.columns)
input_data = np.array([[age, sex_num, cp_num, resting_bp, cholesterol, 
                        fasting_bs, resting_ecg_num, max_hr, exang_num, 
                        oldpeak, slope_num]])

# Scale
input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    
    if pred == 1:
        st.error(f"⚠️ Heart Failure Risk: {prob*100:.1f}%")
    else:
        st.success(f"✅ No Heart Failure: {(1-prob)*100:.1f}%")

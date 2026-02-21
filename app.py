import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Load saved objects
model = joblib.load("heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.title("❤️ Heart Disease Prediction System")
st.markdown("### AI Powered Medical Risk Assessment")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🩺 Patient Information")

    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

with col2:
    st.subheader("📊 Prediction Result")

if st.button("Predict"):

    input_dict = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        f"Sex_{sex}": 1,
        f"ChestPainType_{chest_pain}": 1,
        f"RestingECG_{resting_ecg}": 1,
        f"ExerciseAngina_{exercise_angina}": 1,
        f"ST_Slope_{st_slope}": 1,
    }

    input_df = pd.DataFrame([input_dict])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    # Scale
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.progress(int(probability * 100))
    st.write(f"Risk Probability: {round(probability * 100, 2)}%")

st.markdown("---")
st.caption("This system is for educational purposes only.")
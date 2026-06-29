import streamlit as st
import joblib
import numpy as np
from datetime import datetime

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# Load Model and Scaler
# ==========================================

model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ==========================================
# Title
# ==========================================

st.header("Diabetes Prediction System")

st.caption(
    "Predict whether a patient is at risk of diabetes using a trained Logistic Regression model."
)

# ==========================================
# Input Fields
# ==========================================

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0
    )

    glucose = st.number_input(
        "Glucose",
        min_value=40.0,
        max_value=250.0,
        value=100.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=20.0,
        max_value=200.0,
        value=70.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )

with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=900.0,
        value=80.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        value=25.0
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.500,
        format="%.3f"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

# ==========================================
# Predict Button
# ==========================================

col1, col2, col3 = st.columns([1,2,1])

with col2:
    predict = st.button(
        "Predict Diabetes",
        use_container_width=True
    )
    st.write("")

# ==========================================
# Prediction
# ==========================================

if predict:

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    # Scale input

    input_scaled = scaler.transform(input_data)

    # Prediction

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    risk = probability[0][1] * 100

    st.divider()

    st.header("Prediction Result")

    st.metric(
        "Estimated Diabetes Risk",
        f"{risk:.2f}%"
    )

    st.progress(risk / 100)

    # Risk Level

    if risk >= 70:
        st.error("Prediction: High Risk of Diabetes")
        st.write(
            "The model predicts a high likelihood of diabetes. Please consult a healthcare professional for further evaluation."
        )

    elif risk >= 40:
        st.warning("Moderate Risk of Diabetes")
        st.write(
            "The model predicts a moderate risk. Regular health check-ups and a healthy lifestyle are recommended."
        )

    else:
        st.success("Prediction: Low Risk of Diabetes")
        

    # ==========================================
    # Model Confidence
    # ==========================================

    st.subheader("Model Confidence")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "No Diabetes",
            f"{probability[0][0]*100:.2f}%"
        )

    with c2:
        st.metric(
            "Diabetes",
            f"{probability[0][1]*100:.2f}%"
        )

   


# ==========================================
# Footer
# ==========================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("Model")
    st.write("Logistic Regression")

with col2:
    st.caption("Accuracy")
    st.write("75.32%")

with col3:
    st.caption("Dataset")
    st.write("Pima Indians Diabetes")
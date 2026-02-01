import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model & scaler
model = load_model("breast_cancer_ann_model.h5")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🩺",
    layout="centered"
)

# Title
st.title("🩺 Breast Cancer Prediction System")
st.subheader("AI-Assisted Diagnostic Support for Doctors")

st.markdown("---")

st.info(
    "This tool predicts whether a breast tumor is **Benign** or **Malignant** "
    "based on clinical measurements. "
    "**For decision support only — not a replacement for medical diagnosis.**"
)

# Input section
st.header("📋 Enter Patient Measurements")

col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input("Radius (Mean)", 0.0, 50.0, 14.0)
    texture_mean = st.number_input("Texture (Mean)", 0.0, 50.0, 20.0)
    perimeter_mean = st.number_input("Perimeter (Mean)", 0.0, 200.0, 90.0)
    area_mean = st.number_input("Area (Mean)", 0.0, 2500.0, 600.0)
    smoothness_mean = st.number_input("Smoothness (Mean)", 0.0, 1.0, 0.1)

with col2:
    compactness_mean = st.number_input("Compactness (Mean)", 0.0, 1.0, 0.12)
    concavity_mean = st.number_input("Concavity (Mean)", 0.0, 1.0, 0.15)
    concave_points_mean = st.number_input("Concave Points (Mean)", 0.0, 1.0, 0.08)
    symmetry_mean = st.number_input("Symmetry (Mean)", 0.0, 1.0, 0.18)
    fractal_dimension_mean = st.number_input("Fractal Dimension (Mean)", 0.0, 1.0, 0.06)

st.markdown("---")

# Remaining features (simplified – fixed average values)
remaining_features = [
    0.4, 1.2, 2.5, 40.0, 0.006,
    0.03, 0.04, 0.01, 0.02, 0.004,
    16.0, 25.0, 105.0, 800.0, 0.14,
    0.30, 0.35, 0.15, 0.30, 0.08
]

# Predict button
if st.button("🔍 Predict Diagnosis"):
    patient_data = np.array([[
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean,
        symmetry_mean, fractal_dimension_mean,
        *remaining_features
    ]])

    patient_scaled = scaler.transform(patient_data)
    prediction = model.predict(patient_scaled)[0][0]

    st.markdown("## 🧾 Prediction Result")

    if prediction >= 0.5:
        st.error(f"🔴 **Malignant Tumor Detected**\n\nConfidence: {prediction*100:.2f}%")
    else:
        st.success(f"🟢 **Benign Tumor Detected**\n\nConfidence: {(1-prediction)*100:.2f}%")

st.markdown("---")
st.caption("Developed using ANN | TensorFlow | Streamlit")

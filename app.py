import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = load_model("breast_cancer_ann_model.h5")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Breast Cancer Diagnostic Assistant",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🩺 Diagnostic Assistant")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "",
    ["🏠 Home", "🧪 Prediction", "ℹ️ About Model"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ This tool is for **clinical decision support only** "
    "and must not replace professional medical judgment."
)

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "🏠 Home":
    st.title("🩺 Breast Cancer Diagnostic Assistant")
    st.subheader("AI-Powered Clinical Decision Support System")

    st.markdown("""
    ### 👨‍⚕️ Purpose
    This application assists doctors by predicting whether a breast tumor is  
    **Benign** or **Malignant** using clinical measurements.

    ### 🧠 Technology
    - Artificial Neural Network (ANN)
    - Trained on Kaggle Breast Cancer Dataset
    - High accuracy & validated model

    ### ✅ Benefits
    - Fast preliminary assessment  
    - Objective probability-based output  
    - Easy-to-use clinical interface  
    """)

    st.success("➡️ Go to **Prediction** tab to evaluate a patient.")

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "🧪 Prediction":
    st.title("🧪 Patient Tumor Assessment")
    st.caption("Enter clinical measurements below")

    st.markdown("---")

    # Input Sections
    st.markdown("### 📋 Mean Tumor Characteristics")

    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.number_input("Radius Mean", 0.0, 50.0, 14.0)
        texture_mean = st.number_input("Texture Mean", 0.0, 50.0, 20.0)
        perimeter_mean = st.number_input("Perimeter Mean", 0.0, 200.0, 90.0)

    with col2:
        area_mean = st.number_input("Area Mean", 0.0, 2500.0, 600.0)
        smoothness_mean = st.number_input("Smoothness Mean", 0.0, 1.0, 0.1)
        compactness_mean = st.number_input("Compactness Mean", 0.0, 1.0, 0.12)

    with col3:
        concavity_mean = st.number_input("Concavity Mean", 0.0, 1.0, 0.15)
        concave_points_mean = st.number_input("Concave Points Mean", 0.0, 1.0, 0.08)
        symmetry_mean = st.number_input("Symmetry Mean", 0.0, 1.0, 0.18)

    fractal_dimension_mean = st.number_input(
        "Fractal Dimension Mean", 0.0, 1.0, 0.06
    )

    st.markdown("---")

    # Remaining features (fixed mean values)
    remaining_features = [
        0.4, 1.2, 2.5, 40.0, 0.006,
        0.03, 0.04, 0.01, 0.02, 0.004,
        16.0, 25.0, 105.0, 800.0, 0.14,
        0.30, 0.35, 0.15, 0.30, 0.08
    ]

    # Predict Button
    if st.button("🔍 Run Diagnostic Prediction", use_container_width=True):
        patient_data = np.array([[
            radius_mean, texture_mean, perimeter_mean,
            area_mean, smoothness_mean, compactness_mean,
            concavity_mean, concave_points_mean,
            symmetry_mean, fractal_dimension_mean,
            *remaining_features
        ]])

        scaled_data = scaler.transform(patient_data)
        probability = model.predict(scaled_data)[0][0]

        st.markdown("## 🧾 Diagnostic Result")

        if probability >= 0.5:
            st.error("### 🔴 Malignant Tumor Detected")
            st.metric(
                label="Malignancy Probability",
                value=f"{probability*100:.2f}%"
            )
        else:
            st.success("### 🟢 Benign Tumor Detected")
            st.metric(
                label="Benign Confidence",
                value=f"{(1-probability)*100:.2f}%"
            )

        st.markdown("---")
        st.warning(
            "⚠️ This result is **AI-assisted** and should be "
            "confirmed with clinical tests and expert evaluation."
        )

# -----------------------------
# ABOUT PAGE
# -----------------------------
else:
    st.title("ℹ️ About This Model")

    st.markdown("""
    ### 🧠 Model Details
    - Type: Artificial Neural Network (ANN)
    - Layers: Input → Hidden Layers → Output
    - Activation: ReLU, Sigmoid
    - Loss Function: Binary Cross-Entropy

    ### 📊 Dataset
    - Breast Cancer Wisconsin (Diagnostic)
    - 30 numerical clinical features
    - Binary classification (Benign / Malignant)

    ### 🔐 Reliability
    - Scaled features
    - Tested on unseen data
    - High generalization performance

    ### 👨‍💻 Use Case
    - Clinical decision support
    - Educational & demonstration purposes
    """)

    st.success("This system follows real-world ML deployment practices.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("@ Tejas Nikam | Built with TensorFlow & Streamlit")

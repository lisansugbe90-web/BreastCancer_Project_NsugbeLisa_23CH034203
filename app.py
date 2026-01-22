import streamlit as st
import pandas as pd
import os
import joblib

# Optional debug: print current dir and list files
print("Current working directory:", os.getcwd())
print("Files in root:", os.listdir('.'))
print("Files in model/:", os.listdir('model') if os.path.exists('model') else "model folder missing!")

model = joblib.load("model/breast_cancer_model.joblib")
scaler = joblib.load("model/scaler.joblib")  # if you have it
st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

st.title("Breast Cancer Tumor Prediction (Educational Demo)")
st.markdown("""
**Important Disclaimer**: This is a student project for learning purposes only.  
It is **NOT** a medical diagnostic tool. Always consult a qualified doctor.
""")

# Load artifacts
model = joblib.load("model/breast_cancer_model.joblib")
scaler = joblib.load("model/scaler.joblib")

# Input fields (mean values, realistic ranges from dataset)
st.subheader("Enter Tumor Mean Features")

radius_mean    = st.number_input("Radius Mean", min_value=5.0, max_value=30.0, value=14.0, step=0.1)
texture_mean   = st.number_input("Texture Mean", min_value=8.0, max_value=40.0, value=19.0, step=0.1)
perimeter_mean = st.number_input("Perimeter Mean", min_value=40.0, max_value=190.0, value=90.0, step=1.0)
area_mean      = st.number_input("Area Mean", min_value=140.0, max_value=2500.0, value=650.0, step=10.0)
smoothness_mean = st.number_input("Smoothness Mean", min_value=0.05, max_value=0.17, value=0.10, step=0.001)

if st.button("Predict Tumor Type"):
    input_data = pd.DataFrame({
        'radius_mean': [radius_mean],
        'texture_mean': [texture_mean],
        'perimeter_mean': [perimeter_mean],
        'area_mean': [area_mean],
        'smoothness_mean': [smoothness_mean]
    })

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]  # prob of Malignant

    if prediction == 1:
        st.error(f"**Predicted: Malignant** (probability: {prob:.1%})")
    else:
        st.success(f"**Predicted: Benign** (probability: {1 - prob:.1%})")

st.markdown("---")
st.caption("Features selected: radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean | Algorithm: Logistic Regression")
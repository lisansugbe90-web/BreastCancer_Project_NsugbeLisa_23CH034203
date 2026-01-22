import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

model = joblib.load("model/breast_cancer_model.pkl")

st.title("Breast Cancer Tumor Prediction (Educational Demo)")
st.markdown("""
**Important Disclaimer**: This is a student project for learning purposes only.  
It is **NOT** a medical diagnostic tool. Always consult a qualified doctor.
""")

st.subheader("Enter Tumor Mean Features")

radius_mean = st.number_input("Radius Mean", 5.0, 30.0, 14.0, 0.1)
texture_mean = st.number_input("Texture Mean", 8.0, 40.0, 19.0, 0.1)
perimeter_mean = st.number_input("Perimeter Mean", 40.0, 190.0, 90.0, 1.0)
area_mean = st.number_input("Area Mean", 140.0, 2500.0, 650.0, 10.0)
smoothness_mean = st.number_input("Smoothness Mean", 0.05, 0.17, 0.10, 0.001)

if st.button("Predict Tumor Type"):
    input_data = pd.DataFrame([{
        "radius_mean": radius_mean,
        "texture_mean": texture_mean,
        "perimeter_mean": perimeter_mean,
        "area_mean": area_mean,
        "smoothness_mean": smoothness_mean
    }])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]  # prob malignant (1)

    if pred == 1:
        st.error(f"**Predicted: Malignant** (probability: {prob:.1%})")
    else:
        st.success(f"**Predicted: Benign** (probability: {(1-prob):.1%})")

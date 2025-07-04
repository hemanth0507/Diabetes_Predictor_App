# -*- coding: utf-8 -*-
"""diabetes_app_py.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cXKbiMODq7nrie_-NM0025F7qje--C4W
"""

import streamlit as st
import numpy as np
import joblib



def load_model():
    return joblib.load("diabetes_model.pkl")

model = load_model()

st.set_page_config(
    page_title="Diabetes Prediction",
    layout="centered",
    page_icon="🩺"
)

st.markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>🧬 Diabetes Prediction App</h1>
    <p style='text-align: center; font-size: 18px;'>Predict whether a person has diabetes based on medical test data using a trained Logistic Regression model.</p>
    <hr>
""", unsafe_allow_html=True)

st.subheader("📝 Enter Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("👴 Age", min_value=10, max_value=100, value=30)

with col2:
    mass = st.number_input("⚖️ BMI (Mass)", min_value=10.0, max_value=70.0, value=30.0)

with col3:
    insu = st.number_input("💉 Insulin Level", min_value=0, max_value=300, value=100)

col4, col5 = st.columns(2)

with col4:
    plas = st.slider("🩸 Plasma Glucose", 0, 200, 100)

with col5:
    pres = st.slider("🫀 Blood Pressure", 0, 150, 70)

predict_btn = st.button("🚀 Predict Now", use_container_width=True)

if predict_btn:
    input_data = np.array([[age, mass, insu, plas, pres]])

    with st.spinner("🔍 Analyzing medical data..."):
        prediction = model.predict(input_data)[0]

    st.markdown("---")
    st.subheader("📊 Prediction Outcome")

    if prediction == "tested_positive":
        st.error("🩺 **Result: Diabetes Positive**", icon="⚠️")
        st.markdown("""
        <div style='background-color:#ffcccc;padding:20px;border-radius:10px'>
            <b>Advice:</b> Please consult a healthcare provider for further evaluation and confirmatory tests.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("🧘 **Result: No Diabetes Detected**", icon="✅")
        st.markdown("""
        <div style='background-color:#d4edda;padding:20px;border-radius:10px'>
            <b>Great News!</b> The patient is likely non-diabetic. Maintain a healthy lifestyle.
        </div>
        """, unsafe_allow_html=True)

with st.expander("📈 Show Model Information"):
    st.markdown("""
    - **Model Used**: Logistic Regression
    - **Features Used**: Age, BMI, Insulin, Plasma Glucose, Blood Pressure
    - **Trained On**: 768 Samples
    - **Accuracy**: ~77%
    """)

st.markdown("""---""")
st.markdown("<p style='text-align:center;'>Made with ❤️ by Hemanth | Powered by Streamlit</p>", unsafe_allow_html=True)

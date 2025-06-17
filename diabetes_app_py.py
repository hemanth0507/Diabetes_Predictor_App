import streamlit as st
import numpy as np
import joblib

# Load trained model from .pkl
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

model = load_model()

# UI Setup
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🧪 Diabetes Prediction App")
st.markdown("This app predicts whether a person has diabetes based on medical input values using a trained Logistic Regression model.")

# Sidebar Input
st.sidebar.header("🔧 Input Patient Data")
age = st.sidebar.slider("Age", 10, 100, 30)
mass = st.sidebar.slider("BMI (Body Mass Index)", 10.0, 70.0, 30.0)
insu = st.sidebar.slider("Insulin Level", 0, 300, 100)
plas = st.sidebar.slider("Plasma Glucose", 0, 200, 100)
pres = st.sidebar.slider("Blood Pressure", 0, 150, 70)

# Predict Button
if st.sidebar.button("🚀 Predict"):
    input_data = np.array([[age, mass, insu, plas, pres]])
    prediction = model.predict(input_data)[0]

    st.subheader("🔍 Prediction Result")
    if prediction == "tested_positive":
        st.error("⚠️ The model predicts: **Diabetes Positive**.")
    else:
        st.success("✅ The model predicts: **No Diabetes**.")

# Footer
st.markdown("---")
st.caption("👨‍⚕️ Developed using Streamlit | Logistic Regression based | Diabetes Prediction Tool")

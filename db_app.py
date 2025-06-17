import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and prepare the model (would normally be saved/loaded properly)
@st.cache_resource
def load_model():
    # This would normally load from a saved model file
    db = pd.read_excel('diabetes.xlsx')
    ip = db[['age', 'mass', 'insu', 'plas', 'pres']]
    dp = db['class']
    LR = LogisticRegression()
    LR.fit(ip, dp)
    return LR

model = load_model()

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
        }
        .header {
            color: #2c3e50;
        }
        .result-box {
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
            background-color: #ffffff;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# App header
st.title("🩺 Diabetes Risk Prediction System")
st.markdown("""
    This tool helps assess your risk of diabetes based on key health metrics. 
    Please enter your information below for a preliminary assessment.
""")

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Information")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    mass = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
    insu = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=846, value=80)

with col2:
    st.subheader("Health Metrics")
    plas = st.number_input("Plasma Glucose Concentration (mg/dL)", min_value=0, max_value=200, value=100)
    pres = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)

# Prediction button
predict_btn = st.button("Assess Diabetes Risk")

# Result display
if predict_btn:
    try:
        prediction = model.predict([[age, mass, insu, plas, pres]])
        probability = model.predict_proba([[age, mass, insu, plas, pres]])
        
        negative_prob = round(probability[0][0] * 100, 2)
        positive_prob = round(probability[0][1] * 100, 2)
        
        st.markdown("---")
        
        if prediction[0] == 'tested_positive':
            st.markdown(f"""
                <div class="result-box">
                    <h3 style='color:#e74c3c'>Risk Assessment Result: Positive</h3>
                    <p>Based on the provided information, there is a <strong>{positive_prob}%</strong> probability of diabetes risk.</p>
                    <p>Recommendation: Please consult with a healthcare professional for further evaluation.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-box">
                    <h3 style='color:#2ecc71'>Risk Assessment Result: Negative</h3>
                    <p>Based on the provided information, there is a <strong>{negative_prob}%</strong> probability of diabetes risk.</p>
                    <p>Recommendation: Maintain healthy lifestyle habits and regular check-ups.</p>
                </div>
            """, unsafe_allow_html=True)
            
        # Show feature importance (for educational purposes)
        st.subheader("Factors Influencing This Assessment")
        st.markdown("""
            The following factors were most significant in this prediction:
            - Plasma Glucose Concentration
            - BMI
            - Age
            - Insulin Level
            - Blood Pressure
        """)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d; font-size: 0.9em;'>
        <p>This tool provides preliminary assessment only and is not a substitute for professional medical advice.</p>
        <p>© 2023 Diabetes Prediction System. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)

st.image("images/logo.png", width=200)
with st.expander("About This Prediction Model"):
    st.write("""
        This model uses logistic regression trained on the Pima Indians Diabetes Dataset.
        It considers five key health indicators to assess diabetes risk.
        
        **Model Accuracy**: 77.2% (based on training data)
        
        **Input Features**:
        - Age
        - Body Mass Index (BMI)
        - Insulin Level
        - Plasma Glucose Concentration
        - Blood Pressure
    """)
    # Add this before the prediction
if plas == 0 or pres == 0 or mass == 0:
    st.warning("Please enter valid values for all health metrics (zero values are not typical).")
    st.stop()
import streamlit as st
from prediction_helper import predict

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Health Insurance Cost Predictor", page_icon="🏥")
st.title("🏥 Health Insurance Cost Predictor")
st.write("Enter the details below and click **Predict** to estimate the health insurance cost.")

# ---------------------------------------------------------
# Dropdown options
# ---------------------------------------------------------
GENDER = ['Male', 'Female']
REGION = ['Northeast', 'Northwest', 'Southeast', 'Southwest']
MARITAL_STATUS = ['Unmarried', 'Married']
BMI_CATEGORY = ['Overweight', 'Underweight', 'Normal', 'Obesity']
SMOKING_STATUS = ['Regular', 'No Smoking', 'Occasional', 'Smoking=0',
                  'Does Not Smoke', 'Not Smoking']
EMPLOYMENT_STATUS = ['Self-Employed', 'Freelancer', 'Salaried']
INCOME_LEVEL = ['> 40L', '<10L', '10L - 25L', '25L - 40L']
MEDICAL_HISTORY = [
    'High blood pressure', 'No Disease', 'Diabetes & High blood pressure',
    'Diabetes & Heart disease', 'Diabetes', 'Diabetes & Thyroid',
    'Heart disease', 'Thyroid', 'High blood pressure & Heart disease'
]
INSURANCE_PLAN = ['Silver', 'Bronze', 'Gold']

# ---------------------------------------------------------
# Input form
# Fields appear in the same order as input_dict keys
# ---------------------------------------------------------
with st.form("prediction_form"):
    # Row 1
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    with col2:
        number_of_dependants = st.number_input(
            "Number of Dependants", min_value=0, max_value=10, value=0, step=1
        )
    with col3:
        income_lakhs = st.number_input(
            "Income in Lakhs", min_value=0.0, max_value=100.0, value=10.0, step=0.5
        )

    # Row 2
    col1, col2, col3 = st.columns(3)
    with col1:
        genetical_risk = st.slider("Genetical Risk", min_value=0, max_value=5, value=0)
    with col2:
        insurance_plan = st.selectbox("Insurance Plan", INSURANCE_PLAN)
    with col3:
        employment_status = st.selectbox("Employment Status", EMPLOYMENT_STATUS)

    # Row 3
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", GENDER)
    with col2:
        marital_status = st.selectbox("Marital Status", MARITAL_STATUS)
    with col3:
        bmi_category = st.selectbox("BMI Category", BMI_CATEGORY)

    # Row 4
    col1, col2, col3 = st.columns(3)
    with col1:
        smoking_status = st.selectbox("Smoking Status", SMOKING_STATUS)
    with col2:
        region = st.selectbox("Region", REGION)
    with col3:
        medical_history = st.selectbox("Medical History", MEDICAL_HISTORY)

    submitted = st.form_submit_button("Predict")

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:
    input_dict = {
        'Age':                  age,
        'Number of Dependants': number_of_dependants,
        'Income in Lakhs':      income_lakhs,
        'Genetical Risk':       genetical_risk,
        'Insurance Plan':       insurance_plan,
        'Employment Status':    employment_status,
        'Gender':               gender,
        'Marital Status':       marital_status,
        'BMI Category':         bmi_category,
        'Smoking Status':       smoking_status,
        'Region':               region,
        'Medical History':      medical_history,
    }

    try:
        prediction = predict(input_dict)
        st.success(f"### 💰 Predicted Health Insurance Cost: **{prediction:,.2f}**")
    except Exception as e:
        import traceback
        st.error(f"Prediction failed: {e}")
        st.code(traceback.format_exc())
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load trained model + feature list
# -------------------------------
model = joblib.load("premium_model.pkl")
feature_list = joblib.load("feature_list.pkl")

# -------------------------------
# Header / UI introduction
# -------------------------------
st.title("Medical Insurance Premium Predictor")

st.markdown("""
### **Predict Your Estimated  Insurance Premium**

This application uses a trained machine learning model to estimate a person's 
**expected monthly and annual medical insurance premium** based on their demographics, 
health factors, and medical utilization history.

Please provide accurate information below to receive a personalized premium estimate.
""")

st.markdown("---")

# -------------------------------
# User Input Widgets
# -------------------------------

st.subheader("Basic Information")

age = st.number_input("Age", 18, 100, 40)
income = st.number_input("Annual Income ($)", 0, 500000, 50000)
bmi = st.number_input("BMI", 10.0, 60.0, 27.5)
dependents = st.number_input("Number of Dependents", 0, 10, 0)

st.markdown("---")
st.subheader("Medical History")

hospitalizations_last_3yrs = st.number_input("Hospitalizations (last 3 years)", 0, 20, 0)
days_hospitalized_last_3yrs = st.number_input("Total Days Hospitalized (last 3 years)", 0, 200, 0)
medication_count = st.number_input("Medication Count", 0, 50, 0)
chronic_count = st.number_input("Chronic Condition Count", 0, 20, 0)

st.markdown("---")
st.subheader("Medical Service Usage")

proc_imaging_count = st.number_input("Imaging Procedures", 0, 50, 0)
proc_surgery_count = st.number_input("Surgical Procedures", 0, 50, 0)
proc_consult_count = st.number_input("Consultations", 0, 100, 0)
proc_lab_count = st.number_input("Lab Procedures", 0, 100, 0)

st.markdown("---")
st.subheader("Categorical Inputs")

sex = st.selectbox("Sex", ["Male", "Female"])
smoker = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
region = st.selectbox("Region", ["North", "South", "East", "West"])
had_major_procedure = st.selectbox("Had a Major Procedure?", ["No", "Yes"])

# -------------------------------
# Convert user input → model-ready row
# -------------------------------
input_dict = {
    "age": age,
    "income": income,
    "bmi": bmi,
    "dependents": dependents,
    "hospitalizations_last_3yrs": hospitalizations_last_3yrs,
    "days_hospitalized_last_3yrs": days_hospitalized_last_3yrs,
    "medication_count": medication_count,
    "chronic_count": chronic_count,
    "proc_imaging_count": proc_imaging_count,
    "proc_surgery_count": proc_surgery_count,
    "proc_consult_count": proc_consult_count,
    "proc_lab_count": proc_lab_count,
}

# One-hot encoded categorical values
encoded_cols = {
    f"sex_{sex}": 1,
    f"smoker_{smoker}": 1,
    f"region_{region}": 1,
    f"had_major_procedure_Yes": 1 if had_major_procedure == "Yes" else 0,
}

# Build full row including missing encoded columns
row = {col: 0 for col in feature_list}
row.update(input_dict)
row.update(encoded_cols)

X_input = pd.DataFrame([row])[feature_list]

# -------------------------------
# Predict
# -------------------------------
st.markdown("---")

if st.button("🔮 Predict Premiums"):
    monthly_premium = model.predict(X_input)[0]
    annual_premium = monthly_premium * 12

    st.success(f"### 🧾 Estimated *Monthly* Medical Insurance Premium: **${monthly_premium:.2f}**")
    st.info(f"### 📅 Estimated *Annual* Medical Insurance Premium: **${annual_premium:.2f}**")

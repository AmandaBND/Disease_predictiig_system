import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# -------------------------------
# Paths
# -------------------------------
MODEL_PATH = os.path.join("Train", "best_model.joblib")
ENCODER_PATH = os.path.join("Train", "label_encoder.joblib")

# Load trained model & encoder
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

st.set_page_config(page_title="Disease Prediction System", page_icon="🩺")
st.title("🩺 Disease Prediction System")
st.write("Fill in patient details to predict possible disease and get specialist recommendation.")

# -------------------------------
# User Inputs
# -------------------------------

# Age with real-world limits
age = st.number_input("Age", min_value=0, max_value=120, step=1)

gender = st.radio("Gender", ["Male", "Female"])

ethnicity = st.selectbox("Ethnicity", ["Caucasian", "Asian", "African American", "Hispanic", "Other"])

family_history = st.selectbox("Family History of Disease", ["Yes", "No"])

smoking = st.selectbox("Smoking Habit", ["No", "Occasional", "Daily"])

alcohol = st.selectbox("Alcohol Consumption", ["No", "Social", "Frequent"])

diet = st.selectbox("Diet Habits", ["Healthy", "Processed Food", "High Sugar", "Balanced"])

activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])

# Height & Weight for BMI calculation
# height = st.number_input("Height (cm)", min_value=50, max_value=250, step=1)
# weight = st.number_input("Weight (kg)", min_value=10, max_value=300, step=1)

# bmi_category = None
# if height > 0 and weight > 0:
#     bmi = weight / ((height / 100) ** 2)
#     if bmi < 18.5:
#         bmi_category = "Underweight"
#     elif 18.5 <= bmi < 25:
#         bmi_category = "Normal"
#     elif 25 <= bmi < 30:
#         bmi_category = "Overweight"
#     else:
#         bmi_category = "Obese"
#     st.write(f"**BMI:** {bmi:.2f} → {bmi_category}")

# BMI calculation
height = st.number_input("Height (cm)", min_value=50, max_value=250, step=1)
weight = st.number_input("Weight (kg)", min_value=10, max_value=300, step=1)

bmi_value = None
if height > 0 and weight > 0:
    bmi_value = weight / ((height / 100) ** 2)
    st.write(f"*BMI:* {bmi_value:.2f}")

# Generic Symptoms
symptoms = st.multiselect(
    "Select Symptoms",
    ["Cough", "Fever", "Chest Pain", "Fatigue"]
)

# Special Symptoms (multiple selection)
special_symptoms = st.multiselect(
    "Special Symptoms (optional)",
    [
        "Wheezing", "Shortness of breath", "Chest tightness", "Coughing at night",
        "Headache", "Dizziness", "Blurred vision", "Nosebleeds",
        "Joint pain", "Sleep apnea", "Back pain", "Daytime fatigue",
        "High fever", "Muscle aches", "Chills and sweats", "Sore throat",
        "Excessive thirst", "Frequent urination", "Slow-healing wounds", "Tingling in hands/feet",
        "Chest pain", "Radiating pain to arm/jaw", "Palpitations", "Cold sweating"
    ]
)
special_symptom1 = special_symptoms[0] if len(special_symptoms) > 0 else np.nan
special_symptom2 = special_symptoms[1] if len(special_symptoms) > 1 else np.nan
special_symptom3 = special_symptoms[2] if len(special_symptoms) > 2 else np.nan

# Extra fields
duration_days = st.number_input("Symptom Duration (days)", min_value=0, max_value=365, step=1)
current_medications = st.text_input("Current Medications (optional)")
pre_existing_conditions = st.text_input("Pre-existing Conditions (optional)")

# -------------------------------
# Build input dataframe
# -------------------------------
input_dict = {
    "Age": [age],
    "Gender": [gender],
    "Ethnicity": [ethnicity],
    "Family_History": [family_history],
    "Smoking": [smoking],
    "Alcohol": [alcohol],
    "Diet_Habits": [diet],
    "Physical_Activity": [activity],
    "Symptom_Cough": ["Yes" if "Cough" in symptoms else "No"],
    "Symptom_Fever": ["Yes" if "Fever" in symptoms else "No"],
    "Symptom_ChestPain": ["Yes" if "Chest Pain" in symptoms else "No"],
    "Symptom_Fatigue": ["Yes" if "Fatigue" in symptoms else "No"],
    "Special_Symptom_1": [special_symptom1],
    "Special_Symptom_2": [special_symptom2],
    "Special_Symptom_3": [special_symptom3],
    "Duration_Days": [duration_days],
    "Current_Medications": [current_medications if current_medications else np.nan],
    "Pre_existing_Conditions": [pre_existing_conditions if pre_existing_conditions else np.nan],
    "BMI_Value": [bmi_value if bmi_value else np.nan],
}

df_input = pd.DataFrame(input_dict)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Disease"):
    try:
        prediction = model.predict(df_input)[0]

        # Decode numeric prediction back to label
        if isinstance(prediction, (int, np.integer)):
            prediction = label_encoder.inverse_transform([prediction])[0]

        st.success(f"Predicted Disease: **{prediction}**")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_input)[0]
            confidence = round(max(probs) * 100, 2)
            st.write(f"Prediction Confidence: {confidence}%")

        # Specialist mapping
        mapping = {
            "Asthma": "Pulmonologist",
            "Hypertension": "Cardiologist",
            "Obesity": "Nutritionist",
            "Influenza": "General Practitioner",
            "Diabetes": "Endocrinologist",
            "Heart Disease": "Cardiologist"
        }
        specialist = mapping.get(prediction, "General Doctor")
        st.info(f"Recommended Specialist: **{specialist}**")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

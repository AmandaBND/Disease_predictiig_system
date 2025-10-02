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

st.set_page_config(page_title="Disease Prediction System", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .main {
        background: #ffffff;
    }
    .stApp {
        background: linear-gradient(135deg, #e8f4f8 0%, #f0f8ff 100%);
    }
    .stButton>button {
        background: linear-gradient(90deg, #0077b6 0%, #023e8a 100%);
        color: white;
        font-weight: bold;
        font-size: 16px;
        border-radius: 10px;
        padding: 12px 30px;
        border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.3);
        transform: translateY(-2px);
        background: linear-gradient(90deg, #023e8a 0%, #0077b6 100%);
    }
    .prediction-card {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        margin: 20px 0;
        font-size: 18px;
    }
    .success-card {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        margin: 15px 0;
        font-size: 20px;
    }
    .info-card {
        background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        margin: 15px 0;
        font-size: 18px;
    }
    .section-header {
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        font-size: 22px;
        font-weight: bold;
        margin-top: 25px;
        margin-bottom: 20px;
        padding: 12px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .bmi-normal {
        background: #16a34a;
        padding: 12px 24px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .bmi-underweight {
        background: #ea580c;
        padding: 12px 24px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .bmi-overweight {
        background: #d97706;
        padding: 12px 24px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .bmi-obese {
        background: #dc2626;
        padding: 12px 24px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    h1 {
        color: #1e3a8a;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    label, .stMarkdown, p {
        color: #1f2937 !important;
        font-weight: 500;
    }
    .stSelectbox label, .stNumberInput label, .stRadio label, .stMultiSelect label, .stTextInput label {
        color: #111827 !important;
        font-weight: 600;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🩺 Disease Prediction System")
st.write("Fill in patient details to predict possible disease and get specialist recommendation.")

# -------------------------------
# User Inputs
# -------------------------------

st.markdown('<div class="section-header">👤 Demographics</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
with col2:
    gender = st.radio("Gender", ["Male", "Female"])
with col3:
    ethnicity = st.selectbox("Ethnicity", ["Caucasian", "Asian", "African American", "Hispanic", "Other"])

col1, col2 = st.columns(2)
with col1:
    family_history = st.selectbox("Family History of Disease", ["Yes", "No"])
with col2:
    st.write("")

st.markdown('<div class="section-header">🚬 Lifestyle Factors</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    smoking = st.selectbox("Smoking Habit", ["No", "Occasional", "Daily"])
with col2:
    alcohol = st.selectbox("Alcohol Consumption", ["No", "Social", "Frequent"])
with col3:
    diet = st.selectbox("Diet Habits", ["Healthy", "Processed Food", "High Sugar", "Balanced"])
with col4:
    activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])

st.markdown('<div class="section-header">📏 Physical Measurements</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    height = st.number_input("Height (cm)", min_value=50, max_value=250, step=1)
with col2:
    weight = st.number_input("Weight (kg)", min_value=10, max_value=300, step=1)

bmi_category = None
if height > 0 and weight > 0:
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:
        bmi_category = "Underweight"
        st.markdown(f'<div class="bmi-underweight">BMI: {bmi:.2f} → {bmi_category}</div>', unsafe_allow_html=True)
    elif 18.5 <= bmi < 25:
        bmi_category = "Normal"
        st.markdown(f'<div class="bmi-normal">BMI: {bmi:.2f} → {bmi_category}</div>', unsafe_allow_html=True)
    elif 25 <= bmi < 30:
        bmi_category = "Overweight"
        st.markdown(f'<div class="bmi-overweight">BMI: {bmi:.2f} → {bmi_category}</div>', unsafe_allow_html=True)
    else:
        bmi_category = "Obese"
        st.markdown(f'<div class="bmi-obese">BMI: {bmi:.2f} → {bmi_category}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-header">🩺 Symptoms</div>', unsafe_allow_html=True)
symptoms = st.multiselect(
    "Select Symptoms",
    ["Cough", "Fever", "Chest Pain", "Fatigue"]
)

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

col1, col2 = st.columns(2)
with col1:
    duration_days = st.number_input("Symptom Duration (days)", min_value=0, max_value=365, step=1)
with col2:
    st.write("")

st.markdown('<div class="section-header">📋 Medical History</div>', unsafe_allow_html=True)
with st.expander("Additional Information (Optional)"):
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
    "BMI_Category": [bmi_category if bmi_category else np.nan],
}

df_input = pd.DataFrame(input_dict)

# -------------------------------
# Prediction
# -------------------------------
st.markdown('<div class="section-header">🔮 Prediction</div>', unsafe_allow_html=True)
if st.button("🔮 Predict Disease"):
    with st.spinner("Analyzing patient data..."):
        try:
            prediction = model.predict(df_input)[0]

            if isinstance(prediction, (int, np.integer)):
                prediction = label_encoder.inverse_transform([prediction])[0]

            st.markdown(f'<div class="success-card"><h2 style="margin:0;">Predicted Disease: {prediction}</h2></div>', unsafe_allow_html=True)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df_input)[0]
                confidence = round(max(probs) * 100, 2)
                
                st.markdown(f'<div class="prediction-card"><h3 style="margin:0;">Confidence Level: {confidence}%</h3></div>', unsafe_allow_html=True)
                st.progress(confidence / 100)

                disease_names = label_encoder.classes_
                prob_df = pd.DataFrame({
                    "Disease": disease_names,
                    "Probability (%)": [round(p * 100, 2) for p in probs]
                }).sort_values("Probability (%)", ascending=False)

                st.markdown("### 📊 All Disease Probabilities")
                st.bar_chart(prob_df.set_index("Disease"))
                
                with st.expander("View Detailed Probabilities"):
                    st.dataframe(prob_df, use_container_width=True)

            mapping = {
                "Asthma": "Pulmonologist",
                "Hypertension": "Cardiologist",
                "Obesity": "Nutritionist",
                "Influenza": "General Practitioner",
                "Diabetes": "Endocrinologist",
                "Heart Disease": "Cardiologist"
            }
            specialist = mapping.get(prediction, "General Doctor")
            
            specialist_icons = {
                "Pulmonologist": "🫁",
                "Cardiologist": "❤️",
                "Nutritionist": "🥗",
                "General Practitioner": "👨‍⚕️",
                "Endocrinologist": "💉",
                "General Doctor": "🩺"
            }
            icon = specialist_icons.get(specialist, "🩺")
            
            st.markdown(f'<div class="info-card"><h3 style="margin:0;">{icon} Recommended Specialist: {specialist}</h3></div>', unsafe_allow_html=True)
            
            st.balloons()

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

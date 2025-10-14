#Preprocess/preprocess.py

import os
from pathlib import Path
import pandas as pd
import numpy as np


# Paths

INPUT_PATH = os.path.join("Data", "Disease_dataset.csv")
OUTPUT_PATH = os.path.join("Preprocess", "processed_dataset.csv")


# Load dataset

df = pd.read_csv(INPUT_PATH)
print("Original shape:", df.shape)


# 1. Drop unwanted columns

DROP_COLS = [
    "Record_ID", "Record_Date", "Living_Area",
    "Risk_Score", "risr score", "Risk_Score_0_10",
    "Severity_Cat", "Specialist",
    "Symptom_Severity", "Symptom_Duration", "Total_Symptoms"  # <- DROPPED
]

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")


# 2. Remove duplicates

df = df.drop_duplicates()
print("After dropping duplicates:", df.shape)


# 3. Handle outliers

if "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").clip(0, 120)

if "Duration_Days" in df.columns:
    df["Duration_Days"] = pd.to_numeric(df["Duration_Days"], errors="coerce")
    df.loc[df["Duration_Days"] < 0, "Duration_Days"] = np.nan


# 4. Normalize text columns

if "Gender" in df.columns:
    df["Gender"] = df["Gender"].astype(str).str.strip().str.capitalize()

if "Diet_Habits" in df.columns:
    df["Diet_Habits"] = df["Diet_Habits"].astype(str).str.strip().str.title()

for col in ["Special_Symptom_1", "Special_Symptom_2", "Special_Symptom_3",
            "Current_Medications", "Pre_existing_Conditions"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().replace(["nan", "NaN", ""], np.nan)

# 5. Convert BMI_Category to numeric BMI_Value

if "BMI_Category" in df.columns:
    def assign_bmi_value(cat):
        if cat == "Underweight":
            return np.random.uniform(15, 18.4)
        elif cat == "Normal":
            return np.random.uniform(18.5, 24.9)
        elif cat == "Overweight":
            return np.random.uniform(25, 29.9)
        elif cat == "Obese":
            return np.random.uniform(30, 40)
        return np.nan

    df["BMI_Value"] = df["BMI_Category"].apply(assign_bmi_value)
    df = df.drop(columns=["BMI_Category"])



# Save cleaned dataset

Path("Preprocess").mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f" Cleaned dataset saved to {OUTPUT_PATH}")
print(" Final shape:", df.shape)

#Train/train.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# Paths

DATA_PATH = os.path.join("Preprocess", "processed_dataset.csv")
MODEL_PATH = os.path.join("Train", "best_model.joblib")
ENCODER_PATH = os.path.join("Train", "label_encoder.joblib")


# Load dataset

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)


# Target & Features

TARGET = "Disease"
if TARGET not in df.columns:
    raise ValueError("❌ Target column 'Disease' not found!")

X = df.drop(columns=[TARGET], errors="ignore")
y = df[TARGET]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
os.makedirs("Train", exist_ok=True)
joblib.dump(label_encoder, ENCODER_PATH)


# Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


# Preprocessing for features

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ]
)


# Models

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    "DecisionTree": DecisionTreeClassifier(
        max_depth=None, random_state=42, class_weight="balanced"
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        use_label_encoder=False
    )
}

best_model = None
best_acc = 0


# Train & Evaluate

for name, model in models.items():
    clf = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(f"{name} Classification Report:\n",
          classification_report(y_test, preds, target_names=label_encoder.classes_))

    if acc > best_acc:
        best_acc = acc
        best_model = clf


# Save best model

joblib.dump(best_model, MODEL_PATH)
print(f"\n Best model saved: {MODEL_PATH} with accuracy {best_acc:.4f}")


#UI/app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os


# Paths

MODEL_PATH = os.path.join("Train", "best_model.joblib")
ENCODER_PATH = os.path.join("Train", "label_encoder.joblib")

# Load trained model & encoder
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

st.set_page_config(page_title="Disease Prediction System", page_icon="🩺", layout="wide")


# Sidebar

with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This **Disease Prediction System** uses machine learning to predict potential diseases based on patient demographics, lifestyle factors, symptoms, and medical history.
    
    **Supported Diseases:**
    - 🫁 Asthma
    - ❤️ Hypertension
    - ⚖️ Obesity
    - 🤧 Influenza
    - 💉 Diabetes
    - 💔 Heart Disease
    
    **Features:**
    - 🔍 Multi-factor analysis
    - 📊 Confidence scoring
    - 👨‍⚕️ Specialist recommendation
    - 📏 BMI calculation
    """)
    
    st.divider()
    
    st.markdown("### 🔒 Privacy")
    st.info("All data is processed locally. No information is stored or transmitted.")
    
    st.divider()
    
    st.markdown("### 📖 Instructions")
    with st.expander("How to use"):
        st.markdown("""
        1. 📝 Fill in **Basic Info** (age, gender, ethnicity, physical measurements)
        2. 🏃 Add **Lifestyle** factors (smoking, alcohol, diet, activity)
        3. 🩺 Select **Symptoms** being experienced
        4. 📋 Add **Medical History** (optional)
        5. 🔮 Click **Predict Disease** to get results
        """)
    
    st.divider()
    
    st.markdown("### ⚠️ Disclaimer")
    st.caption("""
    This system is for informational purposes only and does not replace professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or qualified health provider.
    """)
    
    st.divider()
    
    st.markdown("**Version:** 2.0")
    st.markdown("**Last Updated:** 2025")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Hero section styling */
    .hero-banner {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 16px 48px rgba(79, 70, 229, 0.4);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 8s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    .hero-banner h1 {
        color: white !important;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 4px 8px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    .hero-banner .subtitle {
        color: rgba(255,255,255,0.95) !important;
        font-size: 1.3rem;
        font-weight: 500;
        margin-top: 0.8rem;
        position: relative;
        z-index: 1;
    }
    .hero-banner .badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        margin-top: 1rem;
        font-weight: 600;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.3);
        position: relative;
        z-index: 1;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        font-size: 16px;
        letter-spacing: 0.5px;
        border-radius: 16px;
        padding: 16px 40px;
        border: none;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .stButton>button:active {
        transform: translateY(-2px);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.9) 0%, rgba(124, 58, 237, 0.9) 100%);
        backdrop-filter: blur(20px);
        padding: 32px;
        border-radius: 24px;
        color: white;
        box-shadow: 0 16px 48px rgba(79, 70, 229, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 24px 0;
        font-size: 20px;
        position: relative;
        overflow: hidden;
    }
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 100%;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 100%);
        pointer-events: none;
    }
    
    .success-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.9) 0%, rgba(5, 150, 105, 0.9) 100%);
        backdrop-filter: blur(20px);
        padding: 32px;
        border-radius: 24px;
        color: white;
        box-shadow: 0 16px 48px rgba(16, 185, 129, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 24px 0;
        font-size: 22px;
        position: relative;
        overflow: hidden;
    }
    .success-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 100%;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 100%);
        pointer-events: none;
    }
    
    .info-card {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.9) 0%, rgba(2, 132, 199, 0.9) 100%);
        backdrop-filter: blur(20px);
        padding: 32px;
        border-radius: 24px;
        color: white;
        box-shadow: 0 16px 48px rgba(14, 165, 233, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 24px 0;
        font-size: 20px;
        position: relative;
        overflow: hidden;
    }
    .info-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 100%;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 100%);
        pointer-events: none;
    }
    .section-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%);
        color: white;
        font-size: 20px;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 25px;
        padding: 15px 25px;
        border-radius: 12px;
        box-shadow: 0 6px 15px rgba(59, 130, 246, 0.3);
        border: 1px solid rgba(255,255,255,0.2);
        position: relative;
        overflow: hidden;
    }
    .section-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #fbbf24, #f59e0b, #fbbf24);
    }
    .bmi-normal {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
        padding: 15px 30px;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
        box-shadow: 0 6px 15px rgba(22, 163, 74, 0.3);
        border: 2px solid rgba(255,255,255,0.2);
    }
    .bmi-underweight {
        background: linear-gradient(135deg, #ea580c 0%, #dc2626 100%);
        padding: 15px 30px;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
        box-shadow: 0 6px 15px rgba(234, 88, 12, 0.3);
        border: 2px solid rgba(255,255,255,0.2);
    }
    .bmi-overweight {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
        padding: 15px 30px;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
        box-shadow: 0 6px 15px rgba(217, 119, 6, 0.3);
        border: 2px solid rgba(255,255,255,0.2);
    }
    .bmi-obese {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        padding: 15px 30px;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
        box-shadow: 0 6px 15px rgba(220, 38, 38, 0.3);
        border: 2px solid rgba(255,255,255,0.2);
    }
    h1 {
        color: #1e3a8a;
        font-weight: 800;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
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
    /* Input field styling */
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput div[data-baseweb="input"] > div > input,
    .stTextInput div[data-baseweb="input"] > div > input {
        border-radius: 8px !important;
        border: 2px solid #e5e7eb !important;
        transition: all 0.3s ease;
    }
    .stSelectbox div[data-baseweb="select"]:hover > div,
    .stNumberInput div[data-baseweb="input"]:hover > div > input,
    .stTextInput div[data-baseweb="input"]:hover > div > input {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6, #1d4ed8) !important;
    }
    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #3b82f6, transparent);
        margin: 2rem 0;
    }
    /* Symptom badge styling */
    .symptom-badge {
        display: inline-block;
        padding: 8px 16px;
        margin: 4px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    .symptom-badge-special {
        background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%);
        box-shadow: 0 2px 8px rgba(168, 85, 247, 0.3);
    }
    .symptom-container {
        margin: 20px 0;
        padding: 20px;
        background: rgba(59, 130, 246, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 10px 0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
    }
    /* Better spacing for sections */
    .stMultiSelect, .stSelectbox, .stNumberInput, .stTextInput, .stRadio {
        margin-bottom: 1rem;
    }
    /* Sidebar styling for better contrast */
    [data-testid="stSidebar"] {
        background-color: #1f2937 !important;
    }
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] h3 {
        color: #f9fafb !important;
    }
    [data-testid="stSidebar"] .stMarkdown strong {
        color: #fbbf24 !important;
    }
    [data-testid="stSidebar"] hr {
        background: linear-gradient(90deg, transparent, #60a5fa, transparent) !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: #374151 !important;
        border: 1px solid #4b5563 !important;
    }
    [data-testid="stSidebar"] .st-emotion-cache-16idsys p,
    [data-testid="stSidebar"] .st-emotion-cache-16idsys {
        color: #e5e7eb !important;
    }
</style>
""", unsafe_allow_html=True)

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <h1>🩺 Disease Prediction System</h1>
    <p class="subtitle">AI-Powered Medical Diagnosis Assistance</p>
    <div class="badge">✨ Powered by Machine Learning</div>
</div>
""", unsafe_allow_html=True)

st.write("")


# Progress Tracking

if 'form_progress' not in st.session_state:
    st.session_state.form_progress = 0


# User Inputs with Tabs


tab1, tab2, tab3, tab4 = st.tabs(["👤 Basic Info", "🚬 Lifestyle", "🩺 Symptoms", "📋 Medical History"])

with tab1:
    st.markdown('<div class="section-header">👤 Demographics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("🎂 Age", min_value=0, max_value=120, step=1, help="Enter patient's age in years (0-120)")
    with col2:
        gender = st.radio("⚧️ Gender", ["Male", "Female"], help="Select patient's biological gender")
    with col3:
        ethnicity = st.selectbox("🌍 Ethnicity", ["Caucasian", "Asian", "African American", "Hispanic", "Other"], help="Select patient's ethnic background")
    
    col1, col2 = st.columns(2)
    with col1:
        family_history = st.selectbox("👪 Family History of Disease", ["Yes", "No"], help="Does the patient have a family history of chronic diseases?")
    with col2:
        st.write("")
    
    st.divider()
    
    st.markdown('<div class="section-header">📏 Physical Measurements</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        height = st.number_input("📐 Height (cm)", min_value=50, max_value=250, step=1, help="Enter height in centimeters")
    with col2:
        weight = st.number_input("⚖️ Weight (kg)", min_value=10, max_value=300, step=1, help="Enter weight in kilograms")
    
    bmi_category = None
    if height > 0 and weight > 0:
        bmi = weight / ((height / 100) ** 2)
        
        st.markdown("### 📊 BMI Gauge")
        col_a, col_b, col_c = st.columns([1,2,1])
        
        with col_b:
            if bmi < 18.5:
                bmi_category = "Underweight"
                st.markdown(f'<div class="bmi-underweight">BMI: {bmi:.2f} → {bmi_category}</div>', unsafe_allow_html=True)
                st.metric("BMI Status", bmi_category, f"{bmi:.1f}", delta_color="inverse")
                st.info("⚠️ BMI is below healthy range. Consider nutritional consultation.")
            elif 18.5 <= bmi < 25:
                bmi_category = "Normal"
                st.markdown(f'<div class="bmi-normal">BMI: {bmi:.2f} → {bmi_category}</div>', unsafe_allow_html=True)
                st.metric("BMI Status", bmi_category, f"{bmi:.1f}", delta_color="normal")
                st.success("✅ BMI is within healthy range!")
            elif 25 <= bmi < 30:
                bmi_category = "Overweight"
                st.markdown(f'<div class="bmi-overweight">BMI: {bmi:.2f} → {bmi_category}</div>', unsafe_allow_html=True)
                st.metric("BMI Status", bmi_category, f"{bmi:.1f}", delta_color="inverse")
                st.warning("⚠️ BMI indicates overweight. Lifestyle changes recommended.")
            else:
                bmi_category = "Obese"
                st.markdown(f'<div class="bmi-obese">BMI: {bmi:.2f} → {bmi_category}</div>', unsafe_allow_html=True)
                st.metric("BMI Status", bmi_category, f"{bmi:.1f}", delta_color="inverse")
                st.error("⚠️ BMI indicates obesity. Medical consultation recommended.")
            
            bmi_ranges = ["<18.5", "18.5-25", "25-30", ">30"]
            bmi_colors = ["#ea580c", "#16a34a", "#d97706", "#dc2626"]
            current_range_idx = 0 if bmi < 18.5 else (1 if bmi < 25 else (2 if bmi < 30 else 3))
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; margin-top: 15px;">
                {''.join([f'<div style="flex: 1; height: 20px; background: {bmi_colors[i]}; opacity: {"1" if i == current_range_idx else "0.3"}; margin: 0 2px; border-radius: 4px;"></div>' for i in range(4)])}
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 11px; color: #666;">
                {''.join([f'<div style="flex: 1; text-align: center;">{r}</div>' for r in bmi_ranges])}
            </div>
            """, unsafe_allow_html=True)
    else:
        if height == 0 or weight == 0:
            st.info("ℹ️ Please enter both height and weight to calculate BMI.")

with tab2:
    st.markdown('<div class="section-header">🚬 Lifestyle Factors</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        smoking = st.selectbox("🚬 Smoking Habit", ["No", "Occasional", "Daily"], help="Patient's smoking frequency")
    with col2:
        alcohol = st.selectbox("🍺 Alcohol Consumption", ["No", "Social", "Frequent"], help="Patient's alcohol consumption pattern")
    with col3:
        diet = st.selectbox("🍽️ Diet Habits", ["Healthy", "Processed Food", "High Sugar", "Balanced"], help="Primary dietary pattern")
    with col4:
        activity = st.selectbox("🏃 Physical Activity", ["Low", "Moderate", "High"], help="Level of regular physical exercise")

with tab3:
    st.markdown('<div class="section-header">🩺 Symptoms</div>', unsafe_allow_html=True)
    symptoms = st.multiselect(
        "🩹 Select Symptoms",
        ["Cough", "Fever", "Chest Pain", "Fatigue"],
        help="Select all primary symptoms the patient is experiencing"
    )
    
    if symptoms:
        st.markdown('<div class="symptom-container">', unsafe_allow_html=True)
        st.markdown("**Selected Primary Symptoms:**")
        badges_html = ''.join([f'<span class="symptom-badge">{s}</span>' for s in symptoms])
        st.markdown(badges_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    special_symptoms = st.multiselect(
        "🔬 Special Symptoms (optional)",
        [
            "Wheezing", "Shortness of breath", "Chest tightness", "Coughing at night",
            "Headache", "Dizziness", "Blurred vision", "Nosebleeds",
            "Joint pain", "Sleep apnea", "Back pain", "Daytime fatigue",
            "High fever", "Muscle aches", "Chills and sweats", "Sore throat",
            "Excessive thirst", "Frequent urination", "Slow-healing wounds", "Tingling in hands/feet",
            "Chest pain", "Radiating pain to arm/jaw", "Palpitations", "Cold sweating"
        ],
        help="Additional disease-specific symptoms (select up to 3)"
    )
    
    if special_symptoms:
        st.markdown('<div class="symptom-container">', unsafe_allow_html=True)
        st.markdown("**Selected Special Symptoms:**")
        badges_html = ''.join([f'<span class="symptom-badge symptom-badge-special">{s}</span>' for s in special_symptoms])
        st.markdown(badges_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    special_symptom1 = special_symptoms[0] if len(special_symptoms) > 0 else np.nan
    special_symptom2 = special_symptoms[1] if len(special_symptoms) > 1 else np.nan
    special_symptom3 = special_symptoms[2] if len(special_symptoms) > 2 else np.nan
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        duration_days = st.number_input("⏱️ Symptom Duration (days)", min_value=0, max_value=365, step=1, help="How many days has the patient been experiencing these symptoms?")
    with col2:
        st.write("")

with tab4:
    st.markdown('<div class="section-header">📋 Medical History</div>', unsafe_allow_html=True)
    with st.expander("📝 Additional Information (Optional)"):
        current_medications = st.text_input("💊 Current Medications (optional)", help="List any medications the patient is currently taking")
        pre_existing_conditions = st.text_input("🏥 Pre-existing Conditions (optional)", help="Any known chronic conditions or medical diagnoses")


# Build input dataframe & Calculate Progress


# Calculate form completion
filled_fields = 0
total_fields = 10

if age > 0:
    filled_fields += 1
if gender:
    filled_fields += 1
if ethnicity:
    filled_fields += 1
if family_history:
    filled_fields += 1
if smoking:
    filled_fields += 1
if alcohol:
    filled_fields += 1
if diet:
    filled_fields += 1
if activity:
    filled_fields += 1
if height > 0 and weight > 0:
    filled_fields += 1
if len(symptoms) > 0 or duration_days > 0:
    filled_fields += 1

progress_percentage = (filled_fields / total_fields) * 100

st.markdown('<div class="section-header">📊 Form Completion</div>', unsafe_allow_html=True)
st.progress(progress_percentage / 100)
st.write(f"**{progress_percentage:.0f}% Complete** ({filled_fields}/{total_fields} sections filled)")

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
    "BMI_Value": [bmi if bmi else np.nan],
}

df_input = pd.DataFrame(input_dict)


# Prediction

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


# Footer

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🩺 Disease Prediction System**")
    st.caption("AI-powered medical diagnosis assistance")
with col2:
    st.markdown("**👥 Team**")
    st.caption("Developed by Medical AI Team")
with col3:
    st.markdown("**📧 Contact**")
    st.caption("For support: support@example.com")

st.caption("© 2025 Disease Prediction System. All rights reserved. | Version 2.0")


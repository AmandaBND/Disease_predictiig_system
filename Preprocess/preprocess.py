import os
from pathlib import Path
import pandas as pd
import numpy as np

# -------------------------------
# Paths
# -------------------------------
INPUT_PATH = os.path.join("Data", "Disease_dataset.csv")
OUTPUT_PATH = os.path.join("Preprocess", "processed_dataset.csv")

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(INPUT_PATH)
print("Original shape:", df.shape)

# -------------------------------
# 1. Drop unwanted columns
# -------------------------------
KEEP_COLS = [
    "Age", "Gender", "Ethnicity", "Family_History", "Smoking", "Alcohol",
    "Diet_Habits", "Physical_Activity",
    "Symptom_Cough", "Symptom_Fever", "Symptom_ChestPain", "Symptom_Fatigue",
    "Special_Symptom_1", "Special_Symptom_2", "Special_Symptom_3",
    "Duration_Days", "Current_Medications", "Pre_existing_Conditions",
    "BMI_Category",  # computed in app, but may exist in dataset
    "Disease"
]
df = df[[col for col in KEEP_COLS if col in df.columns]]

# -------------------------------
# 2. Remove duplicates
# -------------------------------
df = df.drop_duplicates()
print("After dropping duplicates:", df.shape)

# -------------------------------
# 3. Handle outliers
# -------------------------------
if "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").clip(0, 120)

if "Duration_Days" in df.columns:
    df["Duration_Days"] = pd.to_numeric(df["Duration_Days"], errors="coerce")
    df.loc[df["Duration_Days"] < 0, "Duration_Days"] = np.nan

# -------------------------------
# 4. Normalize text columns
# -------------------------------
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].astype(str).str.strip().str.capitalize()

if "Diet_Habits" in df.columns:
    df["Diet_Habits"] = df["Diet_Habits"].astype(str).str.strip().str.title()

for col in ["Special_Symptom_1", "Special_Symptom_2", "Special_Symptom_3",
            "Current_Medications", "Pre_existing_Conditions"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().replace(["nan", "NaN", ""], np.nan)

# -------------------------------
# Save cleaned dataset
# -------------------------------
Path("Preprocess").mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Cleaned dataset saved to {OUTPUT_PATH}")
print("📊 Final shape:", df.shape)

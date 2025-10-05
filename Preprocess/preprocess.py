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
DROP_COLS = [
    "Record_ID", "Record_Date", "Living_Area",
    "Risk_Score", "risr score", "Risk_Score_0_10",
    "Severity_Cat", "Specialist",
    "Symptom_Severity", "Symptom_Duration", "Total_Symptoms"  # <- DROPPED
]

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

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


# -------------------------------
# Save cleaned dataset
# -------------------------------
Path("Preprocess").mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f" Cleaned dataset saved to {OUTPUT_PATH}")
print(" Final shape:", df.shape)

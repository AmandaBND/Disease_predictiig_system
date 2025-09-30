# 🩺 Disease Prediction System

A machine learning project to predict diseases (Asthma, Hypertension, Obesity, Influenza, Diabetes, Heart Disease) 
based on patient demographics, lifestyle, and symptoms.


## 🧹 Data Preprocessing Techniques

- Dropped irrelevant columns (IDs, metadata, risk scores).
- Removed duplicate rows to avoid bias.
- Clipped unrealistic values:
  - Age → [0, 120]
  - Negative duration → NaN
- Normalized categorical values:
  - Gender standardized (Male/Female)
  - Diet habits capitalized (Vegetarian/Non-Vegetarian)
- Cleaned text fields (replaced "nan"/"NaN"/empty → NaN).
- Converted BMI category → numeric BMI value.
- Saved cleaned dataset for model training.



# System Diagram

              ┌─────────────────────────┐
              │  Preprocessed Dataset   │
              │ processed_dataset.csv   │
              └────────────┬────────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Split Features (X)│
                 │ and Target (y)    │
                 └─────────┬─────────┘
                           │
                           ▼
              ┌───────────────────────────┐
              │ Encode Target (LabelEncoder)│
              │ e.g. Flu→0, Covid→1, etc. │
              └────────────┬──────────────┘
                           │
                           ▼
               ┌─────────────────────────┐
               │ Train/Test Split (80/20)│
               └────────────┬────────────┘
                           │
                           ▼
     ┌───────────────────────────────────────────────┐
     │                 Preprocessing                 │
     │-----------------------------------------------│
     │ Numeric Columns: Impute (median) + Scale      │
     │ Categorical Columns: Impute (mode) + One-Hot  │
     └────────────────────────┬─────────────────────┘
                              │
                              ▼
           ┌─────────────────────────────┐
           │       Train Models          │
           │-----------------------------│
           │ RandomForestClassifier      │
           │ HistGradientBoostingClassifier │
           └─────────────┬──────────────┘
                         │
                         ▼
               ┌────────────────────┐
               │ Evaluate Models     │
               │ Accuracy + Report   │
               └─────────┬──────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
                   
   │ Best Model? │── Yes ─────────▶ Save Model  │
                                  best_model.joblib 
                                   


##  Project Structure

##  Run Locally
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess data
python Preprocess/preprocess.py

# 3. Train model
python Train/train.py

# 4. Launch Streamlit app
streamlit run UI/app.py


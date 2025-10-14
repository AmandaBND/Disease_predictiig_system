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

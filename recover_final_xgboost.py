
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/raw/diabetic_data.csv"
SPLIT_DIR = ROOT / "data/processed/split"
MODEL_DIR = ROOT / "models"

TARGET = "readmitted_30"
PATIENT_ID = "patient_nbr"
ID_COLUMNS = ["encounter_id", PATIENT_ID]
FEATURES_TO_REMOVE = ["examide", "citoglipton", "weight", "payer_code"]
THRESHOLD = 0.15
RANDOM_STATE = 42

# Load data and recreate the target exactly as in the notebook.
df = pd.read_csv(DATA_PATH)
df[TARGET] = (df["readmitted"] == "<30").astype(int)

train_patients = pd.read_csv(SPLIT_DIR / "train_patients.csv")[PATIENT_ID]
test_patients = pd.read_csv(SPLIT_DIR / "test_patients.csv")[PATIENT_ID]

overlap = set(train_patients).intersection(set(test_patients))
if overlap:
    raise ValueError(f"Patient split overlap detected: {len(overlap)}")

train_df = df[df[PATIENT_ID].isin(train_patients)].copy()
test_df = df[df[PATIENT_ID].isin(test_patients)].copy()

y_train = train_df[TARGET].copy()
y_test = test_df[TARGET].copy()

X_train = train_df.drop(columns=[TARGET] + ID_COLUMNS)
X_test = test_df.drop(columns=[TARGET] + ID_COLUMNS)

X_train = X_train.drop(columns=["readmitted"], errors="ignore")
X_test = X_test.drop(columns=["readmitted"], errors="ignore")

X_train = X_train.drop(columns=FEATURES_TO_REMOVE, errors="ignore")
X_test = X_test.drop(columns=FEATURES_TO_REMOVE, errors="ignore")

X_train = X_train.replace("?", np.nan)
X_test = X_test.replace("?", np.nan)

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Train and test feature columns do not match.")

numerical_features = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]
categorical_features = [
    col for col in X_train.columns if col not in numerical_features
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features),
])

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])

print("Training encounters:", len(X_train))
print("Test encounters:", len(X_test))
print("Input features:", X_train.shape[1])
print("Positive training examples:", int(y_train.sum()))
print("Positive test examples:", int(y_test.sum()))

xgb_pipeline.fit(X_train, y_train)
probabilities = xgb_pipeline.predict_proba(X_test)[:, 1]
predictions = (probabilities >= THRESHOLD).astype(int)

print("\nHeld-out test metrics at threshold", THRESHOLD)
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 4))
print("F1:", round(f1_score(y_test, predictions, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print("PR-AUC:", round(average_precision_score(y_test, probabilities), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))

# Save to new names; preserve every existing baseline artifact.
final_model_path = MODEL_DIR / "xgboost_final.joblib"
threshold_path = MODEL_DIR / "xgboost_final_threshold.joblib"

if final_model_path.exists() or threshold_path.exists():
    raise FileExistsError(
        "A final artifact already exists. Inspect it before replacing anything."
    )

joblib.dump(xgb_pipeline, final_model_path)
joblib.dump(THRESHOLD, threshold_path)

print("\nSaved final pipeline:", final_model_path)
print("Saved threshold:", threshold_path)

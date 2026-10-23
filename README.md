# AI Hospital Intelligence

## AI-Driven Predictive Data Mining for 30-Day Hospital Readmission and Hospital Resource Optimization

A research prototype that predicts the probability of 30-day hospital readmission using patient encounter information from the UCI Diabetes 130-US Hospitals dataset.

> **Current semester scope:** 30-day readmission prediction. Hospital resource forecasting and optimization are planned future extensions and are not claimed as implemented.

## 1. Project Objectives

- Predict whether a patient encounter is associated with readmission within 30 days.
- Compare Logistic Regression, Random Forest, and XGBoost.
- Evaluate performance using accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC.
- Expose model predictions through a FastAPI backend.
- Provide an interactive Streamlit interface for individual and CSV batch predictions.
- Record model evaluation results for project documentation.

## 2. Technology Stack

- **Language:** Python
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn, XGBoost
- **Model persistence:** joblib
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **Dataset:** UCI Diabetes 130-US Hospitals

## 3. Dataset and Target

The dataset contains 101,766 hospital encounters involving 71,518 patients.

The target is defined as:

- `1`: The dataset's `readmitted` value is `<30`.
- `0`: Otherwise.

The data contains repeated encounters for some patients. The project uses a patient-level train/test split to prevent the same patient from appearing in both partitions.

The reconstructed final-model evaluation used:

- Training encounters: 81,477
- Test encounters: 20,289
- Raw input features: 43
- Encoded features: 2,391

## 4. Data Preprocessing

The model pipeline uses:

- Missing-value handling for numerical and categorical columns.
- Median imputation and standard scaling for numerical features.
- Categorical imputation using an `Unknown` value.
- One-hot encoding with unknown categories ignored.
- Conversion of `?` values to missing values.

The saved final XGBoost pipeline includes preprocessing, so it accepts the original 43 input features.

## 5. Models

Three model families are available in the application:

1. Logistic Regression — baseline model.
2. Random Forest — baseline model.
3. XGBoost — selected final model.

The final XGBoost configuration is:

| Parameter | Value |
|---|---:|
| `n_estimators` | 200 |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `random_state` | 42 |
| Classification threshold | 0.15 |

The threshold determines how predicted probabilities are converted into binary classifications. It is a research setting, not a clinically approved cutoff.

## 6. Recovered Final XGBoost Evaluation

The following results come from the recovered model's fresh evaluation on the held-out test set.

| Metric | Result |
|---|---:|
| Accuracy | 79.12% |
| Precision | 22.72% |
| Recall | 39.70% |
| F1-score | 0.2890 |
| ROC-AUC | 0.6868 |
| PR-AUC | 0.2357 |

Confusion matrix:

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 15,191 | 2,929 |
| Actual positive | 1,308 | 861 |

Accuracy alone is insufficient for this imbalanced classification task. Recall, precision, F1-score, ROC-AUC, and PR-AUC are also important.

The older notebook evaluation contains different metrics from a previous run. Those results should not be combined with the recovered model's metrics as though they came from the same experiment.

Detailed recovered-model metrics are recorded in `reports/model_evaluation_report.json`.

## 7. Project Structure

```text
ai-hospital-intelligence/
├── backend/
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
│       └── split/
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── components.py
│   ├── model_features.txt
│   └── test_patients.csv
├── models/
│   ├── logistic_regression_baseline.joblib
│   ├── preprocessor.joblib
│   ├── random_forest_baseline.joblib
│   ├── xgboost_baseline.joblib
│   ├── xgboost_final.joblib
│   └── xgboost_final_threshold.joblib
├── notebooks/
├── reports/
│   ├── create_evaluation_report.py
│   └── model_evaluation_report.json
├── .venv/
├── README.md
└── README_backup.md
```

## 8. Setup

Run commands from the project root in PowerShell.

Activate the existing virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages if they are not already installed:

```powershell
python -m pip install -r frontend/requirements.txt
```

Ensure the dataset and model artifacts are present before starting the application.

## 9. Run the Backend

Open a terminal at the project root and activate the virtual environment.

```powershell
python -m uvicorn backend.main:app --reload
```

The API is available at:

- Base URL: `http://127.0.0.1:8000`
- Health endpoint: `http://127.0.0.1:8000/health`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

Keep the backend terminal running.

## 10. Run the Frontend

Open a second terminal at the project root.

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run .\frontend\app.py
```

Open the local Streamlit URL printed in the terminal, normally:

`http://localhost:8501`

The interface supports model selection, threshold adjustment, individual predictions, CSV batch predictions, and CSV output downloads.

## 11. Prediction API

The prediction endpoint is:

`POST /predict`

Example request structure:

```json
{
  "model": "xgboost",
  "threshold": 0.15,
  "patient": {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[70-80)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 3,
    "medical_specialty": "InternalMedicine",
    "num_lab_procedures": 40,
    "num_procedures": 0,
    "num_medications": 10,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "diag_1": "428",
    "diag_2": "276",
    "diag_3": "250",
    "number_diagnoses": 5,
    "max_glu_serum": "None",
    "A1Cresult": "None",
    "metformin": "No",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "acetohexamide": "No",
    "glipizide": "No",
    "glyburide": "No",
    "tolbutamide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "insulin": "No",
    "glyburide-metformin": "No",
    "glipizide-metformin": "No",
    "glimepiride-pioglitazone": "No",
    "metformin-rosiglitazone": "No",
    "metformin-pioglitazone": "No",
    "change": "No",
    "diabetesMed": "No"
  }
}
```

This is an illustrative request structure. Use the application's actual accepted category values and provide all required patient features.

The API returns the selected model, predicted class, estimated probability, threshold, and a research-prototype disclaimer.

## 12. Model Artifacts

- `models/xgboost_final.joblib`: Final preprocessing and XGBoost pipeline.
- `models/xgboost_final_threshold.joblib`: Saved classification threshold.
- `models/logistic_regression_baseline.joblib`: Logistic Regression baseline.
- `models/random_forest_baseline.joblib`: Random Forest baseline.
- `models/preprocessor.joblib`: Standalone preprocessor for the baseline models.

The API expects 43 raw features for the final XGBoost pipeline.

## 13. Limitations and Responsible Use

- This project is an academic research prototype, not a clinically validated medical device.
- Predictions are estimates and must not replace professional clinical judgement.
- The dataset represents historical encounters and may not generalize to other hospitals or current clinical populations.
- The selected threshold involves a trade-off between precision and recall.
- Evaluation on a held-out dataset does not establish clinical utility or prospective effectiveness.
- Patient privacy, fairness, calibration, external validation, and clinical review would require additional work before real-world deployment.

## 14. Future Work

Planned extensions include:

- Explainability using SHAP.
- Hospital resource-demand forecasting.
- Resource optimization and allocation.
- An integrated hospital intelligence dashboard.
- Additional validation, calibration, fairness analysis, and clinical review.

These are future extensions and are not represented as completed features in the current semester's readmission prediction scope.

## 15. Evaluation Report

See `reports/model_evaluation_report.json` for the recorded metrics and evaluation configuration.

The report contains the recovered model's fresh evaluation results, which are distinct from the older results stored in the notebook.

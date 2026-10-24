
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"

app = FastAPI(
    title="Hospital Readmission Prediction API",
    version="1.1.0",
)

FEATURES = [
    "race", "gender", "age", "admission_type_id",
    "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "medical_specialty",
    "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient",
    "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "metformin",
    "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide",
    "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "change", "diabetesMed",
]

BASELINE_MODEL_FILES = {
    "logistic_regression": "logistic_regression_baseline.joblib",
    "random_forest": "random_forest_baseline.joblib",
}

DEFAULT_THRESHOLD = 0.15

models = {}
load_errors = {}
final_threshold = DEFAULT_THRESHOLD


class PredictionRequest(BaseModel):
    patient: dict
    model: Literal[
        "logistic_regression", "random_forest", "xgboost"
    ] = "xgboost"
    threshold: float | None = Field(default=None, ge=0, le=1)


@app.on_event("startup")
def load_models():
    global models, load_errors, final_threshold

    models = {}
    load_errors = {}
    final_threshold = DEFAULT_THRESHOLD

    # Load the final XGBoost pipeline. It includes preprocessing.
    try:
        final_model_path = MODEL_DIR / "xgboost_final.joblib"
        threshold_path = MODEL_DIR / "xgboost_final_threshold.joblib"

        final_pipeline = joblib.load(final_model_path)
        saved_threshold = float(joblib.load(threshold_path))

        if not 0 <= saved_threshold <= 1:
            raise ValueError("Saved threshold must be between 0 and 1.")

        if not hasattr(final_pipeline, "feature_names_in_"):
            raise ValueError("Final pipeline has no recorded input feature names.")

        expected_features = list(final_pipeline.feature_names_in_)
        if set(expected_features) != set(FEATURES):
            raise ValueError(
                "Final XGBoost feature schema does not match the API schema."
            )

        models["xgboost"] = final_pipeline
        final_threshold = saved_threshold

    except Exception as exc:
        load_errors["xgboost"] = str(exc)

    # Load the existing baseline models.
    # They use the separately saved preprocessor.
    try:
        preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    except Exception as exc:
        preprocessor = None
        load_errors["preprocessor"] = str(exc)

    for name, filename in BASELINE_MODEL_FILES.items():
        try:
            models[name] = {
                "classifier": joblib.load(MODEL_DIR / filename),
                "preprocessor": preprocessor,
            }
            if preprocessor is None:
                raise RuntimeError("Baseline preprocessor is unavailable.")
        except Exception as exc:
            models.pop(name, None)
            load_errors[name] = str(exc)


class PredictionRequest(BaseModel):
    patient: dict
    model: Literal[
        "logistic_regression", "random_forest", "xgboost"
    ] = "xgboost"
    threshold: float | None = Field(default=None, ge=0, le=1)


@app.get("/")
def root():
    return {
        "message": "Hospital Readmission Prediction API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if models else "error",
        "available_models": list(models.keys()),
        "xgboost_final_loaded": "xgboost" in models,
        "xgboost_threshold": final_threshold,
        "load_errors": load_errors,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    model_name = request.model

    if model_name not in models:
        raise HTTPException(
            status_code=503,
            detail={
                "message": f"Model '{model_name}' is unavailable.",
                "load_errors": load_errors,
            },
        )

    patient = request.patient
    missing = set(FEATURES) - set(patient)
    extra = set(patient) - set(FEATURES)

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": sorted(missing),
                "extra_features": sorted(extra),
            },
        )

    threshold = (
        request.threshold
        if request.threshold is not None
        else final_threshold if model_name == "xgboost"
        else DEFAULT_THRESHOLD
    )

    # Keep the original feature order and treat literal '?' as missing.
    patient_df = pd.DataFrame(
        [{feature: patient[feature] for feature in FEATURES}]
    ).replace("?", float("nan"))

    try:
        if model_name == "xgboost":
            # The final pipeline performs preprocessing internally.
            pipeline = models["xgboost"]
            probability = float(pipeline.predict_proba(patient_df)[0, 1])
        else:
            # Baseline models require the separately saved preprocessor.
            bundle = models[model_name]
            preprocessor = bundle["preprocessor"]
            classifier = bundle["classifier"]

            expected = list(preprocessor.feature_names_in_)
            ordered_df = patient_df[expected]
            processed = preprocessor.transform(ordered_df)

            probability = float(classifier.predict_proba(processed)[0, 1])

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    predicted_class = int(probability >= threshold)

    return {
        "model": model_name,
        "predicted_class": predicted_class,
        "prediction": (
            "30-day readmission predicted"
            if predicted_class
            else "30-day readmission not predicted"
        ),
        "probability": probability,
        "probability_percent": round(probability * 100, 2),
        "threshold": threshold,
        "disclaimer": (
            "Research prototype only. This result is not a diagnosis "
            "or a substitute for professional clinical judgment."
        ),
    }

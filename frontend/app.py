import pandas as pd
import streamlit as st

from api_client import APIClientError, check_backend, predict_patient
from components import render_patient_form


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 30-Day Hospital Readmission Prediction")

st.caption(
    "AI-driven research prototype for predicting readmission risk "
    "using trained machine-learning models."
)

st.warning(
    "Research prototype only. Predictions are not medical diagnoses "
    "and must not be used as the sole basis for clinical decisions."
)


# --------------------------------------------------
# SIDEBAR: MODEL SETTINGS AND BACKEND STATUS
# --------------------------------------------------

with st.sidebar:
    st.header("Prediction settings")

    model = st.selectbox(
        "Select trained model",
        ["xgboost", "logistic_regression", "random_forest"],
        format_func=lambda x: {
            "xgboost": "XGBoost",
            "logistic_regression": "Logistic Regression",
            "random_forest": "Random Forest",
        }[x],
    )

    threshold = st.slider(
        "Classification threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.15,
        step=0.01,
        help=(
            "Provisional threshold. Validate this value against "
            "your model evaluation before interpreting results."
        ),
    )

    st.divider()
    st.subheader("Backend status")

    try:
        health = check_backend()

        if health.get("status") == "ok":
            st.success("Backend connected")
        else:
            st.error("Backend is not ready")

        available_models = health.get("available_models", [])

        st.write(
            "Loaded models:",
            ", ".join(available_models) or "None",
        )

        if model not in available_models:
            st.warning("The selected model is unavailable.")

    except APIClientError as exc:
        st.error(str(exc))
        available_models = []


# --------------------------------------------------
# TABS
# --------------------------------------------------

single_tab, batch_tab = st.tabs(
    [
        "Single patient prediction",
        "CSV batch prediction",
    ]
)


# --------------------------------------------------
# HELPER: DISPLAY PREDICTION
# --------------------------------------------------

def display_prediction(result):
    probability = result["probability_percent"]

    st.subheader("Prediction result")

    if result["predicted_class"] == 1:
        st.error("30-day readmission predicted")
    else:
        st.success("30-day readmission not predicted")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Estimated probability",
        f"{probability:.2f}%",
    )

    col2.metric(
        "Classification threshold",
        f"{result['threshold']:.2f}",
    )

    col3.metric(
        "Model",
        result["model"].replace("_", " ").title(),
    )

    st.progress(
        min(max(probability / 100, 0.0), 1.0)
    )

    st.caption(result.get("disclaimer", ""))


# --------------------------------------------------
# HELPER: CONVERT PANDAS VALUES TO JSON-SAFE VALUES
# --------------------------------------------------

def make_json_safe(value):
    """
    Convert pandas/NumPy values into JSON-compatible Python values.

    Missing values become None, which requests serializes as JSON null.
    NumPy scalar values are converted to native Python values.
    """

    if value is None or value is pd.NA:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        value = value.item()

        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

    return value


# --------------------------------------------------
# TAB 1: SINGLE PATIENT PREDICTION
# --------------------------------------------------

with single_tab:
    st.markdown(
        "Enter the patient attributes required by the trained model."
    )

    with st.form("single_patient_form"):
        patient = render_patient_form()

        submitted = st.form_submit_button(
            "Predict 30-day readmission risk",
            type="primary",
            disabled=model not in available_models,
        )

    if submitted:
        with st.spinner(
            "Requesting prediction from the backend..."
        ):
            try:
                safe_patient = {
                    feature: make_json_safe(value)
                    for feature, value in patient.items()
                }

                result = predict_patient(
                    patient=safe_patient,
                    model=model,
                    threshold=threshold,
                )

                display_prediction(result)

            except APIClientError as exc:
                st.error(str(exc))


# --------------------------------------------------
# TAB 2: CSV BATCH PREDICTION
# --------------------------------------------------

with batch_tab:
    st.markdown("### Upload patient data")

    st.write(
        "Upload a CSV containing the same 43 feature columns "
        "used by the trained preprocessor. Missing values are "
        "sent as null and handled by the preprocessing pipeline."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="batch_csv",
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            st.write(
                f"Rows: {len(batch_df)} | "
                f"Columns: {len(batch_df.columns)}"
            )

            st.dataframe(
                batch_df.head(10),
                use_container_width=True,
            )

            # Expected model input schema: 43 features.
            expected_columns = [
                "race",
                "gender",
                "age",
                "admission_type_id",
                "discharge_disposition_id",
                "admission_source_id",
                "time_in_hospital",
                "medical_specialty",
                "num_lab_procedures",
                "num_procedures",
                "num_medications",
                "number_outpatient",
                "number_emergency",
                "number_inpatient",
                "diag_1",
                "diag_2",
                "diag_3",
                "number_diagnoses",
                "max_glu_serum",
                "A1Cresult",
                "metformin",
                "repaglinide",
                "nateglinide",
                "chlorpropamide",
                "glimepiride",
                "acetohexamide",
                "glipizide",
                "glyburide",
                "tolbutamide",
                "pioglitazone",
                "rosiglitazone",
                "acarbose",
                "miglitol",
                "troglitazone",
                "tolazamide",
                "insulin",
                "glyburide-metformin",
                "glipizide-metformin",
                "glimepiride-pioglitazone",
                "metformin-rosiglitazone",
                "metformin-pioglitazone",
                "change",
                "diabetesMed",
            ]

            missing = sorted(
                set(expected_columns) - set(batch_df.columns)
            )

            extra = sorted(
                set(batch_df.columns) - set(expected_columns)
            )

            if missing or extra:
                if missing:
                    st.error(
                        "Missing columns: "
                        + ", ".join(missing)
                    )

                if extra:
                    st.error(
                        "Unexpected columns: "
                        + ", ".join(extra)
                    )

            elif batch_df.empty:
                st.error(
                    "The uploaded CSV contains no patient rows."
                )

            elif len(batch_df) > 500:
                st.error(
                    "Upload at most 500 rows per batch."
                )

            elif st.button(
                "Run batch predictions",
                type="primary",
                disabled=model not in available_models,
                key="run_batch_predictions",
            ):
                results = []

                progress = st.progress(0.0)
                status = st.empty()

                for index, (_, row) in enumerate(
                    batch_df.iterrows()
                ):
                    status.write(
                        f"Processing patient {index + 1} "
                        f"of {len(batch_df)}..."
                    )

                    try:
                        # Convert NaN/pandas values to JSON-safe values.
                        patient_data = {
                            feature: make_json_safe(value)
                            for feature, value in row.items()
                        }

                        result = predict_patient(
                            patient=patient_data,
                            model=model,
                            threshold=threshold,
                        )

                        results.append(
                            {
                                "predicted_class": result[
                                    "predicted_class"
                                ],
                                "prediction": result[
                                    "prediction"
                                ],
                                "probability_percent": result[
                                    "probability_percent"
                                ],
                                "model": result["model"],
                                "error": None,
                            }
                        )

                    except APIClientError as exc:
                        st.error(
                            f"Row {index + 1} failed: {exc}"
                        )

                        results.append(
                            {
                                "predicted_class": None,
                                "prediction": "Error",
                                "probability_percent": None,
                                "model": model,
                                "error": str(exc),
                            }
                        )

                    progress.progress(
                        (index + 1) / len(batch_df)
                    )

                # Combine input rows and prediction results.
                output_df = pd.concat(
                    [
                        batch_df.reset_index(drop=True),
                        pd.DataFrame(results),
                    ],
                    axis=1,
                )

                success_count = sum(
                    result["prediction"] != "Error"
                    for result in results
                )

                status.success(
                    f"Batch processing finished: "
                    f"{success_count}/{len(batch_df)} "
                    f"predictions succeeded."
                )

                st.dataframe(
                    output_df,
                    use_container_width=True,
                )

                # Export all rows, including errors if any.
                csv_bytes = output_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download prediction results",
                    data=csv_bytes,
                    file_name="readmission_predictions.csv",
                    mime="text/csv",
                    key="download_batch_results",
                )

        except (
            ValueError,
            pd.errors.ParserError,
            UnicodeDecodeError,
        ) as exc:
            st.error(
                f"Unable to read CSV: {exc}"
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Research decision-support prototype. Model predictions "
    "may be incorrect and are not a substitute for clinical judgment."
)

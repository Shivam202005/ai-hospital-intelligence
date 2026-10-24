from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# DATASET AND MODEL INPUT SCHEMA
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "diabetic_data.csv"

NUMERIC_FIELDS = {
    "time_in_hospital": ("Days in hospital", 1, 14, 4),
    "num_lab_procedures": ("Number of lab procedures", 0, 100, 40),
    "num_procedures": ("Number of procedures", 0, 20, 1),
    "num_medications": ("Number of medications", 0, 100, 15),
    "number_outpatient": ("Outpatient visits (past year)", 0, 100, 0),
    "number_emergency": ("Emergency visits (past year)", 0, 100, 0),
    "number_inpatient": ("Inpatient visits (past year)", 0, 100, 0),
    "number_diagnoses": ("Number of diagnoses", 1, 16, 5),
}

MEDICATION_FIELDS = [
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
]

MEDICATION_OPTIONS = ["No", "Steady", "Up", "Down"]

# Used if the dataset is not available.
FALLBACK_OPTIONS = {
    "race": [
        "Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Unknown"
    ],
    "gender": ["Female", "Male", "Unknown/Invalid"],
    "age": [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ],
    "admission_type_id": [str(i) for i in range(1, 9)],
    "discharge_disposition_id": [str(i) for i in range(1, 30)],
    "admission_source_id": [str(i) for i in range(1, 26)],
    "medical_specialty": ["Unknown"],
    "diag_1": ["Unknown"],
    "diag_2": ["Unknown"],
    "diag_3": ["Unknown"],
    "max_glu_serum": ["None", "Norm", ">200", ">300"],
    "A1Cresult": ["None", "Norm", ">7", ">8"],
    "change": ["No", "Ch"],
    "diabetesMed": ["Yes", "No"],
}


@st.cache_data
def load_dataset_options():
    """
    Load observed category values from the original dataset.
    The literal '?' is displayed as 'Unknown' in the UI and
    converted back to '?' before sending to the model.
    """
    if not DATA_PATH.exists():
        return FALLBACK_OPTIONS

    df = pd.read_csv(DATA_PATH, low_memory=False)

    category_fields = [
        "race",
        "gender",
        "age",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "medical_specialty",
        "diag_1",
        "diag_2",
        "diag_3",
        "max_glu_serum",
        "A1Cresult",
        "change",
        "diabetesMed",
    ]

    options = {}

    for field in category_fields:
        values = df[field].dropna().astype(str).unique().tolist()

        # Keep only values observed in the dataset.
        # '?' is represented in the UI as 'Unknown'.
        cleaned = [
            "Unknown" if value == "?" else value
            for value in values
        ]

        # Remove duplicates while preserving order.
        options[field] = list(dict.fromkeys(cleaned))

        # Stable sorting for diagnosis codes and IDs.
        if field in {
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
        }:
            options[field] = sorted(
                options[field],
                key=lambda value: (
                    value == "Unknown",
                    int(value) if value.isdigit() else 9999,
                ),
            )
        elif field.startswith("diag_"):
            options[field] = sorted(
                options[field],
                key=lambda value: (value == "Unknown", value),
            )

    return options


def choose_category(field, options, label=None):
    """Render a selectbox and map the UI's Unknown back to dataset '?'. """
    available = options.get(field, ["Unknown"])

    if not available:
        available = ["Unknown"]

    default_index = 0

    if field == "race" and "Caucasian" in available:
        default_index = available.index("Caucasian")
    elif field == "gender" and "Female" in available:
        default_index = available.index("Female")
    elif field == "age" and "[70-80)" in available:
        default_index = available.index("[70-80)")
    elif field in {
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    } and "1" in available:
        default_index = available.index("1")
    elif field in {"change", "diabetesMed"}:
        preferred = "No" if field == "change" else "Yes"
        if preferred in available:
            default_index = available.index(preferred)

    selected = st.selectbox(
        label or field.replace("_", " ").title(),
        available,
        index=default_index,
        key=f"field_{field}",
    )

    return "?" if selected == "Unknown" else selected


def render_patient_form():
    options = load_dataset_options()
    patient = {}

    st.subheader("Patient demographics")

    c1, c2, c3 = st.columns(3)

    with c1:
        patient["race"] = choose_category("race", options, "Race")

    with c2:
        patient["gender"] = choose_category("gender", options, "Gender")

    with c3:
        patient["age"] = choose_category("age", options, "Age group")

    st.subheader("Admission details")

    c1, c2, c3 = st.columns(3)

    with c1:
        patient["admission_type_id"] = choose_category(
            "admission_type_id", options, "Admission type ID"
        )

    with c2:
        patient["discharge_disposition_id"] = choose_category(
            "discharge_disposition_id", options, "Discharge disposition ID"
        )

    with c3:
        patient["admission_source_id"] = choose_category(
            "admission_source_id", options, "Admission source ID"
        )

    patient["medical_specialty"] = choose_category(
        "medical_specialty", options, "Medical specialty"
    )

    st.subheader("Diagnoses")

    st.caption(
        "Choose diagnosis codes observed in the dataset. "
        "Use Unknown when the code is unavailable."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        patient["diag_1"] = choose_category(
            "diag_1", options, "Primary diagnosis"
        )

    with c2:
        patient["diag_2"] = choose_category(
            "diag_2", options, "Secondary diagnosis"
        )

    with c3:
        patient["diag_3"] = choose_category(
            "diag_3", options, "Additional diagnosis"
        )

    st.subheader("Utilization")

    columns = st.columns(4)

    for index, (field, (label, minimum, maximum, default)) in enumerate(
        NUMERIC_FIELDS.items()
    ):
        with columns[index % 4]:
            patient[field] = st.number_input(
                label,
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=1,
                key=f"numeric_{field}",
            )

    st.subheader("Lab results")

    c1, c2 = st.columns(2)

    with c1:
        patient["max_glu_serum"] = choose_category(
            "max_glu_serum", options, "Maximum glucose serum"
        )

    with c2:
        patient["A1Cresult"] = choose_category(
            "A1Cresult", options, "HbA1c result"
        )

    st.subheader("Diabetes medication status")

    columns = st.columns(3)

    for index, field in enumerate(MEDICATION_FIELDS):
        with columns[index % 3]:
            patient[field] = st.selectbox(
                field.replace("-", " ").title(),
                MEDICATION_OPTIONS,
                index=0,
                key=f"med_{field}",
            )

    c1, c2 = st.columns(2)

    with c1:
        patient["change"] = choose_category(
            "change", options, "Medication changed"
        )

    with c2:
        patient["diabetesMed"] = choose_category(
            "diabetesMed", options, "Diabetes medication prescribed"
        )

    return patient
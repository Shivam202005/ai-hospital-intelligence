# Data Preprocessing Decisions

## Purpose

This document records the preprocessing decisions made after reviewing the initial EDA and feature analysis.

No preprocessing implementation is performed at this stage.

---

## 1. Identifier and Target Columns

| Column | Decision | Reason |
|---|---|---|
| encounter_id | Remove from model features | Unique encounter identifier |
| patient_nbr | Remove from model features | Patient identifier; may be retained temporarily for patient-level grouping |
| readmitted | Remove from features | Original outcome variable |
| readmitted_30 | Target | Binary 30-day readmission target |

---

## 2. Demographic Features

| Column | Decision | Reason |
|---|---|---|
| race | Keep | Potentially useful demographic feature |
| gender | Keep | Retain initially despite weak standalone relationship |
| age | Keep | Potentially important predictor |

---

## 3. Administrative Features

| Column | Decision | Reason |
|---|---|---|
| admission_type_id | Keep | Admission characteristics may provide predictive information |
| admission_source_id | Keep initially | May contain useful admission information |
| discharge_disposition_id | Review before implementation | Potential temporal leakage depending on prediction timing |
| payer_code | Review | High missingness and administrative nature |
| medical_specialty | Review | High missingness but potentially useful |

---

## 4. High-Missingness Features

| Column | Decision | Reason |
|---|---|---|
| weight | Provisionally remove | Approximately 96.86% values are missing |
| medical_specialty | Review | Approximately 49.08% missing |
| payer_code | Review | Approximately 39.56% missing |

Missing values represented by `?` will be handled during the implementation phase.

---

## 5. Numerical Features

The following features are retained as candidate features:

- time_in_hospital
- num_lab_procedures
- num_procedures
- num_medications
- number_outpatient
- number_emergency
- number_inpatient
- number_diagnoses

Final feature selection will be performed after preprocessing and model experiments.

---

## 6. Diagnosis Features

The following diagnosis features require further review:

- diag_1
- diag_2
- diag_3

Considerations:

- Low percentage of missing values
- Categorical diagnosis codes
- Potentially high cardinality
- Encoding strategy needs to be decided before implementation
- Rare categories may need special handling

No encoding is implemented at this stage.

---

## 7. Missing Value Strategy

Provisional strategy:

- Convert `?` into proper missing values.
- Avoid dropping all rows containing missing values.
- Remove extremely high-missingness features when justified.
- Use `Unknown` for appropriate categorical missing values.
- Consider median imputation for numerical variables where required.

Final strategy will be implemented using training-data-only statistics to avoid data leakage.

---

## 8. Categorical Encoding

Provisional strategy:

- Low-cardinality categorical features → One-Hot Encoding
- Diagnosis/medication codes → further cardinality analysis before deciding encoding
- Rare categories may be grouped if required

No encoding is implemented at this stage.

---

## 9. Feature Scaling

Scaling will be model-specific:

- Logistic Regression → StandardScaler
- Random Forest → scaling generally not required
- XGBoost → scaling generally not required
- LightGBM → scaling generally not required
- CatBoost → scaling generally not required

Model-specific preprocessing pipelines will be considered during implementation.

---

## 10. Temporal Leakage Review

`discharge_disposition_id` requires special attention.

If the system is intended to predict readmission **before discharge**, information only known at or after discharge should not be used.

The final decision will depend on the defined prediction time.

---

## 11. Current Status

### Completed

- Initial EDA reviewed
- Feature analysis reviewed
- Candidate features identified
- Missing-value issues identified
- Potential leakage identified
- Preliminary preprocessing strategy documented

### Not yet implemented

- Missing-value handling
- Encoding
- Scaling
- Feature selection
- Model training

These will be implemented in the next development stage after the team finalizes the decisions.

## 6. Diagnosis Features

| Column | Decision | Reason |
|---|---|---|
| diag_1 | Keep | Primary diagnosis; potentially important predictor |
| diag_2 | Keep | Secondary diagnosis; potentially useful predictor |
| diag_3 | Keep | Additional diagnosis; potentially useful predictor |

Diagnosis features have relatively low missingness:
- diag_1: 0.02%
- diag_2: 0.35%
- diag_3: 1.40%

However, diagnosis codes may have high cardinality and sparse categories. Therefore, they should not be blindly one-hot encoded.

Before implementation, diagnosis-code cardinality and frequency should be reviewed. Rare categories may require grouping or another suitable representation/encoding strategy.

## 7. Medication Features

The medication-related features are retained as candidate features for the initial modeling stage.

These include individual diabetes medications, medication combinations, `change`, and `diabetesMed`.

### Decision

- Keep medication features initially.
- No medication columns are removed at this review stage.
- No encoding is implemented yet.
- Medication frequency/variation should be examined during preprocessing before final feature selection.
- `change` and `diabetesMed` will be treated as categorical features.
- Rare or near-constant medication categories may be reviewed later based on frequency analysis.

Final medication feature selection will be based on preprocessing and model experiments rather than EDA alone.

## 8. Glucose and HbA1c Features

| Column | Decision | Reason |
|---|---|---|
| max_glu_serum | Keep as candidate | Potentially useful clinical feature, but has substantial missingness |
| A1Cresult | Keep as candidate | Potentially useful clinical feature, but has substantial missingness |

Both features are categorical and should be handled as categorical variables during preprocessing.

Missing values will be handled during the preprocessing stage rather than dropping rows.

The final decision will be based on missing-value analysis and model performance.

## 9. Treatment Status Features

| Column | Decision | Reason |
|---|---|---|
| change | Keep | Potentially useful indicator of diabetes medication changes |
| diabetesMed | Keep | Potentially useful indicator of diabetes medication usage |

Both features will be treated as categorical variables during preprocessing.

No encoding or transformation is implemented at this stage.

Final usefulness will be evaluated during model development.



## 10. Missing-Value Decisions

### weight

**Decision: Remove**

Reason:
- Approximately 96.86% of values are missing.
- The feature contains insufficient observed information for reliable modeling.
- Imputing such a large proportion of missing values is not considered appropriate.

### medical_specialty

**Decision: Keep as a candidate**

Reason:
- Approximately 49.08% of values are missing.
- Despite high missingness, medical specialty may contain useful predictive information.
- Missing values should not cause patient rows to be removed.
- During preprocessing, missing values will be represented using an `Unknown` category.
- Final usefulness will be evaluated during model development.

### payer_code

**Decision: Remove**

Reason:
- Approximately 39.56% of values are missing.
- The feature is primarily administrative/insurance-related.
- It provides less direct clinical information than the main patient and utilization features.
- Removing it simplifies preprocessing and reduces the impact of high missingness.

This decision can be revisited during model experimentation if required.

### race

**Decision: Keep**

Reason:
- Approximately 2.23% of values are missing.
- Missingness is relatively low.
- Race is retained as a demographic candidate feature.
- Missing values will be represented as an `Unknown` category.
- Final usefulness will be evaluated during model development.


### gender

**Decision: Keep**

Reason:
- Gender is a demographic feature.
- Male and female groups show similar 30-day readmission rates, but weak standalone association is not sufficient reason for removal.
- Unknown/Invalid values will be handled as a categorical value.
- Final usefulness will be evaluated during model development.

### age

**Decision: Keep**

Reason:
- Age is represented as 10 age-group categories.
- Readmission rates show some variation across age groups.
- Age may provide useful predictive information.
- Age groups will be encoded appropriately during preprocessing rather than treated as arbitrary numeric values.

### Diagnosis Features

| Column | Missing-Value Decision |
|---|---|
| diag_1 | Replace missing `?` with `Unknown` |
| diag_2 | Replace missing `?` with `Unknown` |
| diag_3 | Replace missing `?` with `Unknown` |

The diagnosis features have relatively low missingness, so rows will not be removed because of missing diagnosis values.

### Glucose and HbA1c Features

| Column | Missing-Value Decision |
|---|---|
| max_glu_serum | Replace missing `?` with `Unknown` |
| A1Cresult | Replace missing `?` with `Unknown` |

These features have substantial missingness. Rather than dropping patient encounters, missing values will be represented as an explicit categorical `Unknown` value during preprocessing.

The final contribution of these features will be evaluated during model development.

## 11. Administrative Admission Features

### admission_type_id

**Decision: Keep**

Reason:
- Represents the type of hospital admission.
- Can provide useful admission-related information.
- Will be treated as a categorical feature during preprocessing.

### admission_source_id

**Decision: Keep**

Reason:
- Represents the source of admission/referral.
- May provide useful information for prediction.
- Will be treated as a categorical feature during preprocessing.

### discharge_disposition_id

**Decision: Remove from primary model**

Reason:
- Discharge disposition may only be known later in the hospital encounter.
- Using it for a pre-discharge readmission prediction model can introduce temporal/data leakage.
- The primary model should use information available at the defined prediction time.

The feature may be investigated separately in an experimental analysis if required, but it will not be included in the primary clinically meaningful model.

## 12. Final Missing-Value Strategy

Based on the EDA, the following strategy has been decided:

### Remove

- `weight` — approximately 96.86% missing
- `payer_code` — approximately 39.56% missing and primarily administrative

### Keep and represent missing values as `Unknown`

- `medical_specialty`
- `race`
- `diag_1`
- `diag_2`
- `diag_3`
- `max_glu_serum`
- `A1Cresult`

Categorical missing values represented by `?` will be converted to an explicit `Unknown` category during preprocessing.

Rows will not be dropped solely because these categorical values are missing.

### Numerical Features

Numerical candidate features will not receive unnecessary imputation at this stage because the current EDA does not show `?` missing values in these variables.

If missing numerical values are encountered during implementation, the handling strategy will be determined within the preprocessing pipeline using training-data-only statistics.

### Important

These are preprocessing decisions made after reviewing the initial EDA. No preprocessing has been implemented yet.

## 13. Categorical Encoding Strategy

### Standard Categorical Features

The following categorical features will initially use One-Hot Encoding:

- race
- gender
- age
- admission_type_id
- admission_source_id
- medical_specialty
- max_glu_serum
- A1Cresult
- change
- diabetesMed

One-Hot Encoding is preferred because these categories do not have a meaningful numerical ordering.

### Diagnosis Features

`diag_1`, `diag_2`, and `diag_3` require separate analysis because diagnosis codes may create a high-dimensional and sparse feature space.

Before final encoding:
- Analyze cardinality
- Analyze category frequency
- Review rare diagnosis codes
- Select an appropriate representation/encoding strategy

### Medication Features

Medication features will initially be treated as categorical variables.

One-Hot Encoding will be considered as the initial approach, while rare or near-constant medication features will be reviewed before final modeling.

No encoding is implemented at this stage.


## 14. Feature Scaling Strategy

Scaling will be model-specific.

| Model | Scaling |
|---|---|
| Logistic Regression | StandardScaler |
| Random Forest | Not required |
| XGBoost | Not required |
| LightGBM | Not required |
| CatBoost | Not required |

Logistic Regression will use StandardScaler for numerical features.

Tree-based models generally do not require feature scaling.

Scaling and other preprocessing transformations will be fitted only on the training data and then applied to validation/test data to prevent data leakage.

Model-specific preprocessing pipelines will be used during implementation.


## 15. Final Feature Decision Summary

| Feature / Group | Decision | Planned Handling |
|---|---|---|
| encounter_id | Remove | Identifier |
| patient_nbr | Remove from model | Retain temporarily only for patient-level grouping if required |
| readmitted | Remove | Original target |
| readmitted_30 | Keep | Binary target |
| race | Keep | Unknown + One-Hot Encoding |
| gender | Keep | Categorical + One-Hot Encoding |
| age | Keep | One-Hot Encoding |
| weight | Remove | Excessive missingness |
| admission_type_id | Keep | Categorical + One-Hot Encoding |
| discharge_disposition_id | Remove from primary model | Avoid temporal leakage |
| admission_source_id | Keep | Categorical + One-Hot Encoding |
| time_in_hospital | Keep | Numerical |
| payer_code | Remove | High missingness + limited direct clinical value |
| medical_specialty | Keep initially | Unknown + categorical encoding |
| num_lab_procedures | Keep | Numerical |
| num_procedures | Keep | Numerical |
| num_medications | Keep | Numerical |
| number_outpatient | Keep | Numerical |
| number_emergency | Keep | Numerical |
| number_inpatient | Keep | Numerical |
| diag_1 | Keep | Unknown + further encoding analysis |
| diag_2 | Keep | Unknown + further encoding analysis |
| diag_3 | Keep | Unknown + further encoding analysis |
| number_diagnoses | Keep | Numerical |
| max_glu_serum | Keep initially | Unknown + categorical encoding |
| A1Cresult | Keep initially | Unknown + categorical encoding |
| Medication features | Keep initially | Categorical encoding + frequency review |
| change | Keep | Categorical + One-Hot Encoding |
| diabetesMed | Keep | Categorical + One-Hot Encoding |



## 16. Day 4 Review Conclusion

The initial EDA and feature analysis have been reviewed.

The team has identified:
- Features to remove
- Features to retain
- Missing-value handling strategy
- Categorical encoding strategy
- Numerical scaling strategy
- Potential temporal leakage in `discharge_disposition_id`
- Diagnosis and medication features requiring further frequency/cardinality analysis

No preprocessing implementation has been performed during this review stage.

The finalized decisions will be implemented in the preprocessing stage using reproducible model-specific pipelines.
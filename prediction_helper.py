import pandas as pd
from joblib import load

# ---------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------
model_rest   = load("artifacts/model_rest.joblib")
model_young  = load("artifacts/model_young.joblib")
scaler_rest  = load("artifacts/scaler_rest.joblib")   # dict: {"scaler", "cols_to_scale"}
scaler_young = load("artifacts/scaler_young.joblib")  # dict: {"scaler", "cols_to_scale"}

YOUNG_AGE_CUTOFF = 25

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
RISK_SCORES = {
    "diabetes": 6,
    "heart disease": 8,
    "high blood pressure": 6,
    "thyroid": 5,
    "no disease": 0,
    "none": 0,
}
MIN_RISK, MAX_RISK = 0, 14

PLAN_ENC = {"Bronze": 0, "Silver": 1, "Gold": 2}
INCOME_LEVEL_ENC = {"<10L": 0, "10L - 25L": 1, "25L - 40L": 2, "> 40L": 3}

# The 21 columns the model was trained on (matches model.feature_names_in_)
EXPECTED_COLUMNS = [
    'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan',
    'genetical_risk', 'normalized_risk_score',
    'gender_Male',
    'region_Northwest', 'region_Southeast', 'region_Southwest',
    'marital_status_Unmarried',
    'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight',
    'smoking_status_No Smoking', 'smoking_status_Not Smoking',
    'smoking_status_Occasional', 'smoking_status_Regular',
    'smoking_status_Smoking=0',
    'employment_status_Salaried', 'employment_status_Self-Employed',
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def calculate_normalized_risk_score(medical_history: str) -> float:
    """Normalized risk score in [0, 1] from a medical_history string."""
    if not isinstance(medical_history, str) or not medical_history.strip():
        medical_history = "No Disease"
    parts = [p.strip().lower() for p in medical_history.split(" & ")]
    d1 = parts[0] if len(parts) > 0 and parts[0] else "none"
    d2 = parts[1] if len(parts) > 1 and parts[1] else "none"
    total = RISK_SCORES.get(d1, 0) + RISK_SCORES.get(d2, 0)
    return (total - MIN_RISK) / (MAX_RISK - MIN_RISK)


def build_input_df(input_dict: dict) -> pd.DataFrame:
    """Turn the Streamlit form dict (human-readable keys) into a single-row DataFrame."""
    row = {col: 0 for col in EXPECTED_COLUMNS}

    # Numeric / derived features
    row["age"]                   = input_dict["Age"]
    row["number_of_dependants"]  = input_dict["Number of Dependants"]
    row["genetical_risk"]        = input_dict["Genetical Risk"]
    row["income_lakhs"]          = input_dict["Income in Lakhs"]
    row["normalized_risk_score"] = calculate_normalized_risk_score(
        input_dict["Medical History"]
    )

    # One-hot / ordinal features
    for key, value in input_dict.items():
        if key == "Insurance Plan":
            row["insurance_plan"] = PLAN_ENC.get(value, 0)

        elif key == "Gender" and value == "Male":
            row["gender_Male"] = 1

        elif key == "Region":
            m = {
                "Northwest": "region_Northwest",
                "Southeast": "region_Southeast",
                "Southwest": "region_Southwest",
                # "Northeast" is the dropped baseline
            }
            if value in m:
                row[m[value]] = 1

        elif key == "Marital Status" and value == "Unmarried":
            row["marital_status_Unmarried"] = 1

        elif key == "BMI Category":
            m = {
                "Obesity":     "bmi_category_Obesity",
                "Overweight":  "bmi_category_Overweight",
                "Underweight": "bmi_category_Underweight",
                # "Normal" is the dropped baseline
            }
            if value in m:
                row[m[value]] = 1

        elif key == "Smoking Status":
            m = {
                "No Smoking":     "smoking_status_No Smoking",
                "Not Smoking":    "smoking_status_Not Smoking",
                "Occasional":     "smoking_status_Occasional",
                "Regular":        "smoking_status_Regular",
                "Smoking=0":      "smoking_status_Smoking=0",
                "Does Not Smoke": "smoking_status_No Smoking",   # alias
            }
            if value in m:
                row[m[value]] = 1

        elif key == "Employment Status":
            m = {
                "Salaried":      "employment_status_Salaried",
                "Self-Employed": "employment_status_Self-Employed",
                # "Freelancer" is the dropped baseline
            }
            if value in m:
                row[m[value]] = 1

        # 'Age', 'Number of Dependants', 'Income in Lakhs', 'Genetical Risk',
        # 'Medical History' already handled above.

    return pd.DataFrame([row])[EXPECTED_COLUMNS]


def predict(input_dict: dict) -> float:
    """Predict insurance cost for one user submission."""
    df = build_input_df(input_dict)
    age = df["age"].iloc[0]

    if age <= YOUNG_AGE_CUTOFF:
        model, bundle = model_young, scaler_young
    else:
        model, bundle = model_rest, scaler_rest

    scaler          = bundle["scaler"]
    scaler_cols_all = list(getattr(scaler, "feature_names_in_", bundle["cols_to_scale"]))
    model_cols      = list(model.feature_names_in_)

    # Build the exact DataFrame the scaler was fit on (fill any missing column with 0)
    df_for_scaler = df.copy()

    # income_level is required by the scaler but not by the model — reconstruct it
    if "income_level" in scaler_cols_all and "income_level" not in df_for_scaler.columns:
        # The dropdown value is in input_dict["Income Level"] (or "Income in Lakhs"
        # if you have that widget). Use whichever exists.
        raw_income = input_dict.get("Income Level", None)
        if raw_income is None:
            # Fallback: infer a bucket from income_lakhs
            lakhs = input_dict.get("Income in Lakhs", 0)
            if   lakhs < 10:  raw_income = "<10L"
            elif lakhs < 25:  raw_income = "10L - 25L"
            elif lakhs < 40:  raw_income = "25L - 40L"
            else:             raw_income = "> 40L"
        df_for_scaler["income_level"] = INCOME_LEVEL_ENC.get(raw_income, 0)

    # Any other missing scaler columns: fill with 0
    for col in scaler_cols_all:
        if col not in df_for_scaler.columns:
            df_for_scaler[col] = 0

    # Scale the full scaler feature set
    scaled = scaler.transform(df_for_scaler[scaler_cols_all])

    # Copy scaled values back ONLY for columns the model uses
    df_scaled = df.copy()
    for col, vals in zip(scaler_cols_all, scaled.T):
        if col in df_scaled.columns:
            df_scaled[col] = vals

    # Align to model's expected feature set/order — this drops income_level
    df_final = df_scaled.reindex(columns=model_cols, fill_value=0)

    return float(model.predict(df_final)[0])
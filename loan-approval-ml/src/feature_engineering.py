from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create financial features without leaking target information."""
    engineered = df.copy()

    applicant_income = engineered["ApplicantIncome"].fillna(0)
    coapplicant_income = engineered["CoapplicantIncome"].fillna(0)
    loan_amount = engineered["LoanAmount"].fillna(0)
    loan_term = engineered["Loan_Amount_Term"].fillna(360)

    engineered["Total_Income"] = applicant_income + coapplicant_income
    engineered["Loan_to_Total_Income"] = loan_amount / engineered["Total_Income"].replace(0, np.nan)
    engineered["Loan_to_Applicant_Income"] = loan_amount / applicant_income.replace(0, np.nan)
    engineered["Income_per_Loan_Amount"] = engineered["Total_Income"].replace(0, np.nan) / loan_amount.replace(0, np.nan)
    engineered["Loan_Term_Years"] = loan_term / 12
    engineered["EMI_Estimate"] = np.where(
        loan_term > 0,
        (loan_amount * 0.12 / 12) / (1 - (1 + 0.12 / 12) ** (-loan_term)),
        0,
    )

    engineered["Dependents_Num"] = engineered["Dependents"].replace({"0": 0, "1": 1, "2": 2, "3+": 3, np.nan: 0}).astype(float)

    engineered["Applicant_Income_per_1000"] = applicant_income / 1000
    engineered["Coapplicant_Income_per_1000"] = coapplicant_income / 1000
    engineered["Combined_Income_per_1000"] = engineered["Total_Income"] / 1000

    engineered["Property_Area_Code"] = engineered["Property_Area"].map({"Urban": 2, "Semiurban": 1, "Rural": 0}).fillna(1)

    return engineered


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the dataset for model training or inference."""
    required_columns = [
        "Loan_ID",
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area",
    ]
    validate_required_columns(df, required_columns)
    prepared = df.copy()
    for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]:
        prepared[col] = prepared[col].fillna("Unknown")
    prepared["Dependents"] = prepared["Dependents"].fillna("0").astype(str)
    prepared["Loan_Amount_Term"] = prepared["Loan_Amount_Term"].fillna(360)
    prepared["LoanAmount"] = prepared["LoanAmount"].fillna(prepared["LoanAmount"].median())
    prepared["Credit_History"] = prepared["Credit_History"].fillna(prepared["Credit_History"].mode().iloc[0])
    return engineer_financial_features(prepared)

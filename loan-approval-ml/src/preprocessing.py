from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Loan_Status"
ID_COLUMN = "Loan_ID"


def clean_loan_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and clean loan data before feature engineering."""
    cleaned = df.copy()

    for col in ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area", "Loan_Status"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype("string").str.strip()
            cleaned[col] = cleaned[col].replace({"nan": np.nan, "NaN": np.nan, "None": np.nan})

    if "Dependents" in cleaned.columns:
        cleaned["Dependents"] = cleaned["Dependents"].fillna("0")
        cleaned["Dependents"] = cleaned["Dependents"].replace({"3+": "3+", "3 +": "3+"})

    if "Loan_ID" in cleaned.columns:
        cleaned["Loan_ID"] = cleaned["Loan_ID"].astype(str)

    return cleaned


def get_numerical_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in {ID_COLUMN, TARGET_COLUMN} and pd.api.types.is_numeric_dtype(df[col])
    ]


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in {ID_COLUMN, TARGET_COLUMN} and (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]))
    ]


def make_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numerical_columns = get_numerical_columns(df)
    categorical_columns = get_categorical_columns(df)

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def split_features_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataframe.")

    X = df.drop(columns=[target_column, ID_COLUMN], errors="ignore")
    y = df[target_column].copy()
    return X, y

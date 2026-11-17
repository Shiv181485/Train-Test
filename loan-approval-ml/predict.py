from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import prepare_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "LP_Test.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "test_predictions.csv"


def load_model(model_path: Path):
    with open(model_path, "rb") as file:
        return pickle.load(file)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run train.py first.")

    test_df = pd.read_csv(DATA_PATH)
    model = load_model(MODEL_PATH)
    prepared = prepare_dataset(test_df)
    X_test = prepared.drop(columns=["Loan_ID"], errors="ignore")

    probabilities = model.predict_proba(X_test)
    classes = model.named_steps["model"].classes_
    positive_idx = 1 if 1 in classes else 0
    approval_probability = probabilities[:, positive_idx]
    rejection_probability = 1 - approval_probability
    predicted_numeric = np.where(approval_probability >= 0.5, 1, 0)
    predicted_status = pd.Series(predicted_numeric).map({0: "N", 1: "Y"})

    output = pd.DataFrame(
        {
            "Loan_ID": test_df["Loan_ID"].tolist(),
            "Predicted_Loan_Status": predicted_status.tolist(),
            "Approval_Probability": approval_probability.tolist(),
            "Rejection_Probability": rejection_probability.tolist(),
            "Confidence_Level": [
                "High" if value >= 0.8 else "Medium" if value >= 0.6 else "Low" for value in approval_probability.tolist()
            ],
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(output)} predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

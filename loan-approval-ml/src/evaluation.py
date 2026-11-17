from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def compute_classification_metrics(y_true, y_pred, y_proba=None, positive_label="Y"):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
    }
    if y_proba is not None:
        metrics["ROC_AUC"] = roc_auc_score((y_true == positive_label).astype(int), y_proba)
    else:
        metrics["ROC_AUC"] = float("nan")
    return metrics


def build_confusion_matrix(y_true, y_pred, positive_label="Y"):
    labels = ["N", "Y"] if positive_label == "Y" else [positive_label, "N"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=["Actual_N", "Actual_Y"], columns=["Pred_N", "Pred_Y"])
    return cm_df


def create_train_prediction_table(loan_ids, y_true, y_pred, y_prob):
    return pd.DataFrame(
        {
            "Loan_ID": loan_ids,
            "Actual_Loan_Status": y_true,
            "Predicted_Loan_Status": y_pred,
            "Prediction_Probability": y_prob,
            "Prediction_Result": ["Correct" if actual == pred else "Incorrect" for actual, pred in zip(y_true, y_pred)],
        }
    )


def create_test_prediction_table(loan_ids, y_pred, approval_prob, rejection_prob):
    confidence = []
    for value in approval_prob:
        if value >= 0.8:
            confidence.append("High")
        elif value >= 0.6:
            confidence.append("Medium")
        else:
            confidence.append("Low")

    return pd.DataFrame(
        {
            "Loan_ID": loan_ids,
            "Predicted_Loan_Status": y_pred,
            "Approval_Probability": approval_prob,
            "Rejection_Probability": rejection_prob,
            "Confidence_Level": confidence,
        }
    )


def save_model_comparison(results_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.evaluation import compute_classification_metrics, create_test_prediction_table, create_train_prediction_table, save_model_comparison
from src.feature_engineering import prepare_dataset
from src.preprocessing import TARGET_COLUMN, make_preprocessor, split_features_target

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")
    return prepare_dataset(df)


def get_model_candidates() -> dict[str, object]:
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", None),
            ("model", LogisticRegression(max_iter=4000, class_weight='balanced', random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", None),
            ("model", DecisionTreeClassifier(random_state=42, max_depth=None, min_samples_leaf=5, class_weight='balanced')),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", None),
            ("model", RandomForestClassifier(random_state=42, n_estimators=300, class_weight='balanced', min_samples_leaf=2, n_jobs=-1)),
        ]),
        "Extra Trees": Pipeline([
            ("preprocessor", None),
            ("model", ExtraTreesClassifier(random_state=42, n_estimators=400, class_weight='balanced', n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", None),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("preprocessor", None),
            ("model", HistGradientBoostingClassifier(random_state=42, max_depth=6, learning_rate=0.05, max_iter=300)),
        ]),
        "SVM": Pipeline([
            ("preprocessor", None),
            ("model", SVC(probability=True, kernel='rbf', C=2.0, gamma='scale', random_state=42, class_weight='balanced')),
        ]),
        "KNN": Pipeline([
            ("preprocessor", None),
            ("model", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", None),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                use_label_encoder=False,
            )),
        ]),
        "LightGBM": Pipeline([
            ("preprocessor", None),
            ("model", LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "CatBoost": Pipeline([
            ("preprocessor", None),
            ("model", CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                loss_function="Logloss",
                random_seed=42,
                verbose=False,
            )),
        ]),
    }


def prepare_pipeline(model_name: str, X: pd.DataFrame) -> Pipeline:
    pipeline = get_model_candidates()[model_name]
    pipeline.steps[0] = ("preprocessor", make_preprocessor(X))
    return pipeline


def run_model_comparison(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    pipelines = {}

    for model_name in get_model_candidates():
        pipeline = prepare_pipeline(model_name, X)
        pipelines[model_name] = pipeline

        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=None,
        )

        baseline_fit = pipeline.fit(X, y)
        classes = baseline_fit.named_steps["model"].classes_
        positive_idx = 1 if 1 in classes else 0

        oof_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, positive_idx]
        oof_pred = np.where(oof_prob >= 0.5, 1, 0)
        metrics = compute_classification_metrics(y, oof_pred, y_proba=oof_prob, positive_label=1)

        results.append(
            {
                "Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "ROC_AUC": metrics["ROC_AUC"],
                "CV_Accuracy_Mean": cv_results["test_accuracy"].mean(),
                "CV_Accuracy_Std": cv_results["test_accuracy"].std(),
                "CV_F1_Mean": cv_results["test_f1"].mean(),
                "CV_F1_Std": cv_results["test_f1"].std(),
            }
        )

    comparison = pd.DataFrame(results).sort_values(["F1", "ROC_AUC", "Accuracy"], ascending=False).reset_index(drop=True)
    return comparison, pipelines


def tune_best_model(X: pd.DataFrame, y: pd.Series, best_model_name: str, best_pipeline: Pipeline) -> Pipeline:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if best_model_name == "Random Forest":
        param_grid = {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 8, 12],
            "model__min_samples_leaf": [1, 2, 4],
            "model__class_weight": [None, "balanced"],
        }
        search = RandomizedSearchCV(best_pipeline, param_distributions=param_grid, n_iter=12, cv=cv, scoring="f1", random_state=42, n_jobs=-1)
        grid = GridSearchCV(best_pipeline, param_grid={
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 3],
            "model__class_weight": ["balanced"],
        }, cv=cv, scoring="f1", n_jobs=-1)
    elif best_model_name == "Logistic Regression":
        param_grid = {
            "model__C": [0.01, 0.1, 1.0, 5.0],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"],
        }
        search = RandomizedSearchCV(best_pipeline, param_distributions=param_grid, n_iter=8, cv=cv, scoring="f1", random_state=42, n_jobs=-1)
        grid = GridSearchCV(best_pipeline, param_grid={
            "model__C": [0.1, 1.0, 5.0],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs"],
        }, cv=cv, scoring="f1", n_jobs=-1)
    else:
        param_grid = {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 8, 12],
            "model__min_samples_leaf": [1, 2, 4],
        }
        search = RandomizedSearchCV(best_pipeline, param_distributions=param_grid, n_iter=8, cv=cv, scoring="f1", random_state=42, n_jobs=-1)
        grid = GridSearchCV(best_pipeline, param_grid={
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 3],
        }, cv=cv, scoring="f1", n_jobs=-1)

    search.fit(X, y)
    grid.fit(X, y)
    best_estimator = search.best_estimator_ if search.best_score_ >= grid.best_score_ else grid.best_estimator_
    return best_estimator


def generate_training_outputs(best_model: Pipeline, X: pd.DataFrame, y: pd.Series, train_df: pd.DataFrame) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classes = best_model.named_steps["model"].classes_
    positive_idx = 1 if 1 in classes else 0

    oof_prob = cross_val_predict(best_model, X, y, cv=cv, method="predict_proba")[:, positive_idx]
    oof_pred = np.where(oof_prob >= 0.5, 1, 0)

    prediction_table = create_train_prediction_table(
        loan_ids=train_df["Loan_ID"].tolist(),
        y_true=train_df["Loan_Status"].tolist(),
        y_pred=oof_pred.tolist(),
        y_prob=oof_prob.tolist(),
    )
    prediction_table["Actual_Loan_Status"] = prediction_table["Actual_Loan_Status"].map({0: "N", 1: "Y"})
    prediction_table["Predicted_Loan_Status"] = prediction_table["Predicted_Loan_Status"].map({0: "N", 1: "Y"})
    prediction_table["Prediction_Result"] = [
        "Correct" if actual == pred else "Incorrect" for actual, pred in zip(prediction_table["Actual_Loan_Status"], prediction_table["Predicted_Loan_Status"])
    ]
    return prediction_table


def generate_test_outputs(best_model: Pipeline, X_test: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    classes = best_model.named_steps["model"].classes_
    positive_idx = 1 if 1 in classes else 0

    prob = best_model.predict_proba(X_test)
    approval_prob = prob[:, positive_idx]
    rejection_prob = 1 - approval_prob
    predicted = np.where(approval_prob >= 0.5, 1, 0)
    predicted_labels = pd.Series(predicted).map({0: "N", 1: "Y"})

    return create_test_prediction_table(
        loan_ids=test_df["Loan_ID"].tolist(),
        y_pred=predicted_labels.tolist(),
        approval_prob=approval_prob.tolist(),
        rejection_prob=rejection_prob.tolist(),
    )


def save_best_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(model, file)


def main() -> None:
    train_path = DATA_DIR / "LP_Train.csv"
    test_path = DATA_DIR / "LP_Test.csv"

    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    train_df["Loan_Status"] = train_df["Loan_Status"].map({"Y": 1, "N": 0}).astype(int)
    y_train = train_df[TARGET_COLUMN].copy()
    X_train = train_df.drop(columns=[TARGET_COLUMN, "Loan_ID"], errors="ignore")
    X_test = test_df.drop(columns=["Loan_ID"], errors="ignore")

    model_comparison, candidate_pipelines = run_model_comparison(X_train, y_train)
    best_model_name = model_comparison.iloc[0]["Model"]
    best_pipeline = candidate_pipelines[best_model_name]

    tuned_model = tune_best_model(X_train, y_train, best_model_name, best_pipeline)
    save_best_model(tuned_model, MODELS_DIR / "best_model.pkl")

    train_predictions = generate_training_outputs(tuned_model, X_train, y_train, train_df)
    train_predictions.to_csv(OUTPUTS_DIR / "train_predictions.csv", index=False)

    test_predictions = generate_test_outputs(tuned_model, X_test, test_df)
    test_predictions.to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)

    model_comparison.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    print("Best model:", best_model_name)
    print(model_comparison.head(10).to_string(index=False))
    print(f"Saved model to {MODELS_DIR / 'best_model.pkl'}")
    print(f"Train predictions: {len(train_predictions)} rows")
    print(f"Test predictions: {len(test_predictions)} rows")


if __name__ == "__main__":
    main()

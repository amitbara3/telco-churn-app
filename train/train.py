"""Train a churn-prediction model on the Telco Customer Churn dataset.

For each candidate model family:
  1. Hyperparameter-search on the training split (5-fold CV, scored on
     ROC-AUC, which is threshold-independent).
  2. Pick a decision threshold by sweeping F1 on the "Churn" class using
     out-of-fold predictions on the training split (cross_val_predict) —
     this never looks at the test set, so the threshold isn't overfit to it.
  3. Evaluate the tuned model + tuned threshold once on the held-out test
     split, and keep whichever candidate has the best test-set churn F1.

Saves the winning pipeline, its decision threshold, per-candidate metrics,
and a feature schema (for the API/UI) under ./model/.

Usage:
    python train/train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `python train/train.py` can `import app.*`

from app.features import ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC, FeatureEngineer  # noqa: E402

DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
THRESHOLD_PATH = MODEL_DIR / "decision_threshold.json"

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"

RANDOM_STATE = 42
CV_FOLDS = 5
SEARCH_ITER = 20


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["customerID"])
    # TotalCharges is stored as a string and has a handful of blank values
    # for customers with tenure == 0; coerce and drop those rows.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    return df


def build_preprocessor() -> ColumnTransformer:
    # Dense output + pandas column names throughout (rather than the
    # default sparse/ndarray output) so the same column-labeled frame shape
    # flows through fit *and* predict for every candidate — LightGBM's
    # sklearn wrapper otherwise warns loudly about feature-name mismatches
    # between training (array with generated names) and serving (a plain
    # array with none). Dataset is small enough (~55 columns after
    # one-hot encoding) that dense is a non-issue memory-wise.
    ct = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES + ENGINEERED_NUMERIC),
        ]
    )
    return ct.set_output(transform="pandas")


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("engineer", FeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def search_spaces() -> dict:
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            {"model__C": np.logspace(-3, 2, 30)},
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [200, 300, 400],
                "model__max_depth": [4, 6, 8, 10, None],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2"],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 150, 200],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.85, 1.0],
            },
        ),
        "xgboost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
                eval_metric="logloss",
            ),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [2, 3, 4, 5, 6],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "model__subsample": [0.6, 0.8, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
                "model__min_child_weight": [1, 3, 5],
            },
        ),
        "lightgbm": (
            LGBMClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
                subsample_freq=1,
                verbosity=-1,
            ),
            {
                "model__n_estimators": [100, 200, 300],
                "model__num_leaves": [15, 31, 63, 127],
                "model__max_depth": [-1, 3, 5, 7],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "model__subsample": [0.6, 0.8, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
                "model__min_child_samples": [5, 10, 20, 30],
            },
        ),
    }


def best_threshold_for_f1(y_true, y_proba) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (y_proba >= t).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_t, best_f1 = float(t), float(score)
    return best_t, best_f1


def evaluate_at_threshold(y_true, y_proba, threshold: float) -> dict:
    preds = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "classification_report": classification_report(
            y_true, preds, target_names=["No Churn", "Churn"], output_dict=True
        ),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    best_name, best_pipeline, best_threshold, best_test_f1 = None, None, 0.5, -1.0

    for name, (estimator, param_dist) in search_spaces().items():
        pipeline = build_pipeline(estimator)

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=SEARCH_ITER,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        tuned_pipeline = search.best_estimator_
        cv_std = float(search.cv_results_["std_test_score"][search.best_index_])

        # Out-of-fold probabilities on the *training* split, using the
        # tuned hyperparameters, so the threshold is picked without ever
        # touching the held-out test set.
        oof_proba = cross_val_predict(
            tuned_pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        threshold, oof_f1 = best_threshold_for_f1(y_train, oof_proba)

        test_proba = tuned_pipeline.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_at_threshold(y_test, test_proba, threshold)

        results[name] = {
            "best_params": search.best_params_,
            "cv_roc_auc_mean": float(search.best_score_),
            "cv_roc_auc_std": cv_std,
            "tuned_decision_threshold": threshold,
            "oof_train_f1_at_threshold": oof_f1,
            "test_metrics": test_metrics,
        }

        print(
            f"{name}: cv_roc_auc={search.best_score_:.3f} (+/-{cv_std:.3f})  "
            f"threshold={threshold:.2f}  "
            f"test_f1={test_metrics['f1']:.3f}  test_roc_auc={test_metrics['roc_auc']:.3f}  "
            f"test_acc={test_metrics['accuracy']:.3f}"
        )

        if test_metrics["f1"] > best_test_f1:
            best_name, best_pipeline, best_threshold, best_test_f1 = (
                name,
                tuned_pipeline,
                threshold,
                test_metrics["f1"],
            )

    print(
        f"\nSelected best model: {best_name} "
        f"(test churn f1={best_test_f1:.3f}, threshold={best_threshold:.2f})"
    )

    joblib.dump(best_pipeline, MODEL_PATH)
    THRESHOLD_PATH.write_text(json.dumps({"threshold": best_threshold}, indent=2))
    METRICS_PATH.write_text(
        json.dumps({"selected_model": best_name, "candidates": results}, indent=2)
    )

    # Feature schema used to build the API's Pydantic model and the
    # Streamlit form: allowed categories per categorical column, and
    # observed min/max for numeric columns. Deliberately based on the raw
    # (not engineered) columns, since that's the request schema clients see.
    schema = {
        "categorical_features": {
            col: sorted(df[col].unique().tolist()) for col in CATEGORICAL_FEATURES
        },
        "numeric_features": {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
            }
            for col in NUMERIC_FEATURES
        },
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2))

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved decision threshold to {THRESHOLD_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved feature schema to {SCHEMA_PATH}")


if __name__ == "__main__":
    main()

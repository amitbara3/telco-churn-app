"""Train a churn-prediction model on the Telco Customer Churn dataset.

Loads the raw CSV, builds a preprocessing + classification pipeline,
evaluates a few candidate models, and saves the best one (plus metrics
and a feature schema used by the API/UI) under ./model/.

Usage:
    python train/train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"

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
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def candidate_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}
    # Select by F1 on the "Churn" class rather than ROC-AUC or accuracy: this
    # dataset is imbalanced (~27% churn), and for a churn-prevention use case
    # missing an actual churner (low recall) is more costly than a false
    # alarm. ROC-AUC is threshold-independent and ends up nearly tied across
    # candidates here, which hides that gap.
    best_name, best_pipeline, best_score = None, None, -1.0

    for name, estimator in candidate_models().items():
        pipeline = Pipeline(
            steps=[("preprocess", build_preprocessor()), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "classification_report": classification_report(
                y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True
            ),
        }
        results[name] = metrics
        print(f"{name}: accuracy={metrics['accuracy']:.3f} "
              f"f1={metrics['f1']:.3f} roc_auc={metrics['roc_auc']:.3f}")

        if metrics["f1"] > best_score:
            best_name, best_pipeline, best_score = name, pipeline, metrics["f1"]

    print(f"\nSelected best model: {best_name} (churn f1={best_score:.3f})")

    joblib.dump(best_pipeline, MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps({"selected_model": best_name, "candidates": results}, indent=2)
    )

    # Feature schema used to build the API's Pydantic model and the
    # Streamlit form: allowed categories per categorical column, and
    # observed min/max for numeric columns.
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
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved feature schema to {SCHEMA_PATH}")


if __name__ == "__main__":
    main()

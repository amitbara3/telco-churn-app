"""Train the churn-prediction model on the Telco Customer Churn dataset.

The model is CatBoost using its native categorical handling — it consumes
the raw category strings directly via `cat_features`, with no one-hot
encoding step at all. It was picked after comparing twelve model families
(tree ensembles, boosting libraries, a linear baseline, and two neural
nets) under this same CV/threshold-tuning protocol; it won on both churn
F1 and balanced accuracy. That comparison lives in git history rather than
here — see README for the results table and the reasoning.

Pipeline:
  1. Hyperparameter search on the training split (5-fold stratified CV,
     scored on ROC-AUC, which is threshold-independent).
  2. Pick a decision threshold by sweeping F1 on the "Churn" class over
     out-of-fold predictions on the *training* split (cross_val_predict) —
     this never looks at the test set, so the threshold isn't overfit to it.
  3. Evaluate once on the held-out test split.

Saves the fitted pipeline, its decision threshold, metrics, feature
importances, and a feature schema (for the API/UI) under ./model/.

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
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `python train/train.py` can `import app.*`

from app.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    ENGINEERED_CATEGORICAL,
    NUMERIC_FEATURES,
    CustomerSegmentFeature,
    FeatureEngineer,
)

DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
THRESHOLD_PATH = MODEL_DIR / "decision_threshold.json"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"

TARGET = "Churn"

# Columns CatBoost should treat as categorical (raw strings, not encoded).
NATIVE_CATEGORICAL_COLUMNS = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL

# cat_features is passed at fit() time rather than on the constructor:
# as a constructor arg it hits a real CatBoost/sklearn interop bug, where
# CatBoost's get_params() doesn't round-trip the list in a way that
# satisfies sklearn's clone() equality check — and RandomizedSearchCV
# clones the pipeline per fold/candidate. Passing it via fit() sidesteps
# clone() entirely.
FIT_PARAMS = {"model__cat_features": NATIVE_CATEGORICAL_COLUMNS}

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


def build_pipeline() -> Pipeline:
    """Feature engineering, then K-Means segmentation, then the classifier.

    No ColumnTransformer/one-hot step — CatBoost consumes the raw
    categorical strings directly (see FIT_PARAMS).
    """
    return Pipeline(
        steps=[
            ("engineer", FeatureEngineer()),
            ("segment", CustomerSegmentFeature(random_state=RANDOM_STATE)),
            (
                "model",
                CatBoostClassifier(
                    random_state=RANDOM_STATE, verbose=False, thread_count=1
                ),
            ),
        ]
    )


def search_space(scale_pos_weight_options: list[float]) -> dict:
    return {
        "model__iterations": [100, 200, 300],
        "model__depth": [3, 4, 5, 6, 7],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        # Native Python floats, not numpy.float64: CatBoost's
        # get_params()/constructor round-trip doesn't preserve numpy scalar
        # dtype, which fails sklearn's clone() check.
        "model__l2_leaf_reg": [float(x) for x in np.logspace(-1, 2, 10)],
        "model__scale_pos_weight": scale_pos_weight_options,
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
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "classification_report": classification_report(
            y_true, preds, target_names=["No Churn", "Churn"], output_dict=True
        ),
    }


def extract_feature_importance(pipeline: Pipeline) -> pd.Series:
    """CatBoost's own feature importances, normalized to sum to 1 and
    labelled with the column names it was fit on."""
    model = pipeline.named_steps["model"]
    values = np.asarray(model.feature_importances_, dtype=float)
    if values.sum() > 0:
        values = values / values.sum()
    return pd.Series(values, index=model.feature_names_).sort_values(ascending=False)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    scale_pos_weight_options = [1.0, float(np.sqrt(ratio)), ratio]
    print(
        f"Class imbalance ratio (neg/pos) on train: {ratio:.3f}  "
        f"scale_pos_weight options: {[round(v, 3) for v in scale_pos_weight_options]}"
    )

    search = RandomizedSearchCV(
        build_pipeline(),
        param_distributions=search_space(scale_pos_weight_options),
        n_iter=SEARCH_ITER,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train, **FIT_PARAMS)
    pipeline = search.best_estimator_
    cv_std = float(search.cv_results_["std_test_score"][search.best_index_])

    # Out-of-fold probabilities on the *training* split, using the tuned
    # hyperparameters, so the threshold is picked without ever touching the
    # held-out test set.
    oof_proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1,
        params=FIT_PARAMS,
    )[:, 1]
    threshold, oof_f1 = best_threshold_for_f1(y_train, oof_proba)

    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_at_threshold(y_test, test_proba, threshold)

    print(
        f"\nCV ROC-AUC: {search.best_score_:.4f} (+/-{cv_std:.4f})\n"
        f"Tuned decision threshold: {threshold:.2f} "
        f"(out-of-fold train F1 {oof_f1:.4f})\n"
        f"Test: f1={test_metrics['f1']:.4f}  "
        f"balanced_acc={test_metrics['balanced_accuracy']:.4f}  "
        f"acc={test_metrics['accuracy']:.4f}  "
        f"roc_auc={test_metrics['roc_auc']:.4f}"
    )
    print(f"Best params: {search.best_params_}")

    joblib.dump(pipeline, MODEL_PATH)
    THRESHOLD_PATH.write_text(json.dumps({"threshold": threshold}, indent=2))
    METRICS_PATH.write_text(
        json.dumps(
            {
                "model": "catboost_native",
                "best_params": search.best_params_,
                "cv_roc_auc_mean": float(search.best_score_),
                "cv_roc_auc_std": cv_std,
                "tuned_decision_threshold": threshold,
                "oof_train_f1_at_threshold": oof_f1,
                "test_metrics": test_metrics,
            },
            indent=2,
        )
    )

    importance = extract_feature_importance(pipeline)
    IMPORTANCE_PATH.write_text(
        json.dumps(
            [
                {"feature": feat, "importance": round(float(val), 6)}
                for feat, val in importance.items()
            ],
            indent=2,
        )
    )
    print(f"\nTop 5 features: {list(importance.head(5).items())}")

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

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved decision threshold to {THRESHOLD_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved feature importance to {IMPORTANCE_PATH}")
    print(f"Saved feature schema to {SCHEMA_PATH}")


if __name__ == "__main__":
    main()

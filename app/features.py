"""Row-wise feature engineering, shared between training and serving.

This is a scikit-learn Pipeline step (not a one-off script) specifically so
that train/train.py and the live API always derive these columns the same
way — there is no separate "recompute this by hand in the API" step to
drift out of sync with training.

Every derived feature here is computed from a single row's own raw values
(no cross-row aggregates/statistics), so it's safe to use inside
cross-validation without leaking information across folds.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ADDON_SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# For these columns, one of the raw categories ("No internet/phone service")
# is 100% determined by another feature (InternetService == "No" /
# PhoneService == "No" respectively) — it's not new information, just the
# same bit re-encoded once per column. Collapsing it into "No" removes 7
# structurally redundant one-hot columns that a feature-importance check
# showed carrying ~0 signal, without losing anything: InternetService and
# PhoneService already carry that bit directly.
NO_SERVICE_TO_NO_COLUMNS = {
    "OnlineSecurity": "No internet service",
    "OnlineBackup": "No internet service",
    "DeviceProtection": "No internet service",
    "TechSupport": "No internet service",
    "StreamingTV": "No internet service",
    "StreamingMovies": "No internet service",
    "MultipleLines": "No phone service",
}

TENURE_BUCKET_EDGES = [-1, 12, 24, 48, 60, 1000]
TENURE_BUCKET_LABELS = ["0-12", "12-24", "24-48", "48-60", "60+"]

NEW_CUSTOMER_TENURE_MONTHS = 3

ENGINEERED_CATEGORICAL = ["tenure_bucket"]
ENGINEERED_NUMERIC = [
    "num_addon_services",
    "avg_charge_per_tenure",
    "charges_delta",
    "is_new_customer",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds a few derived columns to the raw Telco feature set, and
    collapses structurally-redundant categories (see
    NO_SERVICE_TO_NO_COLUMNS above)."""

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X["num_addon_services"] = (X[ADDON_SERVICE_COLUMNS] == "Yes").sum(axis=1)
        X["avg_charge_per_tenure"] = X["TotalCharges"] / (X["tenure"] + 1)
        # Positive => billed less than tenure * current rate would suggest,
        # i.e. they've been on a discount/promo at some point in their
        # tenure. A common telecom churn trigger is that discount expiring.
        X["charges_delta"] = X["tenure"] * X["MonthlyCharges"] - X["TotalCharges"]
        X["is_new_customer"] = (X["tenure"] <= NEW_CUSTOMER_TENURE_MONTHS).astype(int)
        X["tenure_bucket"] = pd.cut(
            X["tenure"], bins=TENURE_BUCKET_EDGES, labels=TENURE_BUCKET_LABELS
        ).astype(str)

        for col, redundant_value in NO_SERVICE_TO_NO_COLUMNS.items():
            X[col] = X[col].replace(redundant_value, "No")

        return X


class AverageProbabilityEnsemble(BaseEstimator):
    """Averages predict_proba across a set of already-fitted pipelines.

    Used when a simple probability-average blend of several tuned model
    families generalizes better than any single one of them — a standard
    way to gain ground once individually-tuned candidates have converged
    to similar performance.
    """

    def __init__(self, fitted_pipelines: list):
        self.fitted_pipelines = fitted_pipelines

    def predict_proba(self, X):
        probas = np.stack([p.predict_proba(X) for p in self.fitted_pipelines])
        return probas.mean(axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

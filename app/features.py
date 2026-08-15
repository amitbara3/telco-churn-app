"""Row-wise feature engineering, shared between training and serving.

This is a scikit-learn Pipeline step (not a one-off script) specifically so
that train/train.py and the live API always derive these columns the same
way — there is no separate "recompute this by hand in the API" step to
drift out of sync with training.

Every derived feature here is computed from a single row's own raw values
(no cross-row aggregates/statistics), so it's safe to use inside
cross-validation without leaking information across folds.
"""
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

TENURE_BUCKET_EDGES = [-1, 12, 24, 48, 60, 1000]
TENURE_BUCKET_LABELS = ["0-12", "12-24", "24-48", "48-60", "60+"]

ENGINEERED_CATEGORICAL = ["tenure_bucket"]
ENGINEERED_NUMERIC = ["num_addon_services", "avg_charge_per_tenure"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds a few derived columns to the raw Telco feature set."""

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["num_addon_services"] = (X[ADDON_SERVICE_COLUMNS] == "Yes").sum(axis=1)
        X["avg_charge_per_tenure"] = X["TotalCharges"] / (X["tenure"] + 1)
        X["tenure_bucket"] = pd.cut(
            X["tenure"], bins=TENURE_BUCKET_EDGES, labels=TENURE_BUCKET_LABELS
        ).astype(str)
        return X

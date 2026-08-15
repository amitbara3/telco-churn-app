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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Single source of truth for the raw request schema and its column order.
# Both train.py (what the model is fit on) and model.py (what a live
# request is turned into) import this — previously they each hardcoded
# their own copy, and the two lists silently drifted out of order. That
# was harmless for one-hot pipelines (ColumnTransformer selects columns by
# name, order-independent) but broke CatBoost's native-categorical mode,
# which resolves cat_features to positional indices at fit time: a
# differently-ordered DataFrame at serve time silently fed categorical
# strings into columns CatBoost expected to be numeric.
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
RAW_FEATURE_ORDER = CATEGORICAL_FEATURES + NUMERIC_FEATURES

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
HIGH_RISK_TENURE_MONTHS = 12

# PaymentMethod values that require the customer to actively take an
# action each cycle, vs. being billed automatically — a commonly-cited
# churn signal in telecom (an independent benchmark on this same dataset,
# arXiv:2607.10260, engineered an equivalent "payment stability" feature).
MANUAL_PAYMENT_METHODS = {"Electronic check", "Mailed check"}

# Engineered columns CatBoost must treat as categorical. The engineered
# numeric columns aren't enumerated anywhere: CatBoost consumes the whole
# frame and treats everything not listed as categorical as numeric, so
# FeatureEngineer.transform() below is the single definition of what
# exists.
ENGINEERED_CATEGORICAL = ["tenure_bucket", "customer_segment"]


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
        # Same signal as charges_delta, normalized to a fraction of expected
        # spend rather than an absolute dollar amount — trees may split on
        # the ratio differently than on the raw delta.
        X["discount_ratio"] = X["charges_delta"] / (X["tenure"] * X["MonthlyCharges"] + 1)
        X["is_new_customer"] = (X["tenure"] <= NEW_CUSTOMER_TENURE_MONTHS).astype(int)
        X["has_streaming"] = (
            (X["StreamingTV"] == "Yes") | (X["StreamingMovies"] == "Yes")
        ).astype(int)
        # Explicit interaction between the two strongest individual
        # predictors (Contract, tenure) — trees can already learn this via
        # splits, but giving it a dedicated column costs nothing and helps
        # the linear candidate (Logistic Regression) capture it directly.
        X["high_risk_new_customer"] = (
            (X["Contract"] == "Month-to-month") & (X["tenure"] <= HIGH_RISK_TENURE_MONTHS)
        ).astype(int)
        X["manual_payment"] = X["PaymentMethod"].isin(MANUAL_PAYMENT_METHODS).astype(int)
        X["tenure_bucket"] = pd.cut(
            X["tenure"], bins=TENURE_BUCKET_EDGES, labels=TENURE_BUCKET_LABELS
        ).astype(str)

        for col, redundant_value in NO_SERVICE_TO_NO_COLUMNS.items():
            X[col] = X[col].replace(redundant_value, "No")

        return X


class CustomerSegmentFeature(BaseEstimator, TransformerMixin):
    """K-Means customer segmentation (tenure, MonthlyCharges, TotalCharges,
    num_addon_services) as an additional categorical feature — inspired by
    an independent benchmark on this same dataset (arXiv:2607.10260) that
    used K=3 segments for its churn-risk/value analysis.

    Unlike FeatureEngineer's row-wise columns, this genuinely needs
    fitting (cluster centers learned from data), so it's its own pipeline
    step placed right after FeatureEngineer. Cross-validation machinery
    calls .fit() separately per fold, so the cluster centers are always
    learned from that fold's training rows only — no leakage from
    validation/test rows into where the cluster boundaries land.
    """

    CLUSTER_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "num_addon_services"]

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None) -> "CustomerSegmentFeature":
        self.scaler_ = StandardScaler().fit(X[self.CLUSTER_COLUMNS])
        scaled = self.scaler_.transform(X[self.CLUSTER_COLUMNS])
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
        ).fit(scaled)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        scaled = self.scaler_.transform(X[self.CLUSTER_COLUMNS])
        X["customer_segment"] = "segment_" + pd.Series(
            self.kmeans_.predict(scaled), index=X.index
        ).astype(str)
        return X

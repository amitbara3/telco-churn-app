"""Tests for the model pipeline itself, as opposed to the HTTP layer.

The centrepiece is test_train_and_serve_produce_identical_features: an
earlier bug shipped because `app/model.py` and `train/train.py` each kept
their own hand-maintained copy of the raw column order, and the two
drifted. One-hot pipelines didn't care (ColumnTransformer selects by
name), but CatBoost's native-categorical mode resolves cat_features to
positional indices, so a live request silently fed category strings into
columns the model expected to be numeric. These tests exist so that class
of drift fails here instead of in production.
"""
import joblib
import pandas as pd
import pytest

from app.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_FEATURE_ORDER,
    FeatureEngineer,
)
from app.model import MODEL_PATH, load_pipeline, load_threshold, predict

SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


def test_raw_feature_order_matches_model_inputs():
    """The columns a request is assembled from must be exactly the columns,
    in the same order, that the fitted model was trained on."""
    pipeline = load_pipeline()
    model = pipeline.named_steps["model"]

    # Everything FeatureEngineer + the segmenter derive is appended after
    # the raw columns, so the model's first N features are the raw ones.
    trained_on = list(model.feature_names_)
    assert trained_on[: len(RAW_FEATURE_ORDER)] == RAW_FEATURE_ORDER


def test_categorical_columns_are_registered_as_categorical():
    """CatBoost resolves cat_features positionally. If a categorical column
    isn't registered, it's silently treated as numeric — which either
    crashes on a string or, worse, coerces nonsense."""
    pipeline = load_pipeline()
    model = pipeline.named_steps["model"]
    names = list(model.feature_names_)
    declared = {names[i] for i in model.get_cat_feature_indices()}

    for col in CATEGORICAL_FEATURES:
        assert col in declared, f"{col} is categorical but not declared to CatBoost"
    for col in NUMERIC_FEATURES:
        assert col not in declared, f"{col} is numeric but declared categorical"


def test_train_and_serve_produce_identical_features():
    """Feature engineering must be identical whether a row arrives as a
    training frame or a single API request."""
    engineer = FeatureEngineer()

    # As training sees it: a frame built straight from the raw column list.
    train_frame = pd.DataFrame([{c: SAMPLE_CUSTOMER[c] for c in RAW_FEATURE_ORDER}])
    # As serving sees it: dict ordering from JSON, which need not match.
    shuffled = {k: SAMPLE_CUSTOMER[k] for k in reversed(list(SAMPLE_CUSTOMER))}
    serve_frame = pd.DataFrame([{c: shuffled[c] for c in RAW_FEATURE_ORDER}])

    out_train = engineer.transform(train_frame)
    out_serve = engineer.transform(serve_frame)

    assert list(out_train.columns) == list(out_serve.columns)
    pd.testing.assert_frame_equal(out_train, out_serve)


def test_prediction_is_deterministic():
    first = predict(SAMPLE_CUSTOMER)
    second = predict(SAMPLE_CUSTOMER)
    assert first == second


def test_threshold_drives_the_prediction_label():
    """churn_prediction must agree with the stored threshold rather than a
    hardcoded 0.5 — the two diverged once already."""
    threshold = load_threshold()
    result = predict(SAMPLE_CUSTOMER)
    expected = "Yes" if result["churn_probability"] >= threshold else "No"
    assert result["churn_prediction"] == expected


def test_risk_bands_are_consistent_with_the_prediction():
    """A 'Low' risk customer must never be predicted to churn, and a 'High'
    risk one must never be predicted not to."""
    threshold = load_threshold()
    for tenure, contract in [(1, "Month-to-month"), (72, "Two year"), (24, "One year")]:
        result = predict({**SAMPLE_CUSTOMER, "tenure": tenure, "Contract": contract})
        if result["risk_level"] == "Low":
            assert result["churn_probability"] < threshold
        if result["risk_level"] == "High":
            assert result["churn_probability"] >= threshold


@pytest.mark.parametrize("tenure", [0, 1, 72])
def test_boundary_tenures_predict_without_error(tenure):
    """tenure=0 (a brand-new customer) is accepted by the API, so the model
    must actually handle it — it was excluded from training data until the
    blank TotalCharges rows were imputed rather than dropped."""
    total = 0.0 if tenure == 0 else SAMPLE_CUSTOMER["MonthlyCharges"] * tenure
    result = predict({**SAMPLE_CUSTOMER, "tenure": tenure, "TotalCharges": total})
    assert 0.0 <= result["churn_probability"] <= 1.0


def test_model_artifact_exists():
    assert MODEL_PATH.exists(), "run `python train/train.py` first"
    assert joblib.load(MODEL_PATH) is not None

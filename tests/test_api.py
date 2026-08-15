import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

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


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_response():
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in ("Yes", "No")
    assert body["risk_level"] in ("Low", "Medium", "High")


def test_predict_rejects_invalid_category():
    bad_customer = {**SAMPLE_CUSTOMER, "InternetService": "Satellite"}
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    incomplete = dict(SAMPLE_CUSTOMER)
    del incomplete["tenure"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenure", -1),
        ("tenure", 1000),
        ("MonthlyCharges", -1.0),
        ("MonthlyCharges", 10_000.0),
        ("TotalCharges", -1.0),
        ("TotalCharges", 1_000_000.0),
    ],
)
def test_predict_rejects_out_of_range_numbers(field, value):
    """A tree model clamps rather than extrapolates, so an absurd input
    returns a confident-looking number instead of failing. Reject at the
    edge rather than answer a question the model can't actually answer."""
    response = client.post("/predict", json={**SAMPLE_CUSTOMER, field: value})
    assert response.status_code == 422


def test_model_info_reports_what_is_deployed():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "catboost_native"
    assert body["calibration"] == "sigmoid"
    assert 0.0 < body["decision_threshold"] < 1.0


def test_logger_emits_independently_of_the_root_logger():
    """uvicorn installs handlers on its own loggers and leaves root bare, so
    a module logger that only propagates emits nothing in production while
    still passing a caplog-based test. This asserts the configuration that
    makes it actually emit — the check that would have caught that."""
    logger = logging.getLogger("churn")
    assert logger.handlers, "no handler: nothing will be written under uvicorn"
    assert logger.level <= logging.INFO
    assert logger.propagate is False, "would double-log if root is configured"


def test_prediction_is_logged_without_customer_details(caplog):
    """Logs must carry enough to monitor output drift, but no customer
    attributes — the request body is personal data."""
    logger = logging.getLogger("churn")
    # Attach caplog's handler directly: the logger deliberately does not
    # propagate to root, which is where caplog normally listens.
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="churn"):
            client.post("/predict", json=SAMPLE_CUSTOMER)
    finally:
        logger.removeHandler(caplog.handler)

    records = [r.message for r in caplog.records if '"event": "prediction"' in r.message]
    assert records, "prediction was not logged"
    logged = json.loads(records[0])
    assert set(logged) == {
        "event", "churn_probability", "churn_prediction",
        "risk_level", "tenure_bucket", "latency_ms",
    }
    # None of the raw customer attributes may appear anywhere in the line.
    for field in ("PaymentMethod", "Contract", "MonthlyCharges", "TotalCharges", "gender"):
        assert field not in records[0]


def test_predict_accepts_brand_new_customer():
    """tenure=0 with no charges yet is a real, valid customer state."""
    response = client.post(
        "/predict", json={**SAMPLE_CUSTOMER, "tenure": 0, "TotalCharges": 0.0}
    )
    assert response.status_code == 200
    assert 0.0 <= response.json()["churn_probability"] <= 1.0

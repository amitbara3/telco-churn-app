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


def test_predict_accepts_brand_new_customer():
    """tenure=0 with no charges yet is a real, valid customer state."""
    response = client.post(
        "/predict", json={**SAMPLE_CUSTOMER, "tenure": 0, "TotalCharges": 0.0}
    )
    assert response.status_code == 200
    assert 0.0 <= response.json()["churn_probability"] <= 1.0

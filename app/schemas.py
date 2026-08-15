"""Request/response models for the churn-prediction API.

The Literal values below mirror the exact categories present in the
Telco Customer Churn dataset (see train/train.py / model/feature_schema.json).

Numeric bounds are set from the observed training range with deliberate
headroom, rather than left effectively open. A tree model doesn't
extrapolate — it clamps to the nearest leaf — so an out-of-range input
doesn't error, it silently returns a confident-looking number the model
has no basis for. Rejecting those at the edge is more honest than
answering them.
"""
from typing import Literal

from pydantic import BaseModel, Field

YesNo = Literal["Yes", "No"]

# Training data spans tenure 0-72 (6 years, the dataset's own ceiling) and
# MonthlyCharges 18.25-118.75. The bounds below allow modest headroom for
# genuinely new data while still rejecting nonsense.
MAX_TENURE_MONTHS = 100
MAX_MONTHLY_CHARGES = 500.0
MAX_TOTAL_CHARGES = 50_000.0


class CustomerFeatures(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(
        description="1 if the customer is a senior citizen, else 0"
    )
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(
        ge=0,
        le=MAX_TENURE_MONTHS,
        description="Months the customer has stayed (training data: 0-72)",
    )
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(
        ge=0,
        le=MAX_MONTHLY_CHARGES,
        description="Current total monthly charge (training data: 18.25-118.75)",
    )
    TotalCharges: float = Field(
        ge=0,
        le=MAX_TOTAL_CHARGES,
        description="Total amount charged to date (training data: 0-8684.80)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float = Field(
        description="Calibrated probability of churn, 0-1. Calibrated means "
        "0.7 really does correspond to roughly a 70% churn rate."
    )
    churn_prediction: Literal["Yes", "No"]
    risk_level: Literal["Low", "Medium", "High"]


class ModelInfo(BaseModel):
    """What a running instance is actually serving."""

    model: str
    calibration: str | None
    decision_threshold: float
    test_churn_f1: float | None
    test_balanced_accuracy: float | None
    test_roc_auc: float | None
    expected_calibration_error: float | None

    # `model` collides with pydantic's protected namespace; this field is
    # the model's *name*, so keep it rather than rename the API surface.
    model_config = {"protected_namespaces": ()}

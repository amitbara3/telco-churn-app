"""Request/response models for the churn-prediction API.

The Literal values below mirror the exact categories present in the
Telco Customer Churn dataset (see train/train.py / model/feature_schema.json).
"""
from typing import Literal

from pydantic import BaseModel, Field

YesNo = Literal["Yes", "No"]


class CustomerFeatures(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(
        description="1 if the customer is a senior citizen, else 0"
    )
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Months the customer has stayed")
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
    MonthlyCharges: float = Field(ge=0, description="Current total monthly charge")
    TotalCharges: float = Field(ge=0, description="Total amount charged to date")

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
    churn_probability: float = Field(description="Model probability of churn, 0-1")
    churn_prediction: Literal["Yes", "No"]
    risk_level: Literal["Low", "Medium", "High"]

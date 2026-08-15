"""Streamlit front-end for the churn-prediction API.

Renders a form for the 19 customer features, calls the FastAPI service
running alongside it in the same container, and shows the prediction.
"""
import json
import os
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "model" / "feature_schema.json"
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="centered")


@st.cache_data
def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def main() -> None:
    st.title("📉 Telco Customer Churn Predictor")
    st.caption(
        "FastAPI backend + scikit-learn model, trained on the Kaggle "
        "'Telco Customer Churn' dataset. Fill in a customer profile and "
        "get a live churn prediction."
    )

    if not SCHEMA_PATH.exists():
        st.error(
            "No trained model/schema found. Run `python train/train.py` "
            "before starting the app."
        )
        return

    schema = load_schema()
    cat = schema["categorical_features"]
    num = schema["numeric_features"]

    with st.form("customer_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", cat["gender"])
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda v: "Yes" if v else "No")
            partner = st.selectbox("Has partner", cat["Partner"])
            dependents = st.selectbox("Has dependents", cat["Dependents"])
            tenure = st.slider("Tenure (months)", 0, int(num["tenure"]["max"]), 12)
            phone_service = st.selectbox("Phone service", cat["PhoneService"])
            multiple_lines = st.selectbox("Multiple lines", cat["MultipleLines"])
            internet_service = st.selectbox("Internet service", cat["InternetService"])
            online_security = st.selectbox("Online security", cat["OnlineSecurity"])
            online_backup = st.selectbox("Online backup", cat["OnlineBackup"])

        with col2:
            device_protection = st.selectbox("Device protection", cat["DeviceProtection"])
            tech_support = st.selectbox("Tech support", cat["TechSupport"])
            streaming_tv = st.selectbox("Streaming TV", cat["StreamingTV"])
            streaming_movies = st.selectbox("Streaming movies", cat["StreamingMovies"])
            contract = st.selectbox("Contract", cat["Contract"])
            paperless_billing = st.selectbox("Paperless billing", cat["PaperlessBilling"])
            payment_method = st.selectbox("Payment method", cat["PaymentMethod"])
            monthly_charges = st.number_input(
                "Monthly charges ($)", min_value=0.0, value=round(num["MonthlyCharges"]["mean"], 2)
            )
            total_charges = st.number_input(
                "Total charges ($)", min_value=0.0, value=round(num["TotalCharges"]["mean"], 2)
            )

        submitted = st.form_submit_button("Predict churn")

    if submitted:
        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"Could not reach the prediction API at {API_URL}: {exc}")
            return

        st.divider()
        prob = result["churn_probability"]
        risk_color = {"Low": "green", "Medium": "orange", "High": "red"}[result["risk_level"]]

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn probability", f"{prob:.1%}")
        c2.metric("Prediction", result["churn_prediction"])
        c3.markdown(f"**Risk level:** :{risk_color}[{result['risk_level']}]")
        st.progress(min(max(prob, 0.0), 1.0))


if __name__ == "__main__":
    main()

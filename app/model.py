"""Loads the trained churn model and turns a request into a prediction."""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from app.features import RAW_FEATURE_ORDER

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "churn_model.joblib"
THRESHOLD_PATH = ROOT / "model" / "decision_threshold.json"


@lru_cache(maxsize=1)
def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run `python train/train.py` first."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_threshold() -> float:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"No decision threshold found at {THRESHOLD_PATH}. Run `python train/train.py` first."
        )
    return json.loads(THRESHOLD_PATH.read_text())["threshold"]


def predict(features: dict) -> dict:
    pipeline = load_pipeline()
    threshold = load_threshold()
    row = pd.DataFrame([{col: features[col] for col in RAW_FEATURE_ORDER}])

    probability = float(pipeline.predict_proba(row)[0, 1])
    prediction = "Yes" if probability >= threshold else "No"

    # Risk bands are anchored to the tuned decision threshold rather than
    # fixed at 0.33/0.66, so "Medium" always straddles the actual Yes/No
    # cutoff instead of drifting out of sync with it (the threshold can be
    # well below 0.5 — e.g. ~0.29 for the current model).
    low_cut = threshold / 2
    high_cut = (1 + threshold) / 2
    if probability < low_cut:
        risk = "Low"
    elif probability < high_cut:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "churn_probability": round(probability, 4),
        "churn_prediction": prediction,
        "risk_level": risk,
    }

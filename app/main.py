"""FastAPI service that serves churn predictions from the trained model."""
import json
import logging
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model import load_metadata, load_pipeline, load_threshold, predict
from app.schemas import CustomerFeatures, ModelInfo, PredictionResponse

logger = logging.getLogger("churn")


def _configure_logging() -> None:
    """Give this logger its own stdout handler.

    uvicorn installs handlers on its own loggers and leaves the root logger
    bare, so a module logger that merely propagates emits nothing at all
    under `uvicorn app.main:app` — the logs look fine locally under pytest
    (caplog attaches a root handler) and are silently absent in production.
    """
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # Already emitting via our own handler; don't also hand it to root.
    logger.propagate = False


_configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load (and cache) the model + decision threshold once at startup
    # instead of on first request.
    load_pipeline()
    load_threshold()
    yield


app = FastAPI(
    title="Telco Customer Churn API",
    description="Predicts whether a telecom customer is likely to churn.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    """What's actually deployed. Lets you confirm which model and threshold
    a running instance is serving without shelling into the container."""
    return ModelInfo(**load_metadata())


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerFeatures) -> PredictionResponse:
    started = time.perf_counter()
    try:
        result = predict(customer.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Log the prediction, not the customer. Enough to monitor the output
    # distribution for drift — the thing that silently degrades a churn
    # model once the calibration stops matching reality — without writing
    # anyone's contract, payment method or charges to disk.
    logger.info(
        json.dumps(
            {
                "event": "prediction",
                "churn_probability": result["churn_probability"],
                "churn_prediction": result["churn_prediction"],
                "risk_level": result["risk_level"],
                "tenure_bucket": _tenure_bucket(customer.tenure),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
    )
    return PredictionResponse(**result)


def _tenure_bucket(tenure: int) -> str:
    """Coarse tenure band for drift monitoring — deliberately not the raw
    value, which is closer to identifying."""
    if tenure <= 12:
        return "0-12"
    if tenure <= 24:
        return "12-24"
    if tenure <= 48:
        return "24-48"
    return "48+"

"""FastAPI service that serves churn predictions from the trained model."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model import load_pipeline, load_threshold, predict
from app.schemas import CustomerFeatures, PredictionResponse


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


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerFeatures) -> PredictionResponse:
    try:
        result = predict(customer.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(**result)

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.predictor import CustomerRiskPredictor, build_predictor

class CustomerRiskRequest(BaseModel):
    age: int = Field(...,ge=18,le=100)
    monthly_income: float = Field(...,ge=0)
    credit_score: int = Field(...,ge=300,le=850)
    months_as_customer: int = Field(...,ge=0)
    num_previous_purchases: int = Field(..., ge=0)
    avg_payment_delay_days: float = Field(..., ge=0)
    num_late_payments: int = Field(..., ge=0)
    outstanding_balance: float = Field(..., ge=0)
    employment_type: str
    channel: str

class PredictionResponse(BaseModel):
    risk_score: float
    risk_label: int
    risk_segment: str
    model_name: str | None
    model_version: str | None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class ModelInfoResponse(BaseModel):
    model_name: str | None
    model_version: str | None
    algorithm: str | None
    main_metric: dict[str,Any] | None
    feature_columns: list[str] | None

app_state: dict[str,CustomerRiskPredictor | None] = {
    "predictor": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["predictor"] = build_predictor(model_version="v1")
    yield
    app_state["predictor"] = None

app = FastAPI(
    title="Customer Risk Scoring API",
    description="API for customer risk scoring a trained ML model",
    version = "1.0.0",
    lifespan=lifespan,
)


@app.get("/health",response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded = app_state["predictor"] is not None
    )

@app.get("/model-info",response_model=ModelInfoResponse)
def model_info()->ModelInfoResponse:
    predictor = app_state["predictor"]

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    metadata = predictor.metadata

    return ModelInfoResponse(
        model_name=metadata.get("model_name"),
        model_version=metadata.get("model_version"),
        algorithm=metadata.get("algorithm"),
        main_metric=metadata.get("main_metric"),
        feature_columns=metadata.get("feature_columns"),
    )

@app.post("/predict",response_model=PredictionResponse)
def predict(request: CustomerRiskRequest) -> PredictionResponse:
    predictor = app_state["predictor"]

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        prediction = predictor.predict_one(request.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=400,detail=str(error)) from error

    return PredictionResponse(**prediction)
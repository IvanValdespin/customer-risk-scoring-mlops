from fastapi.testclient import TestClient
from src.api.main import app

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True

def test_model_info_endpoint():
    with TestClient(app) as client:
        response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "customer_risk_model"
    assert data["model_version"] == "v1"
    assert data["algorithm"] == "RandomForestClassifier"
    assert "feature_columns" in data
    assert isinstance(data["feature_columns"], list)

def test_predict_endpoint_returns_valid_prediction():

    payload = {
        "age": 34,
        "monthly_income": 18000,
        "credit_score": 620,
        "months_as_customer": 24,
        "num_previous_purchases": 8,
        "avg_payment_delay_days": 6,
        "num_late_payments": 2,
        "outstanding_balance": 12000,
        "employment_type": "formal",
        "channel": "store",
    }

    with TestClient(app) as client:
        response = client.post("/predict",json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "risk_score" in data
    assert "risk_label" in data
    assert "risk_segment" in data
    assert "model_name" in data
    assert "model_version" in data

    assert 0 <= data["risk_score"] <= 1
    assert data["risk_label"] in [0, 1]
    assert data["risk_segment"] in ["low", "medium", "high"]
    assert data["model_version"] == "v1"

def test_predict_endpoint_validates_bad_payload():
    payload = {
        "age": 34,
        "monthly_income": 18000,
        # credit_score missing intentionally
        "months_as_customer": 24,
        "num_previous_purchases": 8,
        "avg_payment_delay_days": 6,
        "num_late_payments": 2,
        "outstanding_balance": 12000,
        "employment_type": "formal",
        "channel": "store",
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
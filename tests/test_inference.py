import pytest
from src.inference.predictor import build_predictor

def sample_customer() -> dict:
    return {
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


def test_predictor_returns_valid_prediction():
    predictor = build_predictor(model_version="v1")

    prediction = predictor.predict_one(sample_customer())

    assert "risk_score" in prediction
    assert "risk_label" in prediction
    assert "risk_segment" in prediction
    assert "model_name" in prediction
    assert "model_version" in prediction

    assert 0 <= prediction["risk_score"] <= 1
    assert prediction["risk_label"] in [0, 1]
    assert prediction["risk_segment"] in ["low", "medium", "high"]
    assert prediction["model_version"] == "v1"


def test_predictor_fails_with_missing_feature():
    predictor = build_predictor(model_version="v1")

    customer = sample_customer()
    customer.pop("credit_score")

    with pytest.raises(ValueError, match="Missing required features"):
        predictor.predict_one(customer)
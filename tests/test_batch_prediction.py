from pathlib import Path

import pandas as pd
from src.data.generate_synthetic_data import generate_customer_risk_data
from src.inference.batch_predict import load_scoring_data, run_batch_prediction
from src.training.train_model import FEATURE_COLUMNS

def test_load_scoring_data_accepts_valid_file(tmp_path):
    df = generate_customer_risk_data(n_samples=50, random_state=42)
    input_path = tmp_path / "scoring_data.csv"
    df.to_csv(input_path, index=False)

    loaded_df = load_scoring_data(input_path)

    assert len(loaded_df) == 50
    assert set(FEATURE_COLUMNS).issubset(loaded_df.columns)


def test_run_batch_prediction_creates_output_file(tmp_path):
    df = generate_customer_risk_data(n_samples=50, random_state=42)
    input_path = tmp_path / "scoring_data.csv"
    output_path = tmp_path / "predictions.csv"

    df.to_csv(input_path, index=False)

    run_batch_prediction(
        input_path=input_path,
        output_path=output_path,
        model_version="v1",
    )

    assert output_path.exists()

    output_df = pd.read_csv(output_path)

    assert len(output_df) == 50
    assert "risk_score" in output_df.columns
    assert "risk_label" in output_df.columns
    assert "risk_segment" in output_df.columns
    assert "model_version" in output_df.columns

import argparse
import json
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
import numpy as np
from src.training.train_model import FEATURE_COLUMNS
from src.utils.paths import METADATA_DIR, MODELS_DIR, PREPROCESSORS

class CustomerRiskPredictor:
    def __init__(self,
                 model_path: Path,
                 preprocessor_path : Path,
                 metadata_path: Path,
                 ) ->None:
        
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.metadata_path = metadata_path

        self.model = None
        self.preprocessor = None
        self.metadata: dict[str,Any] = {}

        self.load_artifacts()

    def load_artifacts(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")

        if not self.preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor artifact not found: {self.preprocessor_path}"
            )
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata artifact not found: {self.metadata_path}")

        self.model= joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

        with self.metadata_path.open("r",encoding="utf-8") as file:
            self.metadata = json.load(file)

    def validate_input(self, input_data: dict[str,Any]) ->None:
        missing_features = set(FEATURE_COLUMNS) - set(input_data.keys())

        if missing_features:
            raise ValueError(f"Missing required features: {sorted(missing_features)}")

    def predict_one(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self.validate_input(input_data)

        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

        processed_input = self.preprocessor.transform(input_df)

        if hasattr(processed_input, "toarray"):
            processed_input = processed_input.toarray()

        processed_input = np.asarray(processed_input)

        if processed_input.ndim == 1:
            processed_input = processed_input.reshape(1, -1)

        risk_score = float(self.model.predict_proba(processed_input)[:, 1][0])
        risk_label = int(self.model.predict(processed_input)[0])

        risk_segment = self._risk_segment(risk_score)

        return {
            "risk_score": round(risk_score, 4),
            "risk_label": risk_label,
            "risk_segment": risk_segment,
            "model_name": self.metadata.get("model_name"),
            "model_version": self.metadata.get("model_version"),
        }

    @staticmethod
    def _risk_segment(risk_score: float) -> str:
        if risk_score >= 0.70:
            return "high"
        if risk_score >= 0.40:
            return "medium"
        return"low"

def build_predictor(model_version: str = "v1") -> CustomerRiskPredictor:
    model_path = MODELS_DIR / f"customer_risk_model_{model_version}.joblib"
    preprocessor_path = PREPROCESSORS / f"preprocessor_{model_version}.joblib"
    metadata_path = METADATA_DIR / f"model_metadata_{model_version}.json"

    return CustomerRiskPredictor(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        metadata_path=metadata_path,
    )


def main() ->None:

    
    parser = argparse.ArgumentParser(description="Run a single customer risk prediction.")
    parser.add_argument("--model-version",type=str,default="v1")

    args = parser.parse_args()

    predictor = build_predictor(model_version=args.model_version)

    sample_customer = {
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

    prediction = predictor.predict_one(sample_customer)

    print(json.dumps(prediction,indent=2))
    

if __name__ == "__main__": 
    main()
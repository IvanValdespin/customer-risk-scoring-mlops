import argparse
from pathlib import Path
import pandas as pd
from src.inference.predictor import build_predictor
from src.training.train_model import FEATURE_COLUMNS
from src.utils.paths import PREDICTIONS_DIR, RAW_DATA_DIR


def load_scoring_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input scoring file not found: {input_path}")

    df = pd.read_csv(input_path)

    missing_features = set(FEATURE_COLUMNS) - set(df.columns)

    if missing_features:
        raise ValueError(f"Missing required features: {sorted(missing_features)}")

    return df

def run_batch_prediction(
        input_path: Path,
        output_path: Path,
        model_version: str,
) -> None:

    df = load_scoring_data(input_path)

    predictor = build_predictor(model_version=model_version)

    predictions = []

    for _,row,in df.iterrows():
        input_data = row[FEATURE_COLUMNS].to_dict()
        prediction = predictor.predict_one(input_data)
        predictions.append(prediction)

    predictions_df = pd.DataFrame(predictions)

    output_df = pd.concat(
        [
            df.reset_index(drop=True),
            predictions_df.reset_index(drop=True),
        ],
        axis=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path,index=False)

    print(f"Batch prediction completed succesfully.")
    print(f"Input rows: {len(df)}")
    print(f"Output file: {output_path}")


def main() -> None:

    parser = argparse.ArgumentParser(description="Run batch customer risk predictions.")
    parser.add_argument("--input-file", type=str, default="customer_risk_data.csv")
    parser.add_argument("--output-file", type=str, default="customer_risk_predictions.csv")
    parser.add_argument("--model-version", type=str, default="v1")

    args = parser.parse_args()

    input_path = RAW_DATA_DIR / args.input_file
    output_path = PREDICTIONS_DIR / args.output_file

    run_batch_prediction(
        input_path=input_path,
        output_path=output_path,
        model_version=args.model_version,
    )


if __name__ == "__main__":
    main()
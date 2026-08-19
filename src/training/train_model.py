import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.paths import METADATA_DIR,MODELS_DIR,PREPROCESSORS,RAW_DATA_DIR

TARGET_COLUMN = "risk_label"
ID_COLUMN = "customer_id"


NUMERIC_FEATURES = [
    "age",
    "monthly_income",
    "credit_score",
    "months_as_customer",
    "num_previous_purchases",
    "avg_payment_delay_days",
    "num_late_payments",
    "outstanding_balance",
]

CATEGORICAL_FEATURES = [
    "employment_type",
    "channel",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def load_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}."
            "Run: python -m src.data.generate_synthetic_data"
        )

    return pd.read_csv(input_path)

def validate_training_data(df: pd.DataFrame) -> None:
    required_columns = {ID_COLUMN,TARGET_COLUMN, *FEATURE_COLUMNS}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if df.empty:
        raise ValueError("Training dataset is empty.")

    invalid_target_values = set(df[TARGET_COLUMN].dropna().unique()) - {0,1}

    if invalid_target_values:
        raise ValueError(
            f"Invalid target values found: {sorted(invalid_target_values)}"
            "Expected only 0 an 1"
        )

def build_preprocessor()->ColumnTransformer:

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("encoder",OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric",numeric_pipeline,NUMERIC_FEATURES),
            ("categorical",categorical_pipeline,CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def train_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int,
) -> tuple[RandomForestClassifier,ColumnTransformer]:
    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=random_state,
        n_jobs= -1,
    )

    model.fit(X_train_processed,y_train)

    return model,preprocessor


def evaluate_model(
    model: RandomForestClassifier,
    preprocessor: ColumnTransformer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    X_test_processed = preprocessor.transform(X_test)

    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    return metrics


def save_json( data: dict[str, Any], output_path: Path) ->None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w",encoding="utf-8") as file:
        json.dump(data,file,indent=2)

def save_artifacts(
    model: RandomForestClassifier,
    preprocessor: ColumnTransformer,
    metrics: dict[str, Any],
    model_name: str,
    model_version: str,
    random_state: int,
    input_path: Path,
    train_rows: int,
    test_rows: int,
) -> None:
    
    MODELS_DIR.mkdir(parents=True,exist_ok=True)
    PREPROCESSORS.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True,exist_ok=True)

    model_path = MODELS_DIR / f"{model_name}_{model_version}.joblib"
    preprocessor_path = PREPROCESSORS / f"preprocessor_{model_version}.joblib"
    metrics_path = METADATA_DIR / f"metrics_{model_version}.json"
    metadata_path = METADATA_DIR / f"model_metadata_{model_version}.json"

    joblib.dump(model,model_path)
    joblib.dump(preprocessor,preprocessor_path)
    save_json(metrics,metadata_path)

    metadata = {
        "model_name": model_name,
        "model_version": model_version,
        "algorithm":"RandomForestClassifier",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features":NUMERIC_FEATURES,
        "categorical_features":CATEGORICAL_FEATURES,
        "input_data_path": str(input_path),
        "train_rows": train_rows,
        "test_rows":test_rows,
        "random_state": random_state,
        "artifacts": {
            "model_path": str(model_path),
            "preprocessor_path": str(preprocessor_path),
            "metrics_path": str(metrics_path),
        },
        "main_metric": {
            "name": "roc_auc",
            "value": metrics["roc_auc"],
        },
    }

    save_json(metadata,metadata_path)
    print("Artifacts saved successfully:")
    print(f"Model: {model_path}")
    print(f"Preprocessor: {preprocessor_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Metadata: {metadata_path}")



def main() ->None:
    parser = argparse.ArgumentParser(description="Train customer risk scoring model.")
    parser.add_argument("--input-file",type=str,default="customer_risk_data.csv")
    parser.add_argument("--model-name",type=str,default="customer_risk_model")
    parser.add_argument("--model-version",type=str,default="v1")
    parser.add_argument("--test-size",type=float,default=0.2)
    parser.add_argument("--random-state",type=int,default=42)

    args = parser.parse_args()
    input_path = RAW_DATA_DIR / args.input_file

    df = load_data(input_path)
    validate_training_data(df)

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train,X_test,y_train,y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model,preprocessor = train_model(
        X_train=X_train,
        y_train = y_train,
        random_state=args.random_state,
    )

    metrics = evaluate_model(
        model=model,
        preprocessor=preprocessor,
        X_test=X_test,
        y_test=y_test,
    )

    print("Model evaluation metrics:")
    print(json.dumps(metrics,indent=2))

    save_artifacts(
        model=model,
        preprocessor=preprocessor,
        metrics=metrics,
        model_name=args.model_name,
        model_version=args.model_version,
        random_state=args.random_state,
        input_path=input_path,
        train_rows=len(X_train),
        test_rows=len(X_test),
    )

if __name__ == "__main__":
    main()
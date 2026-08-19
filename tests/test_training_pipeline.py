from src.data.generate_synthetic_data import generate_customer_risk_data
from src.training.train_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    evaluate_model,
    train_model,
    validate_training_data
)
from sklearn.model_selection import train_test_split

def test_validate_training_data_accepts_valid_data():
    df = generate_customer_risk_data(n_samples=200,random_state=42)

    validate_training_data(df)

def test_train_model_returns_model_and_preprocessor():
    df = generate_customer_risk_data(n_samples=300,random_state=42)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model,preprocessor = train_model(
        X_train=X_train,
        y_train=y_train,
        random_state=42,
    )

    assert model is not None
    assert preprocessor is not None

    X_test_processed = preprocessor.transform(X_test)
    predictions = model.predict(X_test_processed)

    assert len(predictions) == len(X_test)

def test_evaluate_model_returns_expected_metrics():
    df = generate_customer_risk_data(n_samples=300, random_state=42)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model, preprocessor = train_model(
        X_train=X_train,
        y_train=y_train,
        random_state=42,
    )

    metrics = evaluate_model(
        model=model,
        preprocessor=preprocessor,
        X_test=X_test,
        y_test=y_test,
    )

    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "confusion_matrix",
        "classification_report",
    }

    assert expected_metrics.issubset(metrics.keys())
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1
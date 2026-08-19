from src.data.generate_synthetic_data import generate_customer_risk_data


def test_generate_customer_risk_data_shape():

    df = generate_customer_risk_data(n_samples=100,random_state=42)

    assert df.shape[0] == 100
    assert"risk_label" in df.columns
    assert "customer_id" in df.columns

def test_generate_customer_risk_target_values():
    df = generate_customer_risk_data(n_samples=100,random_state=42)

    valid_values = {0,1}
    actual_values = set(df["risk_label"].unique())

    assert actual_values.issubset(valid_values)

def test_generate_customer_risk_data_required_columns():
    df= generate_customer_risk_data(n_samples=100,random_state=42)

    expected_columns = {
        "customer_id",
        "age",
        "monthly_income",
        "credit_score",
        "months_as_customer",
        "num_previous_purchases",
        "avg_payment_delay_days",
        "num_late_payments",
        "outstanding_balance",
        "employment_type",
        "channel",
        "risk_label",
    }

    assert set(df.columns) == expected_columns
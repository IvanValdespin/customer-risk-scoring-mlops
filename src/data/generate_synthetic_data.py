import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.paths import RAW_DATA_DIR

def generate_customer_risk_data(n_samples: int = 5000, random_state: int = 42) ->pd.DataFrame:
    """
    Generate a synthetic customer risk dataset.

    Target:
        risk_label = 1 means high risk
        risk_label = 0 means normal/low risk
    """
    rng = np.random.default_rng(random_state)

    customer_id = [f"CUST_{i:06d}" for i in range(1, n_samples + 1)]
    age = rng.integers(18,70, size=n_samples)
    monthly_income = rng.normal(loc=18000, scale=7000, size=n_samples).clip(4000,80000).round(2)
    credit_score = rng.normal(loc=650,scale=90, size=n_samples).clip(300,850).round(0)
    months_as_customer = rng.integers(1, 120, size=n_samples)
    num_previous_purchases = rng.poisson(lam=6, size=n_samples).clip(0, 40)
    avg_payment_delay_days = rng.exponential(scale=5, size=n_samples).clip(0, 60).round(1)
    num_late_payments = rng.poisson(lam=1.2, size=n_samples).clip(0, 15)
    outstanding_balance = rng.normal(loc=12000, scale=8000, size=n_samples).clip(0, 120000).round(2)

    employment_type = rng.choice(
        ["formal","informal","self_employed","unemployed"],
        size=n_samples,
        p=[0.55,0.25,0.15,0.05],
    )

    channel = rng.choice(
        ["store","online","app","call_center"],
        size=n_samples,
        p=[0.55,0.20,0.15,0.10],
    )

    # Synthetic risk logic.
    # This creates a realistic-ish probability of high risk.
    risk_score_raw = (
        -0.006 * (credit_score - 650)
        -0.000035 * (monthly_income - 18000)
        +0.055 * avg_payment_delay_days
        +0.22 * num_late_payments
        +0.000025 * outstanding_balance
        -0.008 * months_as_customer
        +0.35 * (employment_type == "unemployed")
        +0.18 * (employment_type == "informal")
        +0.08 * (channel == "call_center")
    )

    probability = 1 / (1 + np.exp(-risk_score_raw))
    risk_label = rng.binomial(1, probability)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "monthly_income": monthly_income,
            "credit_score": credit_score.astype(int),
            "months_as_customer": months_as_customer,
            "num_previous_purchases": num_previous_purchases,
            "avg_payment_delay_days": avg_payment_delay_days,
            "num_late_payments": num_late_payments,
            "outstanding_balance": outstanding_balance,
            "employment_type": employment_type,
            "channel": channel,
            "risk_label": risk_label,
        }
    )

    return df


def save_dataset(df:pd.DataFrame,output_path: Path) -> None:
    output_path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(output_path,index=False)

def main() ->None:
    parser = argparse.ArgumentParser(description="Generate synthetic customer risk data")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--output-file", type=str, default="customer_risk_data.csv")
    parser.add_argument("--random-state",type=int,default=42)


    args = parser.parse_args()

    df = generate_customer_risk_data(
        n_samples= args.n_samples,
        random_state=args.random_state,
    )

    output_path = RAW_DATA_DIR / args.output_file
    save_dataset(df,output_path)

    print(f"Dataset generated successfully: {output_path}")
    print(f"Shape: {df.shape}")
    print("Target distribution:")
    print(df["risk_label"].value_counts(normalize=True).round(3))

if __name__ == "__main__":
    main()
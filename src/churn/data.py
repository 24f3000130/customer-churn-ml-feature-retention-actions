from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import requests

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
REQUIRED = {"customerID", "tenure", "MonthlyCharges", "TotalCharges", "Contract", "Churn"}

def validate(df: pd.DataFrame, require_target: bool = True) -> None:
    required = REQUIRED if require_target else REQUIRED - {"Churn"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty")
    if require_target and not set(df["Churn"].dropna().unique()).issubset({"Yes", "No", 1, 0}):
        raise ValueError("Churn must contain Yes/No or 1/0")

def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate(df)
    return df

def download(output: str | Path = "data/raw/telco_churn.csv", force: bool = False) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if force or not output.exists():
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
        output.write_bytes(response.content)
    validate(pd.read_csv(output))
    return output

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/telco_churn.csv")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(download(args.output, args.force))

if __name__ == "__main__":
    main()

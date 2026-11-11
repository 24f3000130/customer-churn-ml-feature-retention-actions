from __future__ import annotations
import pandas as pd

TARGET = "Churn"
DROP_COLUMNS = ["customerID"]
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

def _normalize_numeric_columns(work: pd.DataFrame) -> pd.DataFrame:
    """Keep scoring dtypes consistent even when CSV values are stored as text."""
    if "SeniorCitizen" in work:
        mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, True: 1, False: 0}
        normalized = work["SeniorCitizen"].map(lambda value: mapping.get(value, value))
        work["SeniorCitizen"] = pd.to_numeric(normalized, errors="coerce")
    for column in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if column in work:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    return work

def prepare_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    work = _normalize_numeric_columns(df.copy())
    work["AvgMonthlySpend"] = work["TotalCharges"] / work["tenure"].replace(0, 1)
    work["IsNewCustomer"] = (work["tenure"] <= 6).astype(int)
    work["ServiceCount"] = sum(
        (work[c] == "Yes").astype(int)
        for c in ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        if c in work
    )
    y = work.pop(TARGET).map({"Yes": 1, "No": 0}).fillna(df[TARGET]).astype(int)
    return work.drop(columns=DROP_COLUMNS, errors="ignore"), y

def prepare_scoring_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = _normalize_numeric_columns(df.copy())
    work["AvgMonthlySpend"] = work["TotalCharges"] / work["tenure"].replace(0, 1)
    work["IsNewCustomer"] = (work["tenure"] <= 6).astype(int)
    work["ServiceCount"] = sum(
        (work[c] == "Yes").astype(int)
        for c in ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        if c in work
    )
    return work.drop(columns=[TARGET, *DROP_COLUMNS], errors="ignore")

def align_model_numeric_types(frame: pd.DataFrame, pipeline) -> pd.DataFrame:
    """Coerce every column routed to the fitted model's numeric transformer."""
    aligned = frame.copy()
    preprocessor = pipeline.named_steps["preprocess"]
    numeric_columns = next(
        columns for name, _, columns in preprocessor.transformers_ if name == "num"
    )
    for column in numeric_columns:
        if column in aligned:
            aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
    return aligned

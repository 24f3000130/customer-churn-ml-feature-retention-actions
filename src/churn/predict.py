from __future__ import annotations
import argparse
import joblib
import pandas as pd
from churn.data import validate
from churn.features import prepare_scoring_frame

def score(input_path, model_path="models/churn_model.joblib", output_path="predictions.csv"):
    raw = pd.read_csv(input_path); validate(raw, require_target=False)
    artifact = joblib.load(model_path)
    X = prepare_scoring_frame(raw).reindex(columns=artifact.feature_columns)
    probability = artifact.pipeline.predict_proba(X)[:, 1]
    result = raw[["customerID"]].copy() if "customerID" in raw else pd.DataFrame(index=raw.index)
    result["churn_probability"] = probability
    result["churn_prediction"] = (probability >= artifact.threshold).astype(int)
    result["risk_band"] = pd.cut(probability, [-1,.3,.6,1], labels=["Low","Medium","High"])
    result.to_csv(output_path, index=False); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--model",default="models/churn_model.joblib"); p.add_argument("--output",default="predictions.csv"); a=p.parse_args(); print(score(a.input,a.model,a.output).head())
if __name__ == "__main__": main()


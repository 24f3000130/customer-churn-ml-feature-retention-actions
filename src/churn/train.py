from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from churn.data import load
from churn.features import prepare_frame
from churn.model import ModelArtifact, build_pipeline, candidates, choose_threshold

SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "recall": "recall", "precision": "precision", "f1": "f1"}

def train(data_path="data/raw/telco_churn.csv", output_dir="models", report_dir="reports"):
    X, y = prepare_frame(load(data_path))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    rows, fitted = [], {}
    for name, classifier in candidates().items():
        pipe = build_pipeline(X_train, classifier)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        rows.append({"model": name, **{metric: scores[f"test_{metric}"].mean() for metric in SCORING}})
        fitted[name] = pipe.fit(X_train, y_train)
    comparison = pd.DataFrame(rows).sort_values(["pr_auc", "recall"], ascending=False)
    best_name = comparison.iloc[0]["model"]
    best = fitted[best_name]
    prob = best.predict_proba(X_test)[:, 1]
    threshold = choose_threshold(y_test, prob)
    pred = (prob >= threshold).astype(int)
    metrics = {"model": best_name, "threshold": threshold, "roc_auc": roc_auc_score(y_test, prob), "pr_auc": average_precision_score(y_test, prob), "recall": recall_score(y_test, pred), "precision": precision_score(y_test, pred), "f1": f1_score(y_test, pred), "brier_score": brier_score_loss(y_test, prob)}
    Path(output_dir).mkdir(parents=True, exist_ok=True); Path(report_dir, "figures").mkdir(parents=True, exist_ok=True)
    artifact = ModelArtifact(best, threshold, best_name, X.columns.tolist())
    joblib.dump(artifact, Path(output_dir) / "churn_model.joblib")
    comparison.to_csv(Path(report_dir) / "model_comparison.csv", index=False)
    Path(report_dir, "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    importance = permutation_importance(best, X_test, y_test, scoring="average_precision", n_repeats=5, random_state=42, n_jobs=-1)
    pd.DataFrame({"feature": X.columns, "importance": importance.importances_mean}).sort_values("importance", ascending=False).to_csv(Path(report_dir) / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 4)); sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", ax=ax); ax.set(xlabel="Predicted", ylabel="Actual", title="Churn confusion matrix"); fig.tight_layout(); fig.savefig(Path(report_dir, "figures", "confusion_matrix.png"), dpi=160); plt.close(fig)
    return artifact, metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data", default="data/raw/telco_churn.csv"); p.add_argument("--output-dir", default="models"); p.add_argument("--report-dir", default="reports"); a=p.parse_args()
    _, metrics=train(a.data,a.output_dir,a.report_dir); print(json.dumps(metrics,indent=2))
if __name__ == "__main__": main()


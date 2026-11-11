# Customer Churn Prediction Pipeline

An end-to-end machine-learning project that predicts telecom customer churn and turns model scores into retention actions. It includes reproducible data acquisition, validation, feature engineering, class-imbalance handling, model comparison, explainability, tests, and a Streamlit scoring app.

## Business question

Which active customers are most likely to leave, and which risk factors can a retention team act on? The model outputs a churn probability and a risk band. Because missing a likely churner is costly, model selection prioritizes recall and PR-AUC while still reporting ROC-AUC, precision, F1 and calibration-oriented Brier score.

The Streamlit dashboard also produces a staff action list with outreach priority, recommended retention measures, and a plain-language reason. Recommendations use actionable account and service fields—for example contract type, support enrollment, payment method, and tenure—and are included in the downloadable scored-customer CSV. They are suggestions for staff review; the app does not contact customers or apply offers automatically.

## Data

The project uses the public **IBM Telco Customer Churn** sample (7,043 customers, 21 fields). `python -m churn.data` downloads a pinned copy from IBM's public GitHub repository and validates its schema.

## Architecture

```text
IBM CSV -> schema validation -> train/test split -> preprocessing
  -> Logistic Regression / Random Forest / HistGradientBoosting
  -> cross-validation & threshold selection -> model artifact
  -> batch/API-style scoring and Streamlit dashboard
```

All transformations live inside the saved scikit-learn/imbalanced-learn pipeline, preventing train/serve skew and leakage. SMOTE-NC is represented conservatively here with `RandomOverSampler` after preprocessing because one-hot encoded categories are sparse; only training folds are resampled.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
python -m churn.data
python -m churn.train
streamlit run app.py
```

Keep `PYTHONPATH="$PWD/src"` set while working in the project. Run tests with `pytest -q`. Train on a custom CSV with `python -m churn.train --data path/to/file.csv`. Score a file with `python -m churn.predict --input customers.csv --output predictions.csv`.

## Outputs

- `models/churn_model.joblib`: fitted preprocessing, classifier and decision threshold
- `reports/model_comparison.csv`: cross-validated model metrics
- `reports/test_metrics.json`: held-out test results and selected threshold
- `reports/feature_importance.csv`: permutation importance for model explanations
- `reports/figures/`: churn distribution, confusion matrix, ROC and precision-recall curves

## Responsible use

This is a portfolio project, not a production decision system. Predictions should support—not replace—human judgment. Before deployment, validate data drift, subgroup performance, intervention costs, consent, privacy, and whether retention offers create unfair outcomes.

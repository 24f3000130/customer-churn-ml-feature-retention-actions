from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class ModelArtifact:
    pipeline: Pipeline
    threshold: float
    model_name: str
    feature_columns: list[str]

def candidates(random_state: int = 42):
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=350, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=random_state),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=.06, max_leaf_nodes=20, random_state=random_state),
    }

def build_pipeline(X: pd.DataFrame, classifier) -> Pipeline:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    pre = ColumnTransformer([
        ("num", SkPipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", SkPipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    return Pipeline([("preprocess", pre), ("balance", RandomOverSampler(random_state=42)), ("classifier", classifier)])

def choose_threshold(y_true, probabilities, beta: float = 2.0) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return float(thresholds[min(int(np.nanargmax(score)), len(thresholds) - 1)]) if len(thresholds) else .5


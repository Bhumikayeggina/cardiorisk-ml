"""
Step 12: fit the chosen final model on the whole training set and evaluate it
ONCE on the sealed test set, with bootstrap confidence intervals.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.pipeline import Pipeline

from src.imbalance import tuned_model
from src.preprocessing import build_preprocessor


def fit_final_pipeline(X, y, features, name="xgb", seed=42):
    """Preprocessing + tuned, unweighted model, fitted on the full training set."""
    pipe = Pipeline([
        ("prep", build_preprocessor(features, derive_chronic_count=False)),
        ("model", tuned_model(name, seed=seed)),
    ])
    return pipe.fit(X[list(features)], y)


def _metrics(y, p, t):
    flag = p >= t
    tp = (flag & (y == 1)).sum(); fp = (flag & (y == 0)).sum()
    fn = (~flag & (y == 1)).sum(); tn = (~flag & (y == 0)).sum()
    return {"roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p),
            "recall": tp / (tp + fn),
            "specificity": tn / (tn + fp),
            "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "flagged_pct": 100 * flag.mean()}


def evaluate_with_ci(y, p, threshold, n_boot=500, seed=0):
    """Point estimates on the full test set + 95% bootstrap intervals."""
    y = np.asarray(y); p = np.asarray(p)
    point = _metrics(y, p, threshold)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        boots.append(_metrics(y[idx], p[idx], threshold))
    boots = pd.DataFrame(boots)
    return pd.DataFrame({"estimate": pd.Series(point),
                         "ci_low": boots.quantile(0.025),
                         "ci_high": boots.quantile(0.975)})

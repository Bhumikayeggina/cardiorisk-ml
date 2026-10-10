"""
Step 10: out-of-fold (OOF) probabilities and decision-threshold analysis.

OOF = every training row is predicted by a model that never saw it, so the
curves and thresholds below are honest estimates made on TRAINING data only.
The test set stays sealed until the final evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.imbalance import tuned_model
from src.preprocessing import build_preprocessor

FOLD_JOBS = {"logreg": 5, "dtree": 5, "xgb": 1, "rf": 1}


def oof_probabilities(name, X, y, features, seed=42):
    """5-fold out-of-fold P(CHD/MI) for the tuned, unweighted model."""
    pipe = Pipeline([
        ("prep", build_preprocessor(features, derive_chronic_count=False)),
        ("model", tuned_model(name, seed=seed)),
    ])
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, X[list(features)], y, cv=cv,
                             method="predict_proba", n_jobs=FOLD_JOBS.get(name, 1))[:, 1]


def operating_point(y, p, t):
    """Metrics if everyone with probability >= t is flagged."""
    y = np.asarray(y)
    flag = np.asarray(p) >= t
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum()); tn = int((~flag & (y == 0)).sum())
    return {"threshold": float(t),
            "recall": tp / (tp + fn),
            "specificity": tn / (tn + fp),
            "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
            "flagged_pct": 100 * flag.mean(),
            "flagged_per_case_found": (tp + fp) / tp if tp else float("nan")}


def threshold_for_recall(y, p, target):
    """Highest threshold that still catches at least `target` of the cases."""
    _, recall, thr = precision_recall_curve(y, p)
    idx = np.where(recall[:-1] >= target)[0].max()
    return thr[idx]


def threshold_table(y, p, targets=(0.70, 0.80, 0.90)):
    rows = [{"goal": f"recall >= {t:.0%}", **operating_point(y, p, threshold_for_recall(y, p, t))}
            for t in targets]
    prec, rec, thr = precision_recall_curve(y, p)                # best F2 (recall counts 4x)
    f2 = 5 * prec[:-1] * rec[:-1] / np.maximum(4 * prec[:-1] + rec[:-1], 1e-12)
    rows.append({"goal": "max F2", **operating_point(y, p, thr[np.argmax(f2)])})
    return pd.DataFrame(rows)


def net_benefit(y, p, thresholds):
    """Decision-curve net benefit of flagging everyone with p >= t."""
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    out = []
    for t in thresholds:
        flag = p >= t
        tp = (flag & (y == 1)).sum(); fp = (flag & (y == 0)).sum()
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def net_benefit_treat_all(y, thresholds):
    prev = float(np.mean(y))
    return np.array([prev - (1 - prev) * t / (1 - t) for t in thresholds])

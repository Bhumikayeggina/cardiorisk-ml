"""
Step 14: subgroup audit of the final model on the TEST predictions.

Descriptive audit at the fixed decision threshold: discrimination (within-group
ROC-AUC), sensitivity (recall), false-positive rate, and calibration
(mean predicted vs observed risk), with Wilson 95% intervals.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

AGE = {1.0: "18-24", 2.0: "25-34", 3.0: "35-44", 4.0: "45-54", 5.0: "55-64", 6.0: "65+"}
RACE = {1.0: "White", 2.0: "Black", 3.0: "AI/AN", 4.0: "Asian", 5.0: "NHPI",
        6.0: "Other", 7.0: "Multiracial", 8.0: "Hispanic"}

ORDERS = {
    "sex": ["Male", "Female"],
    "age": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    "income": ["<$25k", "$25-50k", "$50-100k", "$100k+", "Unknown"],
    "race_ethnicity": ["White", "Black", "AI/AN", "Asian", "NHPI", "Other",
                       "Multiracial", "Hispanic", "Unknown"],
}


def make_groups(X):
    """Group labels built from the raw test inputs (race is NOT a model input)."""
    income = (pd.cut(X["INCOME3"], bins=[0, 4, 6, 8, 11],
                     labels=["<$25k", "$25-50k", "$50-100k", "$100k+"])
              .astype(object).fillna("Unknown"))
    return {
        "sex": X["SEXVAR"].map({1.0: "Male", 2.0: "Female"}).fillna("Unknown"),
        "age": X["_AGE_G"].map(AGE).fillna("Unknown"),
        "income": income,
        "race_ethnicity": X["_RACE"].map(RACE).fillna("Unknown"),
    }


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    denom = 1 + z ** 2 / n
    centre = (ph + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def group_table(y, p, groups, t, order=None, min_cases=100):
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "g": groups.values})
    rows = []
    for g, d in df.groupby("g"):
        yy, pp = d["y"].to_numpy(), d["p"].to_numpy()
        flag = pp >= t
        n, pos = len(yy), int(yy.sum())
        neg = n - pos
        tp = int((flag & (yy == 1)).sum())
        fp = int((flag & (yy == 0)).sum())
        r_lo, r_hi = wilson(tp, pos)
        f_lo, f_hi = wilson(fp, neg)
        o_lo, o_hi = wilson(pos, n)
        rows.append({
            "group": g, "n": n, "cases": pos,
            "observed_rate": pos / n, "obs_lo": o_lo, "obs_hi": o_hi,
            "mean_predicted": pp.mean(),
            "roc_auc": roc_auc_score(yy, pp) if 0 < pos < n else np.nan,
            "recall": tp / pos if pos else np.nan, "recall_lo": r_lo, "recall_hi": r_hi,
            "fpr": fp / neg if neg else np.nan, "fpr_lo": f_lo, "fpr_hi": f_hi,
            "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "flagged_pct": 100 * flag.mean(),
            "note": "few cases: interpret with caution" if pos < min_cases else "",
        })
    tab = pd.DataFrame(rows).set_index("group")
    if order is not None:
        tab = tab.reindex([o for o in order if o in tab.index])
    return tab

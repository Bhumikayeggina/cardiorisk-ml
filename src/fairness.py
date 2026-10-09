"""
Phase 10: Subgroup fairness — recall/specificity broken down by demographic group.
"""

import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix


def subgroup_metrics(y_true, y_pred, group_labels: pd.Series) -> pd.DataFrame:
    """group_labels: a Series aligned with y_true/y_pred, e.g. df_test['SEXVAR']."""
    rows = []
    for group, idx in pd.Series(range(len(y_true))).groupby(group_labels.values):
        y_t = [y_true[i] for i in idx]
        y_p = [y_pred[i] for i in idx]
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else float("nan")
        rows.append({
            "group": group,
            "n": len(idx),
            "recall": recall_score(y_t, y_p, zero_division=0),
            "specificity": specificity,
        })
    return pd.DataFrame(rows).sort_values("group")

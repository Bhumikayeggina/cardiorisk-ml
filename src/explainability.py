"""
Phase 12: SHAP explainability — global feature importance and local
per-prediction explanations.
"""

import shap
import matplotlib.pyplot as plt


def get_explainer(model, X_background):
    """Use TreeExplainer for tree-based models (fast); falls back to a
    general Explainer for anything else (slower, sample X_background down
    for speed on large data)."""
    try:
        return shap.TreeExplainer(model)
    except Exception:
        return shap.Explainer(model, X_background)


def plot_global_summary(explainer, X, save_path: str = None):
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return shap_values


def plot_local_explanation(explainer, X_row, save_path: str = None):
    shap_values = explainer(X_row)
    shap.plots.waterfall(shap_values[0], show=False)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

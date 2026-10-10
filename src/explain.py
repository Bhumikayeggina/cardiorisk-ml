"""
Step 15: SHAP explanations for the final pipeline, rolled up to the original
questions (one-hot and missing-indicator columns are summed back onto the
question they came from, which is valid because SHAP values are additive).

SHAP values are in log-odds units: positive pushes risk up, negative down.
"""

import numpy as np
import pandas as pd
import shap

from src.feature_selection import source_feature


def explain(pipe, X, features):
    """Returns (per-question SHAP DataFrame, base value, model margin per row)."""
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    A = prep.transform(X[list(features)])
    if hasattr(A, "toarray"):
        A = A.toarray()
    names = prep.named_steps["encode"].get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    sv = np.asarray(explainer.shap_values(A))
    sources = [source_feature(n) for n in names]
    agg = pd.DataFrame(sv, columns=sources, index=X.index).T.groupby(level=0).sum().T
    agg = agg[list(features)]

    base = float(np.ravel(explainer.expected_value)[0])
    margin = model.predict(A, output_margin=True)
    return agg, base, margin

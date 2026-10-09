"""
Phase 7/9: Model zoo, hyperparameter grids, and ensemble builders.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier

BASE_MODELS = {
    "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "knn": KNeighborsClassifier(),
    "svm": SVC(probability=True, class_weight="balanced"),
    "dtree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "rf": RandomForestClassifier(class_weight="balanced", random_state=42),
    "xgb": XGBClassifier(eval_metric="logloss", random_state=42),
}

# Starting search spaces — narrow/widen based on your Phase 7 baseline results.
PARAM_GRIDS = {
    "svm": {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"], "model__gamma": ["scale", "auto"]},
    "rf": {"model__n_estimators": [200, 400, 600], "model__max_depth": [None, 10, 20],
           "model__min_samples_split": [2, 5, 10]},
    "xgb": {"model__n_estimators": [200, 400, 600], "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 7]},
}


def build_voting_ensemble(fitted_estimators: list[tuple]) -> VotingClassifier:
    return VotingClassifier(estimators=fitted_estimators, voting="soft")


def build_stacking_ensemble(fitted_estimators: list[tuple]) -> StackingClassifier:
    return StackingClassifier(estimators=fitted_estimators, final_estimator=LogisticRegression(max_iter=1000))

"""Voting and tree-based ensemble models for spam classification."""
from typing import Any, Dict, Optional, Union
import numpy as np
from scipy.sparse import csr_matrix


class SpamVotingEnsemble:
    """Soft- or hard-voting ensemble for fitted or unfitted binary classifiers."""
    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None, voting: str = "soft"):
        if not models:
            raise ValueError("At least one base model is required")
        if voting not in {"soft", "hard"}:
            raise ValueError("voting must be 'soft' or 'hard'")
        self.models = dict(models)
        self.weights = dict(weights) if weights is not None else {name: 1.0 for name in models}
        if any(float(self.weights.get(name, 1.0)) < 0 for name in self.models):
            raise ValueError("Voting weights must be non-negative")
        if sum(float(self.weights.get(n, 1.0)) for n in self.models) <= 0:
            raise ValueError("At least one voting weight must be positive")
        self.voting = voting
        self.classes_ = np.array([0, 1])

    def fit(self, X: Union[np.ndarray, csr_matrix], y: np.ndarray) -> "SpamVotingEnsemble":
        for model in self.models.values():
            model.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        total = np.zeros((X.shape[0], 2), dtype=float)
        denom = 0.0
        for name, model in self.models.items():
            weight = float(self.weights.get(name, 1.0))
            if weight == 0:
                continue
            if hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(X), dtype=float)
            elif hasattr(model, "decision_function"):
                scores = np.asarray(model.decision_function(X)).ravel()
                p1 = 1.0 / (1.0 + np.exp(-np.clip(scores, -35, 35)))
                proba = np.column_stack((1 - p1, p1))
            else:
                labels = np.asarray(model.predict(X)).astype(int)
                proba = np.column_stack((labels == 0, labels == 1)).astype(float)
            if proba.shape != total.shape:
                raise ValueError(f"Model {name!r} returned invalid probability shape {proba.shape}")
            total += weight * proba
            denom += weight
        if denom <= 0:
            raise ValueError("Total voting weight must be positive")
        return total / denom

    def predict(self, X: Union[np.ndarray, csr_matrix], threshold: float = 0.5) -> np.ndarray:
        if self.voting == "soft":
            if not 0 <= threshold <= 1:
                raise ValueError("threshold must be in [0, 1]")
            return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
        votes = np.zeros((X.shape[0], 2), dtype=float)
        for name, model in self.models.items():
            weight = float(self.weights.get(name, 1.0))
            if weight == 0:
                continue
            try:
                pred = np.asarray(model.predict(X, threshold=threshold)).ravel().astype(int)
            except TypeError:
                pred = np.asarray(model.predict(X)).ravel().astype(int)
            votes[:, 0] += weight * (pred == 0)
            votes[:, 1] += weight * (pred == 1)
        return (votes[:, 1] > votes[:, 0]).astype(int)


def build_tree_based_ensembles(random_state: int = 42) -> Dict[str, Any]:
    """Create Random Forest and Gradient Boosting classifier candidates."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_features="sqrt", class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=2,
            random_state=random_state, subsample=0.8),
    }

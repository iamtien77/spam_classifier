"""Core from-scratch classifiers for the spam email project."""
from typing import Any, Dict, Optional, Union
import numpy as np
from scipy.sparse import csr_matrix, issparse

ArrayLike = Union[np.ndarray, csr_matrix]


def _as_2d(X: ArrayLike):
    # Task 1 exposes its own CSRMatrixScratch type; convert it without densifying.
    if not issparse(X) and hasattr(X, "to_scipy_csr"):
        X = X.to_scipy_csr()
    if issparse(X):
        X = X.tocsr().astype(np.float64)
    else:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must contain samples and features")
    return X


def _class_sample_weights(y: np.ndarray, class_weight: Optional[str]) -> np.ndarray:
    n = len(y)
    weights = np.ones(n, dtype=np.float64)
    if class_weight == "balanced":
        classes, counts = np.unique(y, return_counts=True)
        per_class = {c: n / (len(classes) * count) for c, count in zip(classes, counts)}
        weights = np.asarray([per_class[v] for v in y], dtype=np.float64)
        weights /= max(weights.mean(), 1e-12)
    elif isinstance(class_weight, dict):
        weights = np.asarray([float(class_weight.get(v, 1.0)) for v in y])
        weights /= max(weights.mean(), 1e-12)
    elif class_weight not in (None, "none"):
        raise ValueError("class_weight must be None, 'balanced', or a mapping")
    return weights


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionFromScratch:
    """Binary logistic regression trained using batch gradient descent."""
    def __init__(self, learning_rate: float = 0.1, max_iter: int = 300, C: float = 1.0,
                 class_weight: Optional[str] = "balanced", random_state: int = 42):
        if learning_rate <= 0 or max_iter < 1 or C <= 0:
            raise ValueError("learning_rate, max_iter and C must be positive")
        self.learning_rate, self.max_iter, self.C = float(learning_rate), int(max_iter), float(C)
        self.class_weight, self.random_state = class_weight, random_state
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.loss_history_: list = []
        self.classes_ = np.array([0, 1])

    def fit(self, X: ArrayLike, y: np.ndarray) -> "LogisticRegressionFromScratch":
        X = _as_2d(X)
        y = np.asarray(y).ravel().astype(np.float64)
        if len(y) != X.shape[0] or not set(np.unique(y)).issubset({0.0, 1.0}):
            raise ValueError("y must be a binary vector (0/1) matching X")
        n, p = X.shape
        sw = _class_sample_weights(y, self.class_weight)
        w = np.zeros(p, dtype=np.float64)
        b = 0.0
        self.loss_history_ = []
        for _ in range(self.max_iter):
            z = np.asarray(X @ w).reshape(-1) + b
            prob = _sigmoid(z)
            err = sw * (prob - y)
            grad_w = np.asarray(X.T @ err).reshape(-1) / n + w / (self.C * n)
            grad_b = float(err.mean())
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b
            if _ % max(1, self.max_iter // 30) == 0 or _ == self.max_iter - 1:
                z2 = np.clip(np.asarray(X @ w).reshape(-1) + b, -35, 35)
                losses = np.logaddexp(0.0, z2) - y * z2
                loss = float(np.mean(sw * losses) + np.dot(w, w) / (2.0 * self.C * n))
                self.loss_history_.append(loss)
        self.weights_, self.bias_ = w, float(b)
        self.n_features_in_ = p
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        X = _as_2d(X)
        if X.shape[1] != len(self.weights_):
            raise ValueError("X has a different number of features than training data")
        p1 = _sigmoid(np.asarray(X @ self.weights_).reshape(-1) + self.bias_)
        return np.column_stack((1.0 - p1, p1))

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> np.ndarray:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class LinearSVMFromScratch:
    """Linear SVM trained with stochastic subgradient descent on hinge loss."""
    def __init__(self, C: float = 1.0, learning_rate: float = 0.05, max_iter: int = 150,
                 class_weight: Optional[str] = "balanced", random_state: int = 42):
        if C <= 0 or learning_rate <= 0 or max_iter < 1:
            raise ValueError("C, learning_rate and max_iter must be positive")
        self.C, self.learning_rate, self.max_iter = float(C), float(learning_rate), int(max_iter)
        self.class_weight, self.random_state = class_weight, random_state
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.loss_history_: list = []
        self.classes_ = np.array([0, 1])

    def fit(self, X: ArrayLike, y: np.ndarray) -> "LinearSVMFromScratch":
        X = _as_2d(X)
        y = np.asarray(y).ravel().astype(int)
        if len(y) != X.shape[0] or not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("y must be a binary vector (0/1) matching X")
        n, p = X.shape
        ys = np.where(y == 1, 1.0, -1.0)
        sw = _class_sample_weights(y, self.class_weight)
        w = np.zeros(p, dtype=np.float64)
        b = 0.0
        self.loss_history_ = []
        # Vectorized squared-hinge subgradient descent: avoids thousands of
        # repeated CSR row slices while retaining sparse matrix operations.
        for epoch in range(self.max_iter):
            scores = np.asarray(X @ w).reshape(-1) + b
            margin = np.maximum(0.0, 1.0 - ys * scores)
            coeff = -2.0 * ys * sw * margin
            grad_w = np.asarray(X.T @ coeff).reshape(-1) / n + w / (self.C * n)
            grad_b = float(np.mean(coeff))
            lr = self.learning_rate / np.sqrt(1.0 + epoch * 0.05)
            w -= lr * grad_w
            b -= lr * grad_b
            if epoch % max(1, self.max_iter // 20) == 0 or epoch == self.max_iter - 1:
                scores2 = np.asarray(X @ w).reshape(-1) + b
                hinge = np.maximum(0.0, 1.0 - ys * scores2)
                loss = float(np.mean(sw * hinge ** 2) + np.dot(w, w) / (2 * self.C * n))
                self.loss_history_.append(loss)
        self.weights_, self.bias_ = w, float(b)
        self.n_features_in_ = p
        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        X = _as_2d(X)
        if X.shape[1] != len(self.weights_):
            raise ValueError("X has a different number of features than training data")
        return np.asarray(X @ self.weights_).reshape(-1) + self.bias_

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        p1 = _sigmoid(self.decision_function(X))
        return np.column_stack((1.0 - p1, p1))

    def predict(self, X: ArrayLike, threshold: float = 0.0) -> np.ndarray:
        return (self.decision_function(X) >= threshold).astype(int)


class NaiveBayesClassifier:
    """Multinomial or Complement Naive Bayes for non-negative sparse features."""
    def __init__(self, alpha: float = 1.0, kind: str = "complement"):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if kind not in {"multinomial", "complement"}:
            raise ValueError("kind must be 'multinomial' or 'complement'")
        self.alpha, self.kind = float(alpha), kind
        self.classes_: Optional[np.ndarray] = None
        self.class_log_prior_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: np.ndarray) -> "NaiveBayesClassifier":
        X = _as_2d(X)
        y = np.asarray(y).ravel()
        if len(y) != X.shape[0]:
            raise ValueError("X and y have incompatible lengths")
        if issparse(X):
            if X.data.size and np.min(X.data) < 0:
                raise ValueError("Naive Bayes requires non-negative features")
        elif np.any(X < 0):
            raise ValueError("Naive Bayes requires non-negative features")
        self.classes_, counts = np.unique(y, return_counts=True)
        if len(self.classes_) < 2:
            raise ValueError("At least two classes are required")
        self.class_log_prior_ = np.log(counts / counts.sum())
        class_feature_count = []
        for c in self.classes_:
            rows = X[y == c]
            class_feature_count.append(np.asarray(rows.sum(axis=0)).reshape(-1))
        cf = np.vstack(class_feature_count)
        if self.kind == "multinomial":
            smoothed = cf + self.alpha
            self.feature_log_prob_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
            self._complement_ = False
        else:
            complement = cf.sum(axis=0, keepdims=True) - cf
            smoothed = complement + self.alpha
            probs = smoothed / smoothed.sum(axis=1, keepdims=True)
            # Negative normalized log complement weights; larger score means stronger class evidence.
            weights = -np.log(np.maximum(probs, 1e-300))
            weights /= np.maximum(np.sum(np.abs(weights), axis=1, keepdims=True), 1e-12)
            self.feature_log_prob_ = weights
            self._complement_ = True
        self.n_features_in_ = X.shape[1]
        return self

    def _class_scores(self, X: ArrayLike) -> np.ndarray:
        if self.feature_log_prob_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        X = _as_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than training data")
        scores = np.asarray(X @ self.feature_log_prob_.T)
        # Multinomial stores log P(feature|class); Complement stores positive
        # negative-log complement weights, whose larger values support that class.
        scores = scores + self.class_log_prior_
        return scores

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        scores = self._class_scores(X)
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(np.clip(scores, -700, 700))
        return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-300)

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        spam_idx = int(np.where(self.classes_ == 1)[0][0]) if 1 in self.classes_ else int(np.argmax(self.classes_))
        ham_idx = 1 - spam_idx if len(self.classes_) == 2 else int(np.argmin(self.classes_))
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        return np.where(proba[:, spam_idx] >= threshold, self.classes_[spam_idx], self.classes_[ham_idx]).astype(int)


def get_baseline_models(config: Optional[Any] = None) -> Dict[str, Any]:
    """Return the three required baseline models using project default settings."""
    defaults = getattr(config, "default_tuning_params", {}) if config is not None else {}
    defaults = defaults or {}
    return {
        "logistic_regression": LogisticRegressionFromScratch(
            C=float(defaults.get("lr_C", defaults.get("C", 1.0))),
            learning_rate=float(defaults.get("lr_learning_rate", 0.1)),
            max_iter=int(defaults.get("lr_max_iter", 300)),
            class_weight="balanced", random_state=getattr(config, "random_state", 42)),
        "svm": LinearSVMFromScratch(
            C=float(defaults.get("svm_C", defaults.get("C", 1.0))),
            learning_rate=float(defaults.get("svm_learning_rate", 0.05)),
            max_iter=int(defaults.get("svm_max_iter", 150)),
            class_weight="balanced", random_state=getattr(config, "random_state", 42)),
        "naive_bayes": NaiveBayesClassifier(
            alpha=float(defaults.get("alpha", 1.0)), kind=str(defaults.get("nb_kind", "complement")))
    }

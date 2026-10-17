"""
classifiers.py - Các thuật toán phân loại cốt lõi theo đề bài
Chuẩn hóa theo yêu cầu đề bài (ML project: Classifying Spam Emails - Workflow Bước 2)
=====================================================================================
Đề bài quy định rõ 3 phương pháp phân loại chính (Classification Approach):
1. Logistic Regression:
   - Mô hình nhị phân phổ biến hàng đầu cho bài toán phân loại email spam.
   - Mô hình hóa xác suất một email là spam bằng hàm Sigmoid / Logistic function:
     P(y=1|x) = 1 / (1 + exp(-(w^T * x + b))).
   - Huấn luyện bằng Gradient Descent với backtracking trên Binary Cross-Entropy.
2. Support Vector Machines (SVM):
   - Phương pháp phân loại mạnh mẽ, đặc biệt hiệu quả khi làm việc với không gian đặc trưng nhiều chiều (High-dimensional text).
   - Tìm siêu phẳng lề cực đại (Maximum Margin Hyperplane) phân tách giữa email spam và ham.
   - Huấn luyện bằng Gradient Descent với backtracking trên Hinge Loss và điều chuẩn L2.
3. Naive Bayes:
   - Thuật toán xác suất đơn giản nhưng cực kỳ hiệu quả dựa trên định lý Bayes (Bayes' Theorem).
   - Giả định tính độc lập có điều kiện giữa các đặc trưng khi biết lớp.
   - Cài đặt hỗ trợ cả Multinomial Naive Bayes và Complement Naive Bayes (chuyên trị mất cân bằng lớp).
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy.sparse import csr_matrix


def _safe_dot(X: Any, w: np.ndarray) -> np.ndarray:
    """Tính tích vô hướng ma trận - vector an toàn cho cả CSRMatrixScratch, SciPy CSR và mảng NumPy."""
    w = np.asarray(w, dtype=float).ravel()
    if hasattr(X, "dot"):
        res = X.dot(w)
        if hasattr(res, "toarray"):
            res = res.toarray().ravel()
        return np.asarray(res, dtype=float).ravel()
    return np.asarray(X @ w, dtype=float).ravel()


def _safe_transpose_dot(X: Any, v: np.ndarray) -> np.ndarray:
    """Tính X.T @ v an toàn không cần giải nén ma trận đặc."""
    v = np.asarray(v, dtype=float).ravel()
    if hasattr(X, "transpose_dot"):
        return np.asarray(X.transpose_dot(v), dtype=float).ravel()
    if hasattr(X, "T") and hasattr(X.T, "dot"):
        res = X.T.dot(v)
        if hasattr(res, "toarray"):
            res = res.toarray().ravel()
        return np.asarray(res, dtype=float).ravel()
    return np.asarray(np.asarray(X).T @ v, dtype=float).ravel()


def _get_shape(X: Any) -> Tuple[int, int]:
    """Lấy số hàng và số cột của ma trận đầu vào."""
    if hasattr(X, "shape"):
        return int(X.shape[0]), int(X.shape[1])
    if isinstance(X, (list, tuple)):
        return len(X), len(X[0]) if len(X) > 0 else 0
    raise ValueError(f"Không thể xác định kích thước cho đối tượng kiểu {type(X)}")


def _gradient_descent_backtracking(objective, initial, learning_rate, max_iter, tol=1e-7):
    """Spectral gradient descent với Armijo; không phụ thuộc sklearn.

    Bước học ban đầu được điều chỉnh theo độ cong, tránh đứng gần nghiệm
    khởi tạo trên TF-IDF hoặc phân kỳ khi tăng learning_rate cố định.
    objective trả loss và gradient tại cùng một bộ tham số.
    """
    if learning_rate <= 0 or max_iter < 1:
        raise ValueError("learning_rate và max_iter phải dương.")
    parameters = initial.copy()
    value, gradient = objective(parameters)
    history = [float(value)]
    step = float(learning_rate)
    for _ in range(max_iter):
        norm_sq = float(gradient @ gradient)
        if norm_sq <= tol ** 2:
            break
        for _ in range(60):
            candidate = parameters - step * gradient
            next_value, next_gradient = objective(candidate)
            if np.isfinite(next_value) and next_value <= value - 1e-4 * step * norm_sq:
                break
            step *= .5
        else:
            break
        displacement = candidate - parameters
        gradient_change = next_gradient - gradient
        curvature = float(displacement @ gradient_change)
        next_step = (float(displacement @ displacement) / curvature
                     if curvature > 0 else step * 2)
        parameters, value, gradient = candidate, next_value, next_gradient
        history.append(float(value))
        step = float(np.clip(next_step, 1e-8, 1e8))
    return parameters, history


class LogisticRegressionFromScratch:
    """
    1. LOGISTIC REGRESSION TỰ CÀI ĐẶT TỪ ĐẦU (FROM SCRATCH):
    --------------------------------------------------------
    Phân loại nhị phân dựa trên hàm Logistic/Sigmoid và tối ưu hóa bằng Gradient Descent.
    Hỗ trợ cả ma trận thưa CSRMatrixScratch, scipy.sparse.csr_matrix và mảng numpy.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 300,
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.loss_history_: list = []

    def fit(self, X: Union[np.ndarray, csr_matrix], y: np.ndarray) -> "LogisticRegressionFromScratch":
        n_samples, n_features = _get_shape(X)
        y_arr = np.asarray(y, dtype=float).ravel()

        if self.class_weight == "balanced":
            n_0 = max(1, int(np.sum(y_arr == 0.0)))
            n_1 = max(1, int(np.sum(y_arr == 1.0)))
            w_0 = n_samples / (2.0 * n_0)
            w_1 = n_samples / (2.0 * n_1)
            sample_weights = np.where(y_arr == 1.0, w_1, w_0)
        else:
            sample_weights = np.ones(n_samples, dtype=float)

        reg_lambda = 1.0 / max(1e-5, self.C * n_samples)

        def objective(parameters):
            w, b = parameters[:-1], parameters[-1]
            z = _safe_dot(X, w) + b
            # logaddexp calcule BCE sans clipping du loss ni overflow.
            loss = np.mean(sample_weights * (np.logaddexp(0, z) - y_arr * z))
            loss += .5 * reg_lambda * float(w @ w)
            probabilities = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
            error = sample_weights * (probabilities - y_arr)
            grad_w = reg_lambda * w + _safe_transpose_dot(X, error) / n_samples
            return float(loss), np.r_[grad_w, np.mean(error)]

        parameters, self.loss_history_ = _gradient_descent_backtracking(
            objective, np.zeros(n_features + 1), self.learning_rate, self.max_iter,
        )
        self.weights_, self.bias_ = parameters[:-1], float(parameters[-1])
        return self

    def predict_proba(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Mô hình chưa được huấn luyện (hãy gọi .fit() trước).")
        z = _safe_dot(X, self.weights_) + self.bias_
        z_clipped = np.clip(z, -30.0, 30.0)
        prob_spam = 1.0 / (1.0 + np.exp(-z_clipped))
        prob_ham = 1.0 - prob_spam
        return np.column_stack([prob_ham, prob_spam])

    def predict(self, X: Union[np.ndarray, csr_matrix], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class LinearSVMFromScratch:
    """
    2. SUPPORT VECTOR MACHINES (SVM) TỰ CÀI ĐẶT TỪ ĐẦU (FROM SCRATCH):
    -----------------------------------------------------------------
    Tìm siêu phẳng phân tách lề cực đại (Maximum Margin) bằng giải thuật Gradient Descent
    với hàm mất mát Squared Hinge Loss hoặc Hinge Loss, kết hợp điều chuẩn L2.
    """

    def __init__(
        self,
        C: float = 1.0,
        learning_rate: float = 0.05,
        max_iter: int = 150,
        class_weight: Optional[str] = "balanced",
        loss: str = "squared_hinge",
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        self.C = float(C)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.loss = loss
        self.tol = float(tol)
        self.random_state = random_state
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.loss_history_: list = []
        self.n_features_in_: Optional[int] = None
        self.n_iter_: int = 0

    @property
    def w_(self) -> Optional[np.ndarray]:
        return self.weights_

    @w_.setter
    def w_(self, val: Optional[np.ndarray]) -> None:
        self.weights_ = val

    @property
    def b_(self) -> float:
        return self.bias_

    @b_.setter
    def b_(self, val: float) -> None:
        self.bias_ = val

    def fit(self, X: Union[np.ndarray, csr_matrix], y: np.ndarray) -> "LinearSVMFromScratch":
        n_samples, n_features = _get_shape(X)
        self.n_features_in_ = n_features
        y_arr = np.asarray(y, dtype=float).ravel()
        y_signed = np.where(y_arr == 1.0, 1.0, -1.0)

        if self.class_weight == "balanced":
            n_0 = max(1, int(np.sum(y_arr == 0.0)))
            n_1 = max(1, int(np.sum(y_arr == 1.0)))
            w_0 = n_samples / (2.0 * n_0)
            w_1 = n_samples / (2.0 * n_1)
            sample_weights = np.where(y_arr == 1.0, w_1, w_0)
        else:
            sample_weights = np.ones(n_samples, dtype=float)

        if self.loss not in ("hinge", "squared_hinge"):
            raise ValueError("loss phải là hinge hoặc squared_hinge.")
        reg_lambda = 1.0 / max(1e-5, self.C * n_samples)

        def objective(parameters):
            w, b = parameters[:-1], parameters[-1]
            margins = np.maximum(0., 1 - y_signed * (_safe_dot(X, w) + b))
            if self.loss == "squared_hinge":
                losses = margins ** 2
                derivative = -2 * sample_weights * y_signed * margins
            else:
                losses = margins
                derivative = -sample_weights * y_signed * (margins > 0)
            value = .5 * reg_lambda * float(w @ w) + np.mean(sample_weights * losses)
            grad_w = reg_lambda * w + _safe_transpose_dot(X, derivative) / n_samples
            return float(value), np.r_[grad_w, np.mean(derivative)]

        parameters, self.loss_history_ = _gradient_descent_backtracking(
            objective, np.zeros(n_features + 1), self.learning_rate, self.max_iter,
            tol=min(self.tol, 1e-7),
        )
        self.weights_, self.bias_ = parameters[:-1], float(parameters[-1])
        self.n_iter_ = len(self.loss_history_) - 1
        return self

    def decision_function(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Mô hình chưa được huấn luyện (hãy gọi .fit() trước).")
        return _safe_dot(X, self.weights_) + self.bias_

    def predict_proba(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        z = self.decision_function(X)
        z_clipped = np.clip(z, -30.0, 30.0)
        prob_spam = 1.0 / (1.0 + np.exp(-z_clipped))
        prob_ham = 1.0 - prob_spam
        return np.column_stack([prob_ham, prob_spam])

    def predict(self, X: Union[np.ndarray, csr_matrix], threshold: float = 0.0) -> np.ndarray:
        scores = self.decision_function(X)
        return (scores >= threshold).astype(int)



class NaiveBayesClassifier:
    """
    3. NAIVE BAYES CLASSIFIER (MULTINOMIAL & COMPLEMENT) TỰ CÀI ĐẶT (FROM SCRATCH):
    --------------------------------------------------------------------------------
    Phân loại xác suất dựa trên định lý Bayes, hỗ trợ làm mịn Laplace và giải thuật Complement NB
    chuyên trị tập dữ liệu văn bản mất cân bằng nhãn cao.
    """

    def __init__(self, alpha: float = 1.0, kind: str = "complement"):
        self.alpha = float(alpha)
        self.kind = kind.lower()
        self.classes_: Optional[np.ndarray] = None
        self.class_log_prior_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, csr_matrix], y: np.ndarray) -> "NaiveBayesClassifier":
        n_samples, n_features = _get_shape(X)
        y_arr = np.asarray(y, dtype=int).ravel()
        self.classes_ = np.unique(y_arr)
        if len(self.classes_) != 2:
            self.classes_ = np.array([0, 1], dtype=int)

        self.class_log_prior_ = np.array(
            [np.log(max(1, int(np.sum(y_arr == c))) / n_samples) for c in self.classes_],
            dtype=float,
        )

        feature_log_prob = []
        for c in self.classes_:
            if self.kind == "multinomial":
                mask = (y_arr == c).astype(float)
                counts = _safe_transpose_dot(X, mask)
                smoothed = counts + self.alpha
                total = float(np.sum(smoothed))
                prob = smoothed / total
                feature_log_prob.append(np.log(prob))
            else:
                # Complement Naive Bayes: học trên tập phần bù (y != c)
                mask = (y_arr != c).astype(float)
                counts = _safe_transpose_dot(X, mask)
                smoothed = counts + self.alpha
                total = float(np.sum(smoothed))
                feature_log_prob.append(np.log(smoothed / total))

        self.feature_log_prob_ = np.array(feature_log_prob, dtype=float)
        return self

    def decision_scores(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        if self.feature_log_prob_ is None or self.class_log_prior_ is None:
            raise RuntimeError("Mô hình chưa được huấn luyện (hãy gọi .fit() trước).")

        scores_list = []
        for idx, _ in enumerate(self.classes_):
            if self.kind == "multinomial":
                score = _safe_dot(X, self.feature_log_prob_[idx]) + self.class_log_prior_[idx]
            else:
                # Complement Naive Bayes chuẩn: - sum(x_j * log(theta_complement)) + log(prior)
                score = -_safe_dot(X, self.feature_log_prob_[idx]) + self.class_log_prior_[idx]
            scores_list.append(score)
        return np.column_stack(scores_list)

    def predict_proba(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        scores = self.decision_scores(X)
        max_s = np.max(scores, axis=1, keepdims=True)
        exp_s = np.exp(np.clip(scores - max_s, -30.0, 30.0))
        probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        return probs

    def predict(self, X: Union[np.ndarray, csr_matrix], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


def get_baseline_models(config: Optional[Any] = None) -> Dict[str, Any]:
    """Khởi tạo 3 mô hình cốt lõi theo đúng đề bài quy định."""
    return {
        "logistic_regression": LogisticRegressionFromScratch(
            C=1.0, learning_rate=0.1, max_iter=300, class_weight="balanced"
        ),
        "svm": LinearSVMFromScratch(
            C=1.0, learning_rate=0.05, max_iter=150, class_weight="balanced"
        ),
        "naive_bayes": NaiveBayesClassifier(alpha=1.0, kind="complement"),
    }


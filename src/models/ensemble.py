"""
ensemble.py - Phương pháp kết hợp mô hình (Ensemble Methods)
Chuẩn hóa theo yêu cầu đề bài (ML project: Classifying Spam Emails - Additional Considerations)
=============================================================================================
Đề bài yêu cầu:
"Ensemble Methods: Combine multiple models (e.g., using random forests or boosting)
 to improve generalization and reduce overfitting."

Module này cài đặt:
1. SpamVotingEnsemble:
   - Kết hợp biểu quyết mềm (Soft Voting) hoặc cứng (Hard Voting) giữa 3 mô hình cơ bản:
     Logistic Regression + Support Vector Machines (SVM) + Naive Bayes.
   - Cho phép gán trọng số tối ưu (weights) cho từng mô hình thành viên.
2. RandomForestSpamClassifier & Boosting Wrappers:
   - Ứng dụng giải thuật Bagging (Random Forest) và Boosting (Gradient Boosting / AdaBoost)
     nhằm tăng cường độ khái quát hóa và giảm thiểu hiện tượng quá khớp (overfitting).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.sparse import csr_matrix



class SpamVotingEnsemble:
    """
    BỘ KẾT HỢP BIỂU QUYẾT (VOTING ENSEMBLE) CHO BÀI TOÁN PHÂN LOẠI EMAIL SPAM:
    --------------------------------------------------------------------------
    Tài liệu kỹ thuật: [docs/Algorithms/voting_ensemble.md](../../docs/Algorithms/voting_ensemble.md)

    Kết hợp dự đoán xác suất giữa Logistic Regression, SVM và Naive Bayes.
    """

    def __init__(
        self,
        models: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        voting: str = "soft",
    ):
        if not isinstance(voting, str) or voting.lower() not in ("soft", "hard"):
            raise ValueError("voting phải là 'soft' hoặc 'hard'.")
        self.models = models
        self.weights = weights or {name: 1.0 for name in models}
        self.voting = voting.lower()
        self.classes_ = np.array([0, 1])

    def fit(self, X: Union[np.ndarray, csr_matrix], y: np.ndarray) -> "SpamVotingEnsemble":
        for _, model in self.models.items():
            model.fit(X, y)
        return self

    def _validated_weights(self):
        # Kiểm tra lại cho cấu hình bị thay đổi hoặc bundle được nạp từ file.
        if self.voting not in ("soft", "hard"):
            raise ValueError("voting phải là 'soft' hoặc 'hard'.")
        if not self.models:
            raise ValueError("Chưa cấu hình mô hình thành viên nào trong Ensemble.")
        weights = {name: float(self.weights.get(name, 1.0)) for name in self.models}
        if any(not np.isfinite(weight) or weight < 0 for weight in weights.values()):
            raise ValueError("Trọng số Ensemble phải hữu hạn và không âm.")
        total_weight = sum(weights.values())
        if not np.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("Tổng trọng số Ensemble phải hữu hạn và dương.")
        return weights, total_weight

    def predict_proba(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        weights, total_weight = self._validated_weights()

        accumulated_prob = None
        for name, model in self.models.items():
            w = weights[name]
            if self.voting == "hard":
                # Điểm hard voting là tỷ lệ phiếu có trọng số, không phải soft probability.
                preds = np.asarray(model.predict(X), dtype=float).ravel()
                proba = np.column_stack([1.0 - preds, preds])
            elif hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
            elif hasattr(model, "decision_function"):
                df = model.decision_function(X)
                sp = 1.0 / (1.0 + np.exp(-np.clip(df, -30.0, 30.0)))
                proba = np.column_stack([1.0 - sp, sp])
            else:
                preds = model.predict(X)
                proba = np.column_stack([1.0 - preds, preds])

            if accumulated_prob is None:
                accumulated_prob = w * proba
            else:
                accumulated_prob += w * proba

        return accumulated_prob / total_weight

    def predict(self, X: Union[np.ndarray, csr_matrix], threshold: float = 0.5) -> np.ndarray:
        if self.voting == "soft":
            probs = self.predict_proba(X)[:, 1]
            return (probs >= threshold).astype(int)
        else:
            weights, total_weight = self._validated_weights()
            weighted_votes = None
            for name, model in self.models.items():
                w = weights[name]
                preds = np.asarray(model.predict(X), dtype=float).ravel()
                if weighted_votes is None:
                    weighted_votes = w * preds
                else:
                    weighted_votes += w * preds
            return (weighted_votes >= (total_weight / 2.0)).astype(int)


def build_tree_based_ensembles(
    random_state: int = 42,
) -> Dict[str, Any]:
    """Khởi tạo các mô hình Ensemble dựa trên cây nếu có thư viện phụ trợ."""
    ensembles = {}
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        ensembles["random_forest"] = RandomForestClassifier(n_estimators=100, random_state=random_state)
        ensembles["gradient_boosting"] = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    except ImportError:
        pass
    return ensembles


"""
tuning.py - Tinh chỉnh siêu tham số và tối ưu hóa ngưỡng phân loại
Chuẩn hóa theo yêu cầu đề bài (ML project: Classifying Spam Emails - Hyperparameter Tuning)
===========================================================================================
Đề bài yêu cầu:
"Hyperparameter Tuning: Optimize the model's parameters (e.g., regularization strength,
 learning rate) to achieve better results."

Module này đảm nhiệm:
1. Tối ưu hóa siêu tham số (Hyperparameter Tuning):
   - Logistic Regression: Tinh chỉnh hệ số điều chuẩn C, loại penalty (l1/l2), learning_rate.
   - Support Vector Machines (SVM): Tinh chỉnh hệ số phạt vi phạm lề C, learning_rate, số epoch.
   - Naive Bayes: Tinh chỉnh hệ số làm mịn Laplace/Lidstone alpha.
2. Tối ưu hóa ngưỡng quyết định (Threshold Tuning):
   - Quét lưới 200 giá trị ngưỡng xác suất [0.005, 1.0] trên tập Validation.
   - Tự động dò tìm ngưỡng tối ưu thỏa mãn ràng buộc nghiệp vụ (Recall >= 0.85) và đạt F1-score / F-beta cao nhất.
3. Quy trình lựa chọn mô hình vô địch (Model Selection Workflow).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class ThresholdOptimizerScratch:
    """
    BỘ TỐI ƯU HÓA NGƯỠNG PHÂN LOẠI TỰ XÂY DỰNG (FROM SCRATCH):
    ----------------------------------------------------------
    Dò quét và chọn ngưỡng quyết định tối ưu trên tập Validation nhằm cân bằng giữa
    khả năng bắt trúng thư rác (Recall) và hạn chế tối đa chặn nhầm thư hợp lệ (Precision).
    """

    def __init__(
        self,
        recall_target: float = 0.85,
        beta: float = 2.0,
        n_points: int = 200,
        default_grid: Optional[np.ndarray] = None,
    ):
        """Khởi tạo mục tiêu Recall, hệ số Beta trong F-beta score, và số mốc ngưỡng quét."""
        self.recall_target = float(recall_target)
        self.beta = float(beta)
        if default_grid is not None:
            self.grid = np.asarray(default_grid, dtype=float)
        else:
            self.grid = np.linspace(0.005, 1.0, n_points)

    @staticmethod
    def fbeta_from_precision_recall(precision: float, recall: float, beta: float = 2.0) -> float:
        beta_sq = float(beta) ** 2
        denom = beta_sq * precision + recall
        if denom == 0.0:
            return 0.0
        return float((1.0 + beta_sq) * precision * recall / denom)

    def compute_metrics_at_threshold(
        self,
        y_true: np.ndarray,
        spam_probs: np.ndarray,
        threshold: float,
    ) -> Dict[str, Any]:
        y_arr = np.asarray(y_true, dtype=int).ravel()
        probs = np.asarray(spam_probs, dtype=float).ravel()
        y_pred = (probs >= float(threshold)).astype(int)

        tp = int(np.sum((y_arr == 1) & (y_pred == 1)))
        fp = int(np.sum((y_arr == 0) & (y_pred == 1)))
        tn = int(np.sum((y_arr == 0) & (y_pred == 0)))
        fn = int(np.sum((y_arr == 1) & (y_pred == 0)))

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f_beta = self.fbeta_from_precision_recall(precision, recall, beta=self.beta)

        return {
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "tpr": float(recall),
            "fpr": float(fpr),
            "specificity": float(specificity),
            "f1": float(f1),
            "f2": float(f_beta),
            "f_beta": float(f_beta),
            "target_met": bool(recall >= self.recall_target),
        }

    def metrics_table(
        self,
        y_true: np.ndarray,
        spam_scores: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Xuất bảng chi tiết tất cả các chỉ số ứng với từng mốc threshold."""
        grid = self.grid if thresholds is None else np.asarray(thresholds, dtype=float)
        records = [
            self.compute_metrics_at_threshold(y_true, spam_scores, th)
            for th in grid
        ]
        return pd.DataFrame(records)

    def scan_thresholds(
        self,
        y_true: np.ndarray,
        spam_probs: np.ndarray,
    ) -> pd.DataFrame:
        return self.metrics_table(y_true, spam_probs)

    def select_for_recall_target(
        self,
        threshold_df: pd.DataFrame,
        recall_target: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Chọn ngưỡng đạt Recall >= recall_target và có Accuracy / F1 cao nhất."""
        target = self.recall_target if recall_target is None else float(recall_target)
        work_df = threshold_df.copy()
        work_df["target_met"] = work_df["recall"] >= target

        feasible = work_df[work_df["target_met"] == True]
        if not feasible.empty:
            selected = feasible.sort_values(
                by=["accuracy", "f1", "precision", "fpr"],
                ascending=[False, False, False, True],
            ).iloc[0].copy()
            selected["selection_rule"] = f"Max validation accuracy with recall >= {target:.2f}"
            selected["target_met"] = True
        else:
            selected = work_df.sort_values(
                by=["recall", "accuracy", "f1"],
                ascending=[False, False, False],
            ).iloc[0].copy()
            selected["selection_rule"] = f"Recall target {target:.2f} not reached; selected max recall fallback"
            selected["target_met"] = False

        return selected.to_dict()

    def select_best_threshold(
        self,
        scan_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        return self.select_for_recall_target(scan_df, recall_target=self.recall_target)


def tune_model_hyperparameters(
    model_name: str,
    model_class: Any,
    param_grid: Dict[str, List[Any]],
    X_train: Union[np.ndarray, csr_matrix],
    y_train: np.ndarray,
    X_val: Union[np.ndarray, csr_matrix],
    y_val: np.ndarray,
    scoring_metric: str = "f1",
    recall_target: Optional[float] = 0.85,
    threshold_optimizer: Optional[ThresholdOptimizerScratch] = None,
) -> Tuple[Any, Dict[str, Any], float]:
    """Tìm kiếm siêu tham số kết hợp hàm mục tiêu có ràng buộc nghiệp vụ."""
    import itertools

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))

    best_score = -float("inf")
    best_model = None
    best_params = {}

    y_val_arr = np.asarray(y_val, dtype=int).ravel()
    y_train_arr = np.asarray(y_train, dtype=int).ravel()

    for combo in combinations:
        params = dict(zip(keys, combo))
        model = model_class(**params)
        model.fit(X_train, y_train)

        if threshold_optimizer is not None:
            threshold_info = threshold_optimizer.select_best_threshold(
                threshold_optimizer.metrics_table(y_val, model.predict_proba(X_val)[:, 1])
            )
            y_pred = (model.predict_proba(X_val)[:, 1] >= threshold_info["threshold"]).astype(int)
        else:
            y_pred = np.asarray(model.predict(X_val), dtype=int).ravel()

        tp = int(np.sum((y_val_arr == 1) & (y_pred == 1)))
        fp = int(np.sum((y_val_arr == 0) & (y_pred == 1)))
        fn = int(np.sum((y_val_arr == 1) & (y_pred == 0)))
        tn = int(np.sum((y_val_arr == 0) & (y_pred == 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / max(len(y_val_arr), 1)
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if recall_target is not None:
            # Constrained objective: nếu đạt recall mục tiêu, tối đa hóa accuracy; nếu không đạt, phạt nặng
            if recall >= float(recall_target):
                score = accuracy
            else:
                score = recall - float(recall_target) - 1.0 + 0.01 * accuracy
        elif scoring_metric == "f1":
            score = f1
        elif scoring_metric == "recall":
            score = recall
        else:
            score = accuracy

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    return best_model, best_params, float(best_score)


class ModelSelectionWorkflow:
    """
    QUY TRÌNH SO SÁNH VÀ LỰA CHỌN MÔ HÌNH VÔ ĐỊCH (MODEL SELECTION WORKFLOW):
    --------------------------------------------------------------------------
    So sánh công bằng tất cả các ứng viên mô hình (Default & Tuned) trên tập Validation
    để chọn ra mô hình tối ưu nhất đưa vào đánh giá cuối và triển khai.
    """

    def __init__(self, recall_target: float = 0.85):
        self.recall_target = float(recall_target)

    def select_candidates(self, models, X_val, y_val, optimizer=None):
        """Chọn ứng viên và ngưỡng trên Validation; không nhận dữ liệu Test."""
        if not models:
            raise ValueError("Cần ít nhất một mô hình ứng viên.")
        optimizer = optimizer or ThresholdOptimizerScratch(recall_target=self.recall_target)
        candidates, scans, selections = {}, {}, {}
        for name, model in models.items():
            scans[name] = optimizer.metrics_table(y_val, model.predict_proba(X_val)[:, 1])
            info = optimizer.select_for_recall_target(scans[name], self.recall_target)
            selections[name] = info
            candidates[name] = {
                "branch": "Tuned" if name.endswith(" (Tuned)") else "Default",
                "parameter_source": "Grid Search" if hasattr(model, "tuning_params_") else "Pre-set",
                "params": getattr(model, "tuning_params_", {}),
                "threshold": info["threshold"], "target_met": info["target_met"],
                # Không đạt mục tiêu: ưu tiên recall trước accuracy.
                "objective_score": info["accuracy"] if info["target_met"] else info["recall"],
                "val_accuracy": info["accuracy"], "val_precision": info["precision"],
                "val_recall": info["recall"], "val_f1": info["f1"], "val_f2": info["f2"],
            }
        comparison = self.build_comparison_frame(candidates)
        winner = self.select_final_winner(comparison)
        return winner, selections[winner], comparison, scans[winner]

    def build_comparison_frame(
        self,
        candidates: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """Xây dựng bảng đối sánh chi tiết tất cả các nhánh mô hình."""
        rows = []
        for key, cand in candidates.items():
            row = {
                "candidate_key": cand.get("candidate_key", key),
                "algorithm": cand.get("algorithm", key),
                "branch": cand.get("branch", "Default"),
                "parameter_source": cand.get("parameter_source", "Pre-set"),
                "threshold": float(cand.get("threshold", 0.5)),
                "target_met": bool(cand.get("target_met", True)),
                "objective_score": float(cand.get("objective_score", cand.get("val_f1", 0.0))),
                "validation_accuracy": float(cand.get("validation_accuracy", cand.get("val_accuracy", 0.0))),
                "validation_precision": float(cand.get("validation_precision", cand.get("val_precision", 0.0))),
                "validation_recall": float(cand.get("validation_recall", cand.get("val_recall", 0.0))),
                "validation_f1": float(cand.get("validation_f1", cand.get("val_f1", 0.0))),
                "validation_f2": float(cand.get("validation_f2", cand.get("val_f2", 0.0))),
                "train_val_accuracy_gap": float(cand.get("train_val_accuracy_gap", 0.0)),
                "train_val_f1_gap": float(cand.get("train_val_f1_gap", 0.0)),
                "params": cand.get("params", {}),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(
                by=["target_met", "objective_score", "validation_accuracy", "validation_f1"],
                ascending=[False, False, False, False],
                kind="stable",
            ).reset_index(drop=True)
            df["rank"] = np.arange(1, len(df) + 1)
            df["selected_as_final"] = df["rank"] == 1
        return df

    def compare_candidates(
        self,
        candidate_results: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        df = pd.DataFrame(candidate_results)
        if not df.empty and "f1" in df.columns:
            df = df.sort_values(by="f1", ascending=False).reset_index(drop=True)
        return df

    def select_final_winner(
        self,
        comparison_df: pd.DataFrame,
    ) -> str:
        if not comparison_df.empty:
            if "candidate_key" in comparison_df.columns:
                return str(comparison_df.iloc[0]["candidate_key"])
            if "model_name" in comparison_df.columns:
                return str(comparison_df.iloc[0]["model_name"])
        return "logistic_regression"


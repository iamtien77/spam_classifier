"""
metrics.py - Đánh giá hiệu năng mô hình toàn diện theo đề bài
Chuẩn hóa theo yêu cầu đề bài (ML project: Classifying Spam Emails - Workflow Bước 3)
=====================================================================================
Đề bài quy định rõ:
"Evaluate the model's performance on the testing set using metrics like accuracy,
 precision, recall, and F1-score."

Module này tự cài đặt bộ chỉ số đánh giá From-Scratch độc lập:
1. Confusion Matrix (Ma trận nhầm lẫn: True Positive, False Positive, True Negative, False Negative).
2. Accuracy (Độ chính xác tổng thể).
3. Precision (Độ chuẩn xác - Tỷ lệ email được đoán là spam thực sự là spam).
4. Recall (Độ nhạy - Tỷ lệ bắt trúng email spam trên tổng số email spam thực tế).
5. F1-Score (Trung bình điều hòa giữa Precision và Recall).
6. F-Beta Score (Đo lường linh hoạt ưu tiên Recall với hệ số Beta=2.0).
7. Specificity / True Negative Rate (Khả năng nhận diện chính xác email hợp lệ).
8. ROC-AUC (Diện tích dưới đường cong ROC tính bằng tích phân hình thang Trapezoidal).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class SpamModelEvaluator:
    """
    BỘ ĐÁNH GIÁ HIỆU NĂNG MÔ HÌNH PHÂN LOẠI SPAM TỰ XÂY DỰNG (FROM SCRATCH):
    ----------------------------------------------------------------------
    Tự động tính toán đầy đủ các chỉ số theo yêu cầu đề bài và xuất báo cáo tổng hợp.
    """

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Hàm chia an toàn tránh lỗi ZeroDivisionError."""
        return float(numerator) / float(denominator) if denominator != 0 else default

    def compute_confusion_matrix(
        self,
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
    ) -> np.ndarray:
        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        if not len(y_true_arr) or len(y_true_arr) != len(y_pred_arr):
            raise ValueError("y_true và y_pred phải không rỗng và có cùng độ dài.")
        if not np.all(np.isin(y_true_arr, [0, 1])) or not np.all(np.isin(y_pred_arr, [0, 1])):
            raise ValueError("Nhãn đánh giá phải là 0 hoặc 1.")
        tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
        fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
        fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
        tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
        return np.array([[tn, fp], [fn, tp]], dtype=int)

    def compute_all_metrics(
        self,
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
        y_prob: Optional[np.ndarray] = None,
        beta: float = 2.0,
    ) -> Dict[str, Any]:
        cm = self.compute_confusion_matrix(y_true, y_pred)
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta phải hữu hạn và dương.")
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        fn, tp = int(cm[1, 0]), int(cm[1, 1])
        total = tn + fp + fn + tp

        accuracy = self.safe_divide(tp + tn, total)
        precision = self.safe_divide(tp, tp + fp)
        recall = self.safe_divide(tp, tp + fn)
        f1 = self.safe_divide(2.0 * precision * recall, precision + recall)
        beta_sq = beta ** 2
        f_beta = self.safe_divide((1.0 + beta_sq) * precision * recall, beta_sq * precision + recall)
        specificity = self.safe_divide(tn, tn + fp)
        fpr = self.safe_divide(fp, fp + tn)

        roc_auc = None
        if y_prob is not None:
            probs = np.asarray(y_prob, dtype=float).ravel()
            y_arr = np.asarray(y_true, dtype=int).ravel()
            if (len(probs) != len(y_arr) or not np.all(np.isfinite(probs))
                    or np.any((probs < 0) | (probs > 1))):
                raise ValueError("y_prob phải cùng độ dài với y_true và chứa xác suất hữu hạn thuộc [0, 1].")
            n_pos = int(np.sum(y_arr == 1))
            n_neg = int(np.sum(y_arr == 0))
            if n_pos > 0 and n_neg > 0:
                # Mann–Whitney U: các điểm bằng nhau nhận thứ hạng trung bình.
                order = np.argsort(probs, kind="stable")
                sorted_probs = probs[order]
                starts = np.r_[0, np.flatnonzero(np.diff(sorted_probs)) + 1]
                ends = np.r_[starts[1:], len(probs)]
                ranks = np.empty(len(probs), dtype=float)
                for start, end in zip(starts, ends):
                    ranks[order[start:end]] = (start + 1 + end) / 2.0
                pos_ranks_sum = np.sum(ranks[y_arr == 1])
                roc_auc = float(pos_ranks_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

        return {
            "confusion_matrix": cm,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "f_beta": float(f_beta),
            "specificity": float(specificity),
            "fpr": float(fpr),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
        }

    def build_comparison_summary(
        self,
        model_results: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        rows = []
        for name, metrics in model_results.items():
            rows.append({
                "Mô hình": name,
                "Accuracy": metrics.get("accuracy", 0.0),
                "Precision": metrics.get("precision", 0.0),
                "Recall": metrics.get("recall", 0.0),
                "F1-Score": metrics.get("f1", 0.0),
                "F-Beta": metrics.get("f_beta", 0.0),
                "ROC-AUC": metrics.get("roc_auc", 0.0),
            })
        df = pd.DataFrame(rows)
        if not df.empty and "F1-Score" in df.columns:
            df = df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
        return df

    def audit_evaluation_consistency(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Kiểm toán tính nhất quán của kết quả đánh giá (Evaluation Consistency Audit)."""
        metrics = self.compute_all_metrics(y_true, y_pred, y_prob=y_prob)
        n_samples = len(np.asarray(y_true))
        tp, fp, tn, fn = metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]

        audit_rows = [
            {"Audit Item": "Total Sample Sum Conservation (TP+FP+TN+FN == N)", "Observed": tp + fp + tn + fn, "Expected": n_samples, "Status": "PASS" if (tp + fp + tn + fn == n_samples) else "FAIL"},
            {"Audit Item": "Accuracy Mathematical Consistency", "Observed": round(metrics["accuracy"], 5), "Expected": round((tp + tn) / max(n_samples, 1), 5), "Status": "PASS" if abs(metrics["accuracy"] - (tp + tn) / max(n_samples, 1)) < 1e-4 else "FAIL"},
            {"Audit Item": "Precision in [0, 1] Bounds", "Observed": round(metrics["precision"], 4), "Expected": "0.0 <= P <= 1.0", "Status": "PASS" if 0.0 <= metrics["precision"] <= 1.0 else "FAIL"},
            {"Audit Item": "Recall in [0, 1] Bounds", "Observed": round(metrics["recall"], 4), "Expected": "0.0 <= R <= 1.0", "Status": "PASS" if 0.0 <= metrics["recall"] <= 1.0 else "FAIL"},
            {"Audit Item": "F1-Score in [0, 1] Bounds", "Observed": round(metrics["f1"], 4), "Expected": "0.0 <= F1 <= 1.0", "Status": "PASS" if 0.0 <= metrics["f1"] <= 1.0 else "FAIL"},
            {"Audit Item": "ROC-AUC Valid (if available)", "Observed": round(metrics["roc_auc"], 4) if metrics["roc_auc"] is not None else "N/A", "Expected": ">= 0.50", "Status": "PASS" if (metrics["roc_auc"] is None or metrics["roc_auc"] >= 0.50) else "WARN"},
        ]
        return pd.DataFrame(audit_rows)




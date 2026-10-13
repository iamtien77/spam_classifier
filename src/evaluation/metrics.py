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
        return float(numerator / denominator) if denominator != 0 else float(default)

    def compute_confusion_matrix(
        self,
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
    ) -> np.ndarray:
        """
        LOGIC TÍNH MA TRẬN NHẦM LẪN (CONFUSION MATRIX):
        -----------------------------------------------
        # Bước 1: Chuyển đổi y_true và y_pred về mảng numpy 1D kiểu int.
        # Bước 2: Đếm số lượng mẫu âm tính thật (True Negative - TN): (y_true == 0) & (y_pred == 0).
        # Bước 3: Đếm số lượng mẫu báo động giả (False Positive - FP): (y_true == 0) & (y_pred == 1) [Ham bị chặn nhầm].
        # Bước 4: Đếm số lượng mẫu bỏ sót (False Negative - FN): (y_true == 1) & (y_pred == 0) [Spam lọt lưới].
        # Bước 5: Đếm số lượng mẫu dương tính thật (True Positive - TP): (y_true == 1) & (y_pred == 1) [Bắt đúng Spam].
        # Bước 6: Trả về ma trận 2x2: [[TN, FP], [FN, TP]] dạng np.ndarray(dtype=int).
        """
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_pred = np.asarray(y_pred, dtype=int).ravel()
        if len(y_true) != len(y_pred):
            raise ValueError("y_true và y_pred phải có cùng số mẫu.")
        if not np.all(np.isin(y_true, [0, 1])) or not np.all(np.isin(y_pred, [0, 1])):
            raise ValueError("Nhãn phải là 0 (Ham) hoặc 1 (Spam).")

        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        return np.array([[tn, fp], [fn, tp]], dtype=int)

    def compute_all_metrics(
        self,
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
        y_prob: Optional[np.ndarray] = None,
        beta: float = 2.0,
    ) -> Dict[str, Any]:
        """
        LOGIC TÍNH TOÀN BỘ 4 CHỈ SỐ CỐT LÕI THEO ĐỀ BÀI + CÁC CHỈ SỐ MỞ RỘNG:
        -----------------------------------------------------------------------
        # Bước 1: Gọi compute_confusion_matrix() để lấy TN, FP, FN, TP.
        # Bước 2: Tính Accuracy (Độ chính xác) = (TP + TN) / (TP + TN + FP + FN).
        # Bước 3: Tính Precision (Độ chuẩn xác) = TP / (TP + FP).
        # Bước 4: Tính Recall (Độ thu hồi / TPR) = TP / (TP + FN).
        # Bước 5: Tính F1-Score = 2 * (Precision * Recall) / (Precision + Recall).
        # Bước 6: Tính F-Beta Score = (1 + beta^2) * (Precision * Recall) / (beta^2 * Precision + Recall).
        # Bước 7: Tính Specificity (TNR) = TN / (TN + FP) và FPR = FP / (FP + TN).
        # Bước 8: Nếu có y_prob (xác suất dự đoán):
        #         - Sinh các cặp điểm (FPR, TPR) theo từng mốc ngưỡng.
        #         - Tính ROC-AUC bằng quy tắc tích phân hình thang Trapezoidal:
        #           AUC = sum_i (FPR_{i+1} - FPR_i) * (TPR_i + TPR_{i+1}) / 2.
        # Bước 9: Đóng gói toàn bộ các chỉ số vào Dictionary và trả về.
        """
        confusion_matrix = self.compute_confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix.ravel().tolist()
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta phải là số dương hữu hạn.")

        accuracy = self.safe_divide(tp + tn, tp + tn + fp + fn)
        precision = self.safe_divide(tp, tp + fp)
        recall = self.safe_divide(tp, tp + fn)
        f1 = self.safe_divide(2 * precision * recall, precision + recall)
        beta_squared = beta ** 2
        f_beta = self.safe_divide(
            (1 + beta_squared) * precision * recall,
            beta_squared * precision + recall,
        )
        specificity = self.safe_divide(tn, tn + fp)
        fpr = self.safe_divide(fp, fp + tn)

        # AUC không xác định nếu thiếu xác suất hoặc tập dữ liệu chỉ có một lớp.
        roc_auc = None
        if y_prob is not None:
            labels = np.asarray(y_true, dtype=int).ravel()
            probs = np.asarray(y_prob, dtype=float).ravel()
            if len(probs) != len(labels) or not np.all(np.isfinite(probs)):
                raise ValueError("y_prob phải hữu hạn và có cùng số mẫu với y_true.")

            positives = tp + fn
            negatives = tn + fp
            if positives > 0 and negatives > 0:
                fpr_points = [0.0]
                tpr_points = [0.0]
                # Xét cùng lúc các mẫu có xác suất bằng nhau tại mỗi ngưỡng.
                for threshold in np.unique(probs)[::-1]:
                    predicted_spam = probs >= threshold
                    fpr_points.append(float(np.sum((labels == 0) & predicted_spam) / negatives))
                    tpr_points.append(float(np.sum((labels == 1) & predicted_spam) / positives))

                roc_auc = sum(
                    (fpr_points[i] - fpr_points[i - 1])
                    * (tpr_points[i] + tpr_points[i - 1]) / 2
                    for i in range(1, len(fpr_points))
                )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f_beta": f_beta,
            "specificity": specificity,
            "tpr": recall,
            "fpr": fpr,
            "roc_auc": roc_auc,
            "confusion_matrix": confusion_matrix,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

    def build_comparison_summary(
        self,
        model_results: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        LOGIC TẠO BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG GIỮA CÁC MÔ HÌNH:
        ----------------------------------------------------------
        # Bước 1: Duyệt qua từng mô hình trong model_results (Logistic Regression, SVM, Naive Bayes, Ensemble).
        # Bước 2: Trích xuất các chỉ số chính: Accuracy, Precision, Recall, F1, ROC-AUC.
        # Bước 3: Tạo DataFrame đối chiếu rõ ràng giữa các thuật toán phân loại theo yêu cầu đề bài.
        # Bước 4: Sắp xếp theo thứ tự ưu tiên (F1-score hoặc Accuracy) và trả về bảng tổng hợp.
        """
        metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        rows = []
        for model_name, metrics in model_results.items():
            row = {"model_name": model_name}
            for metric in metric_names:
                row[metric] = metrics.get(metric)
            rows.append(row)

        summary = pd.DataFrame(rows, columns=["model_name"] + metric_names)
        return summary.sort_values(
            by=["f1", "accuracy"], ascending=False,
        ).reset_index(drop=True)

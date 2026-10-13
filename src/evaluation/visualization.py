"""
visualization.py - Trực quan hóa dữ liệu và biểu đồ đánh giá mô hình
Chuẩn hóa theo yêu cầu đề bài (ML project: Classifying Spam Emails)
===================================================================
Module này đảm nhiệm việc xuất bản các biểu đồ trực quan chất lượng cao vào results/figures/:
1. Khám phá dữ liệu (EDA): Phân bố nhãn Ham/Spam, biểu đồ phân phối độ dài văn bản email.
2. Ma trận nhầm lẫn (Confusion Matrix Heatmap): Hiển thị số lượng mẫu và tỷ lệ chuẩn hóa %.
3. Đường cong đánh giá (ROC Curve & Precision-Recall Curve) có đánh dấu điểm ngưỡng tối ưu.
4. Biểu đồ so sánh đa chiều hiệu năng giữa 3 mô hình cốt lõi (Logistic Regression, SVM, Naive Bayes) và Ensemble.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_eda_summary(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """
    LOGIC TRỰC QUAN HÓA KHÁM PHÁ DỮ LIỆU BAN ĐẦU (EDA):
    ---------------------------------------------------
    # Bước 1: Tạo đồ thị 2 subplot:
    #         - Subplot 1: Biểu đồ cột (Bar chart) và biểu đồ tròn thể hiện tỷ lệ % phân bố giữa Ham và Spam.
    #         - Subplot 2: Biểu đồ phân phối độ dài văn bản (Histogram/KDE) so sánh sự khác biệt giữa Ham và Spam.
    # Bước 2: Tùy chỉnh màu sắc trực quan, gắn tiêu đề và chú thích rõ ràng.
    # Bước 3: Lưu hình ảnh độ phân giải cao vào save_path (dpi=300) nếu có yêu cầu.
    """
    pass


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    model_name: str = "Classifier",
    save_path: Optional[Path] = None,
) -> None:
    """
    LOGIC VẼ HEATMAP MA TRẬN NHẦM LẪN (CONFUSION MATRIX HEATMAP):
    -------------------------------------------------------------
    # Bước 1: Khởi tạo figure và axes matplotlib.
    # Bước 2: Tính tỷ lệ phần trăm chuẩn hóa theo từng hàng (cm_normalized = cm / cm.sum(axis=1)).
    # Bước 3: Vẽ ma trận dạng Heatmap bằng seaborn hoặc imshow:
    #         Hiển thị đồng thời cả số lượng tuyệt đối và phần trăm trong từng ô (ví dụ: 'TN: 4320 (98.5%)').
    # Bước 4: Gắn nhãn trục X: 'Predicted Label' [Ham, Spam], trục Y: 'Actual Label' [Ham, Spam].
    # Bước 5: Thêm tiêu đề 'Confusion Matrix - {model_name}' và lưu vào save_path.
    """
    pass


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    best_threshold: Optional[float] = None,
    model_name: str = "Classifier",
    save_path: Optional[Path] = None,
) -> None:
    """
    LOGIC VẼ ĐỒ THỊ ĐÔI ROC CURVE VÀ PRECISION-RECALL CURVE:
    --------------------------------------------------------
    # Bước 1: Tạo figure 2 đồ thị con nằm ngang (1 hàng 2 cột).
    # Bước 2: Đồ thị 1 - ROC Curve:
    #         - Vẽ đường cong FPR vs TPR.
    #         - Vẽ đường tham chiếu ngẫu nhiên đường chéo nét đứt (Chance line y = x).
    #         - Đánh dấu chấm tròn tại vị trí của best_threshold.
    #         - Hiển thị giá trị diện tích dưới đường cong (ROC-AUC).
    # Bước 3: Đồ thị 2 - Precision-Recall Curve:
    #         - Vẽ đường cong Recall vs Precision.
    #         - Đánh dấu chấm tròn tại vị trí của best_threshold.
    #         - Hiển thị Average Precision (AP).
    # Bước 4: Lưu hình ảnh đồ thị đôi vào save_path.
    """
    pass


def plot_models_comparison_bar(
    comparison_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """
    LOGIC VẼ BIỂU ĐỒ CỘT SO SÁNH HIỆU NĂNG GIỮA CÁC MÔ HÌNH THEO ĐỀ BÀI:
    ---------------------------------------------------------------------
    # Bước 1: Chọn các chỉ số chính: Accuracy, Precision, Recall, F1-Score.
    # Bước 2: Vẽ biểu đồ cột nhóm (Grouped Bar Chart) so sánh giữa:
    #         Logistic Regression vs Support Vector Machines vs Naive Bayes vs Ensemble.
    # Bước 3: Ghi chú rõ ràng giá trị số trên đỉnh từng cột.
    # Bước 4: Lưu hình ảnh vào save_path.
    """
    pass

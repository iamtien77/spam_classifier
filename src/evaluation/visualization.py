"""
visualization.py - Trực quan hóa dữ liệu, hiệu năng mô hình và giải thích đặc trưng
Dựa trên logic từ Classification_email_spam.ipynb (Mục 2.3, 3.2, 7.4, 9.1, 10.1, 10.2)
======================================================================================
Module này chịu trách nhiệm:
1. Trực quan hóa EDA ban đầu: Phân bố nhãn, biểu đồ độ dài văn bản, Top N-grams của Ham vs Spam.
2. Trực quan hóa ma trận nhầm lẫn (Confusion Matrix Heatmap) dạng thô và chuẩn hóa %.
3. Vẽ đường cong ROC Curve và Precision-Recall Curve có đánh dấu ngưỡng tối ưu.
4. Biểu đồ quét ngưỡng (Threshold Sweep Plot) phản ánh tương quan đánh đổi giữa Precision và Recall.
5. Biểu đồ đường quét Top K đặc trưng SHAP (SHAP Top-K Sweep Plot).
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_eda_summary(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """
    LOGIC TRỰC QUAN HÓA KHÁM PHÁ DỮ LIỆU BAN ĐẦU:
    ---------------------------------------------
    # Bước 1: Vẽ biểu đồ tròn / biểu đồ cột thể hiện tỷ lệ mất cân bằng giữa Ham và Spam.
    # Bước 2: Vẽ biểu đồ phân bố (Histogram / KDE) so sánh độ dài ký tự và số từ giữa email Ham và Spam.
    # Bước 3: Lưu hình ảnh độ phân giải cao (dpi=300) vào thư mục results/figures/ nếu có save_path.
    """
    pass


def plot_top_ngrams_comparison(
    top_ham_words: List[Tuple[str, int]],
    top_spam_words: List[Tuple[str, int]],
    save_path: Optional[Path] = None,
) -> None:
    """
    LOGIC VẼ BIỂU ĐỒ SO SÁNH TỪ KHÓA ĐẶC TRƯNG GIỮA HAM VÀ SPAM:
    ------------------------------------------------------------
    # Bước 1: Vẽ biểu đồ thanh ngang (Horizontal Bar Chart) cho Top 15 từ khóa xuất hiện nhiều nhất ở lớp Spam.
    # Bước 2: Vẽ biểu đồ song song cho Top 15 từ khóa của lớp Ham để làm nổi bật sự khác biệt ngữ nghĩa.
    # Bước 3: Lưu hình ảnh vào results/figures/.
    """
    pass


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: List[str] = ["Ham", "Spam"],
    normalize: bool = True,
    save_path: Optional[Path] = None,
) -> None:
    """
    LOGIC VẼ HEATMAP MA TRẬN NHẦM LẪN:
    ----------------------------------
    # Bước 1: Tính ma trận tỷ lệ phần trăm chuẩn hóa theo dòng (row-normalized) nếu normalize=True.
    # Bước 2: Dùng heatmap hiển thị trực quan các ô TN, FP, FN, TP kèm số lượng mẫu và tỷ lệ %.
    # Bước 3: Tô màu cảnh báo riêng cho ô False Positive (Ham bị đánh nhầm) và False Negative (Spam lọt lưới).
    # Bước 4: Lưu hình ảnh.
    """
    pass


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    best_threshold: Optional[float] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    LOGIC VẼ BỘ ĐÔI ĐƯỜNG CONG ROC VÀ PRECISION-RECALL:
    --------------------------------------------------
    # Bước 1: Tạo subplot gồm 2 đồ thị nằm ngang:
    #         - Đồ thị 1: ROC Curve (FPR vs TPR) kèm đường chéo tham chiếu ngẫu nhiên (AUC = 0.5).
    #         - Đồ thị 2: Precision-Recall Curve kèm đường baseline tỷ lệ spam gốc.
    # Bước 2: Nếu có best_threshold, chấm điểm nổi bật (highlight dot) vị trí ngưỡng được chọn trên đường cong.
    # Bước 3: Ghi chú diện tích AUC và điểm F-beta tương ứng lên tiêu đề đồ thị.
    # Bước 4: Lưu hình ảnh.
    """
    pass


def plot_threshold_diagnostic(threshold_df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """
    LOGIC VẼ ĐƯỜNG CONG CHẨN ĐOÁN NGƯỠNG PHÂN LOẠI (THRESHOLD SWEEP):
    ---------------------------------------------------------------
    # Bước 1: Trục hoành biểu thị các giá trị ngưỡng phân loại từ 0.0 đến 1.0.
    # Bước 2: Vẽ đồng thời 3 đường chỉ số:
    #         - Đường màu xanh: Precision
    #         - Đường màu cam: Recall
    #         - Đường màu tím: F-beta (beta=2.0)
    # Bước 3: Vẽ đường gióng đứng nét đứt tại ngưỡng tối ưu đã chọn thỏa mãn Recall >= 0.85.
    # Bước 4: Lưu hình ảnh vào results/figures/.
    """
    pass


def plot_shap_top_k_sweep(sweep_df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """
    LOGIC VẼ BIỂU ĐỒ QUÉT TOP K ĐẶC TRƯNG SHAP:
    ------------------------------------------
    # Bước 1: Trục hoành là số lượng đặc trưng Top K được chọn (từ 300 đến 2500).
    # Bước 2: Trục tung là hiệu năng mô hình (Validation F-beta score và Recall).
    # Bước 3: Giúp người nghiên cứu trực quan hóa điểm bão hòa (Elbow Point) của số lượng đặc trưng.
    # Bước 4: Lưu hình ảnh.
    """
    pass

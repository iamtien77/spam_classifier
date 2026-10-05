"""
error_analysis.py - Phân tích lỗi sai chuyên sâu (Error Analysis)
Dựa trên logic từ Classification_email_spam.ipynb (Mục 10.1.4)
=============================================================
Module này chịu trách nhiệm:
1. Lọc và bóc tách hai loại sai lầm điển hình của mô hình:
   - False Positives (FP): Email hợp lệ (Ham) nhưng bị gán nhãn nhầm là Spam (Báo động giả - Nguy cơ mất thư quan trọng).
   - False Negatives (FN): Email Spam nhưng bị bỏ sót thành Ham (Lọt lưới thư rác).
2. Phân tích định tính và định lượng nguyên nhân gây lỗi:
   - Độ dài văn bản, sự xuất hiện của các từ khóa gây hiểu lầm, hoặc các định dạng lách luật.
3. Xuất bảng tổng hợp mẫu lỗi ra file phục vụ việc cải tiến tiền xử lý và đặc trưng.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def extract_misclassified_samples(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    LOGIC TRÍCH XUẤT CÁC MẪU BỊ PHÂN LOẠI SAI TRÊN TẬP TEST:
    --------------------------------------------------------
    # Bước 1: Tạo DataFrame tổng hợp gồm:
    #         - text: Nội dung email gốc
    #         - y_true: Nhãn thực tế (0: Ham, 1: Spam)
    #         - y_pred: Nhãn dự đoán từ mô hình (0 hoặc 1)
    #         - spam_score: Xác suất / điểm số Spam dự đoán
    # Bước 2: Lọc tập False Positives (FP): (y_true == 0) & (y_pred == 1).
    #         Sắp xếp giảm dần theo spam_score (những email hợp lệ nhưng bị mô hình tự tin phán nhầm là Spam).
    # Bước 3: Lọc tập False Negatives (FN): (y_true == 1) & (y_pred == 0).
    #         Sắp xếp tăng dần theo spam_score (những email spam tinh vi nhất lừa được mô hình).
    # Bước 4: Trả về cặp DataFrame (fp_df, fn_df).
    """
    pass


def profile_error_patterns(
    error_df: pd.DataFrame,
    error_type: str = "False Positive",
) -> Dict[str, any]:
    """
    LOGIC PHÂN TÍCH ĐẶC TÍNH CỦA MẪU LỖI:
    -------------------------------------
    # Bước 1: Thống kê độ dài văn bản trung bình và trung vị của các mẫu lỗi so với toàn bộ tập dữ liệu.
    # Bước 2: Đếm tần suất xuất hiện của các từ khóa kích hoạt spam giả định bên trong các mẫu lỗi.
    # Bước 3: Xác định xem lỗi xuất phát từ yếu tố nào:
    #         - Chứa nhiều số điện thoại / link web hợp lệ?
    #         - Văn bản quá ngắn thiếu ngữ cảnh?
    #         - Email chứa nhiều chữ hoa do người gửi nhấn mạnh?
    # Bước 4: Trả về báo cáo tóm tắt đặc trưng lỗi.
    """
    pass


def export_error_report(
    fp_df: pd.DataFrame,
    fn_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    LOGIC XUẤT BÁO CÁO MẪU LỖI RA FILE CSV:
    --------------------------------------
    # Bước 1: Định dạng lại các cột để người dùng dễ đọc và kiểm toán (Audit).
    # Bước 2: Lưu fp_samples.csv và fn_samples.csv vào thư mục results/metrics/.
    # Bước 3: Ghi log hoàn thành để phục vụ quá trình debug và cải tiến mô hình trong tương lai.
    """
    pass

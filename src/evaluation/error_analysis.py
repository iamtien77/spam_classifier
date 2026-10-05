"""
error_analysis.py - Phân tích chuyên sâu các mẫu phân loại sai (Error Analysis)
Chuẩn hóa theo yêu cầu đề bài (ML project: Classifying Spam Emails)
================================================================================
Module này đảm nhiệm việc mổ xẻ nguyên nhân sai lệch của mô hình:
1. False Positive (Báo động giả / Chặn nhầm): Email hợp lệ (Ham) bị đoán nhầm thành Spam.
   - Đây là lỗi nghiêm trọng nhất trong thực tế vì người dùng có thể bị mất thông báo công việc quan trọng.
2. False Negative (Bỏ sót / Lọt lưới): Email rác (Spam) bị đoán nhầm thành email hợp lệ (Ham).
3. Thống kê đặc trưng gây nhiễu: Từ khóa nhạy cảm xuất hiện nhầm, cấu trúc viết hoa, link URLs.
4. Xuất kết quả phân tích ra file CSV tại results/metrics/misclassified_samples.csv.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def extract_misclassified_samples(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    LOGIC TRÍCH XUẤT CÁC MẪU DỰ ĐOÁN SAI:
    -------------------------------------
    # Bước 1: Tạo bản sao DataFrame kiểm tra kèm theo các cột nhãn thực tế, nhãn dự đoán và xác suất.
    # Bước 2: Lọc tập False Positives (FP):
    #         fp_df = df[(y_true == 0) & (y_pred == 1)].
    # Bước 3: Lọc tập False Negatives (FN):
    #         fn_df = df[(y_true == 1) & (y_pred == 0)].
    # Bước 4: Trả về cặp DataFrame (fp_df, fn_df).
    """
    pass


def profile_error_patterns(
    fp_df: pd.DataFrame,
    fn_df: pd.DataFrame,
    save_csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    LOGIC THỐNG KÊ VÀ LẬP HỒ SƠ NGUYÊN NHÂN LỖI SAI:
    -------------------------------------------------
    # Bước 1: Phân tích nhóm False Positives (Ham bị chặn nhầm):
    #         - Độ dài văn bản trung bình, số lượng chữ in hoa, có chứa từ khóa 'free', 'win', 'call' hay không.
    #         - Xác định các từ ngữ dễ gây nhầm lẫn khiến mô hình kích hoạt nhầm.
    # Bước 2: Phân tích nhóm False Negatives (Spam lọt lưới):
    #         - Kiểm tra các thủ thuật lách spam của người gửi (ví dụ: dùng tiếng lóng, viết cố tình sai chính tả,
    #           nội dung quá ngắn không đủ đặc trưng từ vựng).
    # Bước 3: Tổng hợp danh sách tất cả các email dự đoán sai kèm lý do nghi ngờ vào một DataFrame duy nhất.
    # Bước 4: Nếu có save_csv_path: lưu kết quả ra file CSV để phục vụ báo cáo đồ án.
    # Bước 5: Trả về DataFrame phân tích lỗi sai chi tiết.
    """
    pass

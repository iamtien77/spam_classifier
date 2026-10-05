"""
loader.py - Tải và khám phá dữ liệu ban đầu
Dựa trên logic từ Classification_email_spam.ipynb (Mục 2.1)
===========================================================
Module này chịu trách nhiệm:
1. Tìm kiếm và nạp tập dữ liệu thô spam.csv với cơ chế fallback đường dẫn.
2. Xử lý encoding (UTF-8, Latin-1) và trích xuất đúng 2 cột cốt lõi: 'label' và 'text'.
3. Khám phá tổng quan cấu trúc dữ liệu, tỷ lệ mất cân bằng lớp (Ham vs Spam), và missing values.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd


def resolve_data_path(custom_path: Optional[Path] = None) -> Path:
    """
    LOGIC TÌM KIẾM ĐƯỜNG DẪN DỮ LIỆU:
    ----------------------------------
    # Bước 1: Ưu tiên đường dẫn được người dùng truyền vào (custom_path).
    # Bước 2: Duyệt danh sách các vị trí ứng viên (candidates) tiềm năng:
    #         - data/raw/spam.csv
    #         - data/spam.csv
    #         - ../data/spam.csv
    # Bước 3: Kiểm tra sự tồn tại (exists). Nếu tìm thấy file, trả về đường dẫn Path hợp lệ.
    # Bước 4: Nếu không tìm thấy ở bất kỳ đâu, raise FileNotFoundError kèm thông báo chi tiết.
    """
    pass


def load_raw_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    LOGIC ĐỌC VÀ CHUẨN HÓA DỮ LIỆU THÔ BAN ĐẦU:
    ------------------------------------------
    # Bước 1: Gọi resolve_data_path() để xác định vị trí file spam.csv.
    # Bước 2: Đọc file CSV bằng pandas:
    #         - Thử đọc bằng encoding='utf-8'.
    #         - Nếu gặp UnicodeDecodeError, fallback sang encoding='latin-1' (chuẩn của SMS/Email spam dataset).
    # Bước 3: Trích xuất 2 cột đầu tiên (chứa nhãn và nội dung text):
    #         - df = df.iloc[:, [0, 1]].copy()
    #         - Đổi tên cột chuẩn: df.columns = ["label", "text"]
    # Bước 4: Tạo thêm cột tham chiếu:
    #         - df["raw_text"] = df["text"]
    #         - df["label_normalized"] = df["label"].astype(str).str.strip().str.lower()
    # Bước 5: Trả về DataFrame thô sẵn sàng cho bước chia phân tầng (stratified split).
    """
    pass


def explore_raw_data(df: pd.DataFrame) -> dict:
    """
    LOGIC KHÁM PHÁ & THỐNG KÊ TỔNG QUAN:
    ------------------------------------
    # Bước 1: Đếm tổng số dòng, tổng số cột (df.shape).
    # Bước 2: Kiểm tra missing values trên cả 2 cột 'label' và 'text'.
    # Bước 3: Thống kê phân bố nhãn:
    #         - Số lượng và tỷ lệ % của lớp 'ham' (email hợp lệ).
    #         - Số lượng và tỷ lệ % của lớp 'spam' (email rác).
    #         - Đánh giá mức độ mất cân bằng lớp (thông thường ~86% ham và ~14% spam).
    # Bước 4: Thống kê sơ bộ về độ dài ký tự của văn bản (min, median, max, mean).
    # Bước 5: Trả về dictionary tóm tắt các chỉ số thống kê trên để phục vụ logging / hiển thị.
    """
    pass

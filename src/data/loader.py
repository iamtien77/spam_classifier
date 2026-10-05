"""
loader.py - Tải và khám phá dữ liệu ban đầu
Chuẩn hóa theo yêu cầu đề bài (ML project: Classifying Spam Emails)
===================================================================
Module này đảm nhiệm:
1. Xác định vị trí và nạp tập dữ liệu email spam thô (spam.csv) với cơ chế fallback encoding.
2. Trích xuất đúng 2 trường thông tin cốt lõi: 'label' (ham/spam) và 'text' (nội dung email).
3. Khám phá tổng quan cấu trúc dữ liệu, tỷ lệ mất cân bằng lớp (Ham vs Spam), missing values.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


def resolve_data_path(custom_path: Optional[Path] = None) -> Path:
    """
    LOGIC TÌM KIẾM ĐƯỜNG DẪN DỮ LIỆU:
    ----------------------------------
    # Bước 1: Nếu người dùng truyền vào custom_path và đường dẫn tồn tại, trả về custom_path.
    # Bước 2: Thiết lập danh sách các vị trí ứng viên tiềm năng:
    #         - data/raw/spam.csv
    #         - data/spam.csv
    #         - ../data/raw/spam.csv
    #         - ../data/spam.csv
    # Bước 3: Duyệt lần lượt qua từng vị trí ứng viên, kiểm tra bằng candidate.exists().
    # Bước 4: Nếu tìm thấy file, trả về đường dẫn Path hợp lệ.
    # Bước 5: Nếu duyệt hết mà không thấy file nào tồn tại, raise FileNotFoundError kèm thông báo hướng dẫn.
    """
    pass


def load_raw_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    LOGIC ĐỌC VÀ CHUẨN HÓA DỮ LIỆU EMAIL THÔ:
    ------------------------------------------
    # Bước 1: Gọi hàm resolve_data_path(data_path) để định vị file spam.csv.
    # Bước 2: Thử đọc file CSV bằng pandas với encoding='utf-8'.
    # Bước 3: Nếu gặp UnicodeDecodeError, fallback tự động sang encoding='latin-1' hoặc 'windows-1252'.
    # Bước 4: Trích xuất 2 cột đầu tiên (chứa nhãn và nội dung văn bản email).
    # Bước 5: Đặt tên chuẩn cho 2 cột: ['label', 'text'].
    # Bước 6: Chuẩn hóa nhãn lớp về chữ thường và loại bỏ khoảng trắng dư thừa ('ham', 'spam').
    # Bước 7: Tạo thêm cột tham chiếu 'raw_text' lưu trữ nguyên bản nội dung email để trích xuất đặc trưng sau này.
    # Bước 8: Trả về pandas DataFrame đã chuẩn hóa schema ban đầu.
    """
    pass


def explore_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    LOGIC KHÁM PHÁ & THỐNG KÊ TỔNG QUAN DỮ LIỆU:
    ---------------------------------------------
    # Bước 1: Thống kê số lượng dòng (tổng số email) và số lượng cột của dataset.
    # Bước 2: Đếm số lượng giá trị thiếu (missing values / null) trên từng cột.
    # Bước 3: Thống kê phân bố nhãn mục tiêu:
    #         - Số lượng mẫu email hợp lệ (ham) và email rác (spam).
    #         - Tỷ lệ phần trăm (%) của từng lớp để đánh giá mức độ mất cân bằng lớp.
    # Bước 4: Thống kê mô tả về độ dài văn bản email (đếm ký tự và số lượng từ):
    #         - Tính min, max, median, mean của độ dài email.
    # Bước 5: Đóng gói toàn bộ các chỉ số thống kê vào một Dictionary và trả về cho caller.
    """
    pass

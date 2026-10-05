"""
logger.py - Hệ thống ghi log và công cụ đo đạc hiệu năng thực thi
Dựa trên kiến trúc tiện ích từ Classification_email_spam.ipynb
==============================================================
Module này cung cấp:
1. Bộ logger chuẩn ghi đồng thời ra console (Terminal) và file nhật ký (logs/pipeline.log).
2. Decorator time_step đo lường thời gian thực thi của từng phân đoạn trong workflow.
3. Hàm định dạng tiêu đề (section headers) giúp theo dõi tiến trình trực quan.
"""

from functools import wraps
import logging
from pathlib import Path
import time
from typing import Callable


def setup_logger(name: str = "spam_classifier", log_file: Path = Path("logs/pipeline.log")) -> logging.Logger:
    """
    LOGIC THIẾT LẬP LOGGER CHUẨN:
    ------------------------------
    # Bước 1: Tạo thư mục chứa file log nếu chưa tồn tại (log_file.parent.mkdir).
    # Bước 2: Khởi tạo logger với level INFO.
    # Bước 3: Đính kèm StreamHandler để in thông điệp ra màn hình console.
    # Bước 4: Đính kèm FileHandler để lưu toàn bộ nhật ký chạy vào file log_file.
    # Bước 5: Cấu hình định dạng log: [Thời gian - Tên Module - Mức độ]: Thông điệp.
    # Bước 6: Trả về đối tượng logger.
    """
    pass


def log_step(step_name: str) -> Callable:
    """
    LOGIC DECORATOR ĐO THỜI GIAN THỰC THI TỪNG BƯỚC:
    -------------------------------------------------
    # Bước 1: Ghi nhận mốc thời gian bắt đầu trước khi gọi hàm (start_time = time.perf_counter()).
    # Bước 2: In tiêu đề thông báo bắt đầu thực hiện bước step_name.
    # Bước 3: Thực thi hàm mục tiêu.
    # Bước 4: Ghi nhận thời gian kết thúc, tính elapsed_time = end - start.
    # Bước 5: In log hoàn thành kèm thời gian chạy chi tiết (ví dụ: '[HOÀN TẤT] Bước 3 chạy trong 2.45s').
    # Bước 6: Trả về kết quả của hàm gốc.
    """
    pass


def print_section_header(title: str, width: int = 80) -> None:
    """In thanh phân cách và tiêu đề nổi bật giữa các giai đoạn của quy trình huấn luyện."""
    pass

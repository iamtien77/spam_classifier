"""
logger.py - Hệ thống ghi log và công cụ đo đạc hiệu năng thực thi
Dựa trên kiến trúc chuẩn cho dự án Machine Learning
==============================================================
Module này định nghĩa:
1. Bộ logger chuẩn ghi đồng thời ra console (Terminal) và file nhật ký (logs/pipeline.log).
2. Decorator log_step đo lường thời gian thực thi của từng phân đoạn trong workflow.
3. Hàm print_section_header định dạng tiêu đề trực quan giữa các giai đoạn.
"""

from functools import wraps
import logging
from pathlib import Path
import time
from typing import Callable, Optional


def setup_logger(name: str = "spam_classifier", log_file: Optional[Path] = None) -> logging.Logger:
    """
    LOGIC THIẾT LẬP LOGGER CHUẨN:
    ------------------------------
    # Bước 1: Xác định đường dẫn file log (mặc định logs/pipeline.log).
    # Bước 2: Tự động tạo thư mục cha của file log nếu chưa tồn tại (parents=True, exist_ok=True).
    # Bước 3: Lấy đối tượng logger theo tên name và thiết lập mức độ ghi log là logging.INFO.
    # Bước 4: Kiểm tra nếu logger chưa có handlers thì cấu hình:
    #         - StreamHandler để xuất thông điệp trực tiếp ra màn hình Console/Terminal.
    #         - FileHandler (encoding='utf-8') để lưu toàn bộ log thực thi vào file nhật ký.
    #         - Thiết lập Formatter chuẩn: [%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s.
    # Bước 5: Trả về đối tượng logger sẵn sàng sử dụng trong toàn bộ hệ thống.
    """
    pass


def log_step(step_name: str) -> Callable:
    """
    LOGIC DECORATOR ĐO THỜI GIAN THỰC THI TỪNG BƯỚC:
    -------------------------------------------------
    # Bước 1: Khởi tạo decorator bao quanh hàm mục tiêu cần giám sát.
    # Bước 2: In tiêu đề phân đoạn bắt đầu thực thi bước step_name bằng print_section_header.
    # Bước 3: Ghi nhận thời điểm bắt đầu bằng time.perf_counter().
    # Bước 4: Chạy hàm mục tiêu trong khối try/except:
    #         - Nếu thành công: tính thời gian thực thi (elapsed_time = end - start),
    #           ghi log [HOÀN TẤT] kèm thời gian chi tiết (giây).
    #         - Nếu xảy ra lỗi: tính thời gian đến lúc lỗi, ghi log [LỖI] và ném tiếp ngoại lệ (raise).
    # Bước 5: Trả về kết quả thực thi của hàm mục tiêu.
    """
    pass


def print_section_header(title: str, width: int = 80) -> None:
    """
    LOGIC IN TIÊU ĐỀ PHÂN ĐOẠN WORKFLOW:
    ------------------------------------
    # Bước 1: In dòng kẻ phân cách gồm các ký tự '=' với độ rộng width.
    # Bước 2: In tiêu đề phân đoạn in hoa, căn lề đẹp mắt.
    # Bước 3: In dòng kẻ phân cách kết thúc tiêu đề.
    """
    pass

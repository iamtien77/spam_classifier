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
import sys
import time
from typing import Callable, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def setup_logger(name: str = "spam_classifier", log_file: Optional[Path] = None) -> logging.Logger:
    """Thiết lập logger ghi đồng thời Console và File log."""
    if log_file is None:
        log_file = Path("logs/pipeline.log")
    else:
        log_file = Path(log_file)

    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_step(step_name: str) -> Callable:
    """Decorator đo thời gian thực thi của từng phân đoạn pipeline."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("spam_classifier")
            print_section_header(f"BẮT ĐẦU: {step_name}")
            logger.info(f"Đang thực thi: {step_name}...")
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                logger.info(f"[HOÀN TẤT] {step_name} trong {elapsed:.3f} giây.")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.error(f"[LỖI] {step_name} thất bại sau {elapsed:.3f} giây: {e}")
                raise
        return wrapper
    return decorator


def print_section_header(title: str, width: int = 80) -> None:
    """In tiêu đề phân đoạn trực quan."""
    separator = "=" * width
    centered_title = f" {title} ".center(width, "=")
    print(f"\n{separator}\n{centered_title}\n{separator}")


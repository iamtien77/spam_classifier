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
from typing import Any, Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd


def extract_misclassified_samples(
    test_df: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Trích xuất danh sách các mẫu False Positives (FP) và False Negatives (FN)."""
    if isinstance(test_df, list):
        df = pd.DataFrame(test_df)
    else:
        df = test_df.copy()

    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=int).ravel()

    df["actual_label"] = np.where(y_true_arr == 1, "Spam", "Ham")
    df["predicted_label"] = np.where(y_pred_arr == 1, "Spam", "Ham")

    if y_prob is not None:
        probs = np.asarray(y_prob, dtype=float).ravel()
        df["spam_probability"] = np.round(probs, 4)

    fp_mask = (y_true_arr == 0) & (y_pred_arr == 1)
    fn_mask = (y_true_arr == 1) & (y_pred_arr == 0)

    fp_df = df[fp_mask].copy()
    fp_df["error_type"] = "False Positive (Chặn nhầm)"

    fn_df = df[fn_mask].copy()
    fn_df["error_type"] = "False Negative (Lọt lưới)"

    return fp_df, fn_df


def profile_error_patterns(
    fp_df: pd.DataFrame,
    fn_df: pd.DataFrame,
    save_csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Thống kê và phân tích nguyên nhân gây ra các mẫu dự đoán sai."""
    # Giữ schema và ghi CSV cả khi không có lỗi để thay báo cáo lần chạy cũ.
    combined_errors = pd.concat([fp_df, fn_df], ignore_index=True)

    text_col = "text" if "text" in combined_errors.columns else ("raw_text" if "raw_text" in combined_errors.columns else next(iter(combined_errors.columns), "text"))

    suspected_causes = []
    keywords_pattern = re.compile(r"\b(free|win|prize|claim|urgent|cash|call|offer|money|winner)\b", re.IGNORECASE)

    for _, row in combined_errors.iterrows():
        text = str(row.get(text_col, ""))
        err_type = str(row.get("error_type", ""))
        kw_matches = keywords_pattern.findall(text)
        exclamation_count = text.count("!")
        upper_count = sum(1 for c in text if c.isupper())
        total_chars = max(1, len(text))
        upper_ratio = upper_count / total_chars

        cause = []
        if "False Positive" in err_type:
            if kw_matches:
                cause.append(f"Dính từ khóa nhạy cảm ({', '.join(set(kw_matches))})")
            if upper_ratio > 0.2:
                cause.append(f"Tỷ lệ chữ hoa cao ({upper_ratio:.1%})")
            if exclamation_count >= 2:
                cause.append(f"Dấu chấm than ({exclamation_count})")
            if not cause:
                cause.append("Đặc trưng từ vựng trùng lặp với tập Spam")
        else:
            if len(text.split()) < 5:
                cause.append("Email quá ngắn, thiếu từ khóa")
            if not kw_matches:
                cause.append("Spam tinh vi, không dùng từ khóa lộ liễu")
            if not cause:
                cause.append("Xác suất spam dưới ngưỡng quyết định")

        suspected_causes.append("; ".join(cause))

    combined_errors["suspected_cause"] = suspected_causes

    if save_csv_path:
        save_path = Path(save_csv_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        combined_errors.to_csv(save_path, index=False, encoding="utf-8-sig")

    return combined_errors


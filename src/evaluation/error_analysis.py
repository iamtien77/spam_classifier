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
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    if len(test_df) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("test_df, y_true và y_pred phải có cùng số mẫu.")

    df = test_df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    if y_prob is not None:
        probs = np.asarray(y_prob, dtype=float).ravel()
        if len(probs) != len(df):
            raise ValueError("y_prob phải có cùng số mẫu với test_df.")
        df["y_prob"] = probs

    fp_df = df[(df["y_true"] == 0) & (df["y_pred"] == 1)].copy()
    fn_df = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].copy()
    return fp_df, fn_df


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
    fp_df = fp_df.copy()
    fn_df = fn_df.copy()
    fp_df["error_type"] = "FP"
    fn_df["error_type"] = "FN"
    errors = pd.concat([fp_df, fn_df], ignore_index=True)

    # Dùng văn bản gốc để giữ chữ hoa và URL trước bước tiền xử lý.
    text_column = "raw_text" if "raw_text" in errors.columns else "text"
    if text_column in errors.columns:
        texts = errors[text_column].fillna("").astype(str)
    elif errors.empty:
        texts = pd.Series([], dtype=str)
    else:
        raise ValueError("Dữ liệu phải có cột raw_text hoặc text.")

    errors["text_length"] = texts.str.len()
    errors["word_count"] = texts.str.split().str.len()
    errors["uppercase_count"] = [sum(char.isupper() for char in text) for text in texts]
    for keyword in ["free", "win", "call"]:
        errors[f"has_{keyword}"] = texts.str.contains(r"\b" + keyword + r"\b", case=False)
    errors["has_url"] = texts.str.contains(r"https?://|www\.", case=False)
    errors["mean_text_length"] = errors.groupby("error_type")["text_length"].transform("mean")

    # Các lý do dưới đây là dấu hiệu nghi ngờ, không phải kết luận nguyên nhân.
    suspected_reasons = []
    for _, row in errors.iterrows():
        reasons = []
        keywords = [keyword for keyword in ["free", "win", "call"] if row[f"has_{keyword}"]]
        if row["error_type"] == "FP" and keywords:
            reasons.append("Ham chứa từ khóa dễ gây nhầm: " + ", ".join(keywords))
        if row["error_type"] == "FN":
            if row["word_count"] < 5:
                reasons.append("Nội dung ngắn, ít đặc trưng từ vựng")
            if not keywords:
                reasons.append("Thiếu từ khóa spam phổ biến")
        if row["uppercase_count"] >= 3:
            reasons.append("Có nhiều chữ in hoa")
        if row["has_url"]:
            reasons.append("Có liên kết URL")
        suspected_reasons.append("; ".join(reasons) or "Cần xem lại nội dung và đặc trưng mô hình")
    errors["suspected_reason"] = suspected_reasons

    if save_csv_path is not None:
        save_csv_path = Path(save_csv_path)
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        errors.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
    return errors

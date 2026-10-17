"""
predictor.py - Triển khai dịch vụ suy luận phân loại email mới
Chuẩn hóa theo yêu cầu đề bài (ML project: Classifying Spam Emails - Workflow Bước 4: Deployment)
================================================================================================
Đề bài quy định rõ:
"Deploy the trained model to classify new, unseen emails."

Module này đóng gói dịch vụ SpamInferenceService sẵn sàng đưa vào ứng dụng thực tế:
1. Tiếp nhận văn bản email thô mới (Unseen raw email string).
2. Tự động áp dụng toàn bộ chu trình xử lý đã được fit:
   - Làm sạch văn bản (SpamTextProcessor)
   - Trích xuất đặc trưng lai (TF-IDF + Keywords + Numeric stats)
   - Đưa ma trận qua mô hình phân loại tối ưu (Best Classifier)
   - So sánh xác suất với ngưỡng quyết định tối ưu (Best Decision Threshold)
3. Trả về kết quả JSON / Dictionary chi tiết:
   - Nhãn dự đoán: 'SPAM' hoặc 'NOT SPAM / HAM'
   - Điểm số xác suất (Spam Probability [0.0, 1.0])
   - Danh sách các tín hiệu rủi ro bị kích hoạt (Triggered Keywords & Signals).
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from src.models.persistence import load_spam_pipeline


class SpamInferenceService:
    """
    DỊCH VỤ SUY LUẬN VÀ PHÂN LOẠI EMAIL THỜI GIAN THỰC (DEPLOYMENT SERVICE):
    ------------------------------------------------------------------------
    Chịu trách nhiệm nhận chuỗi văn bản email thô mới và trả về dự đoán phân loại tự động.
    """

    def __init__(
        self,
        pipeline_bundle: Optional[Dict[str, Any]] = None,
        bundle_path: Optional[Path] = None,
    ):
        """Khởi tạo dịch vụ từ pipeline bundle đã đóng gói sẵn hoặc nạp từ file .joblib."""
        self.pipeline_bundle = pipeline_bundle
        self.bundle_path = Path(bundle_path) if bundle_path is not None else Path("results/saved_models/spam_pipeline.joblib")
        self.model = None
        self.text_processor = None
        self.feature_builder = None
        self.threshold = 0.5

    def load_service(self) -> "SpamInferenceService":
        """Khởi động dịch vụ và nạp model artifacts."""
        if self.pipeline_bundle is None:
            self.pipeline_bundle = load_spam_pipeline(self.bundle_path)

        model = self.pipeline_bundle.get("model")
        threshold = float(self.pipeline_bundle.get("best_threshold", 0.5))
        feature_builder = self.pipeline_bundle.get("feature_builder")
        if model is None:
            raise ValueError("Không tìm thấy mô hình trong pipeline bundle.")
        if feature_builder is None:
            raise ValueError("Không tìm thấy feature_builder trong pipeline bundle.")
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("best_threshold phải hữu hạn và thuộc [0, 1].")
        # Chỉ công bố trạng thái sau khi bundle đã hợp lệ.
        self.model = model
        self.threshold = threshold
        self.feature_builder = feature_builder
        self.text_processor = self.pipeline_bundle.get("text_processor")
        self.selected_indices = self.pipeline_bundle.get("selected_indices", None)

        return self

    def _extract_signals(self, email_text: str) -> List[str]:
        """Trích xuất danh sách tín hiệu cảnh báo trong email."""
        signals = []
        kw_pattern = re.compile(r"\b(free|win|prize|claim|urgent|cash|call|offer|money|winner|credit|guaranteed|click)\b", re.IGNORECASE)
        found_kw = kw_pattern.findall(email_text)
        if found_kw:
            signals.append(f"Từ khóa nhạy cảm: {', '.join(sorted(set(k.lower() for k in found_kw)))}")

        if re.search(r"https?://\S+|www\.\S+", email_text, re.IGNORECASE):
            signals.append("Chứa liên kết web (URL)")

        exclamations = email_text.count("!")
        if exclamations >= 2:
            signals.append(f"Dấu chấm than lặp lại ({exclamations} lần)")

        dollar_signs = email_text.count("$") + email_text.count("£") + email_text.count("€")
        if dollar_signs > 0:
            signals.append(f"Ký hiệu tiền tệ ({dollar_signs} lần)")

        upper_chars = sum(1 for c in email_text if c.isupper())
        total_chars = max(1, len(email_text))
        if (upper_chars / total_chars) > 0.25 and total_chars > 20:
            signals.append(f"Tỷ lệ chữ in hoa cao ({(upper_chars / total_chars):.1%})")

        return signals

    def predict_single(self, email_text: str) -> Dict[str, Any]:
        """Phân loại một email mới duy nhất."""
        if self.model is None:
            self.load_service()

        signals = self._extract_signals(email_text)

        # Trích xuất đặc trưng
        if self.feature_builder is not None:
            X_input = self.feature_builder.transform([{"text": email_text, "raw_text": email_text}])
            if hasattr(X_input, "to_scipy_csr"):
                X_mat = X_input.to_scipy_csr()
            else:
                X_mat = X_input
            if self.selected_indices is not None:
                sel_idx = np.asarray(self.selected_indices)
                if (sel_idx.ndim != 1 or sel_idx.size == 0
                        or not np.issubdtype(sel_idx.dtype, np.integer)
                        or np.any(sel_idx < 0) or np.any(sel_idx >= X_mat.shape[1])
                        or len(np.unique(sel_idx)) != len(sel_idx)):
                    raise ValueError("selected_indices phải là các chỉ số cột nguyên, hợp lệ và không trùng.")
                X_mat = X_mat[:, sel_idx]
        else:
            raise RuntimeError("Dịch vụ chưa được nạp feature_builder để xử lý văn bản.")

        # Dự đoán xác suất
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_mat)
            prob_spam = float(proba[0, 1])
        elif hasattr(self.model, "decision_function"):
            score = float(self.model.decision_function(X_mat)[0])
            prob_spam = float(1.0 / (1.0 + np.exp(-np.clip(score, -30.0, 30.0))))
        else:
            pred = int(self.model.predict(X_mat)[0])
            prob_spam = float(pred)

        is_spam = bool(prob_spam >= self.threshold)
        label = "SPAM" if is_spam else "NOT SPAM / HAM"

        if prob_spam >= 0.80:
            risk_level = "Rất cao"
        elif is_spam:
            risk_level = "Cao"
        elif prob_spam >= self.threshold * 0.6:
            risk_level = "Trung bình"
        else:
            risk_level = "Thấp"

        return {
            "raw_text": email_text,
            "prediction": label,
            "is_spam": is_spam,
            "spam_probability": round(prob_spam, 4),
            "threshold_used": round(self.threshold, 4),
            "risk_level": risk_level,
            "detected_signals": signals,
        }

    def predict_batch(self, email_list: List[str]) -> pd.DataFrame:
        """Phân loại danh sách nhiều email hàng loạt."""
        results = [self.predict_single(email) for email in email_list]
        return pd.DataFrame(results)


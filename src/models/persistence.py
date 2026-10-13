"""
persistence.py - Lưu trữ và tải Pipeline mô hình hoàn chỉnh
Chuẩn hóa theo yêu cầu đề bài (ML project: Classifying Spam Emails - Workflow Bước 4: Deployment)
================================================================================================
Module này đảm nhiệm:
1. Đóng gói toàn bộ Artifacts của hệ thống thành một file duy nhất (.joblib):
   - Mô hình phân loại tối ưu đã huấn luyện (Trained Classifier)
   - Bộ tiền xử lý văn bản (SpamTextProcessor)
   - Bộ vector hóa đặc trưng (HybridFeatureBuilder / TF-IDF Vectorizers)
   - Bộ chuẩn hóa số học (MaxAbsScaler)
   - Ngưỡng quyết định tối ưu đã chọn (Best Decision Threshold)
   - Metadata cấu hình và danh sách đặc trưng
2. Cung cấp hàm nạp lại pipeline an toàn phục vụ ứng dụng suy luận thời gian thực (Inference Service).
"""

from pathlib import Path
from typing import Any, Dict, Optional
import joblib


def save_spam_pipeline(
    pipeline_bundle: Dict[str, Any],
    save_path: Optional[Path] = None,
) -> Path:
    if save_path is None:
        save_path = Path("results/saved_models/spam_pipeline.joblib")
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline_bundle, save_path, compress=3)
    return save_path


def load_spam_pipeline(
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if model_path is None:
        model_path = Path("results/saved_models/spam_pipeline.joblib")
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file pipeline tại: {model_path}")

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError(f"Dữ liệu pipeline trong {model_path} không đúng định dạng dictionary.")
    return bundle


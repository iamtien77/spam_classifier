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


def archive_run_outputs(directories, archive_root):
    """Copy previous outputs before a new main run; never delete or move originals.

    A failed copy aborts the caller before it starts training/overwriting files.
    An incomplete archive is retained for inspection with no COMPLETE marker.
    """
    import shutil
    from datetime import datetime, timezone
    from uuid import uuid4

    directories = [Path(path) for path in directories]
    populated = [path for path in directories if path.is_dir()
                 and any(item.name != ".gitkeep" for item in path.iterdir())]
    if not populated:
        return None
    destination = Path(archive_root) / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    for source in populated:
        shutil.copytree(source, destination / source.name)
    (destination / "COMPLETE").write_text("Previous outputs copied successfully.\n", encoding="utf-8")
    return destination


def experiment_metadata(config, records):
    """Fingerprint the actual loaded records, configuration and code, not a guessed dataset path."""
    import hashlib
    import json
    import platform
    from dataclasses import asdict
    from datetime import datetime, timezone
    from importlib.metadata import version, PackageNotFoundError

    rows = records.to_dict("records") if hasattr(records, "to_dict") else records
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n").encode("utf-8"))
    packages = {}
    for name in ("numpy", "scipy", "pandas", "joblib", "matplotlib", "seaborn", "scikit-learn"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parents[2]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "packages": packages,
        "config": asdict(config),
        "loaded_records_sha256": digest.hexdigest(), "loaded_rows": len(rows),
        "dataset_origin": "Not supplied; see docs/data_provenance.md",
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [root / "main.py", *sorted((root / "src").rglob("*.py"))]},
    }


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


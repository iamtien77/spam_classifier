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

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import joblib
import numpy as np
from src.config import METRICS_DIR, SAVED_MODELS_DIR


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
        save_path = SAVED_MODELS_DIR / "spam_pipeline.joblib"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline_bundle, save_path, compress=3)
    return save_path


def load_spam_pipeline(
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if model_path is None:
        model_path = SAVED_MODELS_DIR / "spam_pipeline.joblib"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file pipeline tại: {model_path}")

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError(f"Dữ liệu pipeline trong {model_path} không đúng định dạng dictionary.")
    return bundle


def _to_json_native(value: Any) -> Any:
    """Chuyển giá trị NumPy sang kiểu Python thuần để tuần tự hóa JSON an toàn."""
    if isinstance(value, dict):
        return {str(k): _to_json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_json_native(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _serialize_model_weights_entry(name: str, model: Any) -> Dict[str, Any]:
    """Chuẩn hóa thông tin weights trước/trong/sau train của một mô hình."""
    entry: Dict[str, Any] = {
        "branch": "Tuned" if name.endswith(" (Tuned)") else "Default",
        "tuning_params": _to_json_native(getattr(model, "tuning_params_", {})),
    }
    recorded = getattr(model, "weight_history_", None)
    if isinstance(recorded, dict):
        entry.update(_to_json_native(recorded))
        return entry

    if hasattr(model, "weights_") and getattr(model, "weights_", None) is not None:
        w = np.asarray(model.weights_, dtype=float)
        entry.update({
            "model_class": type(model).__name__,
            "training_mode": "iterative_gradient_descent",
            "n_features": int(len(w)),
            "before_train": None,
            "during_train": {
                "note": "Weight snapshots during fit were not recorded for this model.",
                "per_iteration_metrics": [],
                "checkpoint_snapshots": [],
            },
            "after_train": {
                "bias": float(getattr(model, "bias_", 0.0)),
                "weight_l2_norm": float(np.linalg.norm(w)),
                "weights": [float(v) for v in w],
            },
        })
        return entry

    if hasattr(model, "feature_log_prob_") and getattr(model, "feature_log_prob_", None) is not None:
        effective_w, effective_b = (
            model.effective_linear_parameters()
            if hasattr(model, "effective_linear_parameters")
            else (None, None)
        )
        log_prob = np.asarray(model.feature_log_prob_, dtype=float)
        log_prior = np.asarray(model.class_log_prior_, dtype=float)
        entry.update({
            "model_class": type(model).__name__,
            "training_mode": "closed_form_probabilistic",
            "n_features": int(log_prob.shape[1]),
            "before_train": None,
            "during_train": {
                "note": "Closed-form Naive Bayes; weight history not recorded during fit.",
                "per_iteration_metrics": [],
                "checkpoint_snapshots": [],
            },
            "after_train": {
                "class_log_prior": {"0_ham": float(log_prior[0]), "1_spam": float(log_prior[1])},
                "feature_log_prob": {
                    "0_ham": [float(v) for v in log_prob[0]],
                    "1_spam": [float(v) for v in log_prob[1]],
                },
                "effective_linear_weights": [float(v) for v in effective_w] if effective_w is not None else None,
                "effective_bias": effective_b,
            },
        })
        return entry

    entry.update({
        "model_class": type(model).__name__,
        "training_mode": "unknown",
        "before_train": None,
        "during_train": {"per_iteration_metrics": [], "checkpoint_snapshots": []},
        "after_train": None,
    })
    return entry


def build_weights_history_report(
    models: Dict[str, Any],
    ensembles: Optional[Dict[str, Any]] = None,
    feature_names: Optional[Sequence[str]] = None,
    selected_winner: Optional[str] = None,
    best_threshold: Optional[float] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Tạo cấu trúc JSON lưu toàn bộ weights trước, trong và sau khi huấn luyện."""
    if not models:
        raise ValueError("models không được rỗng khi xuất báo cáo weights.")

    ensemble_entries: Dict[str, Any] = {}
    for ens_name, ens in (ensembles or {}).items():
        if hasattr(ens, "_validated_weights"):
            raw_weights, total_weight = ens._validated_weights()
        else:
            raw_weights = {m: float(w) for m, w in getattr(ens, "weights", {}).items()}
            total_weight = sum(raw_weights.values()) or 1.0
        ensemble_entries[ens_name] = {
            "model_class": type(ens).__name__,
            "voting": getattr(ens, "voting", "soft"),
            "members": list(getattr(ens, "models", {}).keys()),
            "raw_weights": {k: float(v) for k, v in raw_weights.items()},
            "normalized_weights": {k: float(v / total_weight) for k, v in raw_weights.items()},
        }

    metadata_summary = None
    if isinstance(run_metadata, dict):
        metadata_summary = {
            "created_at_utc": run_metadata.get("created_at_utc"),
            "loaded_rows": run_metadata.get("loaded_rows"),
            "loaded_records_sha256": run_metadata.get("loaded_records_sha256"),
        }

    return {
        "experiment": "main.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_metadata_summary": metadata_summary,
        "selected_winner": selected_winner,
        "best_threshold": float(best_threshold) if best_threshold is not None else None,
        "n_features": len(feature_names) if feature_names is not None else None,
        "feature_names": [str(name) for name in feature_names] if feature_names is not None else None,
        "models": {
            name: _serialize_model_weights_entry(name, model)
            for name, model in models.items()
        },
        "ensembles": ensemble_entries,
    }


def save_weights_history_json(
    report: Dict[str, Any],
    save_path: Optional[Path] = None,
) -> Path:
    """Ghi file JSON lưu lịch sử weights với chuẩn JSON nghiêm ngặt (không chứa NaN/Inf)."""
    target = METRICS_DIR / "model_weights_history.json" if save_path is None else Path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    target.write_text(payload, encoding="utf-8")
    return target


"""Persistence helpers for saving and loading a trained spam-classification pipeline."""
from pathlib import Path
from typing import Any, Dict, Optional
import joblib


def _default_path() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "saved_models" / "spam_pipeline.joblib"


def save_spam_pipeline(pipeline_bundle: Dict[str, Any], save_path: Optional[Path] = None) -> Path:
    """Validate and persist a pipeline bundle using joblib."""
    if not isinstance(pipeline_bundle, dict) or not pipeline_bundle:
        raise ValueError("pipeline_bundle must be a non-empty dictionary")
    required_groups = ({"model", "final_selected_model", "classifier"},
                       {"best_threshold", "threshold"})
    missing = []
    for options in required_groups:
        if not any(key in pipeline_bundle for key in options):
            missing.append(" or ".join(sorted(options)))
    if missing:
        raise ValueError("pipeline_bundle is missing required key(s): " + ", ".join(missing))
    path = Path(save_path) if save_path is not None else _default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline_bundle, path, compress=3)
    return path


def load_spam_pipeline(model_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and validate a previously saved pipeline bundle."""
    path = Path(model_path) if model_path is not None else _default_path()
    if not path.exists():
        raise FileNotFoundError(f"Spam pipeline was not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("Saved pipeline must contain a dictionary bundle")
    if not any(key in bundle for key in ("model", "final_selected_model", "classifier")):
        raise ValueError("Saved pipeline does not contain a model")
    if not any(key in bundle for key in ("best_threshold", "threshold")):
        raise ValueError("Saved pipeline does not contain a decision threshold")
    return bundle

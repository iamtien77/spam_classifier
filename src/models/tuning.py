"""Hyperparameter tuning, threshold optimization, and validation model selection."""
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score


def _binary_metrics(y_true, y_pred, beta=2.0):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f_beta": float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)),
    }


class ThresholdOptimizerScratch:
    """Scan probability thresholds and choose the best threshold meeting target recall."""
    def __init__(self, recall_target: float = 0.85, beta: float = 2.0, n_points: int = 200):
        if not 0 <= recall_target <= 1 or beta <= 0 or n_points < 2:
            raise ValueError("Invalid recall_target, beta, or n_points")
        self.recall_target = float(recall_target)
        self.beta = float(beta)
        self.grid = np.linspace(0.005, 1.0, int(n_points))
        self.best_threshold_: Optional[float] = None
        self.best_metrics_: Optional[Dict[str, float]] = None

    def compute_metrics_at_threshold(self, y_true: np.ndarray, spam_probs: np.ndarray, threshold: float) -> Dict[str, float]:
        y = np.asarray(y_true).ravel().astype(int)
        p = np.asarray(spam_probs).ravel().astype(float)
        if len(y) != len(p):
            raise ValueError("y_true and spam_probs must have the same length")
        if len(y) == 0 or not np.all(np.isfinite(p)):
            raise ValueError("Inputs must be non-empty and probabilities finite")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        pred = (p >= threshold).astype(int)
        m = _binary_metrics(y, pred, self.beta)
        m["threshold"] = float(threshold)
        tn = int(np.sum((y == 0) & (pred == 0)))
        fp = int(np.sum((y == 0) & (pred == 1)))
        fn = int(np.sum((y == 1) & (pred == 0)))
        tp = int(np.sum((y == 1) & (pred == 1)))
        m.update({"tp": tp, "fp": fp, "tn": tn, "fn": fn})
        return m

    def scan_thresholds(self, y_true: np.ndarray, spam_probs: np.ndarray) -> pd.DataFrame:
        rows = []
        for threshold in self.grid:
            row = self.compute_metrics_at_threshold(y_true, spam_probs, float(threshold))
            row["target_met"] = bool(row["recall"] >= self.recall_target)
            rows.append(row)
        return pd.DataFrame(rows)

    def select_best_threshold(self, scan_df: pd.DataFrame) -> Dict[str, Any]:
        if scan_df is None or scan_df.empty:
            raise ValueError("scan_df must contain threshold results")
        valid = scan_df[scan_df["target_met"].astype(bool)] if "target_met" in scan_df else scan_df[scan_df["recall"] >= self.recall_target]
        target_met = not valid.empty
        if valid.empty:
            # Safety fallback: maximize recall first, then F-beta / F1 and precision.
            valid = scan_df.sort_values(["recall", "f_beta", "f1", "precision"], ascending=False).head(1)
        else:
            valid = valid.sort_values(["f_beta", "f1", "accuracy", "precision"], ascending=False)
        best = valid.iloc[0].to_dict()
        best["best_threshold"] = float(best.get("threshold", 0.5))
        best["target_met"] = bool(target_met)
        self.best_threshold_ = best["best_threshold"]
        self.best_metrics_ = {k: float(v) for k, v in best.items() if k in {"accuracy", "precision", "recall", "f1", "f_beta"}}
        return best


def tune_model_hyperparameters(
    model_name: str,
    model_class: Any,
    param_grid: Dict[str, List[Any]],
    X_train: Union[np.ndarray, csr_matrix],
    y_train: np.ndarray,
    X_val: Union[np.ndarray, csr_matrix],
    y_val: np.ndarray,
    scoring_metric: str = "f1",
) -> Tuple[Any, Dict[str, Any], float]:
    """Manual grid search; each candidate is fitted only on Train and scored on Validation."""
    from sklearn.base import clone
    from sklearn.metrics import get_scorer
    y_train = np.asarray(y_train).ravel()
    y_val = np.asarray(y_val).ravel()
    if len(y_train) != X_train.shape[0] or len(y_val) != X_val.shape[0]:
        raise ValueError("X and y lengths do not match")
    keys = list(param_grid)
    combos = [dict(zip(keys, values)) for values in product(*(param_grid[k] for k in keys))] if keys else [{}]
    best_model, best_params, best_score = None, {}, -np.inf
    valid_metrics = {"accuracy", "precision", "recall", "f1", "f_beta", "roc_auc", "f1_macro", "balanced_accuracy"}
    for params in combos:
        try:
            model = model_class(**params)
        except TypeError:
            model = clone(model_class).set_params(**params)
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            probs = np.asarray(model.predict_proba(X_val))[:, 1]
            pred_default = (probs >= 0.5).astype(int)
        else:
            scores = np.asarray(model.decision_function(X_val)).ravel()
            probs = None
            pred_default = (scores >= 0).astype(int)
        metric = scoring_metric.lower().replace("-", "_")
        if metric in {"roc_auc", "auc"}:
            score = roc_auc_score(y_val, probs if probs is not None else scores)
        elif metric == "f_beta":
            score = fbeta_score(y_val, pred_default, beta=2, zero_division=0)
        elif metric in valid_metrics:
            if metric == "balanced_accuracy":
                from sklearn.metrics import balanced_accuracy_score
                score = balanced_accuracy_score(y_val, pred_default)
            elif metric == "f1_macro":
                score = f1_score(y_val, pred_default, average="macro", zero_division=0)
            else:
                fn = {"accuracy": accuracy_score, "precision": precision_score, "recall": recall_score, "f1": f1_score}[metric]
                score = fn(y_val, pred_default, zero_division=0) if metric != "accuracy" else fn(y_val, pred_default)
        else:
            try:
                score = get_scorer(scoring_metric)(model, X_val, y_val)
            except Exception as exc:
                raise ValueError(f"Unsupported scoring_metric: {scoring_metric}") from exc
        score = float(score)
        if score > best_score:
            best_model, best_params, best_score = model, dict(params), score
    if best_model is None:
        raise RuntimeError(f"No valid model candidate for {model_name}")
    return best_model, best_params, float(best_score)


class ModelSelectionWorkflow:
    """Aggregate validation metrics and select the strongest candidate with recall priority."""
    def __init__(self, recall_target: float = 0.85):
        self.recall_target = float(recall_target)

    def compare_candidates(self, candidate_results: List[Dict[str, Any]]) -> pd.DataFrame:
        if not candidate_results:
            return pd.DataFrame(columns=["model_name", "recall", "f1", "f_beta", "target_met", "rank"])
        rows = []
        for item in candidate_results:
            row = dict(item)
            # Accept common naming differences in results emitted by notebooks.
            aliases = {"model": "model_name", "name": "model_name", "threshold": "best_threshold",
                       "accuracy_score": "accuracy", "precision_score": "precision", "recall_score": "recall",
                       "f1_score": "f1", "fbeta": "f_beta", "f_beta_score": "f_beta", "roc_auc_score": "roc_auc"}
            for old, new in aliases.items():
                if new not in row and old in row:
                    row[new] = row[old]
            row.setdefault("model_name", row.get("key", "unknown"))
            row.setdefault("recall", 0.0)
            row.setdefault("f1", 0.0)
            row.setdefault("f_beta", 0.0)
            row["target_met"] = bool(float(row["recall"]) >= self.recall_target)
            rows.append(row)
        df = pd.DataFrame(rows)
        for col in ("accuracy", "precision", "recall", "f1", "f_beta", "roc_auc"):
            if col not in df:
                df[col] = np.nan
        df = df.sort_values(["target_met", "f_beta", "f1", "accuracy"], ascending=[False, False, False, False], na_position="last").reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        return df

    def select_final_winner(self, comparison_df: pd.DataFrame) -> str:
        if comparison_df is None or comparison_df.empty:
            raise ValueError("comparison_df is empty")
        first = comparison_df.iloc[0]
        return str(first.get("model_name", first.get("model", first.get("name", "unknown"))))

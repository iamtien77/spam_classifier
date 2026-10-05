"""
config.py - Cấu hình trung tâm cho dự án Spam Email Classifier
Chuẩn hóa theo yêu cầu đề bài (ML project: Classifying Spam Emails)
===================================================================
Đề bài yêu cầu:
- 3 mô hình cốt lõi: Logistic Regression, Support Vector Machines (SVM), Naive Bayes
- Workflow: Tiền xử lý, Trích xuất đặc trưng (TF-IDF, word freq, char freq),
            Huấn luyện mô hình, Đánh giá (Accuracy, Precision, Recall, F1), Triển khai.
- Mở rộng: Feature Engineering, Hyperparameter Tuning, Ensemble Methods.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ==============================================================================
# 1. ĐƯỜNG DẪN DỰ ÁN & DỮ LIỆU (PATHS & DIRECTORIES)
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Thư mục dữ liệu
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# File dữ liệu
RAW_DATA_FILE = RAW_DATA_DIR / "spam.csv"
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "spam_clean.csv"

# Thư mục kết quả đầu ra
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
SAVED_MODELS_DIR = RESULTS_DIR / "saved_models"

# Thư mục logs
LOGS_DIR = BASE_DIR / "logs"


def ensure_directories() -> None:
    """Tự động khởi tạo tất cả các thư mục cần thiết trong hệ thống."""
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR, SAVED_MODELS_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. THAM SỐ PHÂN CHIA DỮ LIỆU (DATA SPLIT CONFIG)
# ==============================================================================

RANDOM_STATE: int = 42

# Tỷ lệ phân chia tập dữ liệu: Train (80%) - Validation (10%) - Test (10%)
TRAIN_RATIO: float = 0.80
VAL_RATIO: float = 0.10
TEST_RATIO: float = 0.10

# Nhãn phân loại
LABEL_MAP = {"ham": 0, "spam": 1}
TARGET_NAMES = ["Not Spam (Ham)", "Spam"]


# ==============================================================================
# 3. CẤU HÌNH TRÍCH XUẤT ĐẶC TRƯNG VĂN BẢN (FEATURE ENGINEERING)
# ==============================================================================

TEXT_COLUMN_FOR_MODEL: str = "model_text"
TEXT_FEATURE_MODE: str = "log_count"   # 'count', 'binary', 'log_count'
MIN_DF: int = 2
MAX_UNIGRAM_FEATURES: int = 3000
MAX_BIGRAM_FEATURES: int = 800
MAX_CHAR_FEATURES: int = 500
CHAR_NGRAM_RANGE: Tuple[int, int] = (3, 5)

# Từ khóa nhận diện tín hiệu spam phổ biến
KEYWORD_PATTERNS: Dict[str, str] = {
    "kw_free": r"\bfree\b",
    "kw_win": r"\b(?:win|winner|won|winning)\b",
    "kw_prize": r"\b(?:prize|reward|gift|bonus)\b",
    "kw_claim": r"\b(?:claim|redeem|collect)\b",
    "kw_urgent": r"\b(?:urgent|immediately|act now|important)\b",
    "kw_call": r"\b(?:call|txt|phone|ring)\b",
    "kw_cash": r"\b(?:cash|money|dollar|\$|pound|credit)\b",
    "kw_guarantee": r"\b(?:guarantee|guaranteed)\b",
    "kw_offer": r"\b(?:offer|discount|promo|deal)\b",
    "kw_click": r"\b(?:click|link|visit|website)\b",
}

# Cấu hình các đặc trưng thống kê số học (character frequency, lengths, etc.)
NUMERIC_FEATURE_CONFIG: List[Tuple[str, str]] = [
    ("has_url", "binary"),
    ("has_number", "binary"),
    ("digit_count", "log_minmax"),
    ("special_char_count", "log_minmax"),
    ("exclamation_count", "log_minmax"),
    ("dollar_count", "log_minmax"),
    ("uppercase_ratio", "minmax"),
    ("model_length", "log_minmax"),
]


# ==============================================================================
# 4. THIẾT LẬP MỤC TIÊU ĐÁNH GIÁ & NGƯỠNG (EVALUATION & THRESHOLD)
# ==============================================================================

RECALL_TARGET: float = 0.85
FBETA_BETA: float = 2.0
THRESHOLD_GRID_POINTS: int = 200


# ==============================================================================
# 5. CẤU HÌNH CÁC MÔ HÌNH YÊU CẦU THEO ĐỀ BÀI (LOGISTIC REGRESSION, SVM, NAIVE BAYES)
# ==============================================================================

DEFAULT_MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 300,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    },
    "svm": {
        "C": 1.0,
        "max_iter": 300,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    },
    "naive_bayes": {
        "alpha": 1.0,
    },
}


# ==============================================================================
# 6. EXPERIMENT CONFIG DATACLASS
# ==============================================================================

@dataclass
class SpamExperimentConfig:
    """Dataclass điều phối cấu hình thí nghiệm bài toán phân loại email spam."""
    random_state: int = RANDOM_STATE
    text_column_for_model: str = TEXT_COLUMN_FOR_MODEL
    text_feature_mode: str = TEXT_FEATURE_MODE
    min_df: int = MIN_DF
    max_unigram_features: int = MAX_UNIGRAM_FEATURES
    max_bigram_features: int = MAX_BIGRAM_FEATURES
    max_char_features: int = MAX_CHAR_FEATURES
    char_ngram_range: Tuple[int, int] = CHAR_NGRAM_RANGE
    recall_target: float = RECALL_TARGET
    keyword_patterns: Dict[str, str] = field(default_factory=lambda: KEYWORD_PATTERNS.copy())
    numeric_feature_config: List[Tuple[str, str]] = field(default_factory=lambda: list(NUMERIC_FEATURE_CONFIG))
    default_tuning_params: Dict[str, Any] = field(default_factory=dict)

    def completed_params(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Hợp nhất cấu hình tùy chỉnh vào tham số mặc định."""
        completed = dict(self.default_tuning_params)
        if params:
            completed.update(params)

        completed.setdefault("alpha", 1.0)
        completed.setdefault("threshold", 0.5)
        completed.setdefault("min_df", self.min_df)
        completed.setdefault("max_unigram_features", self.max_unigram_features)
        completed.setdefault("use_bigram", True)
        completed.setdefault("max_bigram_features", self.max_bigram_features)
        completed.setdefault("use_char_ngram", True)
        completed.setdefault("max_char_features", self.max_char_features)
        completed.setdefault("text_feature_mode", self.text_feature_mode)
        completed.setdefault("use_keyword_features", True)
        completed.setdefault("keyword_group_weight", 1.0)
        completed.setdefault("use_numeric_features", True)
        completed.setdefault("numeric_group_weight", 1.0)
        completed.setdefault("use_length_cap", True)
        completed.setdefault("cap_quantile", 0.99)

        completed["max_unigram_features"] = int(completed["max_unigram_features"])
        completed["min_df"] = int(completed["min_df"])
        completed["use_bigram"] = bool(completed["use_bigram"])
        completed["use_char_ngram"] = bool(completed.get("use_char_ngram", False))
        completed["use_keyword_features"] = bool(completed["use_keyword_features"])
        completed["use_numeric_features"] = bool(completed["use_numeric_features"])
        completed["use_length_cap"] = bool(completed["use_length_cap"])
        completed["max_bigram_features"] = int(completed["max_bigram_features"]) if completed["use_bigram"] else 0
        completed["max_char_features"] = int(completed["max_char_features"]) if completed["use_char_ngram"] else 0
        completed["keyword_group_weight"] = float(completed["keyword_group_weight"]) if completed["use_keyword_features"] else 0.0
        completed["numeric_group_weight"] = float(completed["numeric_group_weight"]) if completed["use_numeric_features"] else 0.0
        completed["cap_quantile"] = float(completed["cap_quantile"]) if completed["use_length_cap"] and completed["cap_quantile"] is not None else None
        completed["alpha"] = float(completed["alpha"])
        completed["threshold"] = float(completed["threshold"])
        completed["text_feature_mode"] = str(completed["text_feature_mode"])
        return completed

"""
config.py - Cấu hình trung tâm cho dự án Spam Email Classifier
Dựa trên logic từ Classification_email_spam.ipynb
==============================================================
Module này định nghĩa toàn bộ siêu tham số, đường dẫn, từ khóa regex,
và dataclass SpamExperimentConfig điều phối tham số huấn luyện & trích xuất đặc trưng.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# ==============================================================================
# 1. ĐƯỜNG DẪN DỰ ÁN & DỮ LIỆU (PATHS & DIRECTORIES)
# ==============================================================================

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# Thư mục dữ liệu
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# File dữ liệu text thô (spam.csv gồm 2 cột chính: label ['ham', 'spam'] và text)
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
# 2. THAM SỐ CHUNG & TẬP CHIA (DATA SPLIT CONFIG)
# ==============================================================================

RANDOM_STATE: int = 42

# Tỷ lệ phân chia tập dữ liệu (Stratified Split 80/10/10)
TRAIN_RATIO: float = 0.80
VAL_RATIO: float = 0.10
TEST_RATIO: float = 0.10

# Nhãn phân loại
LABEL_MAP = {"ham": 0, "spam": 1}
TARGET_NAMES = ["Ham", "Spam"]


# ==============================================================================
# 3. CẤU HÌNH TRÍCH XUẤT ĐẶC TRƯNG VĂN BẢN (NLP & N-GRAMS)
# ==============================================================================

TEXT_COLUMN_FOR_MODEL: str = "model_text"
TEXT_FEATURE_MODE: str = "log_count"   # Chế độ biến đổi ma trận đếm: 'count', 'binary', 'log_count'
MIN_DF: int = 2
MAX_UNIGRAM_FEATURES: int = 4000
MAX_BIGRAM_FEATURES: int = 800
MAX_CHAR_FEATURES: int = 800
CHAR_NGRAM_RANGE: Tuple[int, int] = (3, 5)


# ==============================================================================
# 4. TỪ KHÓA ĐẶC TRƯNG SPAM & TÍN HIỆU SỐ HỌC (KEYWORDS & NUMERIC SIGNALS)
# ==============================================================================

# Regex nhận diện các từ khóa/cụm từ báo hiệu spam phổ biến
KEYWORD_PATTERNS: Dict[str, str] = {
    "kw_free": r"\bfree\b",
    "kw_win": r"\b(?:win|winner|won|winning)\b",
    "kw_prize": r"\b(?:prize|reward|gift|bonus)\b",
    "kw_claim": r"\b(?:claim|redeem|collect)\b",
    "kw_urgent": r"\b(?:urgent|immediately|act now|important)\b",
    "kw_call": r"\b(?:call|txt|phone|ring)\b",
    "kw_cash": r"\b(?:cash|money|dollar|\$|pound|credit)\b",
    "kw_guarantee": r"\b(?:guarantee|guaranteed)\b",
}

# Cấu hình các đặc trưng thống kê số học trích xuất từ nội dung email
NUMERIC_FEATURE_CONFIG: List[str] = [
    "num_length",              # Chiều dài văn bản
    "num_word_count",          # Tổng số từ
    "num_uppercase_ratio",     # Tỷ lệ chữ cái in hoa
    "num_exclamation_count",   # Số lượng dấu chấm than (!)
    "num_question_count",      # Số lượng dấu chấm hỏi (?)
    "num_digit_count",         # Số lượng ký tự số
    "num_url_count",           # Số lượng liên kết URL (http/https/www)
    "num_phone_count",         # Số lượng số điện thoại / chuỗi số dài
]


# ==============================================================================
# 5. MỤC TIÊU NGHIỆP VỤ & TỐI ƯU NGƯỠNG (BUSINESS RECALL & THRESHOLD TUNING)
# ==============================================================================

# Ràng buộc nghiệp vụ: Bắt tối thiểu 85% email spam (Recall >= 0.85) để bảo vệ người dùng
RECALL_TARGET: float = 0.85

# Hệ số Beta trong F-beta Score (Beta=2.0 coi trọng Recall gấp đôi so với Precision)
FBETA_BETA: float = 2.0

# Lưới quét ngưỡng phân loại (200 điểm từ 0.005 đến 1.0)
THRESHOLD_GRID_POINTS: int = 200


# ==============================================================================
# 6. THAM SỐ QUÉT CHỌN ĐẶC TRƯNG BẰNG SHAP (FEATURE SELECTION)
# ==============================================================================

SHAP_PRESELECT_TARGET: int = 3000
SHAP_TOP_K_CANDIDATES: List[int] = [300, 500, 800, 1200, 1600, 2000, 2500]


# ==============================================================================
# 7. THAM SỐ TỐI ƯU SIÊU THAM SỐ VỚI OPTUNA (HYPERPARAMETER TUNING)
# ==============================================================================

NB_OPTUNA_TRIALS: int = 15      # Số lần thử nghiệm tối ưu cho Complement Naive Bayes
SVM_OPTUNA_TRIALS: int = 8      # Số lần thử nghiệm tối ưu cho Linear SVM


# ==============================================================================
# 8. EXPERIMENT CONFIG DATACLASS (ĐIỀU PHỐI TẬP TRUNG TOÀN BỘ WORKFLOW)
# ==============================================================================

@dataclass
class SpamExperimentConfig:
    """
    Dataclass lưu trữ và điều phối toàn bộ tham số cấu hình cho pipeline:
    từ tiền xử lý, n-gram, signal patterns, cho tới hyperparameter tuning.
    """
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
    numeric_feature_config: List[str] = field(default_factory=lambda: list(NUMERIC_FEATURE_CONFIG))
    default_tuning_params: Dict = field(default_factory=dict)

    def completed_params(self, params: Dict = None) -> Dict:
        """
        Gộp các tham số cấu hình tùy chỉnh vào tham số mặc định của thí nghiệm:
        - alpha: hệ số làm mịn Laplace/Lidstone cho Naive Bayes
        - threshold: ngưỡng quyết định nhị phân
        - use_bigram, use_char_ngram, use_keyword_features, use_numeric_features: bật/tắt nhóm đặc trưng
        - cap_quantile: giới hạn độ dài từ ngữ (mặc định 0.99)
        """
        pass

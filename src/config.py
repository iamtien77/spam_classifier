"""
config.py - Cấu hình trung tâm cho toàn bộ dự án Spam Email Classifier

Chứa các thiết lập:
- Đường dẫn thư mục & file (sử dụng pathlib.Path)
- URL tải dữ liệu Spambase (UCI Machine Learning Repository)
- Danh sách 57 tên đặc trưng (features) và nhãn phân loại (target)
- Tham số chung cho tiền xử lý, huấn luyện và cross-validation
- Lưới siêu tham số (Hyperparameter grids) cho GridSearchCV
"""

from pathlib import Path


# 1. ĐƯỜNG DẪN DỰ ÁN & DỮ LIỆU (PATHS & DIRECTORIES)


# Thư mục gốc của dự án (spam_classifier/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Thư mục dữ liệu
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# File dữ liệu
RAW_DATA_FILE = RAW_DATA_DIR / "spambase.data"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "spambase_clean.csv"

# Thư mục kết quả đầu ra
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
SAVED_MODELS_DIR = RESULTS_DIR / "saved_models"

# Thư mục logs
LOGS_DIR = BASE_DIR / "logs"

# URL tải dataset Spambase từ UCI Machine Learning Repository
UCI_SPAMBASE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
)
UCI_SPAMBASE_NAMES_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names"
)


def ensure_directories():
    """Tự động khởi tạo tất cả các thư mục cần thiết nếu chưa tồn tại."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        SAVED_MODELS_DIR,
        LOGS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)



# 2. ĐẶC TRƯNG DỮ LIỆU SPAMBASE (57 FEATURES + 1 TARGET)


# 48 đặc trưng tần suất từ (word_freq_WORD)
WORD_FREQ_FEATURES = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
]

# 6 đặc trưng tần suất ký tự đặc biệt (char_freq_CHAR)
CHAR_FREQ_FEATURES = [
    "char_freq_;",
    "char_freq_(",
    "char_freq_[",
    "char_freq_!",
    "char_freq_$",
    "char_freq_#",
]

# 3 đặc trưng chuỗi ký tự viết hoa liên tiếp (capital_run_length)
CAPITAL_RUN_FEATURES = [
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
]

# Toàn bộ 57 đặc trưng đầu vào
FEATURE_NAMES = WORD_FREQ_FEATURES + CHAR_FREQ_FEATURES + CAPITAL_RUN_FEATURES

# Tên cột mục tiêu (1 = Spam, 0 = Non-Spam / Ham)
TARGET_COLUMN = "is_spam"

# Toàn bộ danh sách 58 cột (57 features + 1 label) dùng khi đọc file gốc chưa có header
ALL_COLUMNS = FEATURE_NAMES + [TARGET_COLUMN]

# Label mapping để hiển thị trực quan
LABEL_MAP = {
    0: "Not Spam (Ham)",
    1: "Spam",
}


# ==============================================================================
# 3. THAM SỐ HUẤN LUYỆN & ĐÁNH GIÁ CHUNG (GENERAL ML PARAMETERS)
# ==============================================================================

# Chia tập dữ liệu Train / Test
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cross-Validation
CV_FOLDS = 5
SCORING_METRIC = "f1"

# Feature Selection & Feature Engineering
K_BEST_FEATURES = 30       # Số đặc trưng chọn lọc qua SelectKBest (ANOVA F-test)
PCA_N_COMPONENTS = 20      # Số thành phần chính giữ lại khi giảm chiều PCA


# ==============================================================================
# 4. LƯỚI SIÊU THAM SỐ (HYPERPARAMETER GRIDS CHO GRIDSEARCHCV)
# ==============================================================================

# Logistic Regression
PARAM_GRID_LOGISTIC_REGRESSION = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"],
    "max_iter": [500],
}

# Support Vector Machine (SVC)
PARAM_GRID_SVM = {
    "C": [0.1, 1.0, 10.0, 100.0],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto", 0.01, 0.1],
}

# Gaussian Naive Bayes (var_smoothing từ 1.0 đến 1e-9)
PARAM_GRID_NAIVE_BAYES = {
    "var_smoothing": [10.0 ** (-i) for i in range(10)],
}

# Random Forest Classifier
PARAM_GRID_RANDOM_FOREST = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# Gradient Boosting Classifier
PARAM_GRID_GRADIENT_BOOSTING = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
}

# Từ điển gom nhóm tất cả param grids
HYPERPARAMETER_GRIDS = {
    "Logistic Regression": PARAM_GRID_LOGISTIC_REGRESSION,
    "SVM": PARAM_GRID_SVM,
    "Naive Bayes": PARAM_GRID_NAIVE_BAYES,
    "Random Forest": PARAM_GRID_RANDOM_FOREST,
    "Gradient Boosting": PARAM_GRID_GRADIENT_BOOSTING,
}

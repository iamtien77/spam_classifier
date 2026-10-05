# 📧 Spam Email Classifier (From-Scratch NLP Pipeline)

## 1. Mô tả bài toán

Xây dựng hệ sinh thái Machine Learning hoàn chỉnh để **phân loại tự động email/tin nhắn** là **"spam"** (thư rác) hay **"ham"** (thư hợp lệ).

Hệ thống được thiết kế theo triết lý **"From Scratch Compliance"** (tự cài đặt các thuật toán cốt lõi từ đầu) dựa trên nghiên cứu và thực nghiệm trong [Classification_email_spam.ipynb](file:///e:/Hoc_May_Tren_truong/spam_classifier/Classification_email_spam.ipynb).

### Dữ liệu đầu vào
- **Tập dữ liệu**: `spam.csv` (chứa các mẫu email/SMS thực tế với 2 trường cốt lõi: `label` ['ham', 'spam'] và `text`).
- **Đặc trưng bài toán**: Dữ liệu văn bản phi cấu trúc (unstructured text) và có tính chất **mất cân bằng lớp cao (Imbalanced Data)**: ~86.6% Ham và ~13.4% Spam.

### Mục tiêu kỹ thuật & Nghiệp vụ cốt lõi
1. **Ràng buộc nghiệp vụ**: Đảm bảo **Recall >= 0.85** (bắt trúng tối thiểu 85% email spam để bảo vệ hộp thư người dùng) trong khi vẫn kiểm soát tối đa tỷ lệ báo động giả False Positive.
2. **Triết lý From-Scratch**: Tự cài đặt các thành phần NLP, tiền xử lý, mô hình hóa và đánh giá:
   - `TfidfVectorizerScratch`: Vector hóa TF-IDF hỗ trợ word/char n-grams.
   - `MaxAbsScalerScratch`: Chuẩn hóa độ lớn cực đại bảo toàn tính thưa (sparsity) của ma trận.
   - `ComplementNaiveBayes`: Naive Bayes phần bù chuyên trị dữ liệu văn bản mất cân bằng lớp.
   - `LinearSVMFromScratch`: SVM tuyến tính tối ưu bằng SGD/Pegasos với hàm mất mát Hinge Loss và Class Weighting.
   - `ThresholdOptimizerScratch`: Tự động dò tìm ngưỡng quyết định tối ưu trên tập Validation với chỉ số F-beta (beta=2.0).
   - `SpamModelEvaluator`: Tự tính Confusion Matrix, ROC-AUC (quy tắc hình thang), MCC, Cohen's Kappa.
3. **Cổng kiểm chứng đặc trưng SHAP (SHAP Feature Selection)**: Dò quét Top-K đặc trưng quan trọng nhất để tinh giản không gian chiều.

---

## 2. Luồng thực thi tổng thể (End-to-End Workflow)

```
[spam.csv] ──> [Early Stratified Split (80/10/10)]
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
   [Train DF]       [Val DF]       [Test DF]
       │               │               │
       ├───────────────┴───────────────┤
       ▼
 [Data Cleaning & Text Preprocessing (Unicode NFKD, Regex tokens, Quantile capping)]
       │
       ▼
 [Hybrid Feature Engineering (Word TF-IDF + Char TF-IDF + Keywords + Scaled Numerics)]
       │
       ▼
 [SHAP Feature Selection (Sweep Top-K: 300 -> 2500 on Validation)]
       │
       ▼
 [Model Training & Optuna Tuning (Complement Naive Bayes vs Linear SVM)]
       │
       ▼
 [Threshold Optimization (Recall Target >= 0.85 & Max F-beta on Validation)]
       │
       ▼
 [Final Test Evaluation & In-depth Error Analysis (FP / FN samples audit)]
       │
       ▼
 [Pipeline Packaging (Joblib) & Deployment Inference Service]
```

---

## 3. Cấu trúc thư mục dự án

```
spam_classifier/
├── main.py                         # Entry point điều phối 13 bước của toàn bộ workflow
├── requirements.txt                # Thư viện dependencies (numpy, pandas, scipy, shap, optuna, joblib, matplotlib)
├── README.md                       # Tài liệu thiết kế hệ thống và ánh xạ logic
├── Classification_email_spam.ipynb # Notebook thực nghiệm gốc (From-Scratch logic)
├── data/                           # Dữ liệu lưu trữ
│   ├── raw/                        # Chứa spam.csv gốc
│   └── processed/                  # Dữ liệu sạch và các tập split
├── notebooks/                      # Notebooks báo cáo thực nghiệm
│   ├── 01_eda.ipynb                # Khám phá dữ liệu & n-grams
│   ├── 02_preprocessing_and_features.ipynb # Xây dựng ma trận đặc trưng lai
│   ├── 03_model_baseline_and_training.ipynb # Huấn luyện CNB và Linear SVM
│   ├── 04_tuning_and_ensemble.ipynb # Tinh chỉnh Optuna & Threshold tuning
│   └── 05_error_analysis_and_deployment.ipynb # Phân tích lỗi & Inference
├── src/                            # Mã nguồn module hóa
│   ├── __init__.py
│   ├── config.py                   # Cấu hình siêu tham số, regex keywords, dataclass SpamExperimentConfig
│   ├── data/                       # Module dữ liệu
│   │   ├── __init__.py
│   │   ├── loader.py               # Nạp dữ liệu thô, fallback encoding, khám phá schema
│   │   └── preprocessing.py        # Chia phân tầng 80/10/10 sớm, làm sạch, SpamTextProcessor
│   ├── features/                   # Module đặc trưng
│   │   ├── __init__.py
│   │   └── feature_engineering.py  # TfidfVectorizerScratch, MaxAbsScalerScratch, HybridBuilder, SHAP Top-K
│   ├── models/                     # Module mô hình Machine Learning
│   │   ├── __init__.py
│   │   ├── classifiers.py          # ComplementNaiveBayes & LinearSVMFromScratch (From Scratch)
│   │   ├── ensemble.py             # SpamVotingEnsemble kết hợp CNB + SVM
│   │   ├── tuning.py               # ThresholdOptimizerScratch, Optuna tuning, ModelSelectionWorkflow
│   │   └── persistence.py          # Lưu/tải toàn bộ pipeline artifact (.joblib)
│   ├── evaluation/                 # Module đánh giá & trực quan hóa
│   │   ├── __init__.py
│   │   ├── metrics.py              # SpamModelEvaluator (Confusion matrix, F-beta, MCC, Trapezoid ROC-AUC)
│   │   ├── visualization.py        # EDA, Heatmap CM, ROC/PR curves, Threshold Diagnostic, SHAP plot
│   │   └── error_analysis.py       # Trích xuất và mổ xẻ mẫu False Positive & False Negative
│   ├── deployment/                 # Module triển khai thực tế
│   │   ├── __init__.py
│   │   └── predictor.py            # SpamInferenceService dự đoán trực tiếp email mới
│   └── utils/                      # Tiện ích bổ trợ
│       ├── __init__.py
│       └── logger.py               # Logger ghi console/file, decorator đo thời gian thực thi
├── results/                        # Lưu trữ kết quả đầu ra tự động
│   ├── figures/                    # Biểu đồ xuất ra (.png)
│   ├── metrics/                    # Bảng số liệu và file phân tích lỗi (.csv)
│   └── saved_models/               # Pipeline bundle đã huấn luyện (.joblib)
└── tests/                          # Kiểm thử tự động (Unit tests)
    ├── __init__.py
    ├── test_data.py
    ├── test_features.py
    ├── test_models.py
    ├── test_evaluation.py
    └── test_deployment.py
```

---

## 4. Chi tiết ánh xạ từng Module với logic trong Notebook

| Module trong cấu trúc | Class / Hàm chính | Ánh xạ mục trong Notebook | Vai trò & Logic thực hiện |
|-----------------------|-------------------|---------------------------|----------------------------|
| [src/config.py](file:///e:/Hoc_May_Tren_truong/spam_classifier/src/config.py) | `SpamExperimentConfig` | Mục 1.2, Cell 9 | Lưu trữ cấu hình N-grams, regex `KEYWORD_PATTERNS`, `RECALL_TARGET = 0.85`, tham số Optuna. |
| [src/data/loader.py](file:///e:/Hoc_May_Tren_truong/spam_classifier/src/data/loader.py) | `load_raw_data()`, `explore_raw_data()` | Mục 2.1, Cell 17-19 | Đọc `spam.csv`, fallback encoding UTF-8/Latin-1, lọc 2 cột `label` & `text`, thống kê mất cân bằng nhãn. |
| [src/data/preprocessing.py](file:///e:/Hoc_May_Tren_truong/spam_classifier/src/data/preprocessing.py) | `stratified_split_dataframe()`, `clean_email_split()`, `SpamTextProcessor` | Mục 2.2, 3.1, 3.2, Cell 21, 32, 39 | Phân tầng 80/10/10 sớm; lọc rác; chuẩn hóa văn bản NFKD, token hóa `<URL>`, `<NUMBER>`, `<CURRENCY>`, cắt đuôi phân vị. |
| [src/features/feature_engineering.py](file:///e:/Hoc_May_Tren_truong/spam_classifier/src/features/feature_engineering.py) | `TfidfVectorizerScratch`, `MaxAbsScalerScratch`, `HybridFeatureBuilderScratch`, `ShapTopKSelectionWorkflow` | Mục 6.1 - 6.4, 7.1 - 7.5, Cell 55, 59, 61, 69, 74 | Vectorizer TF-IDF tự viết; Scaler tự viết; ghép nối ma trận thưa Word + Char + Keywords + Numerics; quét Top-K đặc trưng bằng SHAP. |
| [src/models/classifiers.py](file:///e:/Hoc_May_Tren_truong/spam_classifier/src/models/classifiers.py) | `ComplementNaiveBayes`, `LinearSVMFromScratch` | Mục 6.4.1, 8.5, Cell 64, 85 | CNB giải bài toán mất cân bằng nhãn văn bản; Linear SVM huấn luyện bằng SGD với Hinge Loss và Class Weights. |
| [src/models/tuning.py](file:///e:/Hoc_May_Tren_truong/spam_classifier/src/models/tuning.py) | `ThresholdOptimizerScratch`, `tune_complement_nb_optuna()`, `ModelSelectionWorkflow` | Mục 8.1 - 8.5, Cell 83, 85 | Quét 200 ngưỡng để tối đa hóa F-beta thỏa Recall >= 0.85; tối ưu siêu tham số Optuna; chọn mô hình vô địch. |
| [src/models/ensemble.py](file:///e:/Hoc_May_Tren_truong/spam_classifier/src/models/ensemble.py) | `SpamVotingEnsemble` | Kiến trúc mở rộng | Kết hợp Soft Voting giữa Complement Naive Bayes và Linear SVM. |
| [src/models/persistence.py](file:///e:/Hoc_May_Tren_truong/spam_classifier/src/models/persistence.py) | `save_spam_pipeline()`, `load_spam_pipeline()` | Kiến trúc triển khai | Đóng gói toàn bộ bundle (model, vectorizers, scaler, shap mask, threshold) thành 1 file `.joblib`. |
| [src/evaluation/metrics.py](file:///e:/Hoc_May_Tren_truong/spam_classifier/src/evaluation/metrics.py) | `SpamModelEvaluator` | Mục 1.2, 9.8, 10.1 - 10.5 | Tính Confusion Matrix, Precision, Recall, Specificity, F1, F-beta (2.0), MCC, Trapezoidal ROC-AUC. |
| [src/evaluation/visualization.py](file:///e:/Hoc_May_Tren_truong/spam_classifier/src/evaluation/visualization.py) | `plot_confusion_matrix_heatmap()`, `plot_roc_and_pr_curves()`, `plot_threshold_diagnostic()` | Mục 7.4, 9.1, 10.1, 10.2 | Trực quan hóa Heatmap ma trận nhầm lẫn, ROC/PR curves có chấm ngưỡng tối ưu, biểu đồ quét ngưỡng. |
| [src/evaluation/error_analysis.py](file:///e:/Hoc_May_Tren_truong/spam_classifier/src/evaluation/error_analysis.py) | `extract_misclassified_samples()`, `profile_error_patterns()` | Mục 10.1.4, Cell 106 | Trích xuất và mổ xẻ mẫu False Positive (Ham báo nhầm) và False Negative (Spam lọt lưới). |
| [src/deployment/predictor.py](file:///e:/Hoc_May_Tren_truong/spam_classifier/src/deployment/predictor.py) | `SpamInferenceService` | Mục 1.2, 11.1, Cell 120, 121 | Dịch vụ tiếp nhận email thô mới, tự động chạy toàn bộ tiền xử lý và trả về nhãn + xác suất + tín hiệu kích hoạt. |
| [src/utils/logger.py](file:///e:/Hoc_May_Tren_truong/spam_classifier/src/utils/logger.py) | `setup_logger()`, `log_step()` | Mục 1.1 | Ghi nhật ký thực thi đồng thời ra Terminal và file `logs/pipeline.log`, đo thời gian từng bước. |
| [main.py](file:///e:/Hoc_May_Tren_truong/spam_classifier/main.py) | `run_pipeline()` | Tổng thể quy trình | Entry point kết nối tuần tự 13 giai đoạn từ nạp dữ liệu đến đánh giá và suy luận demo. |

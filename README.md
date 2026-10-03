# 📧 Spam Email Classifier

## 1. Mô tả bài toán

Cho một tập dữ liệu email, xây dựng mô hình Machine Learning để **phân loại tự động** mỗi email là **"spam"** hay **"not spam"** (ham).

Đây là bài toán **phân loại nhị phân (binary classification)** — một trong những bài toán kinh điển của ML, có ứng dụng thực tế rộng rãi trong lọc email, phát hiện gian lận, và phân tích cảm xúc.

### Dữ liệu đầu vào
- **Dataset**: UCI Spambase Dataset (4601 emails, 57 features, 1 label)
- Mỗi email được biểu diễn bằng vector đặc trưng gồm:
  - **Word frequency** (48 features): Tần suất xuất hiện các từ khóa đặc trưng (make, address, free, business, ...)
  - **Character frequency** (6 features): Tần suất ký tự đặc biệt (`;`, `(`, `[`, `!`, `$`, `#`)
  - **Capital run length** (3 features): Thống kê về chuỗi ký tự viết hoa liên tiếp (average, longest, total)

### Đầu ra mong muốn
- Nhãn phân loại: `1 = spam`, `0 = not spam`
- Xác suất dự đoán (probability)
- Bảng so sánh hiệu suất giữa các mô hình
- Biểu đồ trực quan (ROC, Confusion Matrix, Feature Importance)

---

## 2. Xác định yêu cầu

### Yêu cầu chức năng
| # | Yêu cầu | Mô tả |
|---|---------|-------|
| F1 | Tải dữ liệu | Tải Spambase dataset từ UCI hoặc file local |
| F2 | Tiền xử lý | Chia train/test, chuẩn hóa, chọn features quan trọng |
| F3 | Feature Engineering | Tạo features mới (tương tác, tỷ lệ), giảm chiều PCA |
| F4 | Huấn luyện mô hình | Train 3 mô hình: Logistic Regression, SVM, Naive Bayes |
| F5 | Đánh giá | Tính Accuracy, Precision, Recall, F1-score, AUC-ROC |
| F6 | Hyperparameter Tuning | Tối ưu tham số bằng GridSearchCV |
| F7 | Ensemble | Random Forest, Gradient Boosting, Voting Classifier |
| F8 | Phân tích lỗi | Xác định và phân tích các mẫu bị phân loại sai |
| F9 | Trực quan hóa | EDA, Confusion Matrix, ROC Curve, biểu đồ so sánh |
| F10 | Deployment | Lưu model, demo phân loại email mới |

### Yêu cầu phi chức năng
- Code có cấu trúc module rõ ràng, dễ bảo trì
- Logging theo dõi quá trình chạy
- Kết quả (biểu đồ, bảng) được lưu ra thư mục `results/`
- Có file `requirements.txt` để tái tạo môi trường

---

## 3. Phân tích nhiệm vụ từng module

### 3.1. `config.py` — Cấu hình project
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| Đường dẫn dữ liệu | URL dataset UCI, đường dẫn file local, thư mục output |
| Tên features | Danh sách 57 tên feature của Spambase |
| Tham số chung | `test_size=0.2`, `random_state=42`, `cv_folds=5` |
| Hyperparameter grids | Dict tham số cho GridSearchCV (LR, SVM, NB) |

### 3.2. `data/loader.py` — Tải và khám phá dữ liệu
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `load_data()` | Tải Spambase từ UCI (hoặc fallback file local), gán tên cột, trả về DataFrame |
| `explore_data(df)` | In shape, kiểu dữ liệu, tỷ lệ spam/ham, missing values, thống kê mô tả |

### 3.3. `data/preprocessing.py` — Tiền xử lý dữ liệu
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `split_data(df)` | Tách features (X) và label (y), chia train/test theo stratified sampling |
| `scale_features(X_train, X_test)` | Chuẩn hóa bằng StandardScaler (fit trên train, transform trên test) |
| `select_features(X_train, X_test, y_train)` | Chọn K features tốt nhất bằng SelectKBest + ANOVA F-test |

### 3.4. `data/feature_engineering.py` — Tạo features mới
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `create_interaction_features(df)` | Tạo features tương tác: tích word_freq × char_freq |
| `create_ratio_features(df)` | Tạo features tỷ lệ: capital_run_length / tổng số từ |
| `apply_pca(X_train, X_test, n)` | Giảm chiều bằng PCA, giữ lại n thành phần chính |
| `compare_feature_sets(X_orig, X_eng, y)` | So sánh hiệu quả bộ features gốc vs. features mới bằng cross-validation |

### 3.5. `models/classifiers.py` — 3 mô hình phân loại chính
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `get_models()` | Trả về dict `{name: model}` cho Logistic Regression, SVM, Gaussian Naive Bayes |
| `train_and_evaluate(models, X_train, X_test, y_train, y_test)` | Huấn luyện từng model, tính metrics (accuracy, precision, recall, F1, cross-val score) |

### 3.6. `models/tuning.py` — Tối ưu Hyperparameter
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `tune_logistic_regression(X, y)` | GridSearchCV cho LR: `C`, `penalty`, `solver` |
| `tune_svm(X, y)` | GridSearchCV cho SVM: `C`, `kernel`, `gamma` |
| `tune_naive_bayes(X, y)` | GridSearchCV cho NB: `var_smoothing` |
| `tune_all(X, y)` | Gọi cả 3 hàm trên, trả về dict `{name: best_estimator}` |

### 3.7. `models/ensemble.py` — Phương pháp Ensemble
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `train_random_forest(X, y)` | Huấn luyện Random Forest Classifier |
| `train_gradient_boosting(X, y)` | Huấn luyện Gradient Boosting Classifier |
| `train_voting_classifier(estimators, X, y)` | Soft Voting kết hợp 3 mô hình đã tuned |

### 3.8. `models/persistence.py` — Lưu/tải model
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `save_model(model, scaler, path)` | Serialize model + scaler bằng joblib |
| `load_model(path)` | Deserialize model đã lưu |
| `save_pipeline(pipeline, path)` | Lưu toàn bộ pipeline (scaler → selector → model) |
| `load_pipeline(path)` | Tải pipeline đã lưu |

### 3.9. `evaluation/metrics.py` — Tính toán metrics
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `compute_metrics(y_test, y_pred, y_proba)` | Tính accuracy, precision, recall, F1, AUC-ROC |
| `print_classification_reports(results, y_test)` | In classification report (sklearn) cho từng model |
| `build_summary_table(all_results)` | Tạo DataFrame tổng hợp tất cả model, sắp xếp theo F1 |

### 3.10. `evaluation/visualization.py` — Trực quan hóa
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `plot_eda(df)` | Biểu đồ EDA: phân bố nhãn, histogram top features, correlation heatmap |
| `plot_confusion_matrices(results, y_test)` | Confusion Matrix heatmap cho từng model |
| `plot_roc_curves(results, y_test)` | ROC Curve so sánh tất cả model trên cùng 1 biểu đồ |
| `plot_precision_recall_curves(results, y_test)` | Precision-Recall Curve |
| `plot_feature_importance(model, names)` | Bar chart Feature Importance (Random Forest) |
| `plot_model_comparison(summary_df)` | Biểu đồ cột grouped so sánh metrics tất cả model |

### 3.11. `evaluation/error_analysis.py` — Phân tích lỗi
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `analyze_misclassified(X_test, y_test, y_pred, names)` | Tìm mẫu sai, thống kê False Positive vs False Negative |
| `plot_misclassified_distribution(errors_df)` | Biểu đồ phân bố giá trị features của mẫu sai vs đúng |
| `compare_error_patterns(results, X_test, y_test)` | So sánh pattern lỗi giữa các model (mẫu nào cùng sai?) |

### 3.12. `deployment/predictor.py` — Triển khai
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `classify_email(features, model, scaler)` | Nhận vector features → trả về SPAM/NOT SPAM + xác suất |
| `demo_prediction(model, scaler, X_test, y_test)` | Lấy ngẫu nhiên 1 mẫu test, phân loại và in kết quả |

### 3.13. `utils/logger.py` — Logging và tiện ích
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `setup_logger(name, log_file)` | Tạo logger ghi ra console + file |
| `log_step(step_name)` | Decorator đo thời gian chạy từng bước |
| `print_section_header(title)` | In header đẹp phân cách các section |

### 3.14. `main.py` — Entry point
| Bước | Gọi module | Mô tả |
|------|-----------|-------|
| 1 | `data.loader` | Tải + khám phá dữ liệu |
| 2 | `evaluation.visualization` | EDA biểu đồ |
| 3 | `data.preprocessing` | Split, scale, select features |
| 4 | `data.feature_engineering` | Tạo features mới |
| 5 | `models.classifiers` | Huấn luyện 3 mô hình cơ bản |
| 6 | `evaluation.metrics` | Đánh giá metrics |
| 7 | `models.tuning` | Tối ưu hyperparameter |
| 8 | `models.ensemble` | Ensemble methods |
| 9 | `evaluation.error_analysis` | Phân tích lỗi |
| 10 | `evaluation.visualization` | Biểu đồ so sánh tổng hợp |
| 11 | `models.persistence` | Lưu model tốt nhất |
| 12 | `deployment.predictor` | Demo phân loại email mới |

---

## 4. Công nghệ sử dụng

| Thư viện | Mục đích |
|----------|----------|
| `pandas` | Xử lý dữ liệu dạng bảng |
| `numpy` | Tính toán số học |
| `scikit-learn` | Mô hình ML, tiền xử lý, đánh giá |
| `matplotlib` | Vẽ biểu đồ cơ bản |
| `seaborn` | Biểu đồ thống kê nâng cao |
| `joblib` | Lưu/tải model |

---

## 5. Cấu trúc thư mục

```
spam_classifier/
├── main.py                         # Entry point, điều phối toàn bộ workflow
├── requirements.txt                # Thư viện dependencies
├── README.md                       # Tài liệu phân tích và hướng dẫn dự án
├── data/                           # Thư mục lưu trữ dữ liệu (không chứa code)
│   ├── raw/                        # Dữ liệu gốc (spambase.data, emails raw...)
│   └── processed/                  # Dữ liệu đã làm sạch, xử lý sẵn sàng train
├── notebooks/                      # Thử nghiệm tương tác & báo cáo EDA
│   ├── 01_eda.ipynb                # Khám phá dữ liệu (EDA)
│   └── 02_model_experiments.ipynb  # Thí nghiệm huấn luyện mô hình
├── src/                            # Toàn bộ mã nguồn chính của dự án
│   ├── __init__.py
│   ├── config.py                   # Cấu hình đường dẫn, hyperparams, feature names
│   ├── data/                       # Module tải và tiền xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── loader.py               # Tải và khám phá dữ liệu
│   │   └── preprocessing.py        # Tiền xử lý (clean, split, scale, select)
│   ├── features/                   # Module đặc trưng
│   │   ├── __init__.py
│   │   └── feature_engineering.py  # Tạo features mới (tương tác, tỷ lệ, PCA)
│   ├── models/                     # Module thuật toán Machine Learning
│   │   ├── __init__.py
│   │   ├── classifiers.py          # 3 mô hình cơ bản: LR, SVM, NB
│   │   ├── ensemble.py             # Random Forest, Gradient Boosting, Voting
│   │   ├── tuning.py               # GridSearchCV hyperparameter tuning
│   │   └── persistence.py          # Lưu/tải model (.joblib)
│   ├── evaluation/                 # Module đánh giá hiệu năng
│   │   ├── __init__.py
│   │   ├── metrics.py              # Accuracy, Precision, Recall, F1, AUC
│   │   ├── visualization.py        # Biểu đồ EDA, ROC, Confusion Matrix
│   │   └── error_analysis.py       # Phân tích mẫu bị phân loại sai
│   ├── deployment/                 # Module dự đoán thực tế
│   │   ├── __init__.py
│   │   └── predictor.py            # Phân loại email mới từ vector/text
│   └── utils/                      # Tiện ích bổ trợ
│       ├── __init__.py
│       └── logger.py               # Logging và đo thời gian
├── results/                        # Lưu trữ kết quả đầu ra
│   ├── figures/                    # Biểu đồ (.png)
│   ├── metrics/                    # Bảng tổng hợp số liệu (.csv)
│   └── saved_models/               # Model weights đã huấn luyện (.joblib)
└── tests/                          # Kiểm thử tự động (Unit tests)
    ├── __init__.py
    ├── test_data.py                # Tests cho module data
    ├── test_models.py              # Tests cho module models
    └── test_evaluation.py          # Tests cho module evaluation
```

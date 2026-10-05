# 📧 Spam Email Classification System (ML Project Architecture)

Dự án Machine Learning phân loại tự động email/tin nhắn rác (**Spam**) và email hợp lệ (**Not Spam / Ham**), được thiết kế và chuẩn hóa 100% theo đặc tả yêu cầu của đề bài **"Practice 1: Classifying Spam Emails"**.

---

## 1. Mô tả bài toán & Dữ liệu (Problem & Data)

### 1.1. Bài toán (Problem)
- **Mục tiêu**: Cho một tập dữ liệu các email, xác định xem mỗi email là **"spam"** hay **"not spam" (ham)**.
- **Bản chất**: Bài toán học máy có giám sát (Supervised Learning) dạng **Phân loại nhị phân (Binary Classification)**.

### 1.2. Biểu diễn đặc trưng dữ liệu (Feature Representation)
Mỗi email được mô hình hóa thành một vector đặc trưng số học:
- **Tần suất từ vựng (Word Frequency)**: Số lần xuất hiện của các từ khóa nhạy cảm spam (`free`, `win`, `prize`, `claim`, `urgent`, `call`, `cash`, `offer`, v.v.).
- **Tần suất ký tự (Character Frequency)**: Tần suất của các ký tự đặc biệt mang tính kích động (dấu chấm than `!`, ký hiệu tiền tệ `$`, `£`, chuỗi chữ số liên tiếp).
- **Thông tin cấu trúc & văn bản**:
  - Dòng tiêu đề (Subject line) & Phần thân email (Email body).
  - Tỷ lệ chữ cái in hoa (Uppercase ratio), chiều dài email (Length / Word count).
  - Sự hiện diện của liên kết web (URL links) và số điện thoại.

---

## 2. Phương pháp tiếp cận phân loại (Classification Approach)

Hệ thống tích hợp đầy đủ **3 mô hình phân loại cốt lõi** theo đúng đề bài quy định:

| # | Mô hình phân loại | Nguyên lý toán học & Đặc trưng |
|---|-------------------|---------------------------------|
| **1** | **Logistic Regression** | Mô hình hóa xác suất một email là spam bằng hàm Logistic/Sigmoid: $P(y=1\|x) = \frac{1}{1 + e^{-(w^T x + b)}}$. Tối ưu hóa bằng Gradient Descent với hàm mất mát Binary Cross-Entropy và điều chuẩn L2. |
| **2** | **Support Vector Machines (SVM)** | Tìm siêu phẳng lề cực đại (Maximum Margin Hyperplane) phân tách giữa Spam và Ham trong không gian đặc trưng nhiều chiều. Huấn luyện bằng SGD với hàm mất mát Hinge Loss. |
| **3** | **Naive Bayes** | Mô hình xác suất dựa trên định lý Bayes ($P(c\|x) \propto P(c) \prod P(x_i\|c)$) với giả định các đặc trưng độc lập có điều kiện. Triển khai cả Multinomial NB và Complement NB (chuyên trị tập dữ liệu văn bản mất cân bằng nhãn). |

### Các cân nhắc mở rộng (Additional Considerations):
- **Kỹ thuật trích xuất đặc trưng (Feature Engineering)**: Thử nghiệm kết hợp các nhóm đặc trưng (Word TF-IDF, Char N-grams, Keywords, Numeric stats).
- **Tinh chỉnh siêu tham số (Hyperparameter Tuning)**: Tối ưu hóa hệ số điều chuẩn $C$, tốc độ học $learning\_rate$, hệ số làm mịn Laplace $\alpha$.
- **Phương pháp kết hợp mô hình (Ensemble Methods)**: Kết hợp biểu quyết (SpamVotingEnsemble) giữa Logistic Regression, SVM và Naive Bayes, kết hợp Random Forest / Boosting để tăng độ khái quát và chống overfitting.
- **Tối ưu hóa ngưỡng phân loại (Threshold Tuning)**: Dò quét ngưỡng xác suất trên Validation set nhằm đảm bảo yêu cầu nghiệp vụ thực tế ($\text{Recall} \ge 0.85$).

---

## 3. Quy trình thực thi chuẩn (Workflow)

```
[spam.csv Raw Data]
       │
       ▼ (Workflow Bước 1: Data Preprocessing)
[Data Cleaning: Xóa stop words, punctuation, HTML tags/URLs]
       │
       ▼ (Early Stratified Split: Train 80% - Val 10% - Test 10%)
[Trích xuất đặc trưng: Word TF-IDF + Char Freq ('!', '$') + Keywords + Numerics]
       │
       ▼ (Workflow Bước 2: Model Training)
[Huấn luyện 3 Mô hình: Logistic Regression, SVM, Naive Bayes]
       │
       ▼ (Additional Considerations: Tuning & Ensemble)
[Hyperparameter Tuning (Grid Search) + Voting Ensemble + Threshold Tuning]
       │
       ▼ (Workflow Bước 3: Model Evaluation)
[Đánh giá đa chiều trên Test: Accuracy, Precision, Recall, F1-Score, ROC-AUC]
       │
       ▼ (Workflow Bước 4: Model Deployment)
[Đóng gói Pipeline (.joblib) & Triển khai Inference Service dự đoán email mới]
```

---

## 4. Cấu trúc thư mục dự án (Project Structure)

```
spam_classifier/
├── main.py                         # Entry point điều phối toàn diện 4 giai đoạn của workflow
├── requirements.txt                # Thư viện phụ thuộc (pandas, numpy, scikit-learn, matplotlib, seaborn, joblib)
├── README.md                       # Tài liệu thiết kế hệ thống và ánh xạ yêu cầu đề bài
├── data/                           # Lưu trữ dữ liệu
│   ├── raw/                        # spam.csv gốc
│   └── processed/                  # Dữ liệu sạch và các tập split
├── docs/                           # Tài liệu đề bài và hướng dẫn
│   ├── ML project_ Classifying Spam Emails.pdf # Đề bài gốc của môn học
│   └── Classification_email_spam.ipynb         # Notebook thực nghiệm mẫu tham khảo
├── notebooks/                      # Bộ 5 Jupyter Notebooks nghiên cứu thực nghiệm
│   ├── 01_eda.ipynb                # Khám phá dữ liệu, phân bố Spam/Ham, thống kê đặc trưng
│   ├── 02_preprocessing_and_features.ipynb # Làm sạch, TF-IDF, trích xuất đặc trưng lai
│   ├── 03_model_baseline_and_training.ipynb # Huấn luyện 3 mô hình (LR, SVM, Naive Bayes)
│   ├── 04_tuning_and_ensemble.ipynb # Tinh chỉnh siêu tham số, Ensemble, tối ưu ngưỡng
│   └── 05_error_analysis_and_deployment.ipynb # Phân tích lỗi (FP/FN), lưu model và demo suy luận
├── src/                            # Mã nguồn module hóa
│   ├── __init__.py
│   ├── config.py                   # Cấu hình trung tâm (Đường dẫn, siêu tham số, regex, dataclass)
│   ├── data/                       # Module xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── loader.py               # Nạp spam.csv, fallback encoding, khám phá schema
│   │   └── preprocessing.py        # Làm sạch stop words/punctuation/HTML, chia phân tầng 80/10/10
│   ├── features/                   # Module trích xuất đặc trưng
│   │   ├── __init__.py
│   │   └── feature_engineering.py  # TF-IDF Vectorizer, Scaler, Keyword & Numeric extractor, Hybrid Builder
│   ├── models/                     # Module thuật toán Machine Learning
│   │   ├── __init__.py
│   │   ├── classifiers.py          # 3 mô hình: Logistic Regression, Linear SVM, Naive Bayes
│   │   ├── ensemble.py             # SpamVotingEnsemble, Random Forest / Boosting
│   │   ├── tuning.py               # Grid Search tuning, ThresholdOptimizer, ModelSelection
│   │   └── persistence.py          # Đóng gói và nạp toàn bộ pipeline (.joblib)
│   ├── evaluation/                 # Module đánh giá hiệu năng
│   │   ├── __init__.py
│   │   ├── metrics.py              # Đầy đủ metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC
│   │   ├── visualization.py        # Heatmap Confusion Matrix, đồ thị đôi ROC & PR, EDA plots
│   │   └── error_analysis.py       # Bóc tách mẫu phân loại sai: False Positives và False Negatives
│   ├── deployment/                 # Module triển khai thực tế
│   │   ├── __init__.py
│   │   └── predictor.py            # SpamInferenceService dự đoán trực tiếp email mới
│   └── utils/                      # Tiện ích bổ trợ
│       ├── __init__.py
│       └── logger.py               # Logger ghi console & file nhật ký, decorator đo thời gian
├── results/                        # Thư mục lưu kết quả tự động
│   ├── figures/                    # Biểu đồ xuất bản (.png)
│   ├── metrics/                    # Bảng số liệu và file phân tích lỗi (.csv)
│   └── saved_models/               # Pipeline bundle đã đóng gói (.joblib)
└── tests/                          # Bộ kiểm thử tự động (Unit Tests)
    ├── __init__.py
    ├── test_data.py                # Kiểm thử nạp và phân tầng dữ liệu
    ├── test_features.py            # Kiểm thử TF-IDF và ghép nối đặc trưng
    ├── test_models.py              # Kiểm thử 3 mô hình (LR, SVM, Naive Bayes) và Ensemble
    ├── test_evaluation.py          # Kiểm thử Confusion Matrix, F1, ROC-AUC
    └── test_deployment.py          # Kiểm thử dịch vụ dự đoán email đơn lẻ & hàng loạt
```

---

## 5. Ánh xạ chi tiết giữa Yêu cầu đề bài và Module triển khai

| Yêu cầu trong Đề bài PDF | Giai đoạn Workflow | Module triển khai chính | Mô tả logic thực hiện |
|---|---|---|---|
| **Clean data (stop words, punctuation, HTML)** | Workflow 1: Preprocessing | `src/data/preprocessing.py` | `clean_text_basic()`, `SpamTextProcessor`: chuẩn hóa Unicode, loại bỏ stop words, dấu câu, thẻ HTML, token hóa URL/MONEY/NUMBER. |
| **Convert text to numerical features (TF-IDF)** | Workflow 1: Preprocessing | `src/features/feature_engineering.py` | `TfidfVectorizerScratch`: tính toán sublinear TF, smooth IDF, chuẩn hóa vector L2 norm. |
| **Word & Character Frequency features** | Feature Representation | `src/features/feature_engineering.py` | `SpamSignalFeatureExtractor`: đếm tần suất từ khóa nhạy cảm, ký tự đặc biệt (`!`, `$`), chữ in hoa, độ dài. |
| **Split dataset into train and test sets** | Workflow 1: Preprocessing | `src/data/preprocessing.py` | `stratified_split_dataframe()`: chia phân tầng 80% Train, 10% Validation, 10% Test độc lập. |
| **Logistic Regression** | Workflow 2: Model Training | `src/models/classifiers.py` | `LogisticRegressionFromScratch`: hàm Sigmoid, Binary Cross-Entropy Loss, Gradient Descent, class weights. |
| **Support Vector Machines (SVM)** | Workflow 2: Model Training | `src/models/classifiers.py` | `LinearSVMFromScratch`: siêu phẳng lề cực đại, Hinge Loss, Pegasos SGD, class weights. |
| **Naive Bayes** | Workflow 2: Model Training | `src/models/classifiers.py` | `NaiveBayesClassifier`: định lý Bayes, làm mịn Laplace, hỗ trợ Multinomial NB và Complement NB. |
| **Accuracy, Precision, Recall, F1-score** | Workflow 3: Model Evaluation | `src/evaluation/metrics.py` | `SpamModelEvaluator`: tính đầy đủ 4 chỉ số cốt lõi, ma trận nhầm lẫn 2x2, ROC-AUC tích phân hình thang. |
| **Model Deployment** | Workflow 4: Deployment | `src/deployment/predictor.py` | `SpamInferenceService`: tiếp nhận email thô mới, tự động chạy toàn bộ pipeline và trả về nhãn + xác suất + tín hiệu rủi ro. |
| **Feature Engineering combinations** | Additional Considerations | `src/features/feature_engineering.py` | `experiment_feature_combinations()`: thử nghiệm các tổ hợp đặc trưng (Word, Char, Keywords, Numerics). |
| **Hyperparameter Tuning** | Additional Considerations | `src/models/tuning.py` | `tune_model_hyperparameters()`: tối ưu hóa siêu tham số $C$, $learning\_rate$, $\alpha$ trên tập Validation. |
| **Ensemble Methods** | Additional Considerations | `src/models/ensemble.py` | `SpamVotingEnsemble`: Soft/Hard Voting kết hợp 3 mô hình, Random Forest / Boosting. |

---

## 6. Hướng dẫn sử dụng & Khởi chạy

### Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### Chạy toàn bộ luồng quy trình điều phối
```bash
python main.py
```

### Khởi động và thực hiện các bước trên Jupyter Notebook
Khởi chạy notebook và thực thi tuần tự từ `01` đến `05`:
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_preprocessing_and_features.ipynb`
3. `notebooks/03_model_baseline_and_training.ipynb`
4. `notebooks/04_tuning_and_ensemble.ipynb`
5. `notebooks/05_error_analysis_and_deployment.ipynb`

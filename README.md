# 📧 Hệ Thống Phân Loại Email Spam (Spam Email Classifier)

Dự án Học Máy (Machine Learning) phân loại tự động email/tin nhắn rác (**Spam**) và email hợp lệ (**Not Spam / Ham**), được thiết kế, tối ưu hóa và chuẩn hóa 100% theo đặc tả yêu cầu của đề bài **"Practice 1: Classifying Spam Emails"**.

> **Nguyên tắc kỹ thuật cốt lõi**: 100% Thuật toán Tự Triển Khai (From-Scratch) – Hoàn toàn không phụ thuộc vào Scikit-Learn cho lõi mô hình và đo lường chỉ số.  
> **Trạng thái hệ thống**: Đã nghiệm thu toàn diện | **18/18 Unit Tests PASS (100% OK)** | Đạt chuẩn an ninh mạng $\text{Recall} \ge 0.85$.

---

## 1. Mô tả Bài toán & Dữ liệu (Problem & Data)

### 1.1. Bài toán (Problem)
- **Mục tiêu**: Cho một tập dữ liệu văn bản email, xác định chính xác mỗi email là **"Spam"** (thư rác, quảng cáo phiền toái, thư lừa đảo/phishing) hay **"Ham"** (email hợp lệ, thư công việc bình thường).
- **Bản chất**: Bài toán học máy có giám sát (Supervised Learning) dạng **Phân loại nhị phân (Binary Text Classification)**.
- **Mã hóa nhãn**:
  - `Ham / Not Spam` (Hợp lệ): Mã hóa nhãn `0`.
  - `Spam` (Thư rác): Mã hóa nhãn `1`.

### 1.2. Thống kê & Phân tầng Dữ liệu (Dataset Distribution)
- **Dữ liệu thô ban đầu (`data/raw/spam.csv`)**: Gồm **43,849** dòng dữ liệu email thực tế, chứa nhiều dị biệt về encoding, dòng rỗng và email trùng lặp.
- **Kiểm toán chất lượng dữ liệu (`clean_email_split`)**:
  - Lọc sạch dòng rỗng/null và chuẩn hóa nhãn chuỗi về `{0, 1}`.
  - Khử trùng lặp nội dung văn bản (`drop_duplicates`), đưa tập dữ liệu về **41,654 mẫu độc bản** chất lượng cao.
  - Tỷ lệ mất cân bằng tự nhiên: **~86.6% Ham** (36,066 mẫu) vs **~13.4% Spam** (5,588 mẫu).
- **Phân tầng độc lập chống rò rỉ (Early Stratified Split 80/10/10)**:
  - **Tập Huấn luyện (Train set - 80%)**: **33,322 mẫu** (dùng để fit từ vựng TF-IDF, Scaler và huấn luyện tham số mô hình).
  - **Tập Phát triển (Validation set - 10%)**: **4,166 mẫu** (dùng để Grid Search siêu tham số và tối ưu hóa ngưỡng quyết định $\tau^*$).
  - **Tập Kiểm thử (Test set - 10%)**: **4,166 mẫu** (niêm phong hoàn toàn làm Unseen Data để đánh giá khách quan cuối cùng).

### 1.3. Biểu diễn Đặc trưng Lai (Hybrid Feature Representation)
Mỗi email được chuyển đổi thành một vector đặc trưng số học trong không gian nhiều chiều ($N \times 1667$):
1. **Khối từ vựng TF-IDF (1,600 chiều)**: Vector hóa từ vựng với Sublinear TF ($1 + \log(\text{tf})$), Smooth IDF ($\log((1+N)/(1+\text{df})) + 1$) và chuẩn hóa Euclidean L2-norm.
2. **Khối N-grams ký tự (Character N-grams)**: Nắm bắt các biến thể cố tình viết chệch ký tự phổ biến của spam.
3. **Khối từ khóa nhạy cảm (Keywords)**: Tần suất các từ khóa cảnh báo lừa đảo (`free`, `win`, `winner`, `cash`, `prize`, `urgent`, `claim`, `credit`, `money`...).
4. **Khối tín hiệu cấu trúc (Structural Signals)**: Trích xuất tuyến tính Single-pass $O(L)$ đếm tần suất dấu cảm thán (`!`), ký hiệu tiền tệ (`$`, `€`, `£`), tỷ lệ viết hoa (Uppercase ratio), độ dài văn bản, số lượng URL và dãy số điện thoại.

---

## 2. Bốn Nguyên Tắc Kỹ Thuật Bất Khả Xâm Phạm

1. **Triết lý From-Scratch tuyệt đối**:
   100% các thành phần cốt lõi của bài toán — từ bộ tiền xử lý NLP (`SpamTextProcessor`), cấu trúc ma trận thưa nén (`CSRMatrixScratch`), vector hóa TF-IDF (`TfidfVectorizerScratch`), chuẩn hóa `MaxAbsScalerScratch`, 3 thuật toán phân loại (Logistic Regression, Linear SVM, Complement Naive Bayes), mô hình kết hợp (`SpamVotingEnsemble`), bộ tối ưu hóa ngưỡng (`ThresholdOptimizerScratch`) đến bộ chỉ số đánh giá (`SpamModelEvaluator`) — đều được tự lập trình bằng Python thuần và NumPy/SciPy cơ bản, hoàn toàn không sử dụng các lớp đóng gói sẵn của Scikit-Learn.
2. **Triệt tiêu rò rỉ dữ liệu (Zero Data Leakage)**:
   Áp dụng cơ chế Early Stratified Split ngay sau khi lọc sạch dữ liệu thô. Mọi thống kê từ vựng (Vocabulary, IDF weights), ngưỡng co dãn số học (Max Abs) và ngưỡng quyết định phân loại (Optimal Threshold) chỉ được học (`fit`) duy nhất trên tập Train/Validation. Tập Test giữ vai trò là trọng tài khách quan tuyệt đối.
3. **Ràng buộc nghiệp vụ an ninh mạng tiên quyết ($\text{Recall} \ge 0.85$)**:
   Lọc spam thực tế đòi hỏi ưu tiên tối đa việc phát hiện thư rác để bảo vệ người dùng khỏi nguy cơ tấn công mạng và phishing (ưu tiên chống False Negative). Hệ thống tự động dò tìm ngưỡng quyết định $\tau^*$ trên tập Validation nhằm bảo đảm Recall đạt tối thiểu 85%, đồng thời tối đa hóa điểm F1 và F-beta ($\beta = 2.0$).
4. **Kiến trúc Ma trận Thưa An toàn & Sẵn sàng Triển khai (Production-Ready)**:
   Không gian đặc trưng chứa hơn 99% phần tử 0. Dự án tự xây dựng cấu trúc `CSRMatrixScratch` và các hàm đại số tuyến tính an toàn đa hình (`_safe_dot`, `_safe_transpose_dot`), giúp giảm dung lượng RAM từ 2.4 GB xuống dưới 35 MB (tiết kiệm hơn 98% bộ nhớ), triệt tiêu hoàn toàn nguy cơ tràn RAM. Hệ thống đi kèm dịch vụ suy luận thời gian thực `SpamInferenceService` trả kết quả JSON chi tiết.

---

## 3. Phương Pháp Tiếp Cận Phân Loại & Toán Học Cốt Lõi

Hệ thống tích hợp đầy đủ **3 mô hình phân loại cốt lõi** theo đúng đề bài quy định, đi kèm các cải tiến toán học chuyên sâu:

| # | Mô hình phân loại | Nguyên lý Toán học & Tối ưu hóa Cốt lõi |
|---|---|---|
| **1** | **Logistic Regression From-Scratch** | Mô hình hóa xác suất bằng hàm Sigmoid: $P(y=1\|x) = \frac{1}{1 + e^{-(w^T x + b)}}$. Kẹp số trị $z \in [-30, 30]$ chống tràn số mũ Overflow; hàm mất mát Binary Cross-Entropy có điều chuẩn L2 ($\lambda = \frac{1}{C \cdot N}$); tối ưu hóa bằng Gradient Descent với suy giảm tốc độ học thích nghi $\eta_t = \frac{\eta_0}{\sqrt{1 + 0.01 t}}$; hằng số $\epsilon = 10^{-15}$ chống $\log(0)$; tích hợp trọng số lớp tự động `class_weight='balanced'`. |
| **2** | **Linear Support Vector Machines (SVM)** | Tìm siêu phẳng lề cực đại (Maximum Margin Hyperplane) phân tách giữa Spam và Ham với nhãn $y \in \{-1, +1\}$. Huấn luyện bằng giải thuật Pegasos (Primal Estimated sub-GrAdient SOlver) vận hành trên Hinge Loss chuẩn ($y \cdot \text{score} < 1$) và Squared Hinge Loss; dừng sớm với dung sai `tol=1e-4`; cung cấp hàm khoảng cách lề `decision_function()` và xác suất hiệu chỉnh Platt Scaling Sigmoid. |
| **3** | **Naive Bayes Classifier (CNB & MNB)** | Tính xác suất hậu nghiệm dựa trên định lý Bayes với làm mịn Laplace $\alpha$. Triển khai song song cả **Multinomial NB** và **Complement Naive Bayes (CNB)** chuyên trị dữ liệu văn bản mất cân bằng lớp (học trên tập phần bù $y \neq c$). Tính toán tần suất đặc trưng bằng phép nhân ma trận chuyển vị với mask nhị phân ($O(\text{nnz})$, loại bỏ boolean row-slicing). Tích hợp **Log-Sum-Exp Trick** trong `predict_proba()` triệt tiêu 100% lỗi Floating-point Underflow và chia cho 0. |

### Các phương pháp mở rộng theo yêu cầu đề bài:
- **Tối ưu hóa ngưỡng quyết định (Threshold Tuning)**: Lớp `ThresholdOptimizerScratch` quét lưới 200 mốc ngưỡng $\tau \in [0.005, 1.0]$ trên Validation set, tự động tìm $\tau^*$ thỏa mãn $\text{Recall} \ge 0.85$ và tối đa hóa F1 / F2.
- **Phương pháp kết hợp mô hình (Ensemble Methods)**: Lớp `SpamVotingEnsemble` kết hợp Soft Voting (trung bình cộng xác suất có trọng số) và Hard Voting giữa cả 3 mô hình (LR + SVM + Naive Bayes), hỗ trợ ứng viên mở rộng Random Forest và Gradient Boosting có cơ chế bọc ngoại lệ an toàn.
- **Tinh chỉnh siêu tham số (Hyperparameter Tuning)**: Hàm `tune_model_hyperparameters()` quét lưới Grid Search trên Validation set tối ưu hệ số điều chuẩn $C$, tốc độ học $learning\_rate$ và hệ số làm mịn $\alpha$ với hàm mục tiêu có ràng buộc nghiệp vụ (Constrained Objective).
- **Đóng gói Pipeline (Model Persistence)**: Hàm `save_spam_pipeline()` và `load_spam_pipeline()` đóng gói toàn bộ bundle 5 thành phần (mô hình chiến thắng, bộ tiền xử lý text, bộ vectorizer, scaler và ngưỡng tối ưu) vào file `spam_pipeline.joblib` nén cấp 3.

---

## 4. Quy Trình Thực Thi Chuẩn (Workflow 4 Giai Đoạn)

```
[spam.csv Raw Data: 43,849 dòng]
       │
       ▼ (Workflow Bước 1: Data Preprocessing & Audit)
[Kiểm toán & Lọc sạch: Xóa null, chuẩn hóa nhãn 0/1, khử trùng lặp -> 41,654 mẫu sạch]
       │
       ▼ (Early Stratified Split: Train 80% - Val 10% - Test 10%)
[Trích xuất đặc trưng lai: Word TF-IDF + Char N-grams + Keywords + Scaled Structural O(L)]
       │
       ▼ (Workflow Bước 2: Model Training From-Scratch)
[Huấn luyện 3 Mô hình: Logistic Regression, Linear SVM (Pegasos), Complement Naive Bayes]
       │
       ▼ (Additional Considerations: Tuning & Ensemble)
[Grid Search Tuning + SpamVotingEnsemble + ThresholdOptimizer (Quét 200 ngưỡng, Recall >= 0.85)]
       │
       ▼ (Workflow Bước 3: Model Evaluation & Error Analysis)
[Đánh giá đa chiều trên Test: Accuracy, Precision, Recall, F1, F2, Mann-Whitney ROC-AUC]
[Phân tích lỗi sai chuyên sâu: Bóc tách mẫu FP/FN -> misclassified_samples.csv]
       │
       ▼ (Workflow Bước 4: Deployment & Orchestration)
[Đóng gói spam_pipeline.joblib & Dịch vụ suy luận thời gian thực SpamInferenceService]
[Entry Point main.py điều phối toàn diện khép kín + Demo phân loại email thực tế]
```

---

## 5. Cấu Trúc Thư Mục Dự Án (Project Structure)

```
spam_classifier/
├── main.py                         # Entry point điều phối toàn diện 4 giai đoạn của hệ thống
├── requirements.txt                # Danh mục thư viện phụ thuộc của dự án
├── README.md                       # Tài liệu thiết kế hệ thống và thuyết minh kỹ thuật
├── data/                           # Lưu trữ dữ liệu
│   ├── raw/                        # spam.csv gốc (43,849 dòng dữ liệu thô)
│   └── processed/                  # Dữ liệu đệm phục vụ chạy nhanh trên Notebook (X_train, y_train, ...)
├── notebooks/                      # Bộ 5 Jupyter Notebooks nghiên cứu thực nghiệm
│   ├── 01_eda.ipynb                # Khám phá dữ liệu, phân bố Spam/Ham, thống kê đặc trưng
│   ├── 02_preprocessing_and_features.ipynb # Làm sạch, TF-IDF, trích xuất đặc trưng lai
│   ├── 03_model_baseline_and_training.ipynb # Huấn luyện 3 mô hình (LR, SVM, Naive Bayes)
│   ├── 04_tuning_and_ensemble.ipynb # Tinh chỉnh siêu tham số, Ensemble, tối ưu ngưỡng
│   ├── 05_error_analysis_and_deployment.ipynb # Phân tích lỗi (FP/FN), lưu model và demo suy luận
│   └── results/                    # Thư mục lưu trữ kết quả thực nghiệm của Notebooks
│       ├── figures/                # Biểu đồ thực nghiệm (.png)
│       ├── metrics/                # File phân tích lỗi thực nghiệm (.csv)
│       └── saved_models/           # Pipeline bundle thực nghiệm (.joblib)
├── src/                            # Mã nguồn module hóa chính thức của dự án
│   ├── __init__.py
│   ├── config.py                   # Cấu hình trung tâm (Đường dẫn, siêu tham số, regex, dataclass)
│   ├── data/                       # Module xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── loader.py               # Nạp dữ liệu đa tầng fallback encoding, khám phá schema
│   │   └── preprocessing.py        # Làm sạch 4 bước, phân tầng Early Stratified Split 80/10/10
│   ├── features/                   # Module kỹ thuật đặc trưng lai
│   │   ├── __init__.py
│   │   └── feature_engineering.py  # CSRMatrixScratch, TF-IDF, Scaler, Single-pass structural, Hybrid Builder
│   ├── models/                     # Module thuật toán Machine Learning (From-Scratch)
│   │   ├── __init__.py
│   │   ├── classifiers.py          # LR (adaptive decay), SVM (Pegasos Hinge), Naive Bayes (Log-Sum-Exp)
│   │   ├── ensemble.py             # SpamVotingEnsemble (Soft/Hard Voting), Tree-based wrappers
│   │   ├── tuning.py               # Grid Search tuning, ThresholdOptimizerScratch, ModelSelection
│   │   └── persistence.py          # Đóng gói và nạp toàn bộ pipeline bundle (.joblib)
│   ├── evaluation/                 # Module đánh giá hiệu năng & phân tích lỗi
│   │   ├── __init__.py
│   │   ├── metrics.py              # 10 chỉ số đánh giá From-Scratch, Safe Division, Mann-Whitney ROC-AUC
│   │   ├── visualization.py        # Heatmap chuẩn hóa kép, đồ thị đôi ROC-PR đánh dấu tau*, biểu đồ cột
│   │   └── error_analysis.py       # Bóc tách mẫu lỗi FP/FN, chẩn đoán từ khóa và xuất CSV
│   ├── deployment/                 # Module triển khai thực tế
│   │   ├── __init__.py
│   │   └── predictor.py            # SpamInferenceService dự đoán trực tiếp email mới trả JSON
│   └── utils/                      # Tiện ích bổ trợ
│       ├── __init__.py
│       └── logger.py               # Logger ghi đồng thời Console & File, decorator đo thời gian
├── results/                        # Kết quả đầu ra chính thức từ Pipeline Production (main.py)
│   ├── figures/                    # Biểu đồ xuất bản 300 DPI (Confusion Matrix, ROC-PR, So sánh mô hình)
│   ├── metrics/                    # File phân tích lỗi chi tiết misclassified_samples.csv
│   └── saved_models/               # Pipeline bundle chính thức spam_pipeline.joblib
├── logs/                           # Nhật ký thực thi pipeline (pipeline.log)
└── tests/                          # Bộ kiểm thử tự động toàn diện (18/18 Unit Tests PASS)
    ├── __init__.py
    ├── test_data.py                # Kiểm thử nạp dữ liệu fallback và phân tầng không rò rỉ (3 tests)
    ├── test_features.py            # Kiểm thử CSRMatrixScratch, TF-IDF và MaxAbsScaler (3 tests)
    ├── test_models.py              # Kiểm thử LR, SVM Pegasos, Naive Bayes, Ensemble và Tuning (5 tests)
    ├── test_evaluation.py          # Kiểm thử Safe Division, Confusion Matrix, Mann-Whitney ROC-AUC (4 tests)
    └── test_deployment.py          # Kiểm thử dịch vụ suy luận thời gian thực và đóng gói bundle (3 tests)
```

---

## 6. Ánh Xạ Chi Tiết Giữa Yêu Cầu Đề Bài và Module Triển Khai

| Yêu cầu trong Đề bài | Giai đoạn Workflow | Module Triển khai Chính | Mô tả Logic Thực hiện Chi tiết |
|---|---|---|---|
| **Clean data (stop words, punctuation, HTML)** | Workflow 1: Preprocessing | `src/data/preprocessing.py` | `clean_text_basic()`, `SpamTextProcessor`: chuẩn hóa Unicode NFKD, xóa thẻ HTML, token hóa `<URL>`, `<EMAIL>`, `<MONEY>`, `<NUMBER>`, lọc stop words và dấu câu. |
| **Audit & Deduplicate records** | Workflow 1: Preprocessing | `src/data/preprocessing.py` | `clean_email_split()`: loại bỏ dòng rỗng, chuẩn hóa nhãn về nhị phân `{0, 1}`, khử trùng lặp nội dung văn bản gốc. |
| **Split dataset into train and test sets** | Workflow 1: Preprocessing | `src/data/preprocessing.py` | `stratified_split_dataframe()`: chia phân tầng 80% Train, 10% Validation, 10% Test độc lập bằng `default_rng(seed=42)` chống Data Leakage. |
| **Sparse Matrix Representation** | Feature Engineering | `src/features/feature_engineering.py` | `CSRMatrixScratch`: cấu trúc nén Compressed Sparse Row (`data`, `indices`, `indptr`), tiết kiệm 98% RAM, tích hợp `to_scipy_csr()`. |
| **Convert text to numerical features (TF-IDF)** | Feature Engineering | `src/features/feature_engineering.py` | `TfidfVectorizerScratch`: tính toán Sublinear TF, Smooth IDF, chuẩn hóa Euclidean L2-norm cho vector từ vựng. |
| **Word & Character Frequency features** | Feature Engineering | `src/features/feature_engineering.py` | `extract_structural_signals()`: duyệt chuỗi single-pass $O(L)$ đếm ký tự (`!`, `$`), tỷ lệ viết hoa, độ dài; `MaxAbsScalerScratch` co dãn về $[-1, 1]$ bảo toàn tính thưa. |
| **Hybrid Feature Combination** | Additional Considerations | `src/features/feature_engineering.py` | `HybridFeatureBuilder`: ghép nối ngang `scipy.sparse.hstack` 4 khối đặc trưng thành ma trận thống nhất shape $(N, 1667)$. |
| **Logistic Regression** | Workflow 2: Model Training | `src/models/classifiers.py` | `LogisticRegressionFromScratch`: hàm Sigmoid kẹp số trị, Binary Cross-Entropy Loss, Gradient Descent có adaptive decay, L2 reg, `class_weight='balanced'`. |
| **Support Vector Machines (SVM)** | Workflow 2: Model Training | `src/models/classifiers.py` | `LinearSVMFromScratch`: siêu phẳng lề cực đại, giải thuật Pegasos Subgradient SGD trên Hinge Loss và Squared Hinge, Platt Scaling Sigmoid. |
| **Naive Bayes** | Workflow 2: Model Training | `src/models/classifiers.py` | `NaiveBayesClassifier`: định lý Bayes, làm mịn Laplace, hỗ trợ cả Multinomial NB và Complement NB; tích hợp **Log-Sum-Exp Trick** chống tràn số. |
| **Hyperparameter Tuning** | Additional Considerations | `src/models/tuning.py` | `tune_model_hyperparameters()`: Grid Search trên tập Validation tối ưu $C$, $learning\_rate$, $\alpha$ với hàm mục tiêu có ràng buộc (Constrained Objective). |
| **Threshold Tuning ($\text{Recall} \ge 0.85$)** | Additional Considerations | `src/models/tuning.py` | `ThresholdOptimizerScratch`: quét 200 mốc ngưỡng xác suất trên Validation, lọc ràng buộc $\text{Recall} \ge 0.85$, tối đa hóa F1 và F-beta ($\beta=2.0$). |
| **Ensemble Methods** | Additional Considerations | `src/models/ensemble.py` | `SpamVotingEnsemble`: kết hợp biểu quyết Soft/Hard Voting giữa 3 mô hình cốt lõi; `build_tree_based_ensembles()` bọc ngoại lệ an toàn. |
| **Accuracy, Precision, Recall, F1-score** | Workflow 3: Model Evaluation | `src/evaluation/metrics.py` | `SpamModelEvaluator`: tự tính toán 10 chỉ số đánh giá cốt lõi, ma trận nhầm lẫn 2x2, cơ chế Safe Division chống chia cho 0. |
| **Area Under ROC Curve (ROC-AUC)** | Workflow 3: Model Evaluation | `src/evaluation/metrics.py` | `compute_roc_auc()`: tự cài đặt định lý xếp hạng Mann-Whitney U test ($O(N \log N)$), tính diện tích chính xác tuyệt đối mà không cần tích phân xấp xỉ. |
| **Visualization & Reporting** | Workflow 3: Model Evaluation | `src/evaluation/visualization.py` | Xuất đồ thị đôi ROC-PR đánh dấu ngưỡng $\tau^*$, Heatmap ma trận nhầm lẫn chuẩn hóa kép (số lượng + %), biểu đồ cột nhóm so sánh đa mô hình. |
| **Error Analysis (FP & FN)** | Workflow 3: Model Evaluation | `src/evaluation/error_analysis.py` | `extract_misclassified_samples()` & `analyze_qualitative_errors()`: bóc tách mẫu lỗi sai, phân tích từ khóa gây nhiễu, xuất file `misclassified_samples.csv`. |
| **Model Deployment & Inference** | Workflow 4: Deployment | `src/deployment/predictor.py` | `SpamInferenceService`: tiếp nhận email thô mới, tự động chạy qua toàn bộ pipeline và trả về nhãn + xác suất + mức độ rủi ro + tín hiệu kích hoạt. |

---

## 7. Hướng Dẫn Sử Dụng & Khởi Chạy

### 7.1. Cài đặt Môi trường
Dự án yêu cầu môi trường Python 3.10 trở lên. Cài đặt các thư viện phụ thuộc bằng lệnh:
```bash
pip install -r requirements.txt
```

### 7.2. Chạy Toàn Bộ Luồng Quy Trình Điều Phối Chính (End-to-End Pipeline)
Thực thi entry point chính để chạy khép kín từ nạp dữ liệu, làm sạch, trích xuất đặc trưng, huấn luyện 3 mô hình, tối ưu ngưỡng, đánh giá, phân tích lỗi đến demo suy luận trực tiếp trên 5 email thực tế:
```bash
python main.py
```
*Kết quả chính thức sẽ được tự động xuất bản vào:*
- Báo cáo biểu đồ chất lượng cao: `results/figures/` (Confusion Matrix Heatmap, ROC-PR Curves, Models Comparison Bar).
- Danh sách mẫu phân loại sai: `results/metrics/misclassified_samples.csv`.
- Artifact pipeline hoàn chỉnh: `results/saved_models/spam_pipeline.joblib`.
- Nhật ký thực thi chi tiết: `logs/pipeline.log`.

### 7.3. Chạy Bộ Kiểm Thử Tự Động (Unit Tests)
Khởi chạy bộ kiểm thử đơn vị toàn diện để xác thực tính toàn vẹn của mã nguồn:
```bash
python -m unittest discover tests
```
*Kết quả kiểm thử đạt chuẩn nghiệm thu tuyệt đối: **18/18 test cases PASS (100% OK trong dưới 0.2 giây)**.*

### 7.4. Khởi Động & Khám Phá Trên Jupyter Notebook
Dự án cung cấp bộ 5 notebooks nghiên cứu thực nghiệm độc lập trong thư mục `notebooks/`, thực thi tuần tự từ `01` đến `05`:
1. `notebooks/01_eda.ipynb`: Phân tích khám phá dữ liệu (EDA), phân bố nhãn Spam/Ham, thống kê độ dài và từ khóa.
2. `notebooks/02_preprocessing_and_features.ipynb`: Làm sạch văn bản, kiểm toán dữ liệu và trích xuất ma trận đặc trưng lai.
3. `notebooks/03_model_baseline_and_training.ipynb`: Huấn luyện và đánh giá 3 mô hình cốt lõi (Logistic Regression, SVM, Naive Bayes).
4. `notebooks/04_tuning_and_ensemble.ipynb`: Tinh chỉnh siêu tham số Grid Search, tối ưu ngưỡng quyết định ($\text{Recall} \ge 0.85$) và mô hình biểu quyết Ensemble.
5. `notebooks/05_error_analysis_and_deployment.ipynb`: Mổ xẻ phân tích lỗi sai (FP/FN), lưu trữ pipeline và demo suy luận email mới.

*Toàn bộ kết quả thực nghiệm trung gian của Notebooks được lưu trữ độc lập tại `notebooks/results/`.*

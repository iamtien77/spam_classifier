# 📓 Quy Trình Nghiên Cứu Thực Nghiệm Trên Jupyter Notebooks

Thư mục này chứa bộ **5 Jupyter Notebooks** nghiên cứu thực nghiệm độc lập, được thiết kế theo chu trình chuẩn mực của bài toán Học Máy phân loại email rác (**Spam Email Classifier**) tuân thủ triết lý **100% From-Scratch**.

> **Nguyên tắc kỹ thuật cốt lõi**:
> - **Triết lý From-Scratch tuyệt đối**: Sử dụng toàn bộ các thuật toán tự lập trình từ module `src/` (không phụ thuộc vào các lớp đóng gói sẵn của Scikit-Learn).
> - **Triệt tiêu rò rỉ dữ liệu (Zero Data Leakage)**: Mọi khâu học từ vựng (Vocabulary), trọng số IDF và ngưỡng co dãn (MaxAbs) chỉ được thực hiện trên tập **Train**; tinh chỉnh siêu tham số và chọn ngưỡng quyết định chỉ thực hiện trên tập **Validation**; tập **Test** được niêm phong hoàn toàn làm Unseen Data cho đánh giá cuối cùng.
> - **Ràng buộc an ninh mạng tiên quyết**: Tối ưu ngưỡng quyết định $\tau^*$ trên tập Validation bảo đảm $\text{Recall} \ge 0.85$, ưu tiên tối đa việc phát hiện và ngăn chặn thư rác/phishing.

---

## 1. Danh Mục & Tóm Tắt 5 Jupyter Notebooks

| # | Tệp Notebook | Trọng tâm Kỹ thuật & Nghiên cứu | Đầu vào (Input) | Đầu ra (Output) |
|---|---|---|---|---|
| **01** | `01_eda.ipynb` | **Khám phá Dữ liệu (EDA)**: Phân tích phân bố nhãn Spam/Ham (~86.6% vs ~13.4%), thống kê độ dài ký tự, từ vựng, tỷ lệ chữ hoa, tần suất từ khóa nhạy cảm và ký tự đặc biệt (`!`, `$`). | `data/raw/spam.csv` | Biểu đồ phân tích EDA, bảng thống kê mô tả đặc trưng. |
| **02** | `02_preprocessing_and_features.ipynb` | **Kiểm toán, Phân tầng & Đặc trưng Lai**: Kiểm toán 4 bước `clean_email_split()`, Early Stratified Split 80/10/10, TF-IDF sublinear From-Scratch, trích xuất cấu trúc Single-pass $O(L)$, co dãn bảo toàn tính thưa `MaxAbsScalerScratch`, ghép nối ma trận thưa nén `CSRMatrixScratch` shape $(N, 1667)$ và đối sánh 5 tổ hợp đặc trưng. | `data/raw/spam.csv` | Cache ma trận đặc trưng `X_train`, `X_val`, `X_test` và nhãn `y` trong `data/processed/`. |
| **03** | `03_model_baseline_and_training.ipynb` | **Huấn luyện 3 Mô hình Baseline From-Scratch**: Huấn luyện Logistic Regression (BCE loss, L2 reg, adaptive decay), Linear SVM (Pegasos subgradient Hinge loss), Complement Naive Bayes (Log-Sum-Exp Trick). Đánh giá baseline trên tập Validation. | `data/processed/` hoặc ma trận sinh từ notebook 02 | Bảng so sánh baseline định lượng, biểu đồ đường cong hàm mất mát (Loss curves). |
| **04** | `04_tuning_and_ensemble.ipynb` | **Tối ưu Siêu tham số, Ngưỡng & Ensemble**: Grid Search siêu tham số có ràng buộc, quét 200 mốc ngưỡng `ThresholdOptimizerScratch` ($\text{Recall} \ge 0.85$), kết hợp mô hình biểu quyết `SpamVotingEnsemble` (Soft & Hard Voting), chọn mô hình chiến thắng và đóng gói pipeline. | `data/processed/` | Ngưỡng tối ưu $\tau^*$, artifact pipeline thực nghiệm `notebooks/results/saved_models/spam_pipeline.joblib`. |
| **05** | `05_error_analysis_and_deployment.ipynb` | **Đánh giá Cuối, Mổ xẻ Lỗi & Demo Suy luận**: Đánh giá khách quan trên Test set (Unseen Data), vẽ Heatmap ma trận nhầm lẫn chuẩn hóa kép và đồ thị đôi ROC-PR, bóc tách lỗi FP/FN, chẩn đoán từ khóa gây nhiễu, demo dịch vụ suy luận thời gian thực `SpamInferenceService`. | `data/processed/` & `notebooks/results/saved_models/` | Báo cáo lỗi `notebooks/results/metrics/misclassified_samples.csv`, 3 biểu đồ 300 DPI trong `notebooks/results/figures/`. |

---

## 2. Luồng Trao Đổi Dữ Liệu Giữa Các Notebooks (Data Flow Pipeline)

```
[data/raw/spam.csv: 43,849 dòng]
       │
       ▼
[01_eda.ipynb] ──> Khám phá phân bố nhãn, độ dài email, từ khóa nhạy cảm
       │
       ▼
[02_preprocessing_and_features.ipynb]
       │   • Lọc sạch & khử trùng lặp (41,654 mẫu sạch)
       │   • Early Stratified Split 80/10/10 (Train: 33,322 | Val: 4,166 | Test: 4,166)
       │   • Fit TfidfVectorizerScratch + MaxAbsScalerScratch trên Train
       │   • Ghép nối ma trận lai CSRMatrixScratch (N, 1667)
       │
       ▼ Xuất Cache trung gian vào data/processed/ (X_train, X_val, X_test, y_train, y_val, y_test)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[03_model_baseline_and_training.ipynb]    [04_tuning_and_ensemble.ipynb]
• Huấn luyện 3 mô hình From-Scratch       • Grid Search Tuning siêu tham số
• So sánh baseline trên Validation        • Quét 200 ngưỡng: Chọn tau* (Recall >= 0.85)
                                          • Soft Voting SpamVotingEnsemble
                                          • Đóng gói spam_pipeline.joblib
                                                 │
                                                 ▼
                                  [05_error_analysis_and_deployment.ipynb]
                                  • Đánh giá khách quan trên Test set (Unseen Data)
                                  • Xuất biểu đồ ROC-PR & Heatmap Confusion Matrix
                                  • Bóc tách mẫu lỗi sai FP/FN -> CSV
                                  • Demo suy luận trực tiếp trên email mới
```

---

## 3. Chi Tiết Kỹ Thuật Từng Jupyter Notebook

### 3.1. `01_eda.ipynb` – Khám Phá Dữ Liệu & Thống Kê Đặc Trưng
- **Mục tiêu**: Nắm bắt đặc điểm cấu trúc dữ liệu thô, phát hiện mất cân bằng lớp và nhận diện các tín hiệu phân biệt giữa Spam và Ham.
- **Kỹ thuật thực hiện**:
  - Nạp dữ liệu an toàn với cơ chế fallback encoding 3 tầng (`utf-8` $\rightarrow$ `latin-1` $\rightarrow$ `windows-1252`).
  - Thống kê tỷ lệ phân bố nhãn: Ham chiếm **~86.6%** (36,066 mẫu), Spam chiếm **~13.4%** (5,588 mẫu) sau khi loại bỏ dị biệt.
  - Phân tích trực quan hóa phân phối độ dài văn bản (Spam có xu hướng ngắn và tập trung hơn Ham).
  - Khảo sát tần suất các ký tự kích động (`!`, `$`, `€`, `£`) và tỷ lệ ký tự in hoa.

### 3.2. `02_preprocessing_and_features.ipynb` – Tiền Xử Lý, Phân Tầng & Kỹ Thuật Đặc Trưng Lai
- **Mục tiêu**: Xây dựng pipeline tiền xử lý văn bản NLP và kiến tạo ma trận đặc trưng lai chuẩn Compressed Sparse Row.
- **Kỹ thuật thực hiện**:
  - Kiểm toán chất lượng dữ liệu 4 bước với `clean_email_split()`: loại bỏ bản ghi rỗng, chuẩn hóa nhãn về `{0, 1}`, khử trùng lặp nội dung văn bản.
  - Áp dụng **Early Stratified Split 80/10/10** trước mọi khâu trích xuất đặc trưng để triệt tiêu Data Leakage.
  - Tiền xử lý NLP với `SpamTextProcessor`: chuẩn hóa Unicode NFKD, xóa thẻ HTML, token hóa `<URL>`, `<EMAIL>`, `<MONEY>`, `<NUMBER>`, lọc stop words và dấu câu.
  - Tự cài đặt `CSRMatrixScratch`: cấu trúc ma trận thưa nén 3 mảng (`data`, `indices`, `indptr`) tiết kiệm hơn 98% RAM, tích hợp phép nhân vô hướng `.dot(w)` và chuyển vị `.transpose_dot(v)`.
  - Tự cài đặt `TfidfVectorizerScratch`: Sublinear TF, Smooth IDF, chuẩn hóa Euclidean L2-norm trên 1,600 từ vựng cốt lõi.
  - Tự cài đặt `MaxAbsScalerScratch`: co dãn các đặc trưng số học về $[-1.0, 1.0]$ bảo toàn nguyên vẹn tính thưa.
  - Thử nghiệm đối sánh 5 tổ hợp đặc trưng (`word_tfidf`, `word_char_frequency`, `word_keywords`, `word_structural`, `hybrid`).

### 3.3. `03_model_baseline_and_training.ipynb` – Huấn Luyện 3 Mô Hình Cốt Lõi From-Scratch
- **Mục tiêu**: Hiện thực hóa và đánh giá baseline 3 thuật toán phân loại theo đề bài trên cùng một tập dữ liệu Train/Validation.
- **Kỹ thuật thực hiện**:
  - **Logistic Regression From-Scratch**: tối ưu bằng Gradient Descent với Binary Cross-Entropy loss có điều chuẩn L2, kẹp số trị $z \in [-30, 30]$ chống tràn số mũ, suy giảm tốc độ học thích nghi $\eta_t = \eta_0 / \sqrt{1 + 0.01 t}$, và tích hợp `class_weight='balanced'`.
  - **Linear Support Vector Machines (SVM) From-Scratch**: giải thuật Pegasos Subgradient SGD trên Hinge Loss và Squared Hinge Loss, tích hợp dừng sớm `tol=1e-4` và Platt Scaling Sigmoid.
  - **Naive Bayes Classifier**: Multinomial NB và Complement Naive Bayes (CNB) chuyên trị mất cân bằng lớp; tính tần suất bằng phép nhân ma trận chuyển vị với mask nhị phân ($O(\text{nnz})$, loại bỏ boolean row-slicing); tích hợp **Log-Sum-Exp Trick** chống tràn số.
  - Sử dụng các hàm đại số tuyến tính thưa an toàn đa hình (`_safe_dot`, `_safe_transpose_dot`) chạy trực tiếp trên ma trận thưa.

### 3.4. `04_tuning_and_ensemble.ipynb` – Tinh Chỉnh Siêu Tham Số, Ngưỡng & Ensemble
- **Mục tiêu**: Tối ưu hóa siêu tham số mô hình, tìm ngưỡng quyết định thỏa mãn ràng buộc nghiệp vụ an ninh mạng và xây dựng mô hình kết hợp.
- **Kỹ thuật thực hiện**:
  - **Grid Search có ràng buộc (Constrained Objective)**: hàm `tune_model_hyperparameters()` dò quét tổ hợp siêu tham số ($C$, $learning\_rate$, $\alpha$) trên tập Validation, phạt nặng các cấu hình không đạt Recall mục tiêu.
  - **Tối ưu hóa ngưỡng phân loại From-Scratch (`ThresholdOptimizerScratch`)**: quét lưới 200 mốc ngưỡng $\tau \in [0.005, 1.0]$ trên Validation set, tính toàn bộ chỉ số từ ma trận nhầm lẫn bằng NumPy thuần, lọc điều kiện tiên quyết $\text{Recall} \ge 0.85$ trước khi tối đa hóa F1 và F-beta ($\beta = 2.0$).
  - **Mô hình kết hợp biểu quyết (`SpamVotingEnsemble`)**: kết hợp Soft Voting (trung bình xác suất có trọng số) và Hard Voting giữa LR, SVM và Naive Bayes.
  - Chọn mô hình chiến thắng (`soft_voting_ensemble`, ngưỡng $\tau^* = 0.465$, Recall Validation đạt $96.37\%$).
  - Đóng gói toàn bộ bundle hoàn chỉnh vào `notebooks/results/saved_models/spam_pipeline.joblib`.

### 3.5. `05_error_analysis_and_deployment.ipynb` – Đánh Giá Cuối, Phân Tích Lỗi & Triển Khai
- **Mục tiêu**: Thẩm định độc lập trên tập Test chưa từng thấy, mổ xẻ nguyên nhân phân loại sai và kiểm thử dịch vụ suy luận trực tiếp.
- **Kỹ thuật thực hiện**:
  - Đánh giá toàn diện 10 chỉ số với `SpamModelEvaluator`: Accuracy, Precision, Recall, Specificity, F1, F2 và diện tích ROC-AUC chính xác tuyệt đối qua định lý thứ hạng Mann-Whitney U ($O(N \log N)$).
  - Trực quan hóa báo cáo: đồ thị đôi ROC-PR Curve đánh dấu ngưỡng tối ưu $\tau^*$ bằng ngôi sao đỏ; Heatmap ma trận nhầm lẫn chuẩn hóa kép (số lượng mẫu + tỷ lệ %); biểu đồ cột nhóm so sánh hiệu năng.
  - **Bóc tách lỗi sai chuyên sâu (`extract_misclassified_samples`)**: lọc toàn bộ mẫu False Positives (Ham bị chặn nhầm) và False Negatives (Spam lọt lưới), xuất file `notebooks/results/metrics/misclassified_samples.csv`.
  - **Chẩn đoán định tính (`analyze_qualitative_errors`)**: thống kê top từ khóa gây kích hoạt giả trong FP và phân phối độ dài ngắn trong FN.
  - **Demo dịch vụ suy luận thời gian thực (`SpamInferenceService`)**: tiếp nhận email thô mới, tự động chạy qua NLP $\rightarrow$ feature builder $\rightarrow$ predict proba $\rightarrow$ so sánh ngưỡng $\tau^*$ $\rightarrow$ trả kết quả JSON chi tiết (nhãn, xác suất, mức độ rủi ro LOW/HIGH/CRITICAL, tín hiệu kích hoạt).

---

## 4. Cấu Trúc Thư Mục Khu Vực Notebooks

```
notebooks/
├── README.md                                 # Hướng dẫn quy trình thực nghiệm (tài liệu này)
├── 01_eda.ipynb                              # Notebook 01: Khám phá dữ liệu & tương quan
├── 02_preprocessing_and_features.ipynb       # Notebook 02: Tiền xử lý, CSR thưa & đặc trưng lai
├── 03_model_baseline_and_training.ipynb       # Notebook 03: Huấn luyện 3 mô hình From-Scratch
├── 04_tuning_and_ensemble.ipynb               # Notebook 04: Grid Search, 200 ngưỡng & Ensemble
├── 05_error_analysis_and_deployment.ipynb     # Notebook 05: Đánh giá Test, phân tích lỗi & demo
└── results/                                  # Thư mục lưu trữ kết quả thực nghiệm độc lập của Notebooks
    ├── figures/                              # Biểu đồ thực nghiệm xuất bản độ phân giải cao
    │   ├── confusion_matrix_heatmap.png      # Heatmap ma trận nhầm lẫn chuẩn hóa kép
    │   ├── roc_pr_curves.png                 # Đồ thị đôi ROC và Precision-Recall Curve
    │   └── models_comparison_bar.png         # Biểu đồ cột nhóm so sánh đa mô hình
    ├── metrics/                              # Báo cáo số liệu thực nghiệm chi tiết
    │   └── misclassified_samples.csv         # Danh sách mẫu email phân loại sai (FP & FN)
    └── saved_models/                         # Pipeline bundle thực nghiệm
        └── spam_pipeline.joblib              # Bundle mô hình, threshold, vectorizer và scaler
```

---

## 5. Nguyên Tắc Kỹ Thuật Khi Chạy Notebooks

1. **Cơ chế đường dẫn động tương đối**:
   Toàn bộ các Notebook sử dụng cơ chế xác định đường dẫn gốc dự án tự động thông qua `Path.cwd().resolve()` hoặc `Path(__file__)`, tuyệt đối không hardcode đường dẫn ổ đĩa cố định (như `C:\` hay `D:\`), bảo đảm khả năng tương thích đa nền tảng trên Windows, Linux và macOS.
2. **Tuân thủ thứ tự thực thi tuần tự**:
   Do Notebook 03, 04, 05 cần sử dụng dữ liệu đã phân tầng và các ma trận đặc trưng được sinh ra từ Notebook 02, người dùng cần chạy tuần tự từ `01` $\rightarrow$ `05`.
3. **Tuyệt đối không vi phạm nguyên tắc chống rò rỉ**:
   Không gọi phương thức `.fit()` của bất kỳ bộ biến đổi đặc trưng nào trên tập Validation hoặc Test. Toàn bộ từ vựng và tham số chuẩn hóa chỉ được học duy nhất trên tập Train.

---

## 6. Hướng Dẫn Khởi Chạy

Từ thư mục gốc của dự án, kích hoạt môi trường ảo Python và khởi động Jupyter Notebook:

```bash
# Khởi động Jupyter Notebook
jupyter notebook notebooks/
```

Hoặc sử dụng JupyterLab:
```bash
jupyter lab notebooks/
```

Mở từng tệp notebook theo thứ tự từ `01_eda.ipynb` đến `05_error_analysis_and_deployment.ipynb`, chọn **Kernel -> Restart & Run All** để thực thi và tái lập toàn bộ kết quả thực nghiệm khoa học.

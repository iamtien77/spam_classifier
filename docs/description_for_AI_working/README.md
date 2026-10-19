# 🧭 Mục Lục Toàn Diện: Hướng Dẫn Kỹ Thuật Dự Án (Description for AI Working)

Thư mục này chứa toàn bộ hệ thống các kịch bản mô tả kỹ thuật chi tiết theo từng phiên bản tiến hóa (`v1` đến `vn`), tái hiện đầy đủ tiến trình lặp (Iterative Progression) của dự án **Spam Email Classifier** tại [`main.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/main.py) và thư mục [`src/`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src).

Mỗi tài liệu được chuẩn hóa thành các hành động kỹ thuật cụ thể theo cấu trúc: `Bước 1. ...`, `Bước 2. ...` quy định rõ đầu vào, logic xử lý thuật toán, kiểm soát lỗi biên, phòng chống rò rỉ dữ liệu (Data Leakage) và số liệu kiểm chứng thực nghiệm.

---

## 1. Bảng Điều Phối Theo Giai Đoạn Lớn (4 Phase Summaries)

| Giai đoạn | File tổng quan giai đoạn | Hàm điều phối trong `main.py` | Module mã nguồn chính |
| :---: | :--- | :--- | :--- |
| **GĐ 1** | 📄 [01_data_preprocessing_and_audit.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/01_data_preprocessing_and_audit.md) | `step_data_preprocessing()` | [`src/data/loader.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/data/loader.py)<br>[`src/data/preprocessing.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/data/preprocessing.py) |
| **GĐ 2** | 📄 [02_feature_engineering_and_selection.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/02_feature_engineering_and_selection.md) | `step_data_preprocessing()` | [`src/features/feature_engineering.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/features/feature_engineering.py) |
| **GĐ 3** | 📄 [03_model_training_and_tuning.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/03_model_training_and_tuning.md) | `step_model_training()`<br>`step_evaluation_and_tuning()` | [`src/models/classifiers.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/classifiers.py)<br>[`src/models/ensemble.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/ensemble.py)<br>[`src/models/tuning.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/tuning.py) |
| **GĐ 4** | 📄 [04_evaluation_and_deployment.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/04_evaluation_and_deployment.md) | `step_evaluation_and_tuning()`<br>`step_deployment_and_demo()` | [`src/evaluation/metrics.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/evaluation/metrics.py)<br>[`src/evaluation/visualization.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/evaluation/visualization.py)<br>[`src/evaluation/error_analysis.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/evaluation/error_analysis.py)<br>[`src/evaluation/experiments.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/evaluation/experiments.py)<br>[`src/models/persistence.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/persistence.py)<br>[`src/deployment/predictor.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/deployment/predictor.py) |

---

## 2. Hệ Thống 37 Phiên Bản Tiến Hóa Chi Tiết (Iterative Progression Files)

### 📊 Nhóm 1: Khám Phá & Phân Tích Dữ Liệu (EDA - 6 bản)
- [EDAv1.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/EDAv1.md): Kế hoạch EDA ban đầu, nạp DataFrame, kiểm tra schema, xem trước nội dung.
- [EDAv2.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/EDAv2.md): Chuẩn hóa nhãn, kiểm tra dữ liệu thiếu và phân bố lớp.
- [EDAv3.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/EDAv3.md): Thống kê độ dài email (ký tự, từ, các phân vị mô tả q95, q99).
- [EDAv4.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/EDAv4.md): Phân tích tín hiệu cấu trúc spam (tần suất số, chữ hoa, dấu !, $, URL).
- [EDAv5.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/EDAv5.md): Phân tích từ vựng, 10 từ khóa nhạy cảm và n-gram giữa hai lớp.
- [EDAv6.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/EDAv6.md): Phân tích tương quan đặc trưng và phát hiện 738 nhóm mâu thuẫn nhãn.

### 🧹 Nhóm 2: Tiền Xử Lý & Kiểm Toán Dữ Liệu (Processing Data - 5 bản)
- [processing-data-v1.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/processing-data-v1.md): Nạp dữ liệu thô, cơ chế giải mã fallback (`utf-8-sig` / `latin-1`), chuẩn hóa schema và sinh `source_id`.
- [processing-data-v2.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/processing-data-v2.md): Bóc tách HTML, thế chỗ token (`urltoken`, `moneytoken`, `numbertoken`), lọc stop-words giữ từ phủ định.
- [processing-data-v3.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/processing-data-v3.md): Kiểm toán 5 bước không chồng lặp (lọc nhãn sai, nội dung rỗng, khử mâu thuẫn, khử trùng lặp $\rightarrow$ $29.028$ mẫu sạch).
- [processing-data-v4.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/processing-data-v4.md): Phân chia phân tầng Train/Val/Test ($80/10/10$, seed 42) bằng thuật toán phần dư lớn nhất.
- [processing-data-v5.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/processing-data-v5.md): Cắt độ dài văn bản (Length Capping: phân vị 99% = $1.361$ từ) chỉ fit trên Train để chống rò rỉ dữ liệu.

### ⚙️ Nhóm 3: Kỹ Thuật Biểu Diễn Đặc Trưng (Feature Engineering - 8 bản)
- [feature-engineering-v1.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/feature-engineering-v1.md): Cài đặt cấu trúc ma trận thưa tự viết `CSRMatrixScratch` (`iter_row`, `dot`, `transpose_dot`).
- [feature-engineering-v2.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/feature-engineering-v2.md): Tự cài đặt thuật toán TF-IDF (`TfidfVectorizerScratch`) với sublinear TF và smooth IDF.
- [feature-engineering-v3.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/feature-engineering-v3.md): Trích xuất khối đặc trưng Word Unigram ($3.000$ từ vựng, $\text{min\_df}=2$).
- [feature-engineering-v4.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/feature-engineering-v4.md): Trích xuất khối đặc trưng Word Bigram ($800$ cặp từ ngữ cảnh).
- [feature-engineering-v5.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/feature-engineering-v5.md): Trích xuất khối đặc trưng Character n-gram ($500$ chuỗi ký tự 3-5).
- [feature-engineering-v6.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/feature-engineering-v6.md): Trích xuất 10 từ khóa nhạy cảm và 8 đặc trưng số co dãn bằng `MaxAbsScalerScratch`.
- [feature-engineering-v7.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/feature-engineering-v7.md): Ghép nối 5 khối đặc trưng lai $4.318$ chiều bằng `HybridFeatureBuilderScratch`.
- [feature-engineering-v8.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/feature-engineering-v8.md): Thử nghiệm lựa chọn đặc trưng Top-K và lưu trữ ma trận đệm `.npz`, `.npy`.

### 🧠 Nhóm 4: Huấn Luyện Mô Hình (Model Training - 5 bản)
- [train-v1.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/train-v1.md): Thiết kế kiến trúc học máy From-Scratch và chuẩn bị đầu vào ma trận thưa.
- [train-v2.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/train-v2.md): Huấn luyện mô hình cơ sở Complement Naive Bayes (Complement counts + Laplace smoothing).
- [train-v3.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/train-v3.md): Huấn luyện Linear Support Vector Machine với Squared Hinge Loss và điều chuẩn $L_2$.
- [train-v4.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/train-v4.md): Huấn luyện Logistic Regression với Spectral Gradient Descent và Armijo Backtracking.
- [train-v5.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/train-v5.md): Xây dựng mô hình tổ hợp biểu quyết `SpamVotingEnsemble` (Soft & Hard Voting).

### 🎯 Nhóm 5: Tối Ưu Siêu Tham Số & Dò Ngưỡng (Tuning - 6 bản)
- [tuning-v1.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/tuning-v1.md): Hạn chế của ngưỡng mặc định 0.5 trong bài toán lọc thư rác thực tế.
- [tuning-v2.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/tuning-v2.md): Thiết kế bộ dò ngưỡng `ThresholdOptimizerScratch` quét 200 mốc phân vị trên Validation set.
- [tuning-v3.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/tuning-v3.md): Thiết lập hàm mục tiêu có ràng buộc nghiệp vụ: Bắt buộc $\text{Recall} \ge 0.85$, tối đa hóa Accuracy.
- [tuning-v4.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/tuning-v4.md): Quét lưới siêu tham số (Grid Search) cho LR ($C$), SVM ($C$) và Naive Bayes ($\alpha$).
- [tuning-v5.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/tuning-v5.md): Quy trình so sánh và xếp hạng 8 mô hình ứng viên trên tập Validation.
- [tuning-v6.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/tuning-v6.md): Chốt mô hình chiến thắng (Winner): Linear SVM Tuned ($C=2.0$, ngưỡng $0.465$).

### 📈 Nhóm 6: Đánh Giá & Triển Khai (Evaluation & Deployment - 7 bản)
- [evaluation-v1.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/evaluation-v1.md): Đánh giá độc lập trên Test set tại ngưỡng đã chốt, tính ma trận nhầm lẫn ($997$ TN, $666$ FP, $180$ FN, $1.059$ TP).
- [evaluation-v2.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/evaluation-v2.md): Tính toán bộ chỉ số mở rộng: Accuracy ($70.85\%$), Precision ($61.39\%$), Recall ($85.47\%$), F1 ($71.46\%$), F2 ($79.25\%$).
- [evaluation-v3.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/evaluation-v3.md): Tự cài đặt và đánh giá ROC-AUC ($81.44\%$) bằng thống kê thứ bậc Wilcoxon có xử lý ties.
- [evaluation-v4.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/evaluation-v4.md): Phân tích lỗi chi tiết (Error Analysis), trích xuất 846 mẫu đoán sai ra `misclassified_samples.csv`.
- [evaluation-v5.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/evaluation-v5.md): Trực quan hóa biểu đồ Confusion Matrix heatmap, ROC curve, PR curve ra `results/figures/`.
- [evaluation-v6.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/evaluation-v6.md): Đóng gói persistence artifact bundle joblib và xác thực toàn vẹn 21 SHA-256 trong `run_summary.json`.
- [evaluation-v7.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/description_for_AI_working/evaluation-v7.md): Xây dựng `SpamInferenceService` suy luận email mới và chạy kịch bản kiểm thử demo.

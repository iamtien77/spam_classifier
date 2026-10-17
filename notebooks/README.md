# Quy trình thực nghiệm trên Jupyter Notebooks

Chạy lần lượt 01 → 02 → 03 → 04 → 05 từ kernel sạch. Các notebook dùng thuật toán from-scratch trong `src/`; không cần chạy `main.py` để tạo artifacts.

`main.py` là thí nghiệm tham chiếu cho README và báo cáo Word. Chuỗi notebook có baseline, lưới tham số và thành phần ensemble riêng nên không yêu cầu kết quả số trùng `main.py`; không trộn kết quả giữa hai thư mục. Dùng Restart & Run All trong Jupyter cho từng notebook và lưu outputs sau khi chạy thành công.

LR/SVM hiện dùng gradient descent với spectral step và Armijo backtracking. SVM squared hinge dùng cùng cách chuẩn hóa L2 `lambda = 1/(C*N)` như nhánh hinge. Đây không phải Pegasos; sigmoid điểm SVM chưa được calibration.

| Notebook | Thực nghiệm và đầu ra |
|---|---|
| 01 — EDA | Kiểm toán dữ liệu, phân bố nhãn, độ dài/ngoại lai, từ khóa, tín hiệu và tương quan. EDA toàn bộ dữ liệu chỉ mang tính mô tả. |
| 02 — Preprocessing & features | Lọc chất lượng và trùng nội dung trước chia tập; fit IDF/scaler/capping trên Train. So sánh 5 tổ hợp đặc trưng bằng cùng Complement NB tại threshold 0.5 trên Validation; quét Top-K và áp dụng đúng thứ tự cột cho ba tập. |
| 03 — Baseline | Train LR, Linear SVM và Complement NB; so sánh Accuracy, Precision, Recall, F1, F2, ROC-AUC trên Validation; xem loss curves và lưu baseline. Không đánh giá Test. |
| 04 — Tuning & ensemble | Grid Search LR/SVM/NB, tối ưu threshold từng cấu hình; so sánh Default–Tuned–Soft Voting–Hard Voting trên Validation và chọn mô hình. Hard Voting giữ threshold 0.5. |
| 05 — Test & deployment | Tính lại metrics cho mọi ứng viên tại threshold đã chốt; giữ nguyên winner từ Validation. Tạo confusion matrix, ROC/PR, bảng so sánh, CSV FP/FN; kiểm tra cache–inference và lưu/nạp; demo 5 email. |

## Chỉ số và tiêu chí chọn

- Đánh giá mô hình báo cáo Accuracy, Precision, Recall, F1, F2 (beta=2), ROC-AUC và confusion matrix. Bảng ablation đặc trưng báo Accuracy, Precision, Recall, F1 và ROC-AUC.
- Logic hiện tại chọn ứng viên đạt Recall mục tiêu trong `SpamExperimentConfig`, sau đó tối đa Accuracy và phân xử bằng F1. Nếu không đạt mục tiêu, ưu tiên Recall rồi Accuracy/F1.
- Đây là tiêu chí hiện có của project, khác mô tả max F1/F2 trong tài liệu cũ. Không gọi `objective_score` của Grid Search là Validation F1.
- Ablation dùng cùng mô hình và threshold cố định để so sánh đóng góp của các nhóm đặc trưng; pipeline tiếp tục dùng hybrid + Top-K. Bảng ablation không tự đổi nhóm đặc trưng.
- Không dùng Test để lựa chọn tham số, ngưỡng hay ứng viên. Phân tích lỗi Test mô tả kết quả đã chốt; nếu dùng để phát triển vòng tiếp theo thì cần tập đánh giá mới.

## Bàn giao và tái lập

Notebook 02 ghi cache **sau Top-K** vào `data/processed/` cùng `notebook_run_id.txt` và lưu `feature_artifacts.joblib` trong `notebooks/results/saved_models/`.
Artifacts chứa config, bộ biến đổi đã fit, thứ tự cột `selected_indices`, danh sách `source_id`, `run_id` và hash SHA-256 của từng ma trận/nhãn/CSV. Marker bị vô hiệu hóa trước khi xuất cache và chỉ công bố lại khi xuất xong; notebook sau xác minh hash trước khi huấn luyện/đánh giá.
Notebook 03 lưu `baseline_models.joblib`; notebook 04 lưu `spam_pipeline.joblib` có các ứng viên, threshold và bảng chọn Validation.
Notebook 05 tự tạo kết quả trong `notebooks/results/metrics/` và `notebooks/results/figures/`, cập nhật metrics Test của bundle.
Không lấy model, builder hoặc CSV lỗi từ `results/` của production. Nếu run_id không khớp, chạy lại chuỗi từ notebook 02.

Các bảng CSV gồm `feature_ablation.csv`, `top_k_sweep.csv`, `baseline_validation.csv`, `hyperparameter_tuning.csv`, `validation_model_selection.csv`, `test_models_comparison.csv`, `misclassified_samples.csv`.
Notebook 05 kiểm tra 10 mẫu Test đầu có cùng đặc trưng, điểm số và quyết định sau lưu/nạp. Đây là kiểm tra parity theo mẫu, không phải chứng minh mọi đầu vào.
Kích thước tập, số chiều và mô hình thắng được tính từ dữ liệu hiện tại; không gán trước số liệu.

## Giới hạn thực nghiệm

- Một split cố định: chưa đánh giá độ ổn định qua nhiều seed, cross-validation hoặc khoảng tin cậy.
- Sigmoid của SVM và điểm Voting chưa được calibration; Hard Voting trả tỷ lệ phiếu có trọng số.
- Grid Search dùng lưới hữu hạn; tuning nhiều lần trên cùng Validation có thể overfit.
- Random Forest/Boosting chưa có thực nghiệm from-scratch. Không thêm sklearn để thay thuật toán lõi.
- Demo email mới không có nhãn chuẩn nên không dùng làm số đo chất lượng.
- Các số liệu/ảnh cũ cần được tạo lại bằng Restart & Run All; có execution count không đủ để chứng minh kết quả khớp mã nguồn hiện tại.

Phiên bản v5. Quy trình so sánh và xếp hạng 8 ứng viên (Model Selection Workflow)
Phiên bản này cài đặt lớp ModelSelectionWorkflow để tập hợp 8 cấu hình mô hình (4 Default và 4 Tuned), so sánh trên cùng tập Validation set và thiết lập bảng xếp hạng tổng thể.

Bước 1. Danh sách 8 mô hình ứng viên được so sánh
Nhánh tham số mặc định (Default Parameters):
1. Linear SVM (Default): C = 1.0, ngưỡng tối ưu tau = 0.465.
2. Logistic Regression (Default): C = 1.0, ngưỡng tối ưu tau = 0.410.
3. Complement Naive Bayes (Default): alpha = 1.0, ngưỡng tối ưu tau = 0.215.
4. SpamVotingEnsemble (Default): Soft voting giữa 3 mô hình mặc định, ngưỡng tối ưu tau = 0.365.

Nhánh tham số tinh chỉnh (Tuned Parameters qua Grid Search):
5. Linear SVM (Tuned): C = 2.0, ngưỡng tối ưu tau = 0.465.
6. Logistic Regression (Tuned): C = 0.5, ngưỡng tối ưu tau = 0.415.
7. Complement Naive Bayes (Tuned): alpha = 2.0, ngưỡng tối ưu tau = 0.220.
8. SpamVotingEnsemble (Tuned): Soft voting giữa 3 mô hình đã tuning, ngưỡng tối ưu tau = 0.385.

Bước 2. So sánh trên 2.903 mẫu của Validation Set
Với từng mô hình trong 8 ứng viên:
1. Dự đoán xác suất trên X_val.
2. Áp dụng ngưỡng tối ưu tương ứng để sinh nhãn dự đoán y_pred.
3. Tính toán toàn bộ bộ chỉ số: accuracy, precision, recall, f1, f2, fpr, target_met.
4. Kiểm tra điều kiện ràng buộc: Cả 8 ứng viên đều đạt target_met = True (Recall >= 0.85).

Bước 3. Xếp hạng thứ tự ưu tiên
Sắp xếp 8 ứng viên theo thứ tự: Validation Accuracy giảm dần -> F1 giảm dần -> Precision giảm dần.
Lập bảng so sánh chi tiết và xuất ra file results/metrics/validation_model_selection.csv.
Tất cả các chỉ số được lưu giữ minh bạch để phục vụ lựa chọn mô hình được chọn.

Phiên bản v4. Quét lưới siêu tham số (Hyperparameter Grid Search) kết hợp Threshold Optimizer
Mục tiêu của bản này là cài đặt hàm tune_model_hyperparameters() để thực hiện tìm kiếm toàn diện trên lưới siêu tham số cho 3 mô hình, kết hợp dò ngưỡng tối ưu trên tập Validation cho từng cấu hình.

Bước 1. Thiết lập không gian tìm kiếm siêu tham số (Param Grid)
Định nghĩa lưới tham số trong main.py:
1. Logistic Regression:
   - C in {0.5, 1.0, 2.0}
   - Cố định learning_rate = 0.1, max_iter = 200, class_weight = 'balanced'.
2. Linear Support Vector Machine:
   - C in {0.5, 1.0, 2.0}
   - Cố định learning_rate = 0.05, max_iter = 150, loss = 'squared_hinge', class_weight = 'balanced'.
3. Complement Naive Bayes:
   - alpha in {0.5, 1.0, 2.0}
   - Cố định kind = 'complement'.

Bước 2. Quy trình huấn luyện và đánh giá trên từng tổ hợp
Hàm tune_model_hyperparameters():
1. Sinh toàn bộ các tổ hợp siêu tham số bằng itertools.product().
2. Với mỗi tổ hợp: Khởi tạo mô hình và gọi fit(X_train, y_train).
3. Dự đoán xác suất trên tập Validation: probs = model.predict_proba(X_val)[:, 1].
4. Đưa xác suất vào ThresholdOptimizerScratch để tìm ngưỡng tối ưu tau tương ứng trên tập Validation.
5. Tính điểm đánh giá theo ràng buộc nghiệp vụ:
   - Nếu Recall >= 0.85: điểm đánh giá là Validation Accuracy.
   - Nếu Recall < 0.85: trừ phạt nặng điểm số để loại bỏ cấu hình vi phạm.

Bước 3. Chọn ra cấu hình tốt nhất cho từng thuật toán
- Logistic Regression (Tuned): Best C = 0.5, ngưỡng tối ưu tau = 0.415.
- Linear SVM (Tuned): Best C = 2.0, ngưỡng tối ưu tau = 0.465.
- Complement Naive Bayes (Tuned): Best alpha = 2.0, ngưỡng tối ưu tau = 0.220.
Toàn bộ các mô hình Tuned này sẵn sàng chuyển sang bước Model Selection tổng thể.

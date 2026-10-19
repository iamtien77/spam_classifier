Phiên bản v5. Xây dựng mô hình tổ hợp biểu quyết (SpamVotingEnsemble)
Mục tiêu của bản này là cài đặt lớp SpamVotingEnsemble để kết hợp dự đoán giữa 3 mô hình nền tảng (Logistic Regression, Linear SVM, Complement Naive Bayes) theo cơ chế Soft Voting và Hard Voting nhằm giảm phương sai dự đoán.

Bước 1. Xác định kiến trúc tổ hợp biểu quyết
Mô hình tổ hợp tiếp nhận từ điển 3 mô hình đã huấn luyện: models = {'lr': model_lr, 'svm': model_svm, 'nb': model_nb}.
Gán vector trọng số kết hợp weights = {'lr': 1.0, 'svm': 1.0, 'nb': 1.0} (trọng số đồng đều).

Bước 2. Cài đặt cơ chế Soft Voting (Bỏ phiếu mềm)
Phương thức predict_proba(X):
1. Thu thập ma trận xác suất từ từng mô hình thành phần: P_lr, P_svm, P_nb.
2. Tính trung bình cộng có trọng số của các ma trận xác suất:
   P_ensemble = (w_lr * P_lr + w_svm * P_svm + w_nb * P_nb) / (w_lr + w_svm + w_nb).
3. Đảm bảo tổng xác suất trên mỗi dòng bằng đúng 1.0.
Phương thức predict(X, threshold): So sánh xác suất lớp Spam với threshold để ra quyết định nhị phân.

Bước 3. Cài đặt cơ chế Hard Voting (Bỏ phiếu cứng theo đa số)
Phương thức predict_hard(X):
1. Lấy nhãn dự đoán nhị phân rời rạc từ từng mô hình: y_pred_lr, y_pred_svm, y_pred_nb.
2. Áp dụng quy tắc đa số phiếu: Nhãn cuối cùng là nhãn nhận được từ 2/3 phiếu bầu trở lên.

Bước 4. Đánh giá tính sẵn sàng của mô hình tổ hợp
SpamVotingEnsemble đóng gói hoàn tất giao diện fit/predict/predict_proba, sẵn sàng tham gia quy trình Model Selection trên tập Validation như một ứng viên độc lập.
Toàn bộ giai đoạn Model Training chính thức hoàn tất.

Phiên bản v4. Huấn luyện Logistic Regression với Spectral Gradient Descent và Armijo Backtracking
Phiên bản này cài đặt và huấn luyện mô hình xác suất Logistic Regression từ đầu bằng Spectral Gradient Descent kết hợp Armijo Backtracking trên hàm mất mát Binary Cross-Entropy có điều chuẩn L2, cung cấp mô hình phân loại có tính giải thích cao.

Bước 1. Xác định hàm mục tiêu Binary Cross-Entropy (BCE)
Mô hình hóa xác suất một email là Spam bằng hàm Sigmoid:
p_i = sigma(w.T @ x_i + b) = 1 / (1 + exp(-(w.T @ x_i + b))).
Thiết lập hàm mất mát BCE có trọng số lớp và điều chuẩn L2:
J(w, b) = - (1 / N) * sum(a_i * [y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i)]) + (lambda / 2) * sum(w_j^2).
Ổn định số học: Sử dụng np.logaddexp(0, -z) để tránh lỗi tràn số khi tính log xác suất.

Bước 2. Cài đặt tối ưu hóa Spectral Gradient Descent
Triển khai lớp LogisticRegressionFromScratch trong src/models/classifiers.py.
Thay thế vòng lặp giảm bước học cố định (dễ bị kẹt tại điểm khởi tạo do giá trị TF-IDF nhỏ) bằng kỹ thuật Spectral Gradient Descent:
1. Tính đạo hàm riêng:
   dJ/dw = lambda * w + (1 / N) * X.T @ (a * (p - y))
   dJ/db = (1 / N) * sum(a * (p - y))
2. Ước lượng bước học tự động theo độ cong Barzilai-Borwein:
   next_step = (displacement @ displacement) / curvature.
3. Kiểm tra điều kiện suy giảm Armijo để co nửa bước học nếu chưa giảm đủ loss.
4. Lặp lại tối đa max_iter = 200 (hoặc 300) vòng lặp.

Bước 3. Dự đoán xác suất hậu nghiệm
Phương thức predict_proba(X): Xuất mảng xác suất 2 cột [1 - p, p], trong đó p là xác suất email là Spam.
Phương thức predict(X, threshold): So sánh p với threshold để phân loại nhị phân.
Sau khi huấn luyện, Logistic Regression được chuyển sang bước tuning và so sánh.

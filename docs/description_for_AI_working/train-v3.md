Phiên bản v3. Huấn luyện Linear Support Vector Machine với Squared Hinge Loss
Phiên bản này cài đặt và huấn luyện Linear SVM từ đầu bằng giải thuật Gradient Descent có Armijo Backtracking trên hàm mất mát Squared Hinge Loss, tìm kiếm siêu phẳng phân tách lề cực đại trên không gian đặc trưng thưa 4.318 chiều.

Bước 1. Xác định hàm mục tiêu Squared Hinge Loss
Chuyển đổi nhãn y in {0, 1} sang nhãn có dấu y_tilde in {-1, +1}.
Thiết lập hàm mục tiêu kết hợp điều chuẩn L2 và trọng số lớp balanced:
J(w, b) = (lambda / 2) * sum(w_j^2) + (1 / N) * sum(a_i * [max(0, 1 - y_tilde_i * (w.T @ x_i + b))]^2)
Trong đó a_i = N / (2 * N_{y_i}) và lambda = 1 / (C * N).
Sử dụng Squared Hinge Loss vì hàm này có đạo hàm bậc nhất liên tục (C^1), cho phép tối ưu trực tiếp bằng Gradient Descent mà tránh điểm gãy của đạo hàm.

Bước 2. Cài đặt thuật toán tối ưu hóa Gradient Descent có Armijo Backtracking
Khởi tạo vector trọng số w = 0 (kích thước 4.318) và bias b = 0.
Sử dụng hàm _gradient_descent_backtracking:
1. Tính điểm số có dấu: z_i = w.T @ x_i + b.
2. Xác định tập các mẫu vi phạm lề: V = {i | 1 - y_tilde_i * z_i > 0}.
3. Tính đạo hàm riêng theo w và b qua phép nhân ma trận thưa transpose_dot.
4. Cập nhật bước học thích ứng theo độ cong Barzilai-Borwein.
5. Co bước học bằng Armijo Backtracking nếu hàm mất mát không giảm đủ.
6. Lặp lại tối đa max_iter = 150 vòng lặp hoặc dừng khi chuẩn gradient norm_sq <= tol^2.

Bước 3. Dự đoán khoảng cách và ánh xạ xác suất Sigmoid
Phương thức decision_function(X): Trả về khoảng cách có dấu w.T @ x + b.
Phương thức predict_proba(X): Đưa điểm z qua hàm Sigmoid 1 / (1 + exp(-z)) để tương thích giao diện thống nhất với bộ dò ngưỡng và Voting Ensemble.
Sau khi huấn luyện, Linear SVM được chuyển sang bước tuning và so sánh.

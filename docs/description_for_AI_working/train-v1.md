Phiên bản v1. Thiết kế kiến trúc học máy From-Scratch và chuẩn bị đầu vào
Phiên bản này thiết lập chuẩn kiến trúc From-Scratch cho toàn bộ các mô hình phân loại trong dự án, xác định giao diện chung (fit, predict, predict_proba) và kiểm tra ma trận đầu vào trước khi huấn luyện.

Bước 1. Xác định nguyên tắc thiết kế From-Scratch
Toàn bộ các mô hình học máy cốt lõi phải được tự viết bằng Python và NumPy/SciPy, không sử dụng các lớp phân loại có sẵn của scikit-learn trong pipeline tham chiếu.
Các mô hình phải tuân thủ chuẩn giao diện scikit-learn-like:
- fit(X, y): Huấn luyện trọng số từ ma trận đặc trưng X và vector nhãn y.
- predict(X, threshold): Dự đoán nhãn nhị phân (0 hoặc 1) dựa trên ngưỡng.
- predict_proba(X): Xuất phân phối xác suất dự đoán cho hai lớp.

Bước 2. Chuẩn bị và kiểm tra ma trận đầu vào
Nạp X_train, y_train từ data/processed/ hoặc nhận trực tiếp từ HybridFeatureBuilderScratch.
Kiểm tra số chiều: X_train phải có đúng 23.223 hàng và 4.318 cột.
Kiểm tra tính không âm: Toàn bộ phần tử x_ij >= 0 (bắt buộc cho Naive Bayes).
Kiểm tra giá trị thiếu: Không chứa NaN hoặc Inf.
Kiểm tra nhãn: y_train chỉ gồm hai giá trị 0 và 1, không có nhãn khác.

Bước 3. Thiết kế chiến lược huấn luyện độc lập 3 họ thuật toán
1. Complement Naive Bayes: Mô hình xác suất thống kê dựa trên tần suất phần bù, đóng vai trò Baseline.
2. Linear Support Vector Machine: Mô hình tìm siêu phẳng lề cực đại với hàm mất mát Squared Hinge, tối ưu bằng Gradient Descent.
3. Logistic Regression: Mô hình xác suất nhị phân với Binary Cross-Entropy và điều chuẩn L2, tối ưu bằng Spectral Gradient Descent có Armijo backtracking.
Giao diện và yêu cầu đầu vào này được dùng khi triển khai từng thuật toán ở các phiên bản tiếp theo.

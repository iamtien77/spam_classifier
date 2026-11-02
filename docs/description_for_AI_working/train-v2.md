Phiên bản v2. Huấn luyện mô hình cơ sở Complement Naive Bayes From-Scratch
Phiên bản này cài đặt và huấn luyện mô hình Complement Naive Bayes (CNB) từ đầu, tận dụng cơ chế học từ tập phần bù của lớp để khắc phục ảnh hưởng của mất cân bằng nhãn và chênh lệch độ dài văn bản.

Bước 1. Xác định ưu thế của Complement Naive Bayes
Multinomial Naive Bayes thông thường dễ bị thiên lệch về lớp Ham chiếm đa số.
Complement Naive Bayes ước lượng tham số từ các mẫu không thuộc lớp đang xét (complement set), giúp mô hình tập trung vào việc nhận diện các từ đặc trưng không xuất hiện ở lớp đối diện.

Bước 2. Cài đặt thuật toán trong hàm fit
Triển khai lớp NaiveBayesClassifier với kind='complement' trong src/models/classifiers.py.
Tính xác suất tiên nghiệm class_log_prior_: ln P(c) = ln(N_c / N).
Với mỗi lớp c in {0, 1}:
1. Lọc các mẫu không thuộc lớp c: {x_i | y_i != c}.
2. Tính tổng giá trị đặc trưng theo cột trên tập phần bù: N_{\setminus c, j} = sum_{i: y_i != c} x_ij.
3. Cộng hệ số làm mịn Laplace: N_{\setminus c, j} + alpha (alpha = 1.0 cho bản mặc định).
4. Chuẩn hóa bằng tổng trọng số của lớp phần bù: theta_{\setminus c, j} = (N_{\setminus c, j} + alpha) / sum_k (N_{\setminus c, k} + alpha).
5. Lưu trọng số đóng góp: w_{c, j} = -ln(theta_{\setminus c, j}).

Bước 3. Dự đoán và xuất xác suất bằng Softmax ổn định số học
Hàm decision_scores(X): Tính điểm số z_c(x) = ln P(c) - sum_j x_j * ln(theta_{\setminus c, j}).
Hàm predict_proba(X):
- Lấy exp(z - max(z)) để tránh tràn số (overflow).
- Chia cho tổng exp trên mỗi dòng để thu được phân phối xác suất p in [0, 1].
Trích xuất cột xác suất lớp 1 (Spam) để chuyển sang module tối ưu hóa ngưỡng trên tập Validation.
CNB sau huấn luyện được dùng làm baseline để so sánh với các mô hình khác.

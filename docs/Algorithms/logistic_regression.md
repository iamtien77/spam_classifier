# Logistic Regression trong bài lab phân loại spam

Tài liệu này giải thích cách em sử dụng Logistic Regression trong bài lab và đối chiếu với phần đã cài đặt trong [classifiers.py](../../src/models/classifiers.py), lớp `LogisticRegressionFromScratch`.

## 1. Mục đích và lý do lựa chọn

Em dùng Logistic Regression để so sánh với baseline Complement Naive Bayes. Mô hình có ranh giới tuyến tính, phù hợp với đặc trưng văn bản dạng ma trận thưa và cho phép xem trọng số của từng đặc trưng.

Với một mẫu có vector đặc trưng $x$, mô hình tính:

$$
z = w^T x + b, \qquad p = \frac{1}{1+e^{-z}}
$$

Trong đó, $w$ là trọng số, $b$ là bias và $p$ là điểm sigmoid cho lớp spam. Điểm này chưa được kiểm tra calibration nên em không xem nó là xác suất rủi ro chính xác trong thực tế.

Hạn chế của mô hình là không tự học được quan hệ phi tuyến phức tạp. Chất lượng dự đoán còn phụ thuộc vào cách tạo đặc trưng.

## 2. Dữ liệu đầu vào và kết quả đầu ra

- `fit(X, y)` nhận ma trận đặc trưng hai chiều và vector nhãn `0` hoặc `1`, có đủ hai lớp.
- Code hỗ trợ NumPy array, SciPy sparse matrix và `CSRMatrixScratch`. Giá trị đầu vào phải hữu hạn.
- `predict_proba(X)` trả ma trận hai cột: điểm ham và điểm spam.
- `predict(X, threshold=0.5)` trả nhãn `1` khi điểm spam lớn hơn hoặc bằng threshold.
- Sau huấn luyện, `weights_`, `bias_` chứa tham số đã học; `loss_history_` lưu các giá trị loss.

## 3. Dữ liệu sử dụng

Em dùng Train để học từ vựng, IDF, các bước biến đổi đặc trưng và tham số mô hình. Validation dùng chọn cấu hình và threshold; Test dùng đánh giá sau khi đã chọn xong.

Trong pipeline chính, đầu vào là các cột đã chọn từ bộ đặc trưng hybrid gồm word TF-IDF, bigram, character n-gram, keyword và numeric features. Số cột phụ thuộc cấu hình và kết quả chọn đặc trưng, không cố định trong thuật toán Logistic Regression.

## 4. Phương pháp huấn luyện và đánh giá

### Hàm mục tiêu

Code sử dụng Binary Cross-Entropy có trọng số mẫu và điều chuẩn L2:

$$
J(w,b)=\frac{1}{N}\sum_i a_i[\log(1+e^{z_i})-y_i z_i]+\frac{\lambda}{2}\|w\|^2
$$

Với `class_weight='balanced'`, $a_i=N/(2N_{y_i})$. Nếu `class_weight=None`, các mẫu có trọng số bằng 1. Hệ số điều chuẩn là $\lambda=1/\max(10^{-5},CN)$. Bias không bị điều chuẩn.

Code tính BCE bằng `np.logaddexp(0, z) - y*z` để tránh tràn số. Khi tính sigmoid, code giới hạn $z$ trong đoạn `[-30, 30]`.

### Cách tối ưu

Hàm `_gradient_descent_backtracking` bắt đầu với trọng số và bias bằng 0. Mỗi vòng lặp cập nhật theo gradient, co bước học theo điều kiện Armijo và điều chỉnh bước tiếp theo dựa trên độ cong theo cách Barzilai-Borwein. Giới hạn số vòng lặp không đồng nghĩa với bảo đảm đã hội tụ.

### Cấu hình và tiêu chí đánh giá

Trong `main.py`, nhánh Default dùng `C=1.0`, `learning_rate=0.1`, `max_iter=200`, `class_weight='balanced'`. Lưu ý constructor của lớp có `max_iter=300`; pipeline truyền lại giá trị 200. Nhánh Tuned thử `C` trong `[0.5, 1.0, 2.0]`.

`ThresholdOptimizerScratch` quét mặc định 200 threshold từ `0.005` đến `1.0` trên Validation. Trong nhóm đạt Recall mục tiêu 0.85, ngưỡng được chọn theo Accuracy, F1, Precision rồi FPR thấp hơn. Nếu không đạt mục tiêu, ưu tiên Recall, Accuracy rồi F1.

Khi so sánh cấu hình và các mô hình cuối, code ưu tiên đạt mục tiêu Recall, sau đó Accuracy và F1; nếu chưa đạt, ưu tiên Recall rồi Accuracy và F1. Em theo dõi thêm Precision, F2, ROC-AUC và confusion matrix để thấy cả lỗi nhận nhầm ham và bỏ sót spam.

## 5. Phần đã cài đặt và cách kiểm tra

Thuật toán được tự cài đặt trong `src/models/classifiers.py`, không gọi lớp Logistic Regression của scikit-learn. `main.py` gọi `fit()` để huấn luyện; `src/models/tuning.py` thực hiện tìm cấu hình và ngưỡng trên Validation.

Các kiểm tra liên quan nằm trong `tests/test_models.py` và `tests/test_model_contracts.py`. Khi báo cáo kiểm thử, em cần ghi kết quả của lần chạy cụ thể; tài liệu này không khẳng định số tests đã pass ở phiên bản hiện tại.

## 6. Kết quả thực nghiệm và giới hạn

Kết quả mỗi lần chạy được đọc từ [bảng lựa chọn Validation](../../results/metrics/validation_model_selection.csv) và [bảng so sánh Test](../../results/metrics/test_models_comparison.csv). Các file này là artifact đã lưu, không thay thế cho việc chạy lại khi code hoặc dữ liệu thay đổi.

Em dùng cùng split và cách tạo đặc trưng khi so sánh với baseline. Các mẫu rất ngắn, chứa từ mới hoặc có ngữ cảnh phức tạp có thể bị dự đoán sai. Một lần chia dữ liệu chưa đủ để kết luận mô hình ổn định trên mọi dữ liệu mới.

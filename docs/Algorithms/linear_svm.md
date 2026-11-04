# Linear SVM trong bài lab phân loại spam

Trong bài lab, dùng Linear SVM để so sánh với Logistic Regression và baseline Naive Bayes. Phần cài đặt nằm trong lớp `LinearSVMFromScratch` của [classifiers.py](../../src/models/classifiers.py).

## 1. Mục đích và lý do lựa chọn

Linear SVM phù hợp với dữ liệu văn bản nhiều chiều và có ranh giới tuyến tính. Mô hình tính decision score:

$$
z=w^T x+b
$$

Score cho biết mẫu nằm về phía nào của siêu phẳng. Khoảng cách có dấu đến siêu phẳng là $z/\|w\|$ khi $w$ khác 0; bản thân $z$ chưa phải khoảng cách hình học đã chuẩn hóa.

Hạn chế của mô hình là khó biểu diễn quan hệ phi tuyến nếu đặc trưng đầu vào chưa thể hiện được những quan hệ đó.

## 2. Dữ liệu đầu vào và kết quả đầu ra

`fit(X, y)` nhận ma trận đặc trưng hữu hạn, không rỗng và nhãn nhị phân có đủ hai lớp. Code hỗ trợ NumPy array, SciPy sparse matrix và `CSRMatrixScratch`. Bản thân SVM không yêu cầu đặc trưng không âm, dù đặc trưng của pipeline hiện tại không âm.

Cần phân biệt hai cách dự đoán trong code:

| Cách gọi                    | Giá trị dùng so sánh                                                | Ngưỡng mặc định    |
| :---------------------------- | :---------------------------------------------------------------------- | :---------------------- |
| `decision_function(X)`      | Trả margin $z$                                                       | Không phân loại      |
| `predict(X, threshold=0.0)` | So sánh trực tiếp margin với threshold                              | 0.0                     |
| `predict_proba(X)`          | Trả hai cột điểm sigmoid từ margin                                 | Không phân loại      |
| Pipeline trong `main.py`    | So sánh `predict_proba(X)[:, 1]` với threshold chọn từ Validation | Được chọn khi chạy |

Vì hai loại threshold khác miền, không truyền trực tiếp threshold đã chọn trên sigmoid score vào `svm.predict()`. Để dùng đúng cách của pipeline:

```python
spam_scores = svm.predict_proba(X)[:, 1]
y_pred = (spam_scores >= selected_threshold).astype(int)
```

Sigmoid dùng margin giới hạn trong `[-30, 30]`. Điểm trả về chưa qua calibration, dù API có tên `predict_proba`.

## 3. Dữ liệu sử dụng

Em huấn luyện bằng Train và dùng Validation chọn tham số, threshold. Test giữ riêng để đánh giá cuối.

Pipeline chính dùng đặc trưng hybrid sau chọn lọc: word TF-IDF, bigram, character n-gram, keyword và numeric features. Từ vựng, IDF và scaler được fit bằng Train. Số chiều thực tế phụ thuộc cấu hình và dữ liệu.

## 4. Phương pháp huấn luyện và đánh giá

### Hàm mục tiêu

Code đổi nhãn thành $t_i=2y_i-1$ và tính mức vi phạm lề:

$$
m_i=\max(0,1-t_i(w^Tx_i+b))
$$

Với cấu hình mặc định `loss='squared_hinge'`, hàm mục tiêu là:

$$
J(w,b)=\frac{\lambda}{2}\|w\|^2+\frac{1}{N}\sum_i a_i m_i^2
$$

Code cũng hỗ trợ `loss='hinge'`, khi đó dùng $m_i$ thay cho $m_i^2$. Với `class_weight='balanced'`, $a_i=N/(2N_{y_i})$; nếu không dùng trọng số lớp thì $a_i=1$. Hệ số điều chuẩn là $\lambda=1/\max(10^{-5},CN)$ và không phạt bias.

### Cách tối ưu

Code khởi tạo tham số bằng 0, sau đó gọi `_gradient_descent_backtracking` để cập nhật theo gradient với Armijo và bước học theo độ cong Barzilai-Borwein. Nhánh Hinge dùng subgradient. Số vòng đã thực hiện nằm trong `n_iter_`, các giá trị loss nằm trong `loss_history_`.

Tham số `tol` được truyền vào hàm tối ưu dưới dạng `min(self.tol, 1e-7)`. Vì vậy `tol` của constructor không được dùng nguyên giá trị làm ngưỡng dừng.

### Cấu hình và tiêu chí đánh giá

Trong `main.py`, nhánh Default dùng `C=1.0`, `learning_rate=0.05`, `max_iter=150`, `class_weight='balanced'`, `loss='squared_hinge'`. Nhánh Tuned thử `C` trong `[0.5, 1.0, 2.0]`.

Bộ chọn ngưỡng quét mặc định 200 giá trị từ `0.005` đến `1.0` trên sigmoid score của Validation. Trong nhóm đạt Recall 0.85, chọn Accuracy cao nhất rồi xét F1, Precision và FPR thấp hơn. Nếu không ngưỡng nào đạt, ưu tiên Recall, Accuracy rồi F1.

Khi chọn cấu hình và mô hình cuối, code ưu tiên đạt Recall mục tiêu, sau đó Accuracy và F1; nếu chưa đạt thì ưu tiên Recall, Accuracy và F1. Test không tham gia lựa chọn. Các chỉ số báo cáo gồm Accuracy, Precision, Recall, F1, F2, ROC-AUC, FPR và confusion matrix.

## 5. Phần đã cài đặt và cách kiểm tra

`LinearSVMFromScratch` tự tính loss, gradient và cập nhật tham số, không gọi `sklearn.svm.LinearSVC`. Đây là cách tối ưu theo trọng số của bài toán primal; code không lưu riêng danh sách support vectors.

`main.py` gọi huấn luyện, còn `src/models/tuning.py` chọn cấu hình và threshold. Các kiểm tra liên quan nằm trong `tests/test_models.py` và `tests/test_model_contracts.py`. Kết quả pass/fail cần được lấy từ lần chạy kiểm thử cụ thể.

## 6. Kết quả thực nghiệm và giới hạn

Em đọc kết quả từ [bảng Validation](../../results/metrics/validation_model_selection.csv) và [bảng Test](../../results/metrics/test_models_comparison.csv). Threshold và mô hình được chọn phụ thuộc lần chạy, không được cố định vào thuật toán.

Khi phân tích kết quả, em cần xem cả spam bị bỏ sót và ham bị nhận nhầm. Các văn bản viết biến dạng, khác ngôn ngữ hoặc khác dữ liệu huấn luyện có thể làm giảm chất lượng. Recall đạt mục tiêu trên Validation không bảo đảm Test cũng đạt mục tiêu đó.

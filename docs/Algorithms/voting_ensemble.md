# Voting Ensemble trong bài lab phân loại spam

Sau khi có các mô hình đơn, em thử Voting Ensemble để xem việc kết hợp dự đoán có cải thiện kết quả hay không. Phần cài đặt nằm trong lớp `SpamVotingEnsemble` của [ensemble.py](../../src/models/ensemble.py).

## 1. Mục đích và lý do lựa chọn

Em kết hợp Logistic Regression, Linear SVM và Complement Naive Bayes vì mỗi thuật toán có cách học khác nhau. Việc kết hợp chỉ có ý nghĩa khi kết quả thực nghiệm cho thấy có lợi so với mô hình đơn.

Đánh đổi là cần chạy dự đoán qua nhiều mô hình, tăng thời gian và bộ nhớ. Nếu các mô hình cùng sai trên một nhóm mẫu, ensemble vẫn có thể dự đoán sai.

## 2. Dữ liệu đầu vào và kết quả đầu ra

Constructor nhận:

- `models`: dictionary chứa tên và đối tượng mô hình thành viên.
- `weights`: dictionary trọng số; mặc định mỗi thành viên có trọng số 1.
- `voting`: chọn `soft` hoặc `hard`.

`fit(X, y)` gọi `fit()` của từng thành viên trên cùng dữ liệu. Trong `main.py`, các thành viên đã được huấn luyện trước khi đưa vào ensemble nên không cần gọi lại `ensemble.fit()`.

`predict_proba(X)` trả hai cột điểm tổng hợp. `predict(X, threshold=0.5)` trả nhãn, nhưng ý nghĩa threshold phụ thuộc cách voting như mô tả bên dưới.

## 3. Dữ liệu sử dụng

Các thành viên nhận cùng ma trận đặc trưng hybrid sau chọn lọc và được fit bằng Train. Validation dùng so sánh ứng viên và chọn threshold; Test dùng đánh giá sau lựa chọn.

Trong `main.py`, nhánh Default dùng trọng số LR:SVM:CNB là `1:1:1.2`. Nhánh Tuned ghép các mô hình đơn đã tuning và dùng trọng số đều `1:1:1`. Code hiện tại không tìm kiếm trọng số ensemble; tên Tuned chỉ việc sử dụng các thành viên đã tuning.

## 4. Cách kết hợp và phương pháp đánh giá

### Soft voting

Với trọng số dương $v_m$ và điểm spam $s_m(x)$ của từng thành viên:

$$
s(x)=\frac{\sum_m v_m s_m(x)}{\sum_m v_m}
$$

Code lấy `predict_proba()` của thành viên nếu có. Nếu chỉ có `decision_function()`, code chuyển điểm qua sigmoid; nếu không có cả hai, code dùng nhãn dự đoán làm điểm 0 hoặc 1.

`predict()` ở chế độ soft trả spam khi $s(x)$ lớn hơn hoặc bằng threshold. Điểm SVM là sigmoid của margin chưa calibration, nên điểm ensemble cũng chưa được xem là xác suất đã hiệu chỉnh.

### Hard voting

Code gọi `predict(X)` mặc định của mỗi thành viên và cộng phiếu có trọng số:

$$
V(x)=\sum_m v_m\hat y_m(x)
$$

`predict()` trả spam nếu $V(x)\ge\frac{1}{2}\sum_m v_m$. Vì dùng dấu lớn hơn hoặc bằng, **hòa phiếu được chọn là spam**. Đây là bỏ phiếu có trọng số; công thức mode không đủ mô tả khi trọng số khác nhau.

Ở chế độ hard, `predict_proba()` trả tỷ lệ phiếu theo trọng số, còn tham số `threshold` của `predict()` không được sử dụng. Code luôn dùng mốc một nửa tổng trọng số. SVM thành viên gọi `predict()` với margin threshold mặc định 0; LR và NB dùng ngưỡng điểm mặc định 0.5. Những ngưỡng này không phải threshold riêng đã chọn từ Validation.

### Cách đánh giá trong pipeline

`main.py` dùng **soft voting**. Bộ chọn ngưỡng quét mặc định 200 mốc từ `0.005` đến `1.0` trên Validation, ưu tiên ngưỡng đạt Recall 0.85 rồi Accuracy, F1, Precision và FPR thấp hơn. Nếu chưa đạt thì ưu tiên Recall, Accuracy và F1.

`ModelSelectionWorkflow` so sánh ensemble với các mô hình đơn theo việc đạt Recall mục tiêu, sau đó Accuracy và F1. Nếu chưa đạt, ưu tiên Recall rồi Accuracy và F1. Test không dùng để chọn lại mô hình. Em báo cáo thêm Precision, F2, ROC-AUC và confusion matrix.

## 5. Phần đã cài đặt và cách kiểm tra

`SpamVotingEnsemble` tự kết hợp điểm và phiếu bằng NumPy. Module còn có hàm tạo Random Forest và Gradient Boosting bằng scikit-learn, nhưng các mô hình cây này không được gọi trong pipeline so sánh chính đang mô tả.

Các kiểm tra liên quan nằm trong `tests/test_models.py`. Khi kiểm tra, cần phân biệt soft score, hard vote, trọng số và trường hợp hòa. Tài liệu này không khẳng định toàn bộ tests hiện tại đã pass.

## 6. Kết quả thực nghiệm và giới hạn

Em đọc kết quả từ [bảng Validation](../../results/metrics/validation_model_selection.csv) và [bảng Test](../../results/metrics/test_models_comparison.csv), sau đó so sánh với từng mô hình đơn trên cùng split.

Ensemble không mặc định tốt hơn mô hình đơn. Sự khác nhau về thang điểm hoặc các lỗi chung giữa thành viên có thể làm giảm lợi ích kết hợp. Code hiện tại chưa kiểm tra đầy đủ mọi trường hợp trọng số không hợp lệ, nên cách dùng được mô tả ở đây áp dụng cho trọng số dương như pipeline chính.

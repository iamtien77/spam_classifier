# Naive Bayes trong bài lab phân loại spam

Em chọn Complement Naive Bayes làm baseline để có kết quả so sánh với Logistic Regression và Linear SVM. Phần cài đặt nằm trong lớp `NaiveBayesClassifier` của [classifiers.py](../../src/models/classifiers.py).

## 1. Mục đích và lý do lựa chọn

Naive Bayes có cách huấn luyện đơn giản, chủ yếu cộng dồn trọng số đặc trưng theo lớp. Mô hình không cần tối ưu lặp bằng gradient nên phù hợp để bắt đầu bài lab và làm proxy model khi thử chọn đặc trưng.

Code hỗ trợ hai biến thể: `multinomial` và `complement`, với `complement` là mặc định. Hạn chế là mô hình chưa biểu diễn tốt ngữ cảnh và sự phụ thuộc giữa các từ.

## 2. Dữ liệu đầu vào và kết quả đầu ra

- `fit(X, y)` nhận ma trận đặc trưng hữu hạn, **không âm**, có đủ hai lớp nhãn `0` và `1`.
- Code hỗ trợ NumPy array, SciPy sparse matrix và `CSRMatrixScratch`.
- `decision_scores(X)` trả ma trận điểm hai cột theo lớp ham và spam.
- `predict_proba(X)` chuyển điểm qua Softmax và trả hai cột điểm đã chuẩn hóa.
- `predict(X, threshold=0.5)` trả spam khi điểm cột 1 lớn hơn hoặc bằng threshold.

Điểm Softmax chưa được kiểm tra calibration nên không được xem là xác suất rủi ro đã hiệu chỉnh.

## 3. Dữ liệu sử dụng

Train dùng tính thống kê đặc trưng và tỷ lệ mỗi lớp. Validation dùng chọn `alpha` và threshold. Test chỉ dùng đánh giá sau lựa chọn.

Baseline đơn giản trong thiết kế dùng word TF-IDF và Complement Naive Bayes. Trong pipeline chính của `main.py`, Naive Bayes nhận cùng đặc trưng hybrid đã chọn như LR và SVM. Module `src/evaluation/experiments.py` có nhánh so sánh word TF-IDF và hybrid trên dữ liệu phát triển.

Các numeric features được chia cho max tuyệt đối học trên Train bằng `MaxAbsScalerScratch`. Với đầu vào không âm, phép biến đổi giữ tính không âm. Tuy nhiên, giá trị của dữ liệu mới có thể lớn hơn 1 nếu vượt max của Train vì scaler không clipping.

## 4. Phương pháp huấn luyện và đánh giá

### Multinomial Naive Bayes

Với lớp $c$, code cộng trọng số đặc trưng của các mẫu thuộc lớp đó:

$$
N_{c,j}=\sum_{i:y_i=c}x_{ij},\qquad
\theta_{c,j}=\frac{N_{c,j}+\alpha}{\sum_k(N_{c,k}+\alpha)}
$$

Sau đó tính:

$$
z_c(x)=\log P(c)+\sum_j x_j\log\theta_{c,j}
$$

Trong đó $P(c)$ là tỷ lệ mẫu thuộc lớp $c$ trong Train và `alpha` phải hữu hạn, lớn hơn 0. Với TF-IDF, các tổng trên là tổng trọng số đặc trưng, không nhất thiết là số lần xuất hiện nguyên.

### Complement Naive Bayes

Code tính thống kê trên các mẫu không thuộc lớp đang xét:

$$
N_{\neg c,j}=\sum_{i:y_i\ne c}x_{ij},\qquad
\theta_{\neg c,j}=\frac{N_{\neg c,j}+\alpha}{\sum_k(N_{\neg c,k}+\alpha)}
$$

Điểm trong phần cài đặt hiện tại là:

$$
z_c(x)=\log P(c)-\sum_j x_j\log\theta_{\neg c,j}
$$

Code lưu log của các trọng số trong `feature_log_prob_` và log tỷ lệ lớp trong `class_log_prior_`. Khi tính Softmax, code trừ điểm lớn nhất của mỗi dòng, giới hạn hiệu trong `[-30, 30]`, lấy hàm mũ rồi chuẩn hóa.

### Cấu hình và tiêu chí đánh giá

Nhánh Default trong `main.py` dùng `kind='complement'`, `alpha=1.0`. Nhánh Tuned thử `alpha` trong `[0.5, 1.0, 2.0]`.

Trên Validation, `ThresholdOptimizerScratch` quét mặc định 200 threshold từ `0.005` đến `1.0`. Trong nhóm đạt Recall 0.85, ưu tiên Accuracy, F1, Precision rồi FPR thấp hơn. Nếu chưa đạt, ưu tiên Recall, Accuracy và F1. Khi so sánh các cấu hình và mô hình cuối, code ưu tiên đạt Recall mục tiêu, sau đó Accuracy và F1; nếu chưa đạt thì ưu tiên Recall rồi Accuracy và F1.

Em theo dõi Accuracy, Precision, Recall, F1, F2, ROC-AUC và confusion matrix, tránh kết luận chỉ từ Accuracy.

## 5. Phần đã cài đặt và cách kiểm tra

Hai biến thể được tự cài đặt trong `NaiveBayesClassifier`, không dùng `sklearn.naive_bayes`. Naive Bayes còn được dùng trong phần thử chọn đặc trưng và là một thành viên của Voting Ensemble.

Các kiểm tra liên quan nằm trong `tests/test_models.py` và `tests/test_model_contracts.py`. Cần kiểm tra cả đầu vào âm, dữ liệu thưa, smoothing và kết quả trên ví dụ nhỏ. Tài liệu này không xác nhận số tests đã pass ở phiên bản hiện tại.

## 6. Kết quả thực nghiệm và giới hạn

Các kết quả đã lưu được đọc từ [bảng Validation](../../results/metrics/validation_model_selection.csv) và [bảng Test](../../results/metrics/test_models_comparison.csv). `alpha`, threshold và số chiều đặc trưng được báo theo từng lần chạy.

Naive Bayes có thể nhận nhầm ham chứa nhiều từ quảng cáo hoặc bỏ sót spam có cách viết mới. Complement Naive Bayes không bảo đảm luôn tốt hơn Multinomial hoặc khắc phục hoàn toàn mất cân bằng lớp. Em dùng baseline làm mốc so sánh và xem thêm các mẫu sai trước khi kết luận.

# Naive Bayes trong bài lab phân loại spam

Nhóm dự kiến chọn Complement Naive Bayes làm baseline để so sánh với Logistic Regression và Linear SVM. Tài liệu này trình bày cách học, yêu cầu dữ liệu và kế hoạch đánh giá trước khi thực hiện bài lab.

## 1. Mục đích và lý do lựa chọn

Naive Bayes huấn luyện tương đối đơn giản, chủ yếu tổng hợp trọng số đặc trưng theo lớp và không cần cập nhật lặp bằng gradient. Nhóm chọn phương pháp này làm mốc so sánh ban đầu và mô hình thử khi lựa chọn đặc trưng.

Nhóm sẽ xem xét hai biến thể Multinomial và Complement, ưu tiên Complement cho baseline. Giả định độc lập giữa các đặc trưng giúp việc tính toán đơn giản hơn, nhưng hạn chế khả năng thể hiện ngữ cảnh và quan hệ giữa các từ.

## 2. Dữ liệu đầu vào và kết quả đầu ra

- Đầu vào gồm ma trận đặc trưng hữu hạn, **không âm**, và nhãn ham = 0, spam = 1; Train cần có đủ hai lớp.
- Dữ liệu có thể ở dạng ma trận đặc hoặc thưa, với cùng ý nghĩa và thứ tự các cột.
- Đầu ra là điểm của hai lớp ham và spam, được chuẩn hóa bằng Softmax.
- Mẫu được dự đoán là spam khi điểm spam lớn hơn hoặc bằng threshold; mốc ban đầu dự kiến là 0,5.

Điểm Softmax có tổng bằng 1, nhưng chưa đủ để xem đó là xác suất đã được hiệu chỉnh cho email thực tế. Nhóm sẽ dùng điểm này để so sánh và chọn ngưỡng trong bài lab.

## 3. Dữ liệu sử dụng

Train dùng để học thống kê đặc trưng và tỷ lệ mỗi lớp. Validation dùng chọn hệ số smoothing và threshold. Test được giữ riêng cho đánh giá cuối.

Baseline dự kiến dùng word TF-IDF. Khi thử thêm bigram, character n-gram, từ khóa hoặc đặc trưng số, nhóm sẽ thay đổi từng nhóm trên cùng split. Khi so sánh thuật toán, các mô hình cần nhận cùng bộ đặc trưng.

Với đặc trưng số không âm, nhóm dự kiến chia từng cột cho giá trị tuyệt đối lớn nhất học từ Train; cột toàn 0 được giữ bằng 0. Phép biến đổi giữ tính không âm, nhưng giá trị mới có thể vượt 1 nếu lớn hơn mức đã gặp trong Train.

## 4. Phương pháp huấn luyện và đánh giá

### Multinomial Naive Bayes

Với lớp $c$, tổng trọng số của đặc trưng $j$ và trọng số sau smoothing được tính như sau:

$$
N_{c,j}=\sum_{i:y_i=c}x_{ij},\qquad
\theta_{c,j}=\frac{N_{c,j}+\alpha}{\sum_k(N_{c,k}+\alpha)}
$$

Điểm của một mẫu $x$ đối với lớp $c$ là:

$$
z_c(x)=\log P(c)+\sum_j x_j\log\theta_{c,j}
$$

Trong đó, $P(c)$ là tỷ lệ mẫu thuộc lớp $c$ trong Train và $\alpha$ là hệ số smoothing hữu hạn, lớn hơn 0. Với TF-IDF, các tổng trên là tổng trọng số, không nhất thiết là số lần xuất hiện nguyên.

### Complement Naive Bayes

Biến thể Complement học thống kê từ các mẫu không thuộc lớp đang xét:

$$
N_{\neg c,j}=\sum_{i:y_i\ne c}x_{ij},\qquad
\theta_{\neg c,j}=\frac{N_{\neg c,j}+\alpha}{\sum_k(N_{\neg c,k}+\alpha)}
$$

Trong phương án của bài lab, nhóm dự kiến kết hợp điểm phần bù với tỷ lệ lớp:

$$
z_c(x)=\log P(c)-\sum_j x_j\log\theta_{\neg c,j}
$$

Nhóm sẽ tính trên miền log để hạn chế vấn đề xác suất quá nhỏ. Khi chuẩn hóa Softmax, dự kiến trừ điểm lớn nhất trong mỗi mẫu, giới hạn hiệu điểm trong $[-30,30]$ rồi lấy hàm mũ và chuẩn hóa.

### Cấu hình và tiêu chí đánh giá

Nhóm dự kiến bắt đầu với Complement và $\alpha=1$, sau đó thử $\alpha\in\{0.5,1,2\}$ trên Validation.

Với mỗi cấu hình, nhóm sẽ thử 200 threshold từ 0,005 đến 1,0. Trong nhóm đạt Recall mục tiêu 0,85, chọn ngưỡng theo Accuracy → F1 → Precision → FPR thấp hơn; nếu chưa đạt thì xét Recall → Accuracy → F1. Khi chọn giữa các cấu hình hoặc mô hình, ưu tiên đạt mục tiêu rồi xét Accuracy → F1; nếu chưa đạt thì xét Recall → Accuracy → F1.

Nhóm sẽ báo cáo Accuracy, Precision, Recall, F1, F2, ROC-AUC, FPR và confusion matrix để thấy cả spam bị bỏ sót lẫn ham bị nhận nhầm.

## 5. Kế hoạch kiểm tra

Nhóm sẽ dùng dữ liệu nhỏ có thể tính tay để đối chiếu tổng trọng số, smoothing, tỷ lệ lớp và điểm của cả hai biến thể. Cùng dữ liệu ở dạng đặc và thưa cần cho kết quả tương đương trong sai số số học.

Các trường hợp cần kiểm tra gồm đặc trưng âm hoặc không hữu hạn, thiếu một lớp nhãn, sai số cột và vector toàn 0. Với vector toàn 0, điểm chỉ còn phụ thuộc tỷ lệ lớp; quy tắc chuyển điểm thành nhãn vẫn cần được áp dụng nhất quán.

## 6. Kế hoạch đánh giá và giới hạn

Nhóm sẽ ghi lại bộ đặc trưng, hệ số smoothing, threshold và thời gian huấn luyện của từng cấu hình. Sau khi chọn bằng Validation, nhóm mới đánh giá Test và đọc các mẫu dự đoán sai.

Naive Bayes có thể nhận nhầm ham chứa nhiều từ quảng cáo hoặc bỏ sót spam có cách viết mới. Complement không bảo đảm luôn tốt hơn Multinomial hay giải quyết hoàn toàn mất cân bằng lớp. Nhóm sẽ dùng baseline làm mốc so sánh thay vì kỳ vọng sẵn một biến thể sẽ thắng.

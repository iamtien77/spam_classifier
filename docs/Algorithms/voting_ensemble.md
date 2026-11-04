# Voting Ensemble trong bài lab phân loại spam

Sau khi có các mô hình đơn, nhóm dự kiến thử Voting Ensemble để xem việc kết hợp dự đoán có mang lại lợi ích hay không. Đây là phần mở rộng của bài lab, được so sánh bằng cùng dữ liệu và tiêu chí với các mô hình đơn.

## 1. Mục đích và lý do lựa chọn

Nhóm dự kiến kết hợp Logistic Regression, Linear SVM và Complement Naive Bayes vì mỗi phương pháp có cách học khác nhau. Ensemble có thể hữu ích nếu các thành viên bổ sung được cho nhau ở những mẫu khó.

Đánh đổi là cần dự đoán qua nhiều mô hình, tăng thời gian và bộ nhớ. Nếu các thành viên cùng sai trên một nhóm mẫu, kết hợp dự đoán chưa chắc giúp cải thiện.

## 2. Dữ liệu đầu vào và kết quả đầu ra

Phương án kết hợp cần xác định:

- Các mô hình thành viên và cách biểu diễn đặc trưng dùng chung.
- Trọng số của từng thành viên, dự kiến bắt đầu từ trọng số đều.
- Cách bỏ phiếu: soft voting hoặc hard voting.

Trọng số phải hữu hạn, không âm và có tổng hữu hạn, lớn hơn 0. Một thành viên có thể mang trọng số 0 nếu các thành viên còn lại vẫn có tổng trọng số dương.

Mỗi thành viên được huấn luyện trên Train trước khi kết hợp. Đầu ra gồm điểm tổng hợp hoặc tỷ lệ phiếu có trọng số và nhãn ham/spam theo quy tắc tương ứng.

## 3. Dữ liệu sử dụng

Các thành viên dự kiến nhận cùng ma trận đặc trưng và cùng split. Bộ biến đổi chỉ học từ Train; Validation dùng so sánh phương án kết hợp và chọn threshold cho soft voting. Test dành cho đánh giá sau lựa chọn.

Nhóm dự kiến so sánh tổ hợp từ các mô hình cấu hình ban đầu với tổ hợp từ các mô hình đã chọn tham số trên Validation. Có thể đặt trước tỷ lệ trọng số LR:SVM:CNB là 1:1:1,2 cho tổ hợp ban đầu và 1:1:1 cho tổ hợp sau tuning. Đây là các phương án thử, chưa phải trọng số tối ưu; tuning mô hình thành viên không đồng nghĩa đã tối ưu trọng số ensemble.

## 4. Cách kết hợp và phương pháp đánh giá

### Soft voting

Với trọng số $v_m\ge0$, tổng trọng số dương và điểm spam $s_m(x)$ của từng thành viên:

$$
s(x)=\frac{\sum_m v_m s_m(x)}{\sum_m v_m}
$$

Nhóm dự kiến dùng điểm sigmoid của LR và SVM, cùng điểm chuẩn hóa của Naive Bayes. Mỗi điểm cần nằm trong khoảng 0–1 và cùng biểu diễn lớp spam trước khi lấy trung bình có trọng số.

Nhãn dự đoán là spam khi $s(x)$ lớn hơn hoặc bằng threshold. Các điểm thành viên chưa được đánh giá calibration nên điểm ensemble cũng chưa được xem là xác suất đã hiệu chỉnh.

### Hard voting

Mỗi thành viên đưa ra một nhãn $\hat y_m(x)\in\{0,1\}$, rồi nhóm tính tổng phiếu spam có trọng số:

$$
V(x)=\sum_m v_m\hat y_m(x)
$$

Quy tắc dự kiến là chọn spam nếu $V(x)\ge\frac{1}{2}\sum_m v_m$, tức **hòa phiếu được chọn là spam**. Tỷ lệ phiếu spam là $V(x)/\sum_m v_m$, không phải xác suất của mẫu.

Nhóm sẽ giữ mốc quyết định hard voting ở một nửa tổng trọng số. Thành viên SVM dùng ngưỡng margin 0; LR và Naive Bayes dùng ngưỡng điểm 0,5. Các ngưỡng thành viên này được đặt trước, khác với threshold riêng được chọn khi đánh giá từng mô hình.

### Cách đánh giá trong pipeline

Nhóm dự kiến dùng soft voting làm phương án kết hợp chính và thử thêm hard voting để đối chiếu. Với soft voting, thử 200 threshold từ 0,005 đến 1,0 trên Validation; ưu tiên ngưỡng đạt Recall mục tiêu 0,85 rồi xét Accuracy → F1 → Precision → FPR thấp hơn. Nếu chưa đạt thì xét Recall → Accuracy → F1. Hard voting giữ mốc 0,5 trên tỷ lệ phiếu.

Khi so sánh ensemble với mô hình đơn, nhóm ưu tiên đạt mục tiêu Recall rồi xét Accuracy → F1; nếu chưa đạt thì xét Recall → Accuracy → F1. Test không dùng để chọn lại mô hình. Nhóm sẽ báo cáo thêm Precision, F2, ROC-AUC, FPR, confusion matrix và chi phí dự đoán.

## 5. Kế hoạch kiểm tra

Nhóm sẽ dùng ví dụ có thể tính tay để đối chiếu trung bình điểm, tổng phiếu và trường hợp hòa. Cần kiểm tra trọng số đều, trọng số khác nhau và một thành viên có trọng số 0.

Các trường hợp không hợp lệ như danh sách thành viên rỗng, trọng số âm, không hữu hạn hoặc tổng bằng 0 cần được phát hiện. Nhóm cũng sẽ kiểm tra nhãn lớp được hiểu thống nhất giữa các thành viên và phân biệt rõ soft score với tỷ lệ phiếu hard voting.

## 6. Kế hoạch đánh giá và giới hạn

Nhóm sẽ ghi lại thành viên, cấu hình, trọng số, cách voting và threshold của mỗi phương án. Ensemble được so sánh với từng mô hình đơn trên cùng split trước khi quyết định có chọn hay không.

Ensemble không mặc định tốt hơn mô hình đơn. Khác biệt về thang điểm và các lỗi chung giữa thành viên có thể làm giảm lợi ích kết hợp. Nếu mức cải thiện không đáng kể so với thời gian và bộ nhớ tăng thêm, nhóm sẽ cân nhắc phương án đơn giản hơn trong phạm vi bài lab.

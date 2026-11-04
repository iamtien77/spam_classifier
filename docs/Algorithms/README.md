# Các thuật toán học máy trong bài lab

Thư mục này trình bày các phương pháp nhóm dự kiến sử dụng cho bài lab **Spam Classifier (Practice 1: Classifying Spam Emails)**. Mục tiêu là hiểu cách từng mô hình học từ đặc trưng văn bản và xây dựng một cách so sánh công bằng.

Xem [design.md](./design.md) để nắm thiết kế theo **5W1H**, vai trò của Train, Validation, Test và cách tổ chức bài nhóm. Các trang bên dưới đi sâu vào từng thuật toán.

Nhóm dự kiến bắt đầu với Complement Naive Bayes, sau đó so sánh Logistic Regression, Linear SVM và thử Voting Ensemble nếu có đủ thời gian. Việc lựa chọn sẽ dựa trên tiêu chí thống nhất trước thí nghiệm, chưa đặt trước mô hình nào tốt nhất.

---

## 4 giai đoạn của pipeline

Nhóm dự kiến tổ chức công việc theo bốn giai đoạn. Mỗi giai đoạn cần có đầu ra rõ ràng để thành viên phụ trách phần sau có thể sử dụng và kiểm tra lại.

| Giai đoạn | Tài liệu tham khảo trong nhóm | Phần việc chính | Nội dung dự kiến |
| :---: | :--- | :--- | :--- |
| **Giai đoạn 1** | [Thiết kế bài lab](./design.md) | Chuẩn bị dữ liệu | Kiểm tra nội dung và nhãn email, xử lý trùng/mâu thuẫn trước khi chia phân tầng Train/Validation/Test. |
| **Giai đoạn 2** | [Thiết kế bài lab](./design.md), [Naive Bayes](./naive_bayes.md) | Tạo đặc trưng | Bắt đầu từ word TF-IDF, sau đó thử nhóm đặc trưng bổ sung; học bộ biến đổi trên Train và chọn số đặc trưng bằng Validation. |
| **Giai đoạn 3** | [Naive Bayes](./naive_bayes.md), [Logistic Regression](./logistic_regression.md), [Linear SVM](./linear_svm.md) | Huấn luyện và lựa chọn | Hoàn thành baseline, so sánh các mô hình đơn, thử một số cấu hình và chọn threshold trên Validation. |
| **Giai đoạn 4** | [Thiết kế bài lab](./design.md), [Voting Ensemble](./voting_ensemble.md) | Đánh giá và thử sử dụng | Đánh giá Test sau khi chốt lựa chọn, đọc mẫu sai, lưu mô hình cùng bộ biến đổi và thử dự đoán email mới. |

---

## 1. Danh Mục Tài Liệu Kỹ Thuật

Mỗi tài liệu gồm sáu phần: mục đích, đầu vào–đầu ra, dữ liệu sử dụng, phương pháp huấn luyện hoặc kết hợp, kế hoạch kiểm tra, kế hoạch đánh giá và giới hạn.

| Tài liệu | Thuật toán | Vai trò trong bài lab | Phương án dự kiến |
| :--- | :--- | :--- | :--- |
| [linear_svm.md](./linear_svm.md) | **Linear Support Vector Machine** | So sánh phương pháp tuyến tính dựa trên lề phân cách | Ưu tiên squared hinge, xem xét điều chuẩn và threshold |
| [logistic_regression.md](./logistic_regression.md) | **Logistic Regression** | So sánh mô hình tuyến tính với Binary Cross-Entropy | Dùng L2, theo dõi loss và quá trình hội tụ |
| [naive_bayes.md](./naive_bayes.md) | **Complement & Multinomial Naive Bayes** | Làm baseline và mô hình thử khi chọn đặc trưng | Bắt đầu với Complement, xem xét ảnh hưởng của smoothing |
| [voting_ensemble.md](./voting_ensemble.md) | **Voting Ensemble** | Xem việc kết hợp có tốt hơn mô hình đơn hay không | Thử soft/hard voting, cân nhắc thêm thời gian và bộ nhớ |

---

## 2. So sánh và lựa chọn mô hình trên tập Validation

Nhóm dự kiến dùng cùng split và bộ đặc trưng khi so sánh thuật toán. Mục tiêu thử nghiệm là Recall từ 0,85 trở lên trên Validation; trong nhóm đạt mục tiêu, ưu tiên Accuracy rồi F1. Nếu chưa đạt, xét Recall → Accuracy → F1 và ghi rõ hạn chế.

Bảng dưới liệt kê các ứng viên dự kiến, không thể hiện thứ hạng:

| Ứng viên | Phương án tham số | Cách chọn ngưỡng | Nội dung cần so sánh |
| :--- | :--- | :--- | :--- |
| Complement Naive Bayes ban đầu | $\alpha=1$ | Chọn trên Validation | Mốc baseline |
| Complement Naive Bayes sau tuning | $\alpha\in\{0.5,1,2\}$ | Chọn cho từng cấu hình | Ảnh hưởng của smoothing |
| Logistic Regression ban đầu | $C=1$ | Chọn trên Validation | Khác biệt so với baseline |
| Logistic Regression sau tuning | $C\in\{0.5,1,2\}$ | Chọn cho từng cấu hình | Ảnh hưởng của điều chuẩn |
| Linear SVM ban đầu | $C=1$, squared hinge | Chọn trên sigmoid score | Khác biệt so với hai phương pháp còn lại |
| Linear SVM sau tuning | $C\in\{0.5,1,2\}$ | Chọn cho từng cấu hình | Điều chuẩn và quá trình hội tụ |
| Soft Voting | Thành viên cấu hình ban đầu hoặc sau tuning | Chọn trên điểm tổng hợp | Lợi ích so với mô hình đơn |
| Hard Voting | Thành viên sau tuning, trọng số đặt trước | Giữ mốc một nửa tổng phiếu | Ảnh hưởng của cách bỏ phiếu |

Nhóm sẽ ghi lại cấu hình và thứ tự thử. Nếu các chỉ số lựa chọn bằng nhau hoàn toàn, giữ ứng viên được xét trước theo thứ tự đã thống nhất.

---

## 3. Kế hoạch đánh giá trên tập Test

Nhóm chỉ đánh giá Test sau khi chốt mô hình, bộ đặc trưng và threshold bằng Validation. Các ứng viên có thể được đặt cạnh nhau trong báo cáo, nhưng điểm Test không được dùng để chọn lại mô hình.

| Chỉ số hoặc nội dung | Câu hỏi cần trả lời | Cách đọc dự kiến |
| :--- | :--- | :--- |
| Accuracy | Bao nhiêu mẫu được dự đoán đúng? | Đọc cùng phân bố nhãn, không dùng một mình |
| Precision | Trong các dự đoán spam, bao nhiêu là spam thật? | Xem nguy cơ nhận nhầm ham |
| Recall | Bao nhiêu spam được phát hiện? | Xem mức bỏ sót spam |
| F1, F2 | Precision và Recall cân bằng ra sao? | F2 đặt trọng số cao hơn cho Recall |
| FPR và số FP | Bao nhiêu ham bị nhận nhầm? | Báo cả tỷ lệ và số mẫu |
| ROC-AUC | Mô hình xếp hạng spam so với ham như thế nào? | Phân biệt với Accuracy tại một threshold |
| Confusion matrix và mẫu sai | Lỗi tập trung ở trường hợp nào? | Đọc cả FP và FN, xem nội dung đã loại thông tin cá nhân |
| Thời gian và bộ nhớ | Phương án nào phù hợp quy mô bài lab? | So sánh trong cùng điều kiện thực hiện |

---

## 4. Nhận xét và lựa chọn kỹ thuật

1. **Bắt đầu từ baseline:** Nhóm chọn Naive Bayes vì cách học đơn giản và ít tốn thời gian. LR và SVM được thử sau để xem cách học khác nhau ảnh hưởng đến phân loại ra sao.
2. **So sánh từng thay đổi:** Giữ cùng split, thử lần lượt thuật toán, nhóm đặc trưng và tham số. Khi thay nhiều yếu tố cùng lúc, nhóm sẽ khó giải thích nguyên nhân chênh lệch.
3. **Giữ đúng phạm vi học tập:** Mục tiêu là hiểu phương pháp, đánh giá có căn cứ và giải thích các mẫu sai. Điểm tốt trên một split chưa đủ để kết luận mô hình phù hợp cho hộp thư thực tế.

---

## 5. Quy Trình Chọn Ngưỡng, Tham Số Và Mô Hình

Bài toán dự kiến có hai nhãn ham = 0 và spam = 1. Baseline ban đầu dùng word TF-IDF và Complement Naive Bayes với $\alpha=1$. Các cấu hình và mục tiêu Recall 0,85 là lựa chọn thử nghiệm của nhóm, không phải kết quả được bảo đảm.

1. **Train:** Học từ vựng, IDF, tham số chuẩn hóa, điểm xếp hạng đặc trưng và tham số mô hình. Nếu thử nhiều số lượng đặc trưng, dùng Validation để lựa chọn; không đặt trước số cột được giữ lại.
2. **Chọn threshold trên Validation:** Với mỗi cấu hình, dự kiến thử 200 ngưỡng từ 0,005 đến 1,0. Trong nhóm đạt Recall mục tiêu, xét Accuracy → F1 → Precision → FPR thấp hơn. Nếu chưa đạt thì xét Recall → Accuracy → F1. Hard voting giữ mốc 0,5 trên tỷ lệ phiếu.
3. **Chọn tham số và mô hình:** Ưu tiên ứng viên đạt Recall mục tiêu, rồi xét Accuracy → F1; nếu chưa đạt thì xét Recall → Accuracy → F1. Ensemble được so sánh như một ứng viên, không mặc định được chọn.
4. **Test:** Giữ nguyên lựa chọn từ Validation để đánh giá cuối. Nếu cần cải tiến dựa trên những lỗi đã quan sát ở Test, nhóm sẽ ghi rõ và cần dữ liệu đánh giá độc lập mới cho lần sau.

Điểm sigmoid, Softmax và tỷ lệ phiếu có ý nghĩa khác nhau; nhóm sẽ không mặc nhiên xem chúng là xác suất rủi ro đã calibration. Các giới hạn cần xem xét gồm email khác nguồn hoặc ngôn ngữ, dữ liệu mất cân bằng và việc thử quá nhiều cấu hình trên cùng Validation.

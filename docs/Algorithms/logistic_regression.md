# Logistic Regression trong bài lab phân loại spam

Nhóm dự kiến dùng Logistic Regression để so sánh với baseline Complement Naive Bayes. Tài liệu này trình bày phương án huấn luyện, cách chọn cấu hình và kế hoạch kiểm tra cho bài lab.

## 1. Mục đích và lý do lựa chọn

Logistic Regression có ranh giới tuyến tính, phù hợp với đặc trưng văn bản dạng ma trận thưa. Nhóm có thể xem trọng số của các đặc trưng để hiểu chúng ảnh hưởng đến dự đoán theo hướng nào.

Với một mẫu có vector đặc trưng $x$, mô hình tính:

$$
z = w^T x + b, \qquad p = \frac{1}{1+e^{-z}}
$$

Trong đó, $w$ là trọng số, $b$ là bias và $p$ là điểm sigmoid cho lớp spam. Nhóm sẽ dùng điểm này để chọn threshold; việc diễn giải thành xác suất thực tế cần có đánh giá calibration riêng.

Hạn chế của mô hình là không tự học được quan hệ phi tuyến phức tạp. Chất lượng dự đoán còn phụ thuộc vào cách biểu diễn văn bản.

## 2. Dữ liệu đầu vào và kết quả đầu ra

- Đầu vào gồm ma trận đặc trưng hai chiều, hữu hạn, không rỗng và vector nhãn ham = 0, spam = 1 có đủ hai lớp.
- Nhóm dự kiến dùng ma trận thưa để tiết kiệm bộ nhớ với dữ liệu văn bản nhiều chiều.
- Đầu ra gồm điểm spam $p$, điểm ham $1-p$ và nhãn dự đoán theo threshold.
- Mốc phân loại ban đầu dự kiến là 0,5; mẫu có điểm spam bằng threshold được xếp vào lớp spam.
- Quá trình huấn luyện cần ghi lại loss để theo dõi mức giảm và dấu hiệu chưa hội tụ.

## 3. Dữ liệu sử dụng

Nhóm sẽ dùng Train để học từ vựng, IDF, tham số biến đổi đặc trưng và trọng số mô hình. Validation dùng chọn cấu hình và threshold; Test chỉ dùng sau khi lựa chọn đã hoàn tất.

Nhóm dự kiến so sánh với Naive Bayes trên cùng bộ đặc trưng. Sau baseline word TF-IDF, có thể thử thêm bigram, character n-gram, từ khóa và đặc trưng số. Số chiều phụ thuộc phương án chọn đặc trưng, không phải một hằng số của Logistic Regression.

## 4. Phương pháp huấn luyện và đánh giá

### Hàm mục tiêu

Nhóm dự kiến dùng Binary Cross-Entropy có trọng số mẫu và điều chuẩn L2:

$$
J(w,b)=\frac{1}{N}\sum_i a_i[\log(1+e^{z_i})-y_i z_i]+\frac{\lambda}{2}\|w\|^2
$$

Ở đây, $N$ là số mẫu Train và $N_{y_i}$ là số mẫu thuộc lớp của mẫu $i$. Khi cân bằng trọng số lớp, $a_i=N/(2N_{y_i})$; nếu không cân bằng thì $a_i=1$. Nhóm dự kiến dùng $\lambda=1/\max(10^{-5},CN)$ với $C>0$, và không điều chuẩn bias.

Để hạn chế tràn số, biểu thức $\log(1+e^z)$ cần được tính theo dạng ổn định số học. Khi tính sigmoid, nhóm dự kiến giới hạn $z$ trong $[-30,30]$.

### Cách tối ưu

Nhóm dự kiến khởi tạo trọng số và bias bằng 0, cập nhật theo gradient rồi giảm bước học nếu chưa thỏa điều kiện Armijo. Bước tiếp theo được điều chỉnh theo độ cong bằng phương pháp Barzilai–Borwein. Việc dừng cần xét gradient và giới hạn số vòng; đạt số vòng tối đa không đồng nghĩa mô hình đã hội tụ.

### Cấu hình và tiêu chí đánh giá

Cấu hình khởi đầu dự kiến là $C=1$, bước học ban đầu 0,1, tối đa 200 vòng và cân bằng trọng số lớp. Nhóm sẽ thử thêm $C\in\{0.5,1,2\}$, giữ các yếu tố khác cố định để dễ so sánh.

Với mỗi cấu hình, nhóm dự kiến quét 200 threshold từ 0,005 đến 1,0 trên Validation. Trong nhóm đạt Recall mục tiêu 0,85, chọn ngưỡng theo Accuracy → F1 → Precision → FPR thấp hơn. Nếu chưa đạt thì xét Recall → Accuracy → F1.

Khi so sánh cấu hình và mô hình cuối, nhóm ưu tiên đạt mục tiêu Recall rồi xét Accuracy → F1; nếu chưa đạt thì xét Recall → Accuracy → F1. Precision, F2, ROC-AUC, FPR và confusion matrix giúp bổ sung góc nhìn về lỗi nhận nhầm ham và bỏ sót spam.

## 5. Kế hoạch kiểm tra

Nhóm sẽ đối chiếu loss và gradient trên ví dụ nhỏ, kiểm tra hướng cập nhật có làm giảm loss hay không và quan sát quá trình hội tụ. Dữ liệu dạng đặc và thưa cần cho kết quả tương đương trong sai số số học.

Nhóm cũng sẽ kiểm tra dữ liệu không hữu hạn, nhãn không hợp lệ, số cột không khớp và điểm nằm sát threshold. Bộ biến đổi học từ Train cần được giữ nguyên khi đánh giá hoặc dự đoán email mới.

## 6. Kế hoạch đánh giá và giới hạn

Mỗi thí nghiệm dự kiến ghi lại cấu hình, bộ đặc trưng, threshold, diễn biến loss và metrics. Nhóm sẽ chọn phương án bằng Validation trước khi đánh giá Test, đồng thời dùng cùng split để so sánh với baseline.

Email rất ngắn, chứa từ mới hoặc có ngữ cảnh phức tạp có thể bị dự đoán sai. Nhóm sẽ đọc các mẫu lỗi để đề xuất thí nghiệm tiếp theo; một split chưa đủ để kết luận mô hình ổn định trên mọi nguồn email.

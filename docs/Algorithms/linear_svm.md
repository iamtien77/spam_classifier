# Linear SVM trong bài lab phân loại spam

Nhóm dự kiến dùng Linear SVM để so sánh với Logistic Regression và baseline Naive Bayes. Phần thiết kế tập trung vào hàm mục tiêu, cách tối ưu và cách dùng điểm mô hình để phân loại email.

## 1. Mục đích và lý do lựa chọn

Linear SVM phù hợp với dữ liệu văn bản nhiều chiều và có ranh giới tuyến tính. Mô hình tính decision score:

$$
z=w^T x+b
$$

Score cho biết mẫu nằm về phía nào của siêu phẳng. Khoảng cách có dấu đến siêu phẳng là $z/\|w\|$ khi $w$ khác 0; bản thân $z$ chưa phải khoảng cách hình học đã chuẩn hóa.

Hạn chế của mô hình là khó biểu diễn quan hệ phi tuyến nếu đặc trưng đầu vào chưa thể hiện được những quan hệ đó. Nhóm sẽ so sánh trên cùng bộ đặc trưng với các mô hình còn lại.

## 2. Dữ liệu đầu vào và kết quả đầu ra

Đầu vào huấn luyện gồm ma trận đặc trưng hữu hạn, không rỗng và nhãn ham = 0, spam = 1 có đủ hai lớp. Nhóm dự kiến dùng ma trận thưa. Bản thân SVM không yêu cầu đặc trưng không âm.

Nhóm sẽ phân biệt hai miền điểm để tránh nhầm threshold:

| Cách biểu diễn | Giá trị dùng so sánh | Ngưỡng dự kiến |
| :--- | :--- | :--- |
| Điểm tuyến tính | $z=w^Tx+b$ | Chỉ biểu diễn điểm |
| Phân loại trực tiếp | So sánh $z$ với ngưỡng margin | Ban đầu là 0 |
| Điểm sigmoid | $s(x)=1/(1+e^{-z})$ | Chỉ biểu diễn điểm trong khoảng 0–1 |
| Phân loại sau chọn ngưỡng | So sánh $s(x)$ với $\tau$ | Chọn trên Validation |

Mốc margin 0 tương ứng với sigmoid 0,5. Threshold chọn trên sigmoid phải được so sánh với sigmoid score, không dùng trực tiếp làm ngưỡng margin.

Quy tắc phân loại dự kiến là spam khi $s(x)\ge\tau$, ngược lại là ham. Khi tính sigmoid, nhóm dự kiến giới hạn margin trong $[-30,30]$ để ổn định số học. Phép biến đổi này chưa phải calibration.

## 3. Dữ liệu sử dụng

Nhóm sẽ huấn luyện bằng Train và dùng Validation chọn tham số, threshold. Test được giữ riêng để đánh giá cuối.

Đầu vào dự kiến bắt đầu từ word TF-IDF, sau đó có thể bổ sung bigram, character n-gram, từ khóa và đặc trưng số. Từ vựng, IDF và tham số chuẩn hóa chỉ được học từ Train. Nếu chọn bớt đặc trưng, cả ba tập cần giữ cùng thứ tự cột.

## 4. Phương pháp huấn luyện và đánh giá

### Hàm mục tiêu

Nhãn được đổi thành $t_i=2y_i-1$. Mức vi phạm lề của mẫu $i$ là:

$$
m_i=\max(0,1-t_i(w^Tx_i+b))
$$

Nhóm dự kiến ưu tiên squared hinge với hàm mục tiêu:

$$
J(w,b)=\frac{\lambda}{2}\|w\|^2+\frac{1}{N}\sum_i a_i m_i^2
$$

Nếu thử hinge, thành phần $m_i^2$ được thay bằng $m_i$. Với trọng số lớp cân bằng, $a_i=N/(2N_{y_i})$; nếu không cân bằng thì $a_i=1$. Nhóm dự kiến dùng $\lambda=1/\max(10^{-5},CN)$ với $C>0$, không điều chuẩn bias. $N$ là số mẫu Train, còn $N_{y_i}$ là số mẫu thuộc lớp của mẫu $i$.

### Cách tối ưu

Nhóm dự kiến tối ưu trực tiếp theo trọng số và bias của bài toán primal, bắt đầu từ 0. Mỗi bước cập nhật dùng gradient, giảm bước theo điều kiện Armijo và điều chỉnh bước tiếp theo bằng Barzilai–Borwein. Nếu thử hinge, nhóm sẽ dùng subgradient tại các đoạn tương ứng.

Nhóm sẽ theo dõi loss và số vòng cập nhật, dừng khi gradient đủ nhỏ hoặc đạt giới hạn số vòng. Ngưỡng gradient dự kiến không lớn hơn $10^{-7}$; nếu chưa hội tụ trong số vòng cho phép, cần ghi rõ khi so sánh.

### Cấu hình và tiêu chí đánh giá

Cấu hình khởi đầu dự kiến là $C=1$, bước học ban đầu 0,05, tối đa 150 vòng, cân bằng trọng số lớp và squared hinge. Nhóm sẽ thử $C\in\{0.5,1,2\}$ trên cùng split.

Với mỗi cấu hình, nhóm dự kiến thử 200 threshold từ 0,005 đến 1,0 trên sigmoid score của Validation. Trong nhóm đạt Recall mục tiêu 0,85, chọn Accuracy cao nhất rồi xét F1, Precision và FPR thấp hơn. Nếu chưa đạt thì xét Recall → Accuracy → F1.

Khi chọn cấu hình và mô hình cuối, nhóm ưu tiên đạt mục tiêu Recall rồi xét Accuracy → F1; nếu chưa đạt thì xét Recall → Accuracy → F1. Test không tham gia lựa chọn. Các chỉ số dự kiến báo cáo gồm Accuracy, Precision, Recall, F1, F2, ROC-AUC, FPR và confusion matrix.

## 5. Kế hoạch kiểm tra

Nhóm sẽ dùng ví dụ nhỏ để kiểm tra dấu của margin, mức vi phạm lề, loss và hướng cập nhật. Cần phân biệt hinge với squared hinge, đồng thời kiểm tra ảnh hưởng của điều chuẩn và trọng số lớp.

Các trường hợp cần thử gồm dữ liệu không hợp lệ, sai số cột, mẫu nằm đúng ngưỡng và dữ liệu có đặc trưng rất nhỏ. Nhóm cũng sẽ đối chiếu kết quả giữa dạng ma trận đặc và thưa, kiểm tra sự tương ứng giữa ngưỡng margin 0 và sigmoid 0,5.

## 6. Kế hoạch đánh giá và giới hạn

Nhóm sẽ ghi cấu hình, diễn biến loss, threshold và metrics của từng thí nghiệm. Mô hình và ngưỡng được chọn bằng Validation trước khi đánh giá Test; không đặt trước một mô hình thắng.

Khi phân tích, nhóm cần xem cả spam bị bỏ sót và ham bị nhận nhầm. Email viết biến dạng, khác ngôn ngữ hoặc khác nguồn dữ liệu có thể làm giảm chất lượng. Đạt Recall mục tiêu trên Validation không bảo đảm Test cũng đạt mục tiêu đó.

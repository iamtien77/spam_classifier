Phiên bản v2. Thiết kế bộ dò ngưỡng tự xây dựng (ThresholdOptimizerScratch)
Mục tiêu của bản này là cài đặt lớp ThresholdOptimizerScratch để quét lưới 200 mốc ngưỡng phân vị xác suất trên tập Validation và tính toán chi tiết toàn bộ bảng chỉ số tương ứng tại từng mốc.

Bước 1. Thiết kế lưới ngưỡng quét (Threshold Grid)
Triển khai lớp ThresholdOptimizerScratch trong src/models/tuning.py.
Xây dựng lưới 200 mốc ngưỡng phân vị đều nhau từ 0.005 đến 1.0:
grid = np.linspace(0.005, 1.0, 200).
Xác thực lưới: Mảng 1 chiều, không rỗng, giá trị hữu hạn và thuộc đoạn [0, 1].

Bước 2. Tính toán ma trận nhầm lẫn và chỉ số tại từng ngưỡng
Phương thức compute_metrics_at_threshold(y_true, probs, threshold):
1. Chuyển xác suất thành nhãn nhị phân: y_pred = (probs >= threshold).astype(int).
2. Tính 4 thành phần ma trận nhầm lẫn: TP, FP, TN, FN.
3. Tính các chỉ số:
   - accuracy = (TP + TN) / total
   - precision = TP / (TP + FP)
   - recall = TP / (TP + FN)
   - fpr = FP / (FP + TN)
   - specificity = TN / (TN + FP)
   - f1 = 2 * precision * recall / (precision + recall)
   - f2 = (1 + 2^2) * precision * recall / (2^2 * precision + recall)

Bước 3. Xuất bảng dữ liệu đánh giá ngưỡng (Metrics Table)
Phương thức metrics_table(y_true, probs):
Duyệt qua toàn bộ 200 mốc ngưỡng, tính toán và lưu vào DataFrame gồm 200 hàng.
DataFrame này cung cấp bức tranh toàn cảnh về sự đánh đổi giữa Precision và Recall khi ngưỡng thay đổi từ thấp đến cao.

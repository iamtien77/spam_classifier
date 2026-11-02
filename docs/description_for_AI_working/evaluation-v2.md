Phiên bản v2. Tính toán bộ chỉ số đo lường hiệu năng toàn diện trên Test
Phiên bản này ứng dụng SpamModelEvaluator để tự tính toán bộ chỉ số đánh giá toàn diện: Accuracy, Precision, Recall, F1-Score, F2-Score, Specificity và False Positive Rate trên tập Test.

Bước 1. Xác định các công thức toán học đo lường
Dựa trên ma trận nhầm lẫn: TN = 997, FP = 666, FN = 180, TP = 1.059.

Bước 2. Tính toán các chỉ số phân loại nhị phân
1. Accuracy (Tỷ lệ dự đoán đúng toàn thể):
   Accuracy = (TP + TN) / (TP + FP + TN + FN) = (1059 + 997) / 2902 = 70,85% (0.70848).
2. Precision (Độ chuẩn xác trên lớp Spam):
   Precision = TP / (TP + FP) = 1059 / (1059 + 666) = 61,39% (0.61391).
3. Recall / Sensitivity (Độ nhạy phát hiện thư rác):
   Recall = TP / (TP + FN) = 1059 / (1059 + 180) = 85,47% (0.85472).
   Xác nhận: Đạt chỉ tiêu nghiệp vụ đề ra ban đầu Recall >= 0.85.
4. F1-Score (Trung bình điều hòa Precision và Recall):
   F1 = 2 * (Precision * Recall) / (Precision + Recall) = 71,46% (0.71457).
5. F2-Score (Đo lường ưu tiên bắt trúng thư rác với beta = 2.0):
   F2 = (1 + 2^2) * (Precision * Recall) / (2^2 * Precision + Recall) = 79,25% (0.79255).
6. Specificity (Tỷ lệ phân loại đúng email hợp lệ):
   Specificity = TN / (TN + FP) = 997 / (997 + 666) = 59,95% (0.59952).
7. False Positive Rate (FPR - Tỷ lệ chặn nhầm email hợp lệ):
   FPR = FP / (FP + TN) = 666 / 1663 = 40,05% (0.40048).

Bước 3. Đánh giá sự đánh đổi thực nghiệm (Trade-off)
Mô hình đạt mục tiêu với tỷ lệ phát hiện thư rác là 85,47%, tuy nhiên tỷ lệ chặn nhầm còn ở mức 40,05%.
Kết luận: Kết quả phản ánh đúng bản chất đánh đổi trong lọc thư rác thực tế, rất phù hợp cho cơ chế cảnh báo hoặc chuyển vào thư mục Spam tạm thời.

Phiên bản v6. Chốt mô hình được chọn (Winner Selection) và chuyển giao sang Test
Phiên bản này phân tích bảng kết quả Validation, chính thức xác định mô hình được chọn (Winner), cố định ngưỡng quyết định tối ưu và chuẩn bị chuyển giao toàn bộ sang khâu đánh giá độc lập.

Bước 1. Phân tích kết quả xếp hạng trên tập Validation
Từ bảng validation_model_selection.csv:
- Hạng 1: Linear SVM (Tuned) - Accuracy: 70,07%, Recall: 85,55%, F1: 70,93%, F2: 79,03%, ngưỡng: 0,465.
- Hạng 2: Linear SVM (Default) - Accuracy: 69,96%, Recall: 85,39%, F1: 70,82%, F2: 78,90%, ngưỡng: 0,465.
- Hạng 3: Logistic Regression (Tuned) - Accuracy: 69,96%, Recall: 85,39%, F1: 70,82%, F2: 78,90%, ngưỡng: 0,415.
- Hạng 4: Logistic Regression (Default) - Accuracy: 69,72%, Recall: 85,47%, F1: 70,67%, F2: 78,87%, ngưỡng: 0,410.
- Hạng 5: SpamVotingEnsemble (Tuned) - Accuracy: 69,07%, Recall: 85,47%, F1: 70,23%, F2: 78,64%, ngưỡng: 0,385.
- Hạng 6: SpamVotingEnsemble (Default) - Accuracy: 68,72%, Recall: 86,36%, F1: 70,21%, F2: 79,08%, ngưỡng: 0,365.
- Hạng 7: Complement Naive Bayes (Tuned) - Accuracy: 63,66%, Recall: 85,23%, F1: 66,69%, F2: 76,70%, ngưỡng: 0,220.
- Hạng 8: Complement Naive Bayes (Default) - Accuracy: 63,42%, Recall: 85,07%, F1: 66,50%, F2: 76,52%, ngưỡng: 0,215.

Bước 2. Quyết định kỹ thuật chọn Winner
Linear SVM (Tuned) đứng đầu bảng xếp hạng, thỏa mãn ràng buộc Recall >= 0.85 (đạt 85,55%) và đạt Validation Accuracy cao nhất (70,07%).
Hệ thống chính thức chọn Linear SVM (Tuned) làm Selected Final Model.
Cấu hình tham số tối ưu:
- Thuật toán: Linear Support Vector Machine
- C = 2.0
- learning_rate = 0.05
- max_iter = 150
- loss = 'squared_hinge'
- class_weight = 'balanced'
- Ngưỡng quyết định đã chốt: tau* = 0.465

Bước 3. Đóng băng cấu hình và chuyển giao
Khóa toàn bộ tham số và ngưỡng tau* = 0.465.
Không cho phép thay đổi ngưỡng hay tham số trong bước đánh giá Test tiếp theo.
Kết quả lựa chọn được chuyển sang bước đánh giá Test.

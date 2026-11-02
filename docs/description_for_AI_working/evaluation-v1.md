Phiên bản v1. Đánh giá độc lập trên Test set và ma trận nhầm lẫn
Phiên bản này thực hiện kiểm chứng khách quan mô hình được chọn Linear SVM (Tuned) trên tập Test độc lập (2.902 mẫu) tại ngưỡng tối ưu đã chốt tau* = 0.465, tính toán chi tiết 4 thành phần ma trận nhầm lẫn.

Bước 1. Nguyên tắc đánh giá độc lập bất biến
Tập Test gồm 2.902 mẫu (1.663 Ham, 1.239 Spam).
Tuyệt đối không sử dụng tập Test để tinh chỉnh lại ngưỡng hay sửa đổi tham số mô hình.
Sử dụng nguyên vẹn Linear SVM (Tuned) và ngưỡng tau* = 0.465 đã được chốt từ tập Validation.

Bước 2. Thực hiện dự đoán trên 2.902 mẫu Test
Gọi hàm tính xác suất / điểm Sigmoid xấp xỉ trên X_test:
probs = model.predict_proba(X_test)[:, 1].
Áp dụng ngưỡng tau* = 0.465:
y_pred = (probs >= 0.465).astype(int).

Bước 3. Tính toán 4 thành phần ma trận nhầm lẫn (Confusion Matrix)
Sử dụng hàm compute_confusion_matrix() trong SpamModelEvaluator.
Quy ước: Hàng là nhãn thực tế, Cột là nhãn dự đoán.
Kết quả ghi nhận:
- True Negatives (TN): 997 mẫu (Email hợp lệ được phân loại đúng là Ham).
- False Positives (FP): 666 mẫu (Email hợp lệ bị chặn nhầm thành Spam).
- False Negatives (FN): 180 mẫu (Email rác bị bỏ sót, gán nhầm thành Ham).
- True Positives (TP): 1.059 mẫu (Email rác được phát hiện chính xác là Spam).
Tổng số mẫu: 997 + 666 + 180 + 1059 = 2.902 mẫu kiểm thử.
Ma trận này là cơ sở để tính toàn bộ các chỉ số hiệu năng ở bản tiếp theo.

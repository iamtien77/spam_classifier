Phiên bản v8. Lựa chọn đặc trưng Top-K và lưu trữ ma trận đệm (Cache Persistence)
Mục tiêu của bản này là hoàn thiện khâu Feature Engineering bằng cách kiểm chứng quy trình lựa chọn đặc trưng Top-K trên tập Validation, khóa thứ tự cột và xuất toàn bộ ma trận đặc trưng ra thư mục cache data/processed/.

Bước 1. Thử nghiệm lựa chọn đặc trưng Top-K (ManualFeatureSelectorScratch)
Xây dựng lớp ManualFeatureSelectorScratch để xếp hạng độ quan trọng của 4.318 cột trên tập Train.
Thử nghiệm các ngưỡng giữ lại K: 1.000, 2.000, 3.000, 4.000 và 4.318.
Dùng mô hình đại diện Complement Naive Bayes để đánh giá nhanh F1-score trên tập Validation.
Kết quả thực nghiệm: Mức K = 4.318 (giữ nguyên 100% đặc trưng) mang lại F1-score và ROC-AUC tốt nhất, do các mô hình Linear SVM và Logistic Regression có cơ chế phạt điều chuẩn L2 tự triệt tiêu trọng số của các cột nhiễu.
Khóa danh sách cột selected_indices = list(range(4318)).

Bước 2. So sánh đối chứng 5 tổ hợp đặc trưng (Ablation Study)
Chạy hàm experiment_feature_combinations() để so sánh 5 nhánh trên tập Validation:
1. Word TF-IDF đơn thuần (3.000 cột).
2. Word + Ký tự tần suất (!, $) (3.002 cột).
3. Word + Character n-grams (3.500 cột).
4. Word + Keywords (3.010 cột).
5. Hybrid toàn diện (4.318 cột).
Kết quả khẳng định nhánh Hybrid toàn diện cho hiệu năng vượt trội nhất.

Bước 3. Lưu trữ ma trận đệm và vector nhãn ra đĩa
Sử dụng hàm export_processed_data() để lưu các ma trận và nhãn vào thư mục data/processed/:
- X_train.npz, X_val.npz, X_test.npz (ma trận thưa SciPy CSR).
- y_train.npy, y_val.npy, y_test.npy (mảng nhãn nhị phân NumPy int32).
Giúp Giai đoạn 3 nạp dữ liệu tức thì trong vài mili-giây mà không cần tiền xử lý hay trích xuất lại từ đầu.
Quy trình Feature Engineering chính thức hoàn tất.

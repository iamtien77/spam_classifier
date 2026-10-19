Phiên bản v1. Phân tích hạn chế của ngưỡng quyết định mặc định 0.5
Mục tiêu của bản này là phân tích cơ sở lý thuyết tại sao ngưỡng xác suất mặc định 0.5 không phù hợp cho bài toán lọc thư rác thực tế, từ đó đặt ra yêu cầu xây dựng bộ tối ưu hóa ngưỡng phân loại chuyên biệt.

Bước 1. Bản chất sự bất đối xứng về chi phí sai sót (Cost Asymmetry)
Trong bài toán spam, hai loại sai sót có mức độ thiệt hại rất khác nhau:
- False Positive (FP): Email công việc hợp lệ bị chặn nhầm vào hòm thư rác (hậu quả: người dùng có thể lỡ thông tin quan trọng, hợp đồng hoặc giao dịch).
- False Negative (FN): Thư rác lọt vào hòm thư chính (hậu quả: gây phiền toái, giảm trải nghiệm).
Tuy nhiên, nếu chỉ tối ưu độ chính xác thông thường (Accuracy) tại ngưỡng 0.5, mô hình có thể chấp nhận bỏ sót nhiều thư rác (Recall thấp) để giảm FP.

Bước 2. Hạn chế của ngưỡng cố định 0.5 trên các mô hình khác nhau
Các thuật toán có thang đo xác suất rất khác nhau:
- Complement Naive Bayes: Xác suất phân phối thường lệch về các giá trị nhỏ (thường < 0.3), ngưỡng 0.5 sẽ khiến mô hình hầu như không bắt được spam nào.
- Linear SVM: Điểm Sigmoid từ khoảng cách đại số chưa qua hiệu chỉnh Platt Scaling, ngưỡng 0.5 không đại diện cho xác suất 50% thực tế.
- Logistic Regression: Dù xuất xác suất chuẩn hơn, ngưỡng 0.5 vẫn chưa cân bằng được mục tiêu phát hiện đa số thư rác.

Bước 3. Đặt ra yêu cầu tối ưu hóa ngưỡng (Threshold Tuning)
Cần xây dựng một thuật toán quét toàn bộ dải ngưỡng trên tập Validation, cho phép điều chỉnh ngưỡng quyết định tau linh hoạt theo mục tiêu nghiệp vụ cụ thể thay vì dùng cố định 0.5.

Phiên bản v7. Triển khai dịch vụ suy luận email mới (SpamInferenceService) và kịch bản Demo
Phiên bản này ứng dụng module predictor.py để xây dựng lớp dịch vụ suy luận SpamInferenceService, cung cấp giao diện lập trình trực quan (Python API) cho phép dự đoán nhãn Spam/Ham trên các email văn bản mới bất kỳ.

Bước 1. Thiết kế dịch vụ suy luận SpamInferenceService
Triển khai lớp SpamInferenceService trong src/deployment/predictor.py.
Phương thức load_service(bundle_path):
- Tự động tìm kiếm file artifact tại results/saved_models/spam_pipeline.joblib nếu không chỉ định đường dẫn.
- Nạp bundle lên bộ nhớ và trích xuất: model, feature_builder, selected_indices, threshold.

Bước 2. Cài đặt API dự đoán email đơn lẻ (predict_single)
Phương thức predict_single(raw_text: str):
1. Chuẩn hóa văn bản đầu vào bằng SpamTextProcessor có trong bundle.
2. Trích xuất vector đặc trưng bằng feature_builder.transform().
3. Lọc đúng 4.318 cột bằng selected_indices.
4. Tính điểm xác suất: spam_prob = model.predict_proba(X_features)[0, 1].
5. So sánh với ngưỡng threshold = 0.465:
   prediction = 'spam' nếu spam_prob >= 0.465, ngược lại là 'ham'.
6. Trả về từ điển kết quả:
   {'prediction': prediction, 'spam_probability': spam_prob, 'threshold_used': 0.465}.

Bước 3. Cài đặt API dự đoán hàng loạt (predict_batch)
Phương thức predict_batch(texts: list[str]):
- Biến đổi đồng thời danh sách email thành ma trận thưa.
- Tính toán ma trận xác suất và nhãn dự đoán theo lô (batch processing).
- Trả về bảng pandas DataFrame chứa các cột: text, prediction, spam_probability, threshold_used.

Bước 4. Thực thi kịch bản kiểm thử Demo trên 5 mẫu email thực tế
Ở phần cuối của main.py, chạy thử nghiệm dự đoán trên 5 mẫu văn bản:
1. Email công việc: "Dear team, please review the meeting notes before Monday." -> Dự đoán: ham.
2. Email quảng cáo trúng thưởng: "URGENT! Claim your free cash prize now!" -> Dự đoán: spam.
3. Email báo cáo dự án: "Please send the weekly project report when ready." -> Dự đoán: ham.
4. Email quà tặng miễn phí: "Free entry into the weekly draw. Text WIN to 80082." -> Dự đoán: spam.
5. Email thông báo dịch vụ: "Your account balance is low, please refill." -> Phân loại và xuất xác suất chi tiết.
Phần demo kết thúc quy trình từ nạp dữ liệu đến suy luận trên văn bản mới.

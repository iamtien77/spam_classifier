Phiên bản v4. Phân tích lỗi chi tiết (Error Analysis) và trích xuất mẫu phân loại sai
Mục tiêu của bản này là ứng dụng module error_analysis.py để lọc toàn bộ 846 mẫu email bị phân loại sai trên tập Test, phân tích đặc điểm ngôn ngữ học và cấu trúc của các mẫu lỗi nhằm định hướng cải tiến hệ thống.

Bước 1. Lọc các mẫu phân loại sai trên tập Test
So sánh nhãn thực tế y_test và nhãn dự đoán y_pred:
Xác định tập các chỉ số sai: error_indices = {i | y_test[i] != y_pred[i]}.
Tổng số mẫu phân loại sai: 846 mẫu (chiếm 29,15% tập Test).
Trong đó gồm 666 mẫu False Positive (Ham bị gán nhầm là Spam) và 180 mẫu False Negative (Spam bị lọt lưới).

Bước 2. Xuất dữ liệu lỗi chi tiết ra file CSV
Sử dụng hàm extract_misclassified_samples():
Trích xuất thông tin chi tiết của 846 mẫu: source_id, true_label, predicted_label, spam_probability, raw_text, clean_text, model_length.
Ghi dữ liệu ra file results/metrics/misclassified_samples.csv để phục vụ kiểm tra thủ công.

Bước 3. Phân tích nguyên nhân các trường hợp False Positive (666 mẫu)
Phân tích văn bản email hợp lệ bị chặn nhầm cho thấy:
- Email chứa nhiều đường link mời họp trực tuyến (Zoom, Google Meet, Teams) bị nhận nhầm thành link tiếp thị (has_url = 1).
- Email trao đổi kỹ thuật hoặc báo cáo tài chính chứa mật độ chữ số và ký hiệu đặc biệt cao.
- Email công việc sử dụng các từ ngữ nhấn mạnh mức độ khẩn cấp (urgent, call, please reply immediately, offer).

Bước 4. Phân tích nguyên nhân các trường hợp False Negative (180 mẫu)
Phân tích văn bản email rác bị bỏ sót cho thấy:
- Thư rác cực ngắn (dưới 10 từ), nội dung chung chung (ví dụ: "Hello, check this out") không chứa các từ khóa tiếp thị truyền thống.
- Thư rác cố tình ngắt chữ hoặc dùng ngôn ngữ đời thường không có tín hiệu cấu trúc bất thường.
Đánh giá này giúp khẳng định các giới hạn hiện tại của mô hình và đưa ra khuyến nghị chỉ nên dùng hệ thống ở chế độ gắn cờ cảnh báo (Flagging).

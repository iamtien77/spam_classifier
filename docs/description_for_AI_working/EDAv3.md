Bản v3. Phân tích thống kê độ dài email và phát hiện phân bố lệch
Mục tiêu của bản này là phân tích chuyên sâu độ dài văn bản email theo số ký tự và số từ, so sánh giữa Ham và Spam, tính toán các quantile quan trọng để phát hiện email ngoại lai (outliers).

Bước 1. Xác định mục tiêu nâng cấp so với EDA v2
Bổ sung các trường độ dài chi tiết: raw_char_length và raw_word_count.
Phân tích thống kê mô tả 5 số (5-number summary) cùng các phân vị cao q90, q95, q99.
So sánh trực quan phân bố độ dài giữa hai lớp Ham và Spam.

Bước 2. Tính toán độ dài văn bản email thô
Đo số lượng ký tự của từng email bằng len(text) lưu vào raw_char_length.
Đo số lượng từ bằng cách tách khoảng trắng len(text.split()) lưu vào raw_word_count.
Tính độ dài trung bình của từ avg_word_length = raw_char_length / raw_word_count.
Ghi nhận các email có độ dài bằng 0 ký tự hoặc chỉ chứa khoảng trắng.

Bước 3. Thống kê mô tả độ dài tổng thể
Tính các chỉ số min, max, mean, std, median và phân vị Q1, Q3 trên toàn bộ dữ liệu.
Tính các phân vị đuôi dài: q95, q99 và q99.9 để nhận diện mức độ lệch phải (right-skewed).
Phát hiện sự tồn tại của các email cực dài (vượt quá vài nghìn từ), có thể gây phình ma trận TF-IDF nếu không được giới hạn.

Bước 4. So sánh phân bố độ dài giữa Ham và Spam
Nhóm dữ liệu theo nhãn Ham và Spam để tính thống kê độ dài riêng biệt.
So sánh mean và median: phân tích xem email Spam có xu hướng dài hơn hay ngắn hơn email Ham.
Quan sát độ biến động (std) của từng lớp: email Spam thường có độ dài tập trung ở các khung từ vựng quảng cáo nhất định, trong khi Ham có độ phân tán lớn hơn.

Bước 5. Trực quan hóa phân bố độ dài
Vẽ biểu đồ phân bố Histogram kết hợp đường cong ước lượng mật độ KDE.
Vẽ biểu đồ Boxplot để quan sát trực quan các điểm ngoại lai (outliers) ở đuôi phân bố.
Rút ra kết luận kỹ thuật: cần áp dụng cơ chế cắt độ dài (Length Capping) ở phân vị 99% để loại bỏ ảnh hưởng tiêu cực của các email ngoại lai siêu dài.

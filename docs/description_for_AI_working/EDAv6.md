Bản v6. Phân tích tương quan đặc trưng và phát hiện mâu thuẫn nhãn
Phiên bản này hoàn thiện quy trình EDA toàn diện, phân tích ma trận tương quan giữa các đặc trưng số với nhãn mục tiêu và kiểm tra mức độ trùng lặp/mâu thuẫn trong tập dữ liệu.

Bước 1. Nội dung bổ sung so với EDA v5
Tính toán hệ số tương quan tuyến tính giữa các tín hiệu cấu trúc và nhãn mục tiêu (0 và 1).
Khảo sát hiện tượng các email có cùng nội dung văn bản nhưng bị gán nhãn trái ngược nhau.

Bước 2. Phân tích ma trận tương quan (Correlation Analysis)
Tính hệ số tương quan Pearson giữa nhãn spam và các đặc trưng số: has_url, has_number, digit_count, special_char_count, exclamation_count, dollar_count, uppercase_ratio, model_length.
Vẽ biểu đồ Heatmap thể hiện ma trận tương quan.
Kết quả cho thấy has_url, digit_count và exclamation_count có hệ số tương quan dương đáng kể với nhãn Spam.

Bước 3. Phát hiện và kiểm toán nhóm nội dung mâu thuẫn nhãn
Chuẩn hóa văn bản bằng Unicode NFKC và gộp khoảng trắng thừa để tạo khóa nội dung đại diện.
Gom nhóm các bản ghi có cùng khóa nội dung và kiểm tra tính nhất quán của nhãn.
Phát hiện hiện tượng nghiêm trọng: có 738 nhóm nội dung văn bản hoàn toàn giống nhau nhưng vừa được gán là Ham vừa được gán là Spam (tổng cộng 1.480 dòng).
Đưa ra kết luận tiền xử lý: phải loại bỏ toàn bộ các nhóm mâu thuẫn này để tránh làm mô hình học các tín hiệu nghịch lý.

Bước 4. Tổng kết toàn diện kết quả EDA và chuyển giao sang Tiền xử lý
Tập hợp toàn bộ các phát hiện từ EDA v1 đến v6:
- Dữ liệu thô có tỷ lệ mất cân bằng nhẹ (57% Ham vs 43% Spam).
- Phân bố độ dài bị lệch phải nặng, cần cắt ngưỡng (length capping) ở q99.
- Tồn tại nhiều bản ghi trùng lặp và mâu thuẫn nhãn cần được làm sạch bằng kiểm toán 5 bước không chồng lặp.
- Các tín hiệu URL, số, ký tự đặc biệt và 10 từ khóa nhạy cảm mang giá trị phân biệt cao, cần được đưa vào kiến trúc đặc trưng lai.
Các kết quả EDA được dùng làm cơ sở thiết kế bước Data Processing.

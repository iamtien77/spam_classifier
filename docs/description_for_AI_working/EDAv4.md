Bản v4. Phân tích các tín hiệu spam cấu trúc (Spam Signal Analysis)
Phiên bản này khảo sát và đo lường tần suất xuất hiện của các tín hiệu cấu trúc đặc trưng trong email như liên kết URL, ký hiệu tiền tệ, chữ số và tỷ lệ chữ hoa giữa Ham và Spam.

Bước 1. Nội dung bổ sung so với EDA v3
Mở rộng phân tích từ độ dài đơn thuần sang các đặc tính cấu trúc đặc trưng của email spam.
Đo lường độ chênh lệch xuất hiện của các tín hiệu số giữa hai lớp để kiểm chứng giả thuyết phân loại.

Bước 2. Phân tích sự xuất hiện của liên kết URL
Định nghĩa mẫu regex nhận diện URL thống nhất: các liên kết bắt đầu bằng http://, https:// hoặc www. kèm nội dung hợp lệ theo sau.
Tính tỷ lệ email chứa URL trong lớp Spam so với lớp Ham.
Kết quả cho thấy email Spam có tỷ lệ chứa URL cao vượt trội so với email Ham thông thường.
Ghi nhận has_url là một cờ nhị phân có khả năng phân biệt hai lớp rõ rệt.

Bước 3. Phân tích sự xuất hiện của chữ số và ký hiệu tiền tệ
Đếm số lượng chữ số (digit_count) và cờ nhị phân has_number trong từng email.
Đếm số lần xuất hiện của ký hiệu tiền tệ (dollar_count: dấu $, £, €).
So sánh tần suất giữa hai lớp: email Spam thường chứa nhiều số điện thoại liên lạc, mã trúng thưởng hoặc giá trị tiền thưởng hơn email Ham.

Bước 4. Phân tích tỷ lệ chữ in hoa và dấu chấm than
Tính tỷ lệ chữ hoa uppercase_ratio = (số chữ hoa) / max(1, tổng số chữ cái).
Đếm số lượng dấu chấm than exclamation_count (!) trong văn bản gốc.
Đánh giá mức độ kích động/khẩn cấp: email Spam lừa đảo hoặc quảng cáo có xu hướng lạm dụng chữ hoa toàn bộ (ALL CAPS) và nhiều dấu chấm than liên tiếp để gây chú ý.

Bước 5. Tổng kết các tín hiệu cấu trúc làm tiền đề cho Feature Engineering
Lập bảng so sánh giá trị trung bình của các tín hiệu: has_url, has_number, digit_count, dollar_count, exclamation_count, uppercase_ratio giữa Ham và Spam.
Khẳng định các tín hiệu này mang tính phân tách cao và cần được trích xuất thành khối đặc trưng số riêng biệt bổ trợ cho TF-IDF.

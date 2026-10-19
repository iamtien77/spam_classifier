Phiên bản v3. Kiểm toán chất lượng dữ liệu 5 bước không chồng lặp (Data Quality Audit)
Mục tiêu của bản này là giải quyết triệt để vấn đề dữ liệu bẩn: lọc nhãn sai, lọc chuỗi rỗng, xử lý 738 nhóm mâu thuẫn nhãn và khử trùng lặp hoàn toàn, bảo toàn nguyên tắc tổng các nhóm loại cộng số mẫu giữ lại bằng đúng số mẫu đầu vào.

Bước 1. Xác định mục tiêu nâng cấp so với v2
Dữ liệu thô 43.849 dòng chứa nhiều lỗi nghiêm trọng: nhãn rác, văn bản trống, bản ghi trùng lặp và cùng nội dung nhưng khác nhãn.
Cần thiết lập quy trình kiểm toán chặt chẽ, không chồng lặp (Mutually Exclusive) để đảm bảo tính minh bạch khoa học.

Bước 2. Bước kiểm toán 1 - Lọc nhãn không hợp lệ
Kiểm tra từng dòng dữ liệu: nhãn có phải là ham hoặc spam hay không.
Loại bỏ tất cả các dòng có nhãn là NaN, chuỗi rỗng, hoặc các ký tự lạ.
Số lượng dòng bị loại ở bước này: 2.195 dòng.
Số dòng còn lại: 41.654 dòng.

Bước 3. Bước kiểm toán 2 - Lọc nội dung văn bản rỗng
Trên các dòng có nhãn hợp lệ, kiểm tra nội dung text sau khi loại bỏ khoảng trắng thừa.
Nếu nội dung rỗng hoàn toàn hoặc chỉ gồm khoảng trắng, loại bỏ bản ghi.
Số lượng dòng bị loại ở bước này: 2.192 dòng.
Số dòng còn lại: 39.462 dòng.

Bước 4. Bước kiểm toán 3 - Khử mâu thuẫn nhãn (Conflict Resolution)
Tạo khóa nội dung chuẩn hóa bằng cách chuyển Unicode NFKC và co cụm khoảng trắng (giữ nguyên hoa/thường và dấu câu).
Tìm các nhóm có cùng khóa nội dung nhưng nhãn bị gán trái ngược nhau (vừa có bản ghi gắn nhãn ham, vừa có bản ghi gắn nhãn spam).
Nguyên tắc khoa học: Loại bỏ toàn bộ các nhóm mâu thuẫn này thay vì tự ý chọn nhãn theo số đông (majority voting) để tránh thiên lệch.
Số lượng nhóm mâu thuẫn phát hiện: 738 nhóm. Tổng số dòng bị loại: 1.480 dòng.
Số dòng còn lại: 37.982 dòng.

Bước 5. Bước kiểm toán 4 & 5 - Khử trùng lặp và xác nhận dữ liệu sạch
Với các nhóm có cùng khóa nội dung và cùng nhãn, giữ lại bản ghi đầu tiên theo source_id và loại bỏ các bản sao thừa.
Số lượng dòng trùng lặp bị loại: 8.954 dòng.
Số lượng mẫu sạch cuối cùng được giữ lại: 29.028 mẫu (16.636 Ham và 12.392 Spam).
Kiểm tra đẳng thức: 2.195 + 2.192 + 1.480 + 8.954 + 29.028 = 43.849 dòng (Khớp 100%).
Xuất báo cáo kiểm toán ra file results/metrics/data_audit.json.

Phiên bản v1. Nạp dữ liệu thô và chuẩn hóa Schema
Mục tiêu của bản này là xây dựng luồng nạp dữ liệu an toàn, xử lý linh hoạt các bảng mã (encoding fallback) và đưa các định dạng cột khác nhau về một cấu trúc schema thống nhất.

Bước 1. Xác định yêu cầu nạp dữ liệu an toàn
Đảm bảo file đầu vào được nạp chính xác từ data/raw/spam.csv.
Xử lý lỗi mã hóa Unicode thường gặp trên các tập dữ liệu email cổ điển.
Không để chương trình bị dừng đột ngột do lỗi byte không tương thích.

Bước 2. Cơ chế giải mã linh hoạt (Encoding Fallback)
Triển khai hàm load_raw_data() trong src/data/loader.py.
Thử đọc file bằng bảng mã utf-8-sig để tự động loại bỏ ký tự BOM nếu có.
Nếu xuất hiện ngoại lệ UnicodeDecodeError, chuyển sang đọc dự phòng bằng bảng mã latin-1.
Ghi log thông báo bảng mã đã được sử dụng thành công.

Bước 3. Chuẩn hóa tên cột
Tự động nhận diện cột nhãn từ một trong các tên: label, category, v1.
Tự động nhận diện cột nội dung từ một trong các tên: text, message, v2, body.
Ném ngoại lệ ValueError nếu thiếu một trong hai cột bắt buộc.
Đổi tên cột thành chuẩn chung gồm label và text.

Bước 4. Tạo trường định danh và bản sao bất biến
Tạo trường source_id dưới dạng số nguyên tăng dần từ 0 để đánh dấu vị trí dòng ban đầu.
Tạo trường raw_text lưu nguyên văn nội dung email thô, chưa qua xử lý làm sạch.
Trường raw_text được giữ cố định để các bước sau đếm chữ in hoa, ký tự đặc biệt và dấu câu.
Kết quả tạo ra bảng dữ liệu chuẩn hóa gồm 4 trường: source_id, label, text, raw_text (tổng cộng 43.849 dòng).

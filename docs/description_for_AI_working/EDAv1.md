Bản v1. Kế hoạch EDA ban đầu
Phiên bản này phác thảo các bước phân tích dữ liệu email spam/ham ở mức nền tảng trước khi notebook được bổ sung thêm trực quan hóa, audit và kiểm chứng chi tiết.

Bước 1. Nạp dữ liệu vào DataFrame
Đọc file dữ liệu gốc data/raw/spam.csv vào DataFrame.
Kiểm tra nhanh số dòng và số cột của dữ liệu thô.
In ra 5 dòng đầu tiên để xác nhận dữ liệu đã được đọc đúng cấu trúc.
Xác định cột chứa nhãn (label/category/v1) và cột chứa nội dung email (text/message/v2).
Nếu tên cột gốc chưa rõ ràng, chuẩn hóa về dạng thống nhất gồm label và text.

Bước 2. Kiểm tra cấu trúc dữ liệu ban đầu
Kiểm tra kích thước ban đầu của bảng dữ liệu (ghi nhận 43.849 dòng).
Kiểm tra danh sách tên cột và các cột không tên (unnamed columns) nếu có.
Kiểm tra kiểu dữ liệu của từng cột để đảm bảo nhãn là dạng chuỗi/phân loại và nội dung là chuỗi văn bản.
Xác nhận kiểu dữ liệu tương thích trước khi chuyển sang bước tiếp theo.

Bước 3. Xem trước nội dung email mẫu
Trích một số mẫu email thực tế đại diện cho từng lớp.
Tạo trường preview rút gọn (100 ký tự đầu) để quan sát nhanh cấu trúc.
Tính toán độ dài ký tự sơ bộ và đếm số lượng từ thô.
So sánh nhanh giữa các email ngắn, trung bình và rất dài.
Ghi nhận sơ bộ sự xuất hiện của các chuỗi URL, chữ số, ký hiệu tiền tệ và chữ in hoa.

Bước 4. Chuẩn hóa nhãn để phục vụ phân tích
Ánh xạ nhãn văn bản ham sang nhãn số 0 và spam sang nhãn số 1.
Tạo thêm cột nhãn chuẩn hóa label_name để tiện đối chiếu trực quan.
Kiểm tra tính toàn vẹn của phép ánh xạ nhãn để tránh nhầm lẫn giữa hai lớp.
Bảo toàn nhãn gốc để phục vụ các phân tích so sánh tiếp theo.

Bước 5. Phân tích phân bố lớp ban đầu
Đếm số lượng bản ghi thuộc lớp Ham và số lượng bản ghi thuộc lớp Spam.
Tính tỷ lệ phần trăm tương đối của từng lớp trong tập dữ liệu thô.
Đánh giá mức độ mất cân bằng nhãn ban đầu.
Ghi nhận kết quả này để định hướng việc chọn hàm mục tiêu và kỹ thuật điều chuẩn ở khâu mô hình hóa.

Bản v2. Phân tích phân bố nhãn và kiểm tra dữ liệu thiếu
Phiên bản này nâng cấp EDA v1, đi sâu vào kiểm tra chất lượng nhãn, phát hiện giá trị thiếu (missing values) và đo lường chính xác tỷ lệ mất cân bằng.

Bước 1. Nội dung bổ sung so với EDA v1
Bổ sung kiểm tra missing values chi tiết trên từng cột.
Xác định các nhãn không hợp lệ (không phải ham hoặc spam).
Đo lường mức độ ảnh hưởng của dữ liệu thiếu đối với bài toán phân loại.

Bước 2. Kiểm tra missing values trên toàn bộ DataFrame
Đếm số lượng giá trị null, NaN, None trên cột nhãn.
Đếm số lượng giá trị null hoặc chuỗi chỉ chứa khoảng trắng trên cột nội dung văn bản.
Tính tỷ lệ phần trăm missing values tương ứng trên tổng số mẫu.
Ghi nhận các trường hợp văn bản trống hoàn toàn để chuẩn bị xử lý ở khâu data cleaning.

Bước 3. Rà soát tính hợp lệ của nhãn phân loại
Trích xuất danh sách các giá trị duy nhất (unique values) trong cột nhãn.
Phát hiện các nhãn bị gõ sai, nhãn không xác định hoặc nhãn chứa ký tự lạ ngoài ham và spam.
Thống kê số lượng dòng vi phạm nhãn không hợp lệ (ghi nhận 2.195 dòng).
Kết luận quy tắc xử lý: các dòng có nhãn không hợp lệ bắt buộc phải bị loại bỏ ở khâu tiền xử lý.

Bước 4. Phân tích phân bố lớp trên dữ liệu hợp lệ
Lọc tạm các dòng có nhãn hợp lệ để tính toán lại phân bố.
Vẽ biểu đồ cột (Bar Chart) và biểu đồ tròn (Pie Chart) thể hiện phân bố nhãn.
Đo lường tỷ lệ chênh lệch giữa hai lớp: lớp Ham chiếm đa số (~57%), lớp Spam chiếm thiểu số (~43%).
Ghi nhận mức độ mất cân bằng nhẹ, xác nhận sự cần thiết của kỹ thuật chia phân tầng (Stratified Split) và trọng số lớp (Class Weights).

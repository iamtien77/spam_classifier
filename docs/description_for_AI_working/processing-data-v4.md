Phiên bản v4. Phân chia tập dữ liệu phân tầng (Stratified Split) an toàn
Mục tiêu của bản này là chia 29.028 mẫu sạch thành 3 tập Train, Validation và Test theo tỷ lệ phân tầng 80/10/10, đảm bảo tính bất biến của tỷ lệ nhãn và tính tái lập hoàn toàn của thực nghiệm.

Bước 1. Xác định vai trò của 3 tập dữ liệu
Tập Train (80%): Dùng để học từ vựng, IDF, tham số co dãn scaler và huấn luyện trọng số mô hình.
Tập Validation (10%): Dùng để so sánh đặc trưng Top-K, quét siêu tham số, tối ưu ngưỡng và chọn mô hình vô địch.
Tập Test (10%): Đánh giá kiểm chứng khách quan cuối cùng tại mô hình và ngưỡng đã chọn.
Tuyệt đối không gộp chung Validation và Test, không dùng Test để chọn ngưỡng.

Bước 2. Cấu hình phân chia và kiểm soát ngẫu nhiên
Sử dụng hàm stratified_split_dataframe() trong src/data/preprocessing.py.
Cố định seed ngẫu nhiên random_state = 42.
Thực hiện chia phân tầng độc lập trên từng lớp Ham và Spam để tỷ lệ phân bố giữa hai lớp được bảo toàn trên cả 3 tập.

Bước 3. Áp dụng thuật toán làm tròn phần dư lớn nhất (Largest Remainder Method)
Khi nhân tỷ lệ 80%, 10%, 10% với số lượng mẫu lẻ, phép làm tròn thông thường có thể làm tổng số mẫu bị lệch.
Áp dụng thuật toán phần dư lớn nhất để phân bổ các mẫu còn dư sao cho tổng số lượng mẫu giữa 3 tập luôn bằng chính xác 29.028 mẫu:
- Tập Train: 23.223 mẫu (13.309 Ham, 9.914 Spam).
- Tập Validation: 2.903 mẫu (1.664 Ham, 1.239 Spam).
- Tập Test: 2.902 mẫu (1.663 Ham, 1.239 Spam).

Bước 4. Đảm bảo nguyên tắc chống rò rỉ dữ liệu (No Data Leakage)
Việc phân chia tập dữ liệu diễn ra hoàn toàn trước các bước fit thống kê (từ điển, IDF, scaler, capping).
Mỗi tập dữ liệu được gắn cờ rõ ràng để các module sau chỉ nhận đúng tập dữ liệu tương ứng.

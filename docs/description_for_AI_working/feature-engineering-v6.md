Phiên bản v6. Trích xuất khối từ khóa nhạy cảm và chuẩn hóa đặc trưng số bằng MaxAbsScalerScratch
Phiên bản này trích xuất 10 cờ từ khóa spam nhạy cảm từ văn bản gốc, thu thập 8 đặc trưng cấu trúc số học và co giãn an toàn về đoạn [0, 1] bằng MaxAbsScaler tự viết nhằm bảo toàn tính thưa và tính không âm.

Bước 1. Trích xuất khối đặc trưng từ khóa nhạy cảm (10 chiều)
Sử dụng danh sách 10 từ khóa đặc trưng: free, win, prize, claim, urgent, call, cash, guarantee, offer, click.
Quét trực tiếp trên trường raw_text bằng regex có biên từ (\b) để tránh nhận nhầm từ con.
Tạo vector nhị phân 10 chiều (1.0 nếu từ khóa xuất hiện, 0.0 nếu không xuất hiện).
Đăng ký khối keyword vào dải chỉ số [4300, 4310).

Bước 2. Thu thập 8 đặc trưng cấu trúc & số học
Sử dụng SpamSignalFeatureExtractor để trích xuất 8 chỉ số:
1. has_url: Có chứa đường link (0 hoặc 1).
2. has_number: Có chứa chữ số (0 hoặc 1).
3. digit_count: Tổng số ký tự số.
4. special_char_count: Tổng số ký tự đặc biệt.
5. exclamation_count: Số dấu chấm than (!).
6. dollar_count: Số ký hiệu tiền tệ ($).
7. uppercase_ratio: Tỷ lệ ký tự hoa trên tổng số chữ cái.
8. model_length: Độ dài email sau khi cắt ngưỡng.

Bước 3. Tự cài đặt MaxAbsScalerScratch
Không sử dụng MinMaxScaler vì phép trừ giá trị nhỏ nhất (x - min) sẽ biến các phần tử giá trị 0 thành số khác không, làm mất cấu trúc thưa.
Cài đặt MaxAbsScalerScratch trong src/features/feature_engineering.py:
x_scaled = x / max(|x_train|).
Chỉ học vector cực đại max(|x_train|) trên tập Train (Anti-leakage).
Transform trên Train, Validation và Test đưa các giá trị về đoạn [0, 1].
Bảo toàn tính không âm bắt buộc để tương thích tuyệt đối với Complement Naive Bayes.
Đăng ký khối numeric vào dải chỉ số [4310, 4318).

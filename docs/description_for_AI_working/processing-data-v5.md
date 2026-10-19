Phiên bản v5. Cắt độ dài văn bản (Length Capping) chống Data Leakage và lưu trữ dữ liệu sạch
Mục tiêu của bản này là hoàn thiện khâu tiền xử lý bằng cách học ngưỡng cắt độ dài tối đa cho email chỉ trên tập Train để loại trừ email ngoại lai (outliers) mà không gây rò rỉ dữ liệu, sau đó xuất các file CSV sạch ra thư mục đệm.

Bước 1. Xác định sự cần thiết của việc cắt độ dài văn bản
Trong tập dữ liệu có một số email có độ dài bất thường (vượt quá vài nghìn từ do đính kèm log hoặc nội dung rác).
Các văn bản quá dài này làm lệch phân vị độ dài và làm phình kích thước từ điển vô ích.
Cần giới hạn độ dài từ vựng tối đa ở phân vị 99% (q = 0.99).

Bước 2. Học ngưỡng cắt độ dài chỉ trên tập Train (Anti-leakage Capping)
Đếm số từ của từng email trên 23.223 mẫu của tập Train.
Sắp xếp mảng độ dài tăng dần và tính giá trị tại phân vị 99%:
cap_word_limit = 1.361 từ.
Lưu giá trị cap_word_limit_ vào bộ xử lý SpamTextProcessor.
Quy tắc bắt buộc: Tập Validation và tập Test không được tính lại ngưỡng này mà phải dùng lại đúng ngưỡng 1.361 từ đã học từ Train để cắt ngắn văn bản.

Bước 3. Áp dụng cắt độ dài và hoàn tất trường model_text
Với mỗi email, nếu số từ vượt quá 1.361 từ, chỉ giữ lại đúng 1.361 từ đầu tiên.
Lưu văn bản sau cắt độ dài vào trường model_text và cập nhật lại biến model_length.
Đây là trường văn bản chuẩn mực cuối cùng được chuyển giao cho các bộ vector hóa TF-IDF ở giai đoạn Feature Engineering.

Bước 4. Xuất các file dữ liệu sạch ra đệm (Cache Persistence)
Sử dụng hàm export_processed_data() để xuất 3 file CSV vào thư mục data/processed/:
- train_clean.csv (chứa 23.223 dòng sạch).
- val_clean.csv (chứa 2.903 dòng sạch).
- test_clean.csv (chứa 2.902 dòng sạch).
Áp dụng cơ chế ghi qua file tạm atomic replace để tránh xung đột file handle trên Windows.
Quy trình Data Processing chính thức hoàn tất.

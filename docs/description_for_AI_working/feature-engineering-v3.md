Phiên bản v3. Trích xuất khối đặc trưng Word Unigram TF-IDF (3.000 chiều)
Mục tiêu của bản này là ứng dụng TfidfVectorizerScratch để trích xuất 3.000 đặc trưng từ đơn (unigram) quan trọng nhất từ văn bản email đã làm sạch, tạo thành khung xương sống cho ma trận đặc trưng.

Bước 1. Chuẩn bị văn bản đầu vào cho Word Unigram
Sử dụng trường văn bản model_text đã được làm sạch và cắt độ dài từ Giai đoạn 1.
Văn bản đã được loại bỏ HTML, chuyển chữ thường, loại dấu câu và lọc stop-words (giữ lại từ phủ định).

Bước 2. Cấu hình tham số cho Word Vectorizer
Khởi tạo TfidfVectorizerScratch với:
- min_df = 2 (từ phải xuất hiện trong ít nhất 2 email trên tập Train).
- max_features = 3.000 (giới hạn 3.000 từ vựng có tần suất cao nhất).
- token_pattern: Tách từ theo khoảng trắng sau khi đã làm sạch.

Bước 3. Fit và transform khối Word Unigram
Gọi phương thức fit trên 23.223 mẫu của tập Train.
Lưu từ điển 3.000 từ vào word_vectorizer.vocabulary_.
Transform tập Train thu được ma trận thưa kích thước (23223, 3000).
Transform tập Validation (2903, 3000) và tập Test (2902, 3000) bằng từ điển và IDF đã fit từ Train.
Đăng ký khối đặc trưng word vào dải chỉ số [0, 3000) trong danh mục đặc trưng tổng thể.

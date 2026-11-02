Phiên bản v4. Trích xuất khối đặc trưng Word Bigram TF-IDF (800 chiều)
Phiên bản này trích xuất 800 đặc trưng cặp từ liền kề (word bigrams) để bổ sung thông tin ngữ cảnh cục bộ mà khối từ đơn (unigram) không thể nắm bắt được.

Bước 1. Xác định sự cần thiết của Word Bigrams
Nhiều từ đơn lẻ xuất hiện phổ biến trong email hợp lệ (ví dụ: free, entry, call, now, cash, customer), nhưng khi ghép thành cặp từ (như free entry, call now, cash prize, urgent call), chúng mang tín hiệu cảnh báo spam rõ rệt.
Bigram giúp mô hình phân biệt được các cụm từ tiếp thị lừa đảo điển hình.

Bước 2. Cấu hình trích xuất Bigrams
Tạo chuỗi cặp từ liên tiếp từ danh sách từ vựng: (word_i, word_{i+1}) ghép nối bằng khoảng trắng.
Khởi tạo TfidfVectorizerScratch cho bigram với:
- min_df = 2.
- max_features = 800 (chọn 800 cặp từ có tần suất cao nhất trên tập Train).

Bước 3. Fit và transform khối Bigrams
Fit bigram vectorizer trên tập Train để xây dựng từ vựng 800 cặp từ và mảng IDF tương ứng.
Transform tập Train, Validation và Test sang ma trận thưa 800 cột.
Đăng ký khối đặc trưng bigram vào dải chỉ số [3000, 3800) trong danh mục đặc trưng tổng thể.

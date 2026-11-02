Phiên bản v7. Ghép nối 5 khối đặc trưng lai (Hybrid Feature Matrix 4.318 chiều)
Phiên bản này tích hợp toàn bộ 5 khối đặc trưng độc lập thành một ma trận đặc trưng lai duy nhất bằng HybridFeatureBuilderScratch, bảo đảm tính nhất quán số chiều và quản lý thứ tự cột chặt chẽ.

Bước 1. Xác định kiến trúc ma trận đặc trưng lai
Ma trận đặc trưng cuối cùng là sự kết hợp theo chiều ngang (horizontal stack) của 5 khối:
1. Word Unigram TF-IDF: 3.000 chiều.
2. Word Bigram TF-IDF: 800 chiều.
3. Character n-gram TF-IDF: 500 chiều.
4. Keyword Binary Signals: 10 chiều.
5. Numeric Structural Scaled: 8 chiều.
Tổng số chiều của ma trận: D = 3000 + 800 + 500 + 10 + 8 = 4.318 đặc trưng.

Bước 2. Cài đặt HybridFeatureBuilderScratch
Triển khai lớp HybridFeatureBuilderScratch trong src/features/feature_engineering.py.
Phương thức fit(train_df):
- Học ngưỡng cắt độ dài trên Train (1.361 từ).
- Fit word_vectorizer trên văn bản Train.
- Fit bigram_vectorizer trên các cặp từ Train.
- Fit char_vectorizer trên các chuỗi con ký tự Train.
- Fit numeric_scaler trên các đặc trưng số của Train.
- Khóa danh sách tên đặc trưng feature_names_ và từ điển vị trí feature_slices_.

Bước 3. Phương thức transform và ghép nối thưa
Với bất kỳ DataFrame đầu vào nào (Train, Val, Test hoặc email mới khi inference):
- Biến đổi văn bản qua từng vectorizer tương ứng.
- Trích xuất từ khóa nhạy cảm và nhân trọng số nhóm keyword_group_weight.
- Trích xuất đặc trưng số và co giãn qua numeric_scaler đã fit.
- Gọi CSRMatrixScratch.hstack() để ghép ngang 5 ma trận thưa thành 1 ma trận duy nhất kích thước (N, 4318).
- Bảo đảm tính thưa và tốc độ thực thi nhanh, không bao giờ giải nén thành mảng đặc.

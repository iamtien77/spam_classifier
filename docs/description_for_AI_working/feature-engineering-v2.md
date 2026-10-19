Phiên bản v2. Tự cài đặt thuật toán TF-IDF từ đầu (TfidfVectorizerScratch)
Mục tiêu của bản này là tự triển khai bộ vector hóa Term Frequency - Inverse Document Frequency (TF-IDF) từ đầu bằng Python thuần, áp dụng kỹ thuật sublinear TF và smooth IDF nhằm giảm độ thiên lệch của các từ lặp lại nhiều lần.

Bước 1. Xác định công thức toán học cho TF-IDF tự viết
Sử dụng Sublinear Term Frequency:
TF(t, d) = 1 + ln(c(t, d)) nếu c(t, d) > 0, ngược lại bằng 0 (với c(t, d) là số lần từ t xuất hiện trong văn bản d).
Sử dụng Smooth Inverse Document Frequency:
IDF(t) = ln((1 + N) / (1 + df(t))) + 1, với N là tổng số văn bản trên tập Train và df(t) là số văn bản Train chứa từ t.
Chuẩn hóa L2 vector đặc trưng: x(t, d) = v(t, d) / sqrt(sum(v^2)).

Bước 2. Cài đặt phương thức fit (chỉ học trên Train set)
Duyệt qua toàn bộ văn bản của tập Train (23.223 mẫu).
Đếm document frequency df(t) cho từng token.
Lọc bỏ các token có df(t) < min_df (cố định min_df = 2 để loại bỏ từ vựng hiếm hoặc lỗi chính tả).
Sắp xếp và chọn ra top max_features token có tần suất cao nhất để xây dựng từ điển vocabulary_.
Tính toán và lưu mảng idf_diag_ chứa giá trị IDF của từng từ trong từ điển.

Bước 3. Cài đặt phương thức transform (áp dụng cho Train, Validation, Test)
Với mỗi văn bản mới, đếm số lần xuất hiện của các từ nằm trong vocabulary_.
Bỏ qua hoàn toàn các từ không nằm trong từ điển (out-of-vocabulary).
Áp dụng công thức Sublinear TF và nhân với trọng số IDF tương ứng.
Chuẩn hóa vector theo chuẩn Euclidean L2. Nếu văn bản không chứa từ nào trong từ điển, giữ nguyên vector toàn số 0.
Xuất kết quả dưới dạng ma trận thưa CSRMatrixScratch.

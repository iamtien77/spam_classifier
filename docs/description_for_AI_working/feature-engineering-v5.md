Phiên bản v5. Trích xuất khối đặc trưng Character n-gram TF-IDF (500 chiều)
Phiên bản này trích xuất 500 đặc trưng chuỗi con ký tự (character n-grams) độ dài từ 3 đến 5 ký tự nhằm chống lại các thủ thuật ngụy trang gõ sai chữ (obfuscation) thường dùng trong thư rác.

Bước 1. Xác định sự cần thiết của Character n-grams
Kẻ gửi thư rác thường cố tình chèn ký tự lạ hoặc gõ sai chính tả để đánh lừa bộ lọc từ vựng (ví dụ: w!n, c@sh, fr3e, v1agra).
Khối word TF-IDF sẽ xem các từ này là từ mới (out-of-vocabulary) và bỏ qua.
Character n-grams trích xuất các chuỗi con ký tự độ dài 3-5, giúp nhận diện được phần gốc của từ bất kể các biến thể gõ sai.

Bước 2. Cấu hình trích xuất Character n-grams
Trượt cửa sổ ký tự qua từng từ trong văn bản để sinh các chuỗi con độ dài 3, 4 và 5 ký tự.
Khởi tạo TfidfVectorizerScratch cho character n-gram với:
- min_df = 2.
- max_features = 500 (chọn 500 chuỗi con ký tự phổ biến nhất trên tập Train).

Bước 3. Fit và transform khối Character n-grams
Fit character vectorizer trên tập Train để học từ vựng 500 n-gram ký tự và IDF.
Transform tập Train, Validation và Test sang ma trận thưa 500 cột.
Đăng ký khối đặc trưng char vào dải chỉ số [3800, 4300) trong danh mục đặc trưng tổng thể.

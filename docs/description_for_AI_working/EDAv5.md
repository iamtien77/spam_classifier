Bản v5. Phân tích tần suất từ vựng, từ khóa nhạy cảm và n-gram
Mục tiêu của bản này là khảo sát không gian từ vựng, xác định các từ khóa spam nhạy cảm xuất hiện dày đặc và phân tích các cụm từ (bigrams) mang tính phân loại cao.

Bước 1. Xác định mục tiêu nâng cấp so với EDA v4
Phân tích sâu vào nội dung từ vựng ngôn ngữ tự nhiên thay vì chỉ xét các biến số cấu trúc.
Lọc bỏ stop-words tạm thời để tìm ra top từ khóa có tần suất xuất hiện cao nhất trong từng lớp.

Bước 2. Trích xuất top từ vựng phổ biến nhất trong lớp Spam
Tách từ văn bản sau khi chuyển chữ thường và loại bỏ dấu câu.
Lọc các stop-words tiếng Anh thông dụng (như the, is, at, which, on).
Đếm tần suất xuất hiện và trích xuất top 30 từ vựng phổ biến nhất của lớp Spam: free, call, claim, prize, win, cash, urgent, mobile, text, guarantee, service.
So sánh với top từ vựng của lớp Ham (thường là: get, come, know, like, time, work, ok, good).

Bước 3. Định hình danh sách từ khóa nhạy cảm (Spam Keywords)
Xác định danh mục 10 từ khóa nhạy cảm có độ phân biệt mạnh nhất: free, win, prize, claim, urgent, call, cash, guarantee, offer, click.
Kiểm tra tỷ lệ xuất hiện của ít nhất một trong các từ khóa này trong lớp Spam so với lớp Ham.
Xác nhận rằng sự hiện diện của các từ này là chỉ báo phân loại rõ rệt.

Bước 4. Phân tích các cặp từ liên tiếp (Bigrams)
Tạo các cặp từ liền kề (word bigrams) từ nội dung email.
Đếm tần suất bigrams trong lớp Spam: free entry, claim prize, call now, cash prize, urgent call, customer service.
Nhận định rằng các từ đơn lẻ như "free" hoặc "entry" có thể xuất hiện trong email bình thường, nhưng cụm từ "free entry" hầu như chỉ xuất hiện trong thư rác quảng cáo.
Kết luận: Cần bổ sung khối đặc trưng Bigram vào mô hình để nắm bắt ngữ cảnh kết hợp từ.

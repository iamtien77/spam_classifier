Phiên bản v2. Xử lý và chuẩn hóa văn bản chuyên sâu cho bài toán Spam
Phiên bản này nâng cấp quá trình làm sạch văn bản, loại bỏ mã HTML, thế chỗ các tín hiệu nhạy cảm bằng token chuyên biệt và lọc stop-words nhưng bảo toàn các từ phủ định.

Bước 1. Nội dung bổ sung so với v1
Chuyển đổi văn bản tự do thành dạng văn bản mô hình (model_text) được chuẩn hóa.
Bảo toàn các tín hiệu có giá trị phân biệt cao (như URL, tiền tệ, số điện thoại) dưới dạng token đại diện.
Tránh làm mất các từ phủ định quan trọng khi loại bỏ stop-words.

Bước 2. Bóc tách thẻ HTML và giải mã ký tự đặc biệt
Sử dụng biểu thức chính quy để loại bỏ toàn bộ các thẻ HTML tags (ví dụ <p>, <br>, <a>, <div>).
Giải mã các HTML entities thông dụng (&amp;, &lt;, &gt;, &quot;) về dạng ký tự tương ứng.
Đảm bảo văn bản không còn dính các đoạn mã HTML rác từ các email tiếp thị.

Bước 3. Thế chỗ tín hiệu bằng Token chuyên biệt
Thay thế tất cả các đường dẫn URL (bắt đầu bằng http://, https:// hoặc www.) bằng token urltoken.
Thay thế các giá trị tiền tệ, ký hiệu đô la, bảng Anh, euro bằng token moneytoken.
Thay thế các dãy số, số điện thoại, mã pin bằng token numbertoken.
Cách làm này giúp mô hình không phải học từng đường link hay số điện thoại cụ thể mà vẫn nắm bắt được dấu hiệu cấu trúc quan trọng.

Bước 4. Chuẩn hóa chữ thường và lọc stop-words có kiểm soát
Chuyển toàn bộ văn bản về chữ thường (lowercase) và chuẩn hóa Unicode NFKC.
Loại bỏ các ký tự dấu câu khỏi văn bản dùng cho mô hình.
Sử dụng danh sách stop-words tiếng Anh cố định (không phụ thuộc thư viện ngoài).
Quy tắc đặc thù: Tuyệt đối giữ lại các từ phủ định như not, no, never vì chúng có thể đảo ngược ý nghĩa câu.
Lưu văn bản sau xử lý vào trường clean_text.

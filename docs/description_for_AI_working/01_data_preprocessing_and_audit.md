# Bản Mô Tả Kỹ Thuật Giai Đoạn 1: Tiền Xử Lý Dữ Liệu & Kiểm Toán Chất Lượng (Data Preprocessing & Audit)

Mục tiêu của tài liệu này là quy định chi tiết từng bước thực thi cho Giai đoạn 1 của hệ thống Spam Classifier: từ việc nạp file CSV dữ liệu thô, kiểm toán chất lượng dữ liệu đa tầng không chồng lặp, khử trùng lặp và mâu thuẫn nhãn, phân chia tập dữ liệu phân tầng (Stratified Split) đến chuẩn hóa văn bản và học ngưỡng cắt độ dài chống rò rỉ thông tin (Data Leakage).

---

### Bước 1. Xác định vị trí dữ liệu thô và cấu hình đường dẫn
- Xác định đường dẫn file dữ liệu đầu vào mặc định tại `data/raw/spam.csv`.
- Đảm bảo cấu hình đường dẫn được quản lý tập trung thông qua [`src/config.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/config.py) (`RAW_DATA_PATH`, `PROCESSED_DATA_DIR`).
- Kiểm tra tính tồn tại của file trước khi đọc. Nếu file không tồn tại, hàm nạp dữ liệu phải ném ra ngoại lệ `FileNotFoundError` kèm thông báo rõ ràng, không được tiếp tục xử lý âm thầm.

### Bước 2. Nạp dữ liệu thô và xử lý mã hóa linh hoạt (Encoding Fallback)
- Sử dụng hàm `load_raw_data()` trong [`src/data/loader.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/data/loader.py) để đọc file CSV.
- Áp dụng cơ chế giải mã an toàn: ưu tiên đọc bằng bảng mã `utf-8-sig` (để tự động xử lý ký tự Byte Order Mark - BOM nếu có).
- Bắt ngoại lệ `UnicodeDecodeError`, nếu thất bại thì lập tức chuyển sang chế độ dự phòng bằng bảng mã `latin-1`.
- Đảm bảo quá trình nạp không bị gián đoạn do lỗi định dạng byte lạ từ các tập dữ liệu email cổ điển.

### Bước 3. Chuẩn hóa Schema và định danh mẫu dữ liệu
- Quét và nhận diện các biến thể tên cột phổ biến trong các bộ dữ liệu Spam:
  - Cột nhãn: nhận diện một trong các tên `label`, `category`, `v1`.
  - Cột nội dung văn bản: nhận diện một trong các tên `text`, `message`, `v2`, `body`.
- Báo lỗi `ValueError` nếu CSV thiếu cột nhãn hoặc cột nội dung bắt buộc, hoặc khi số cột sau chuẩn hóa bị trùng lặp.
- Đưa DataFrame về cấu trúc schema thống nhất gồm 4 trường chuẩn:
  1. `source_id`: Mã định danh nguyên thủy dạng số nguyên (từ 0 đến $N-1$) để theo dõi dấu vết của từng bản ghi qua các bước tiền xử lý.
  2. `label`: Nhãn chuỗi gốc (`ham` hoặc `spam`).
  3. `text`: Nội dung văn bản sẽ được dùng để xử lý mô hình.
  4. `raw_text`: Bản sao bất biến của nội dung gốc, dùng để đếm chữ hoa, dấu câu và các tín hiệu cấu trúc chưa qua làm sạch.
- Kiểm tra số lượng dòng nạp được ở bước thô ban đầu (kết quả thực tế: **43.849 dòng**).

### Bước 4. Kiểm toán chất lượng dữ liệu 5 bước không chồng lặp (Data Quality Audit)
- Triển khai hàm `audit_data_quality()` trong [`src/data/preprocessing.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/data/preprocessing.py).
- Quy định việc kiểm toán theo thứ tự ưu tiên tuần tự và loại trừ lẫn nhau (Mutually Exclusive), đảm bảo: $\text{Dữ liệu giữ lại} + \sum \text{Dữ liệu bị loại} = \text{Dữ liệu thô ban đầu}$.
- Thực hiện 5 bước kiểm toán định lượng:
  1. **Kiểm tra nhãn không hợp lệ:** Loại bỏ các dòng có nhãn là `NaN`, chuỗi rỗng hoặc giá trị không thuộc tập `{'ham', 'spam'}` (thực tế loại **2.195 dòng**).
  2. **Kiểm tra nội dung văn bản rỗng:** Đối với các dòng có nhãn hợp lệ, kiểm tra xem nội dung văn bản sau khi loại bỏ khoảng trắng thừa có rỗng hay không; loại bỏ nếu rỗng (thực tế loại **2.192 dòng**).
  3. **Kiểm tra mâu thuẫn nhãn (Conflict Resolution):** Chuẩn hóa khóa văn bản bằng Unicode NFKC và co cụm khoảng trắng (giữ nguyên hoa/thường và dấu câu). Nếu cùng một nội dung văn bản nhưng xuất hiện ở cả hai nhãn `ham` và `spam`, loại bỏ toàn bộ các nhóm này để tránh làm mô hình học nhãn mâu thuẫn (thực tế phát hiện 738 nhóm mâu thuẫn, loại toàn bộ **1.480 dòng**).
  4. **Khử bản ghi trùng lặp hoàn toàn (Deduplication):** Với các nhóm có cùng nội dung chuẩn hóa và cùng nhãn, giữ lại bản ghi đầu tiên xuất hiện theo `source_id` và loại bỏ các bản sao thừa (thực tế loại **8.954 dòng**).
  5. **Xác nhận tập dữ liệu sạch:** Tính toán và khóa số lượng mẫu sạch cuối cùng sau kiểm toán (thực tế giữ lại **29.028 mẫu**).
- Xuất toàn bộ báo cáo kiểm toán chi tiết ra file JSON tại [`results/metrics/data_audit.json`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/metrics/data_audit.json).

### Bước 5. Phân tích phân bố nhãn sau làm sạch
- Kiểm tra số lượng và tỷ lệ của từng lớp trong $29.028$ mẫu sạch:
  - Lớp `ham`: **16.636 mẫu** (chiếm $\approx 57,31\%$).
  - Lớp `spam`: **12.392 mẫu** (chiếm $\approx 42,69\%$).
- Ghi nhận tỷ lệ mất cân bằng ở mức nhẹ ($\approx 1.34 : 1$). Cấu hình mô hình huấn luyện sau này cần bật `class_weight='balanced'` để bù trừ công bằng.

### Bước 6. Phân chia tập dữ liệu phân tầng an toàn (Stratified Train / Val / Test Split)
- Sử dụng hàm `stratified_split_dataframe()` trong [`src/data/preprocessing.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/data/preprocessing.py).
- Cố định seed ngẫu nhiên (`random_state=42`) để đảm bảo tính tái lập 100% của thực nghiệm.
- Áp dụng tỷ lệ cấu hình: **80% Train, 10% Validation, 10% Test**.
- Sử dụng thuật toán làm tròn phần dư lớn nhất (Largest Remainder Method) trên từng phân lớp để đảm bảo tổng số lượng mẫu sau khi chia bằng đúng $29.028$ mẫu, không bị mất hoặc lệch mẫu do làm tròn số học:
  - **Tập Train (80%):** **23.223 mẫu** ($13.309$ Ham, $9.914$ Spam).
  - **Tập Validation (10%):** **2.903 mẫu** ($1.664$ Ham, $1.239$ Spam).
  - **Tập Test (10%):** **2.902 mẫu** ($1.663$ Ham, $1.239$ Spam).
- Tuyệt đối thực hiện phân chia tập dữ liệu trước mọi bước fit thống kê nhằm ngăn chặn triệt để hiện tượng rò rỉ dữ liệu (Data Leakage).

### Bước 7. Chuẩn hóa và làm sạch văn bản mô hình (SpamTextProcessor)
- Triển khai lớp `SpamTextProcessor` trong [`src/data/preprocessing.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/data/preprocessing.py).
- Thực hiện quy trình chuẩn hóa văn bản qua 5 công đoạn tuần tự:
  1. **Bóc tách thẻ HTML:** Dùng regex nhận diện và loại bỏ toàn bộ các thẻ HTML tags (ví dụ `<p>`, `<a>`, `<br>`), đồng thời giải mã các HTML entities (như `&amp;`, `&lt;`, `&gt;`).
  2. **Thế chỗ các tín hiệu cấu trúc nhạy cảm bằng Token chuyên biệt:**
     - Đường dẫn liên kết (URL bắt đầu bằng `http://`, `https://`, `www.`): thay bằng token `urltoken`.
     - Ký hiệu tiền tệ và giá tiền (`$`, `£`, `€`, `free cash`): thay bằng token `moneytoken`.
     - Dãy số, số điện thoại, mã pin: thay bằng token `numbertoken`.
  3. **Chuẩn hóa chữ thường & Unicode:** Chuyển toàn bộ văn bản về chữ thường (lowercase), chuẩn hóa Unicode về dạng NFKC.
  4. **Lọc dấu câu & Stop-words có kiểm soát:**
     - Loại bỏ các ký tự dấu câu khỏi văn bản dùng cho TF-IDF.
     - Lọc bỏ danh sách stop-words tiếng Anh cố định (không phụ thuộc thư viện ngoài NLTK).
     - **Quy tắc đặc thù bài toán Spam:** Tuyệt đối giữ lại các từ mang ý nghĩa phủ định (như `not`, `no`, `never`) vì chúng đảo ngược ngữ nghĩa câu.
  5. **Tính toán số lượng từ (word_count) và độ dài ký tự:** Lưu lại các trường thống kê `model_length` và `raw_length`.

### Bước 8. Học ngưỡng cắt độ dài văn bản (Length Capping) chống Data Leakage
- Kiểm tra phân bố độ dài từ vựng của email để loại trừ ảnh hưởng tiêu cực của các email ngoại lai (outliers) quá dài làm phình ma trận TF-IDF.
- Sử dụng tham số `cap_quantile = 0.99` (phân vị 99%).
- **Nguyên tắc chống rò rỉ dữ liệu bắt buộc:** Ngưỡng cắt độ dài `cap_word_limit_` chỉ được học duy nhất trên tập **Train**:

  $$
  \text{cap\_word\_limit} = \text{Quantile}_{0.99}(\text{Train lengths}) = \mathbf{1.361\text{ từ}}
  $$

- Tập Validation và tập Test không được tính lại ngưỡng này mà phải dùng lại đúng ngưỡng $1.361$ từ đã fit từ Train để cắt ngắn văn bản.

### Bước 9. Lưu trữ dữ liệu sạch ra đệm (Cache Persistence)
- Sử dụng hàm `export_processed_data()` trong [`src/features/feature_engineering.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/features/feature_engineering.py).
- Ghi dữ liệu sạch của 3 tập ra thư mục `data/processed/`:
  - `train_clean.csv`
  - `val_clean.csv`
  - `test_clean.csv`
- Đảm bảo cơ chế ghi an toàn bằng file tạm (Atomic replace) để chống xung đột file handle trên hệ điều hành Windows.
- Giai đoạn 1 kết thúc thành công khi 3 tập dữ liệu sạch sẵn sàng làm đầu vào cho bộ trích xuất đặc trưng lai ở Giai đoạn 2.

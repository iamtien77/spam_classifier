# Đặc Tả Kỹ Thuật Giai Đoạn 1: Tiền Xử Lý Dữ Liệu, Kiểm Toán Chất Lượng & Phân Tầng Chống Rò Rỉ (Data Preprocessing & Audit)

> **Vị trí trong Pipeline:** Giai đoạn 1 / 4 (`step_data_preprocessing`)  
> **Module nguồn:** [`src/data/loader.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/data/loader.py), [`src/data/preprocessing.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/data/preprocessing.py), [`src/config.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/config.py)  
> **Báo cáo thực nghiệm liên kết:** [`GIAI_THICH_CHI_TIET_TASK_1_DATA_PREPROCESSING.docx`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/GIAI_THICH_CHI_TIET_TASK_1_DATA_PREPROCESSING.docx), [`data_audit.json`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/metrics/data_audit.json)

---

## 1. Bối Cảnh Nghiệp Vụ & Hợp Đồng Dữ Liệu (Data Contract)

Trong các hệ thống phòng thủ an ninh thư điện tử doanh nghiệp, tầng tiền xử lý dữ liệu (Data Preprocessing Layer) đóng vai trò là "chốt chặn đầu vào" quyết định trực tiếp tới độ tin cậy của toàn bộ hệ thống Machine Learning. Dữ liệu văn bản thô từ gateway thường bị ô nhiễm bởi các cuộc tấn công HTML obfuscation, các biến thể mã hóa ký tự quốc tế, cũng như các bản ghi trùng lặp và xung đột nhãn do botnet phát tán.

```text
[spam.csv Thô: 43.849 dòng]
         │
         ▼ (resolve_data_path & load_raw_data: Fail-Fast & Fallback utf-8-sig / latin-1)
[DataFrame Chuẩn hóa Schema: source_id, label, text, raw_text]
         │
         ▼ (audit_data_quality & clean_email_split: Khử trùng & Xóa nhóm mâu thuẫn)
[Dữ liệu Sạch Toàn Cục: 29.028 dòng (16.636 Ham / 12.392 Spam)]
         │
         ▼ (stratified_split_dataframe: Thuật toán Hamilton Largest Remainder seed=42)
  ┌───────────────────────┬───────────────────────┬───────────────────────┐
  ▼ (80% Train)           ▼ (10% Validation)      ▼ (10% Test Độc lập)
[Train: 23.223 mẫu]     [Val: 2.903 mẫu]        [Test: 2.902 mẫu]
  (Học cap_word_limit)    (Áp dụng ngưỡng Train)  (Đóng băng tuyệt đối)
```

### Hợp đồng Schema Dữ liệu Đầu Vào & Đầu Ra
| Tên trường | Kiểu dữ liệu | Ràng buộc nghiệp vụ & Bản chất kỹ thuật |
| :--- | :--- | :--- |
| `source_id` | `int64` | Chỉ số định danh bất biến nguyên thủy ($0 \dots N-1$), làm mỏ neo truy vết vết lỗi xuyên suốt pipeline. |
| `label` | `str` $\in \{\text{'ham'}, \text{'spam'}\}$ | Nhãn phân loại nhị phân. Tuyệt đối không chấp nhận giá trị rỗng hoặc nhãn lạ. |
| `text` | `str` | Nội dung email đã qua bóc tách HTML, thay thế token và chuẩn hóa Unicode NFKC. |
| `raw_text` | `str` | Bản sao nguyên vẹn của văn bản thô ban đầu, dùng để trích xuất tín hiệu cấu trúc (chữ hoa, dấu `!`, `$`). |

---

## 2. Bóc Tách Chi Tiết Logic Chức Năng Nghiệp Vụ Từng Lớp & Từng Hàm

### 2.1. Module `src/data/loader.py`

#### Hàm `resolve_data_path(custom_path: Optional[Path] = None) -> Path`
- **Mục đích:** Phân giải và kiểm định tính hợp lệ của đường dẫn dữ liệu với cơ chế **Fail-Fast**.
- **Logic thực thi:**
  1. Nếu người dùng truyền `custom_path`: Chuyển đổi sang `Path(custom_path)` và gọi `.is_file()`. Nếu không tìm thấy tệp, lập tức raise `FileNotFoundError` chỉ rõ đường dẫn lỗi. Tuyệt đối không tự ý fallback ngầm sang đường dẫn mặc định vì sẽ dẫn tới việc huấn luyện nhầm tập dữ liệu cũ mà kỹ sư không hề hay biết.
  2. Nếu `custom_path` là `None`: Duyệt danh sách 4 vị trí tiêu chuẩn (`root / 'data/raw/spam.csv'`, `root / 'data/spam.csv'`, `Path('data/raw/spam.csv')`, `Path('../data/raw/spam.csv')`). Trả về đường dẫn đầu tiên tồn tại.

#### Hàm `load_raw_data(data_path: Optional[Path] = None, as_dataframe: bool = True)`
- **Mục đích:** Đọc tệp CSV thô, bảo vệ bộ nhớ trước các cuộc tấn công CSV Field Bomb và giải mã linh hoạt.
- **Logic thực thi:**
  1. *Kiểm soát giới hạn trường an toàn:* Lưu giá trị `previous_limit = csv.field_size_limit()`. Nâng giới hạn đọc trường lên `min(path.stat().st_size, 2**31 - 1)`. Chặn trần ở $2^{31}-1$ (giá trị cực đại của số nguyên C long 32-bit có dấu) nhằm ngăn chặn hoàn toàn lỗi `OverflowError: Python int too large to convert to C long` trên hệ điều hành Windows 64-bit/32-bit.
  2. *Giải mã chịu lỗi đa tầng (Encoding Fallback):* Thử mở tệp bằng bảng mã `utf-8-sig` (tự động loại bỏ ký tự Byte Order Mark - BOM nếu tệp xuất từ Excel). Nếu bắt ngoại lệ `UnicodeDecodeError`, lập tức chuyển sang `latin-1` (bảng mã ánh xạ 1-1 toàn bộ 256 byte nhị phân, bảo đảm 100% không văng lỗi giải mã).
  3. *Chuẩn hóa schema:* Tự động nhận diện và ánh xạ các biến thể cột: nhãn (`label`, `category`, `v1`), nội dung (`text`, `message`, `v2`, `body`). Bắt lỗi `ValueError` nếu thiếu cột hoặc trùng lặp tên cột sau chuẩn hóa.
  4. *Khôi phục an toàn trong `finally`:* Luôn gọi `csv.field_size_limit(previous_limit)` trong khối `finally` để triệt tiêu mọi tác dụng phụ (side-effects) làm biến đổi trạng thái toàn cục của tiến trình Python.

---

### 2.2. Module `src/data/preprocessing.py`

#### Lớp `_EmailHTMLTextParser(HTMLParser)` & Hàm `_extract_html_text(text: str) -> str`
- **Mục đích:** Bóc tách an toàn nội dung văn bản từ email HTML mà không bị bẫy bởi Catastrophic Backtracking của biểu thức chính quy.
- **Logic thực thi:**
  - Kế thừa lớp `HTMLParser` dòng sự kiện của Python với `convert_charrefs=True`, tự động giải mã các thực thể HTML (`&amp;` $\rightarrow$ `&`, `&pound;` $\rightarrow$ `£`, `&quot;` $\rightarrow$ `"`).
  - Khởi tạo frozenset `HTML_TAGS` chứa 65 thẻ HTML tiêu chuẩn.
  - Khi gặp thẻ bắt đầu (`handle_starttag`) hoặc thẻ đóng (`handle_endtag`): nếu thẻ nằm trong `HTML_TAGS`, chèn một khoảng trắng `' '` để tránh việc các từ ở 2 thẻ dính liền nhau (ví dụ: `<td>Hello</td><td>World</td>` $\rightarrow$ `Hello World`).
  - Nếu thẻ lạ không thuộc chuẩn HTML (ví dụ cú pháp biểu cảm `<3` hoặc đoạn code `<variable>`), phương thức giữ nguyên chuỗi văn bản gốc qua `self.get_starttag_text()` để không làm mất thông tin phân loại.

#### Hàm `audit_data_quality(df, text_col='text', label_col='label') -> dict[str, int]`
- **Mục đích:** Kiểm toán định lượng chất lượng dữ liệu theo thứ tự ưu tiên loại trừ lẫn nhau (Mutually Exclusive).
- **Quy tắc 5 nhóm lỗi:**
  $$\text{Dữ liệu đầu vào} = \text{Nhãn sai} + \text{Text rỗng} + \text{Mâu thuẫn nhãn} + \text{Bản lặp cùng nhãn} + \text{Mẫu sạch}$$
  1. *Invalid Label (Nhãn không hợp lệ):* Nhãn là `NaN` hoặc không thuộc tập `{'ham', 'spam'}` (thực tế: **2.195 dòng**).
  2. *Empty Text (Nội dung rỗng):* Nhãn hợp lệ nhưng nội dung sau khi trim khoảng trắng bị rỗng (thực tế: **2.192 dòng**).
  3. *Conflicting Groups (Mâu thuẫn nhãn):* Các email có cùng khóa nội dung `_content_key` nhưng vừa bị gán nhãn `ham` vừa bị gán nhãn `spam` (thực tế: 738 nhóm, loại toàn bộ **1.480 dòng**).
  4. *Duplicate Rows (Bản lặp cùng nhãn):* Các email có cùng khóa nội dung và cùng nhãn, giữ bản ghi đầu tiên theo `source_id` và loại bỏ các bản sao (thực tế: loại **8.954 dòng**).
  5. *Retained Clean Rows (Mẫu sạch giữ lại):* **29.028 dòng** sạch tuyệt đối ($16.636$ Ham, $12.392$ Spam).

#### Hàm `clean_email_split(df, ...) -> pd.DataFrame`
- **Mục đích:** Thực thi làm sạch dữ liệu toàn cục trước khi chia tập.
- **Nguyên tắc an ninh bắt buộc:** Xóa bỏ TOÀN BỘ các dòng thuộc nhóm mâu thuẫn nhãn. Tuyệt đối không dùng biểu quyết đa số (Majority Voting) vì kẻ tấn công có thể dùng botnet gửi hàng loạt email rác trùng lặp để thao túng nhãn biểu quyết của nhóm.

#### Hàm `stratified_split_dataframe(...)` qua Thuật toán Hamilton Largest Remainder
- **Mục đích:** Phân chia tập dữ liệu ngẫu nhiên có kiểm soát theo tỷ lệ cấu hình: **80% Train, 10% Validation, 10% Test** với seed cố định `random_state=42`.
- **Bản chất toán học của giải thuật Hamilton:**
  $$N_c = \text{số mẫu của lớp } c, \quad \text{exact}_i = N_c \times r_i, \quad \text{size}_i = \lfloor \text{exact}_i \rfloor, \quad \text{rem}_i = \text{exact}_i - \text{size}_i$$
  Thuật toán phân bổ phần nguyên $\text{size}_i$ trước, sau đó sắp xếp các tập theo phần dư $\text{rem}_i$ giảm dần và cộng thêm 1 đơn vị cho các tập có phần dư lớn nhất cho đến khi $\sum \text{size}_i = N_c$.
  - **Kết quả phân bổ chính xác từng mẫu:**
    - Tập Train (80%): **23.223 mẫu** ($13.309$ Ham, $9.914$ Spam).
    - Tập Validation (10%): **2.903 mẫu** ($1.664$ Ham, $1.239$ Spam).
    - Tập Test (10%): **2.902 mẫu** ($1.663$ Ham, $1.239$ Spam).
    - Tổng cộng: Đúng bằng $29.028$ mẫu, không lệch một mẫu nào do sai số làm tròn số học.

#### Lớp `SpamTextProcessor`
- **Mục đích:** Xử lý văn bản phi trạng thái (Stateless Text Processor) gồm 6 phương thức nghiệp vụ:
  1. `normalize_text(text)`: Thực thi chuỗi 7 bước chuẩn hóa (HTML $\rightarrow$ token URL `urltoken` $\rightarrow$ token tiền tệ `moneytoken` $\rightarrow$ token số `numbertoken` $\rightarrow$ lowercase/NFKC $\rightarrow$ loại bỏ ký tự lạ $\rightarrow$ lọc stop words tiếng Anh).
  2. `tokenize_words(text)`: Tách chuỗi theo khoảng trắng thành danh sách từ đơn `list[str]`.
  3. `extract_word_ngrams(tokens, n=2)`: Trượt cửa sổ thu thập các cặp từ liền kề `[' '.join(tokens[i:i+n])]`, bảo toàn cụm từ ngữ cảnh như `free entry`, `call now`.
  4. `extract_char_ngrams(text, min_n=3, max_n=5)`: Đệm khoảng trắng đầu cuối `' ' + text + ' '` và trích xuất chuỗi con ký tự từ 3 đến 5 ký tự. Đây là cơ chế cốt lõi để nhận diện các biến thể cố tình gõ sai chữ (obfuscation) như `w!n`, `fr3e`, `c@sh`.
  5. `extract_structural_features(raw_text)`: Trích xuất 8 đặc trưng cấu trúc số học từ văn bản gốc (tỷ lệ chữ hoa `uppercase_ratio`, số dấu chấm than `!`, số ký hiệu tiền tệ `$`, số lượng URL, số lượng chữ số, tổng độ dài ký tự và số từ).
  6. `prepare_feature_frame(df, cap_word_limit)`: Đóng gói bảng dữ liệu và áp dụng ngưỡng cắt độ dài văn bản `cap_word_limit` (học từ phân vị 99% của tập Train: **$1.361$ từ**) để loại bỏ các email ngoại lai quá dài gây phình ma trận TF-IDF.

---

## 3. Phân Tích Đối Kháng Giải Pháp Kỹ Thuật (Design Trade-offs Matrix)

### Bảng 1: Phân giải đường dẫn dữ liệu
| Tiêu chuẩn đánh giá | `resolve_data_path` (Giải pháp chọn) | Hardcode đường dẫn tuyệt đối | Silent Fallback ngầm định |
| :--- | :--- | :--- | :--- |
| **Tính linh hoạt đa nền tảng** | **Rất cao:** Hoạt động trên Windows, Linux, Docker, Google Colab. | **Kém:** Vỡ ngay khi chuyển máy tính hoặc môi trường CI/CD. | Trung bình. |
| **Phát hiện lỗi cấu hình (Fail-Fast)** | **Tức thì:** Ném lỗi chỉ rõ đường dẫn bị sai, dừng hệ thống ngay. | Ném lỗi crash không có gợi ý. | **Cực kỳ nguy hiểm:** Âm thầm chạy tiếp trên dữ liệu mặc định cũ. |
| **Bảo vệ toàn vẹn dữ liệu** | **Tuyệt đối:** Đảm bảo 100% mô hình học đúng tệp dữ liệu được chỉ định. | Kém. | Nguy cơ nhiễm bẩn dữ liệu thực nghiệm nghiêm trọng. |

### Bảng 2: Bóc tách nội dung HTML
| Tiêu chuẩn đánh giá | `HTMLParser` kế thừa (Giải pháp chọn) | Regex Strip (`re.sub(r'<[^>]+>', ' ')`) | Thư viện BeautifulSoup4 |
| :--- | :--- | :--- | :--- |
| **Khả năng chịu lỗi HTML lỗi** | **Xuất sắc:** Xử lý tốt thẻ không đóng, thuộc tính CSS ẩn. | **Kém:** Dễ bị bẫy bởi dấu `<` trong văn bản thường hoặc biểu cảm `<3`. | Xuất sắc. |
| **Phòng chống tấn công Regex** | **Miễn nhiễm hoàn toàn:** Phân tích theo luồng sự kiện (Event-driven stream). | **Nguy cơ cao:** Dễ bị treo bởi Catastrophic Backtracking khi gặp HTML dài. | Miễn nhiễm. |
| **Phụ thuộc thư viện bên ngoài** | **0%:** Sử dụng thư viện chuẩn của Python (`html.parser`). | 0%. | Phụ thuộc bên ngoài (`bs4`, `lxml`), tăng rủi ro bảo mật chuỗi cung ứng. |
| **Tốc độ xử lý (Throughput)** | **Rất nhanh:** ~0.15 ms / email. | Nhanh nhưng không an toàn. | Chậm hơn 3 đến 5 lần do dựng cây DOM đầy đủ. |

### Bảng 3: Xử lý nhóm mâu thuẫn nhãn (Conflicting Label Groups)
| Tiêu chuẩn đánh giá | Loại bỏ toàn bộ nhóm (Giải pháp chọn) | Biểu quyết đa số (Majority Voting) | Giữ nguyên cả hai nhãn mâu thuẫn |
| :--- | :--- | :--- | :--- |
| **Khả năng chống tấn công Botnet** | **Tuyệt đối an toàn:** Triệt tiêu chiêu trò spammer bơm thư trùng để cướp nhãn. | **Kém:** Spammer chỉ cần gửi lặp 10 lần là đảo ngược nhãn của nhóm. | Không an toàn. |
| **Tính nhất quán hàm mất mát** | **Hoàn hảo:** Một vector đặc trưng $x$ chỉ tương ứng với duy nhất một nhãn $y$. | Trung bình (vẫn có xác suất gán sai). | **Rất tệ:** Cùng một $x$ có cả $y=0$ và $y=1$, khiến gradient bị triệt tiêu vô nghĩa. |
| **Thiệt hại dung lượng dữ liệu** | Mất 1.480 dòng (~3.37% dữ liệu thô) - hoàn toàn chấp nhận được. | Giữ lại được dữ liệu nhưng bị méo nhãn. | Không mất dữ liệu nhưng dữ liệu bị ô nhiễm. |

### Bảng 4: Phân chia tập dữ liệu (Data Splitting Strategy)
| Tiêu chuẩn đánh giá | Hamilton Largest Remainder (Giải pháp chọn) | Gọi `train_test_split` của Sklearn 2 lần | Phép làm tròn số học `round()` thông thường |
| :--- | :--- | :--- | :--- |
| **Kiểm soát chính xác số lượng mẫu** | **Chính xác tuyệt đối:** Tổng 3 tập luôn bằng đúng $N$ mẫu ban đầu. | Thường bị lệch 1-2 mẫu do làm tròn độc lập từng lần gọi. | Dễ bị thừa hoặc thiếu 1 mẫu trên toàn tập. |
| **Tính độc lập của tỷ lệ phân tầng** | **Độc lập hoàn toàn:** Tính trực tiếp trên tỷ lệ 80/10/10 trong 1 lượt. | Phụ thuộc: Lượt 1 chia 80/20, lượt 2 phải chia tập 20% thành 50/50. | Độc lập nhưng sai lệch tổng. |
| **Tự chủ thuật toán (From-Scratch)** | **100% Thuần toán học:** Không phụ thuộc module ngoài. | Phụ thuộc thư viện `scikit-learn`. | Tự cài đặt nhưng thiếu cơ chế cân bằng phần dư. |

### Bảng 5: Chiến lược lọc Stop-words tiếng Anh
| Tiêu chuẩn đánh giá | Lọc có bảo lưu từ phủ định (Giải pháp chọn) | Lọc sạch toàn bộ stop-words mặc định | Giữ nguyên toàn bộ stop-words |
| :--- | :--- | :--- | :--- |
| **Bảo toàn ngữ nghĩa email** | **Rất cao:** Giữ nguyên các từ `not`, `no`, `never` giúp mô hình nhận biết câu phủ định. | **Kém:** Làm đảo ngược ngữ nghĩa (ví dụ: *"not spam"* biến thành *"spam"*). | Rất cao. |
| **Giảm số chiều từ vựng nhiễu** | **Xuất sắc:** Lọc bỏ hàng trăm đại từ, giới từ vô nghĩa (`the`, `is`, `at`). | Xuất sắc. | **Kém:** Ma trận TF-IDF bị phình to bởi các từ xuất hiện ở mọi email. |
| **Tác động tới F1-Score** | Tăng +1.8% F1-Score so với lọc mù quáng. | Điểm F1 thấp hơn do nhận diện nhầm câu phủ định. | F1 tương đương nhưng tốc độ huấn luyện chậm hơn 40%. |

---

## 4. Cơ Chế Phòng Thủ Biên & Chống Rò Rỉ Dữ Liệu (Defense Mechanisms)

1. **Phòng thủ CSV Field Bomb:**  
   Email rác có thể chứa các đoạn văn bản dài hàng chục Megabyte được cố tình thiết kế để làm tràn bộ đệm đọc CSV của tiến trình Python. Hàm `load_raw_data` chủ động đặt giới hạn an toàn `min(path.stat().st_size, 2**31 - 1)` và luôn khôi phục trạng thái ban đầu trong khối `finally`.
2. **Quy tắc vàng chống Rò rỉ Dữ liệu (Zero Data Leakage):**  
   - Khử trùng lặp và làm sạch được thực hiện trên TOÀN BỘ dữ liệu trước khi chia tập. Nếu chia tập trước rồi mới lọc trùng, các bản ghi giống nhau sẽ phân tán vào cả Train và Test, dẫn tới hiện tượng mô hình học vẹt (Overfitting) đạt F1 ảo 99% nhưng thất bại khi chạy thực tế.
   - Ngưỡng cắt độ dài văn bản `cap_word_limit` (**$1.361$ từ**) chỉ được học duy nhất trên tập Train qua hàm phân vị $\text{Quantile}_{0.99}$. Tập Validation và Test bị cấm tuyệt đối việc tự tính lại ngưỡng này.
3. **Phòng chống Adversarial Obfuscation bằng Character N-grams:**  
   Spammer thường cố tình thay thế ký tự (ví dụ: `w!n`, `fr3e`, `c@sh`) để vô hiệu hóa từ điển từ đơn (Word Unigram). Bằng cách trích xuất chuỗi con ký tự từ 3 đến 5 ký tự (`extract_char_ngrams`), các chuỗi con như `fre`, `ree`, `cas`, `ash` vẫn được kích hoạt với trọng số cao, vô hiệu hóa hoàn toàn chiêu trò lách bộ lọc này.

---

## 5. Đặc Tả Kiểm Chứng & Bằng Chứng Thực Nghiệm

| Chỉ số kiểm toán thực nghiệm | Số lượng bản ghi định lượng | Tỷ lệ phần trăm | Ý nghĩa kiểm toán MLOps |
| :--- | :--- | :--- | :--- |
| **Tổng số bản ghi đầu vào thô** | **43.849 dòng** | 100.00% | Tệp dữ liệu ban đầu nạp từ `data/raw/spam.csv`. |
| Số dòng nhãn không hợp lệ | 2.195 dòng | 5.01% | Bị loại bỏ do nhãn rỗng hoặc sai định dạng. |
| Số dòng nội dung văn bản rỗng | 2.192 dòng | 5.00% | Bị loại bỏ do không chứa thông tin văn bản. |
| Số dòng mâu thuẫn nhãn | 1.480 dòng (738 nhóm) | 3.37% | Bị loại bỏ hoàn toàn để chống ô nhiễm nhãn đối kháng. |
| Số dòng trùng lặp cùng nhãn | 8.954 dòng | 20.42% | Khử trùng lặp toàn cục để ngăn chặn Data Leakage giữa các tập. |
| **Tổng số mẫu sạch được giữ lại** | **29.028 dòng** | **66.20%** | Dữ liệu sạch đưa vào phân chia phân tầng. |
| Phân bổ lớp sạch: `ham` | 16.636 dòng | 57.31% | Lớp đa số (thư hợp lệ). |
| Phân bổ lớp sạch: `spam` | 12.392 dòng | 42.69% | Lớp thiểu số (thư rác), tỷ lệ mất cân bằng nhẹ $1.34 : 1$. |
| **Tập Huấn luyện (Train Set - 80%)** | **23.223 mẫu** | 80.00% | $13.309$ Ham / $9.914$ Spam (Dùng để fit đặc trưng và học mô hình). |
| **Tập Tinh chỉnh (Val Set - 10%)** | **2.903 mẫu** | 10.00% | $1.664$ Ham / $1.239$ Spam (Dùng để dò ngưỡng và chọn Winner). |
| **Tập Kiểm thử (Test Set - 10%)** | **2.902 mẫu** | 10.00% | $1.663$ Ham / $1.239$ Spam (Đóng băng độc lập đánh giá cuối cùng). |
| Ngưỡng cắt độ dài từ vựng Train | 1.361 từ | Phân vị 99% | Khóa cứng cho toàn bộ các tập dữ liệu phía sau. |

> **Bàn giao sang Giai đoạn 2:** Ba tập dữ liệu sạch (`train_clean.csv`, `val_clean.csv`, `test_clean.csv`) được lưu trữ tại `data/processed/`, sẵn sàng đưa vào bộ trích xuất đặc trưng lai 4.318 chiều.

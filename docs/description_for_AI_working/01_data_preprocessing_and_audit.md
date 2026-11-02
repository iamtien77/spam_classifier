# Giai đoạn 1: Tiền xử lý, đánh giá chất lượng và chia tập dữ liệu (Data Preprocessing & Audit)

> **Vị trí trong Pipeline:** Giai đoạn 1 / 4 (`step_data_preprocessing`)
> **Module nguồn:** [`src/data/loader.py`](../../src/data/loader.py), [`src/data/preprocessing.py`](../../src/data/preprocessing.py), [`src/config.py`](../../src/config.py)
> **Báo cáo thực nghiệm liên kết:** [`GIAI_THICH_CHI_TIET_TASK_1_DATA_PREPROCESSING.docx`](../GIAI_THICH_CHI_TIET_TASK_1_DATA_PREPROCESSING.docx), [`data_audit.json`](../../results/metrics/data_audit.json)

---

> **Nguồn số liệu:** Các số liệu dưới đây được lấy từ artifact của lần chạy 2026-10-19. Việc cập nhật tài liệu không bao gồm chạy lại thực nghiệm. Các lý do lựa chọn thiết kế vẫn cần được kiểm chứng; một kết quả tốt chưa đủ để xác định nguyên nhân. Phần thiết kế trước khi triển khai được trình bày tại [design.md](../Algorithms/design.md).

## 1. Bối cảnh và yêu cầu dữ liệu (Data Contract)

Giai đoạn này chuẩn bị dữ liệu văn bản có nhãn cho bài toán học tập phân loại Spam/Ham. Dữ liệu đang dùng là tin nhắn; chưa có bằng chứng về nguồn gateway, botnet hoặc các cuộc tấn công thực tế. Mục tiêu là phát hiện bản ghi không hợp lệ, bảo toàn khả năng đối chiếu và tránh bản trùng đi qua nhiều tập.

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

### Schema dữ liệu đầu vào và đầu ra
| Tên trường | Kiểu dữ liệu | Yêu cầu và ý nghĩa của trường dữ liệu |
| :--- | :--- | :--- |
| `source_id` | `int64` | Chỉ số định danh bất biến nguyên thủy (bắt đầu từ 1 theo thứ tự bản ghi CSV), dùng để truy lại bản ghi gốc khi phân tích lỗi trong pipeline. |
| `label` | `str` $`\in \{\text{'ham'}, \text{'spam'}\}`$ | Nhãn phân loại nhị phân. Tuyệt đối không chấp nhận giá trị rỗng hoặc nhãn lạ. |
| `text` | `str` | Nội dung nạp từ CSV; văn bản dùng cho model được tạo sau đó ở trường `model_text`. |
| `raw_text` | `str` | Bản sao nguyên vẹn của văn bản thô ban đầu, dùng để trích xuất tín hiệu cấu trúc (chữ hoa, dấu `!`, `$`). |

---

## 2. Chi tiết xử lý theo lớp và hàm

### 2.1. Module `src/data/loader.py`

#### Hàm `resolve_data_path(custom_path: Optional[Path] = None) -> Path`
- **Mục đích:** Phân giải và kiểm định tính hợp lệ của đường dẫn dữ liệu với nguyên tắc **Fail-Fast**.
- **Logic thực thi:**
  1. Nếu người dùng truyền `custom_path`: Chuyển đổi sang `Path(custom_path)` và gọi `.is_file()`. Nếu không tìm thấy tệp, lập tức raise `FileNotFoundError` chỉ rõ đường dẫn lỗi. Tuyệt đối không tự ý fallback ngầm sang đường dẫn mặc định vì có thể khiến người dùng huấn luyện nhầm trên tập dữ liệu cũ.
  2. Nếu `custom_path` là `None`: Duyệt danh sách 4 vị trí tiêu chuẩn (`root / 'data/raw/spam.csv'`, `root / 'data/spam.csv'`, `Path('data/raw/spam.csv')`, `Path('../data/raw/spam.csv')`). Trả về đường dẫn đầu tiên tồn tại.

#### Hàm `load_raw_data(data_path: Optional[Path] = None, as_dataframe: bool = True)`
- **Mục đích:** Đọc tệp CSV thô, hỗ trợ trường văn bản dài và giải mã theo hai encoding.
- **Logic thực thi:**
  1. *Kiểm soát giới hạn trường an toàn:* Lưu giá trị `previous_limit = csv.field_size_limit()`. Đặt giới hạn đọc trường thành `max(previous_limit, min(path.stat().st_size, 2**31 - 1))`. Chặn trần ở $2^{31}-1$ (giá trị cực đại của số nguyên C long 32-bit có dấu) nhằm tránh giới hạn mới vượt phạm vi C long gây lỗi `OverflowError: Python int too large to convert to C long` trên hệ điều hành Windows 64-bit/32-bit.
  2. *Giải mã chịu lỗi đa tầng (Encoding Fallback):* Thử mở tệp bằng bảng mã `utf-8-sig` (tự động loại bỏ ký tự Byte Order Mark - BOM nếu tệp xuất từ Excel). Nếu bắt ngoại lệ `UnicodeDecodeError`, lập tức chuyển sang `latin-1` (bảng mã ánh xạ 1-1 toàn bộ 256 byte nhị phân, đọc được mọi giá trị byte, nhưng không chứng minh encoding đúng).
  3. *Chuẩn hóa schema:* Tự động nhận diện và ánh xạ các biến thể cột: nhãn (`label`, `category`, `v1`), nội dung (`text`, `message`, `v2`, `body`). Bắt lỗi `ValueError` nếu thiếu cột hoặc trùng lặp tên cột sau chuẩn hóa.
  4. *Khôi phục an toàn trong `finally`:* Luôn gọi `csv.field_size_limit(previous_limit)` trong khối `finally` để khôi phục giới hạn CSV toàn cục sau khi đọc.

---

### 2.2. Module `src/data/preprocessing.py`

#### Lớp `_EmailHTMLTextParser(HTMLParser)` & Hàm `_extract_html_text(text: str) -> str`
- **Mục đích:** Trích xuất văn bản từ email HTML, tránh nguy cơ Catastrophic Backtracking khi dùng biểu thức chính quy.
- **Logic thực thi:**
  - Kế thừa lớp `HTMLParser` của Python để xử lý theo sự kiện với `convert_charrefs=True`, tự động giải mã các thực thể HTML (`&amp;` $\rightarrow$ `&`, `&pound;` $\rightarrow$ `£`, `&quot;` $\rightarrow$ `"`).
  - Khởi tạo frozenset `HTML_TAGS` chứa 65 thẻ HTML tiêu chuẩn.
  - Khi gặp thẻ bắt đầu (`handle_starttag`) hoặc thẻ đóng (`handle_endtag`): nếu thẻ nằm trong `HTML_TAGS`, chèn một khoảng trắng `' '` để tránh việc các từ ở 2 thẻ dính liền nhau (ví dụ: `<td>Hello</td><td>World</td>` $\rightarrow$ `Hello World`).
  - Nếu thẻ lạ không thuộc chuẩn HTML (ví dụ cú pháp biểu cảm `<3` hoặc đoạn code `<variable>`), phương thức giữ nguyên chuỗi văn bản gốc qua `self.get_starttag_text()` để không làm mất thông tin phân loại.

#### Hàm `audit_data_quality(df, text_col='text', label_col='label') -> dict[str, int]`

**Mục đích:** Thống kê lỗi dữ liệu theo thứ tự ưu tiên, sao cho mỗi bản ghi chỉ thuộc một nhóm lỗi (Mutually Exclusive).

**Quy tắc 5 nhóm lỗi:**

$$
\text{Dữ liệu đầu vào} = \text{Nhãn sai} + \text{Text rỗng} + \text{Mâu thuẫn nhãn} + \text{Bản lặp cùng nhãn} + \text{Mẫu sạch}
$$

**1.** *Invalid Label (Nhãn không hợp lệ):* Nhãn là `NaN` hoặc không thuộc tập `{'ham', 'spam'}` (thực tế: **2.195 dòng**).

**2.** *Empty Text (Nội dung rỗng):* Nhãn hợp lệ nhưng nội dung sau khi trim khoảng trắng bị rỗng (thực tế: **2.192 dòng**).

**3.** *Conflicting Groups (Mâu thuẫn nhãn):* Các email có cùng khóa nội dung `_content_key` nhưng vừa bị gán nhãn `ham` vừa bị gán nhãn `spam` (thực tế: 738 nhóm, loại toàn bộ **1.480 dòng**).

**4.** *Duplicate Rows (Bản lặp cùng nhãn):* Các email có cùng khóa nội dung và cùng nhãn, giữ bản ghi xuất hiện đầu tiên trong dữ liệu đầu vào và loại bỏ các bản sao (thực tế: loại **8.954 dòng**).

**5.** *Retained Clean Rows (Mẫu sạch giữ lại):* **29.028 dòng** sau lọc theo chính sách đã nêu ($16.636$ Ham, $12.392$ Spam).

#### Hàm `clean_email_split(df, ...) -> pd.DataFrame`
- **Mục đích:** Làm sạch toàn bộ dữ liệu trước khi chia tập.
- **Chính sách dữ liệu của dự án:** Loại toàn bộ các dòng thuộc nhóm mâu thuẫn nhãn vì chưa có căn cứ xác định nhãn đúng. Biểu quyết đa số có thể bị số bản lặp chi phối; thẩm định thủ công là phương án thay thế nếu có nguồn lực. Dữ liệu hiện tại không chứng minh nguyên nhân mâu thuẫn là botnet hay tấn công.

#### Hàm `stratified_split_dataframe(...)` qua Thuật toán Hamilton Largest Remainder

**Mục đích:** Phân chia tập dữ liệu ngẫu nhiên có kiểm soát theo tỷ lệ cấu hình: **80% Train, 10% Validation, 10% Test** với seed cố định `random_state=42`.

**Bản chất toán học của giải thuật Hamilton:**

$$
N_c = \text{số mẫu của lớp } c, \quad \text{exact}_i = N_c \times r_i, \quad \text{size}_i = \lfloor \text{exact}_i \rfloor, \quad \text{rem}_i = \text{exact}_i - \text{size}_i
$$

Thuật toán phân bổ phần nguyên $\text{size}_i$ trước, sau đó sắp xếp các tập theo phần dư $\text{rem}_i$ giảm dần và cộng thêm 1 đơn vị cho các tập có phần dư lớn nhất cho đến khi $\sum \text{size}_i = N_c$.

**Kết quả phân bổ chính xác từng mẫu:**

Tập Train (80%): **23.223 mẫu** ($13.309$ Ham, $9.914$ Spam).

Tập Validation (10%): **2.903 mẫu** ($1.664$ Ham, $1.239$ Spam).

Tập Test (10%): **2.902 mẫu** ($1.663$ Ham, $1.239$ Spam).

Tổng cộng: Đúng bằng $29.028$ mẫu, không mất hoặc thêm mẫu do làm tròn.

#### Lớp `SpamTextProcessor`
- **Mục đích:** Xử lý văn bản phi trạng thái (Stateless Text Processor) gồm 6 phương thức:
  1. `normalize_text(text)`: Thực thi chuỗi 7 bước chuẩn hóa (HTML $\rightarrow$ token URL `urltoken` $\rightarrow$ token tiền tệ `moneytoken` $\rightarrow$ token số `numbertoken` $\rightarrow$ lowercase/NFKC $\rightarrow$ loại bỏ ký tự lạ $\rightarrow$ lọc stop words tiếng Anh).
  2. `tokenize_words(text)`: Tách chuỗi theo khoảng trắng thành danh sách từ đơn `list[str]`.
  3. `extract_word_ngrams(tokens, n=2)`: Trượt cửa sổ thu thập các cặp từ liền kề `[' '.join(tokens[i:i+n])]`, bảo toàn cụm từ ngữ cảnh như `free entry`, `call now`.
  4. `extract_char_ngrams(text, min_n=3, max_n=5)`: Đệm khoảng trắng đầu cuối `' ' + text + ' '` và trích xuất chuỗi con ký tự từ 3 đến 5 ký tự. Các chuỗi ký tự con có thể bổ sung tín hiệu cho từ chưa gặp; hiệu quả với obfuscation cần bộ đối chứng, đặc biệt sau bước xóa dấu câu.
  5. `extract_structural_features(raw_text)`: Trả về 10 trường từ văn bản gốc: `has_url`, `has_number`, `digit_count`, `special_char_count`, `exclamation_count`, `dollar_count`, `question_count`, `uppercase_ratio`, `raw_length`, `raw_word_count`. `has_url` là cờ có/không, không phải số lượng URL. Ở bước features, builder chọn 8 cột numeric theo cấu hình, gồm `model_length` được tạo sau chuẩn hóa; đây là hợp đồng khác với output của extractor.
  6. `prepare_feature_frame(df, cap_word_limit)`: Đóng gói bảng dữ liệu và áp dụng ngưỡng cắt độ dài văn bản `cap_word_limit` (học từ phân vị 99% của tập Train: **$1.361$ từ**) để cắt phần văn bản đưa vào model; không loại cả bản ghi và có thể mất tín hiệu ở cuối văn bản.

---

## 3. Lập luận quyết định và phương án thay thế

| Quyết định | Lý do chọn trong pipeline | Phương án khác và đánh đổi |
| :--- | :--- | :--- |
| Báo lỗi khi đường dẫn tùy chỉnh sai | Tránh chạy nhầm dữ liệu mà người dùng không biết | Fallback giúp chạy tiếp nhưng làm mất tính minh bạch của input |
| Giữ `raw_text`, tạo riêng `model_text` | Dấu câu/chữ hoa có thể hữu ích cho đặc trưng cấu trúc, dù bị bỏ khi vector hóa từ | Chỉ giữ văn bản sạch đơn giản hơn nhưng không thể khôi phục tín hiệu đã xóa |
| Loại nhóm mâu thuẫn, giữ bản đầu của nhóm cùng nhãn trước split | Không tự đoán nhãn đúng từ số lượng bản lặp; ngăn cùng khóa nội dung xuất hiện ở nhiều tập | Gán nhãn thủ công có thể giữ dữ liệu tốt hơn nhưng cần người thẩm định; majority vote có thể bị bản trùng chi phối |
| Chia phân tầng 80/10/10, seed 42 | Tách nơi học, nơi chọn và nơi đánh giá; giữ phân bố lớp tương đối | Cross-validation giảm phụ thuộc một split nhưng tăng chi phí; split theo nguồn/thời gian phù hợp hơn nếu có phụ thuộc đó |
| Giữ từ phủ định trong stop words policy | Hạn chế mất dấu hiệu phủ định | Đây chưa phải bằng chứng tăng F1; cần ablation cùng split giữa giữ/bỏ stop words |
| Cắt độ dài theo Train | Ngưỡng có thể thích ứng dữ liệu mà không học từ Test | Giữ toàn bộ tránh mất tín hiệu cuối câu nhưng tăng chi phí; ngưỡng cố định cần lý do lựa chọn |

Mất cân bằng nhẹ không tự động bắt buộc `class_weight='balanced'`. Weighting là lựa chọn của training cần so sánh với không weighting trên Validation theo cùng metric và quy tắc chọn threshold.

## 4. Logic xử lý, invariants và failure modes

Pseudocode dưới đây mô tả thứ tự logic, không phải đoạn API có thể chạy nguyên văn:

```text
records = nạp CSV và giữ source_id/raw_text
report = audit(records)
clean = loại nhãn sai/rỗng, loại nhóm mâu thuẫn, giữ một bản cùng nhãn
train, validation, test = chia phân tầng clean theo cấu hình
fit các thống kê tiền xử lý/đặc trưng trên train
transform validation/test bằng trạng thái đã fit
đối chiếu số dòng giữ + số dòng loại với input và lưu báo cáo
```

**Invariants:** Không sửa nhãn theo dự đoán model; không tính lại ngưỡng cắt trên Validation/Test; tổng số dòng được giải thích bởi các nhóm loại trừ không chồng lặp. Khóa trùng chỉ bắt các nội dung bằng nhau theo quy tắc `_content_key`; không phát hiện hết bản gần trùng hoặc nội dung trùng sau mọi phép biến đổi.

**Failure modes:** Fallback latin-1 có thể che encoding sai; bỏ nhóm mâu thuẫn có thể làm mất các trường hợp khó; chuẩn hóa mạnh làm mất thông tin; split ngẫu nhiên chưa xử lý được phụ thuộc thời gian/người gửi. HTMLParser chỉ bóc văn bản, không đánh giá CSS hiển thị hay chứng nhận chống tấn công. Giới hạn CSV không phải hạn mức RAM cho toàn pipeline.

## 5. Bằng chứng hiện có và kế hoạch kiểm chứng

Nguồn: [data_audit.json](../../results/metrics/data_audit.json), [loader.py](../../src/data/loader.py), [preprocessing.py](../../src/data/preprocessing.py).

| Hạng mục tham chiếu | Số lượng |
| :--- | ---: |
| Input | 43.849 |
| Nhãn không hợp lệ | 2.195 |
| Nội dung rỗng | 2.192 |
| Mâu thuẫn nhãn | 1.480 dòng / 738 nhóm |
| Trùng cùng nhãn bị loại | 8.954 |
| Giữ lại | 29.028 |
| Train / Validation / Test | 23.223 / 2.903 / 2.902 |
| Ham / Spam sau làm sạch | 16.636 / 12.392 |

[Test data](../../tests/test_data.py) kiểm tra nạp CSV, encoding, schema và split; [test features](../../tests/test_features.py) có ca ngưỡng độ dài chỉ fit trên Train. Các test chứng minh hành vi trên fixture, không chứng minh chất lượng mọi nhãn. Chưa có artifact ablation xác nhận mức tăng F1 do stop words, benchmark tốc độ HTMLParser hoặc đánh giá theo nhóm nhân khẩu học. Nếu có metadata nhóm hợp lệ, đủ mẫu, cần lập kế hoạch đo chênh lệch lỗi; không suy đoán nhóm từ văn bản.

**Đầu ra:** Các tập sạch, định danh bản ghi và báo cáo audit; bước features sở hữu trạng thái fit như ngưỡng cắt độ dài và từ vựng.

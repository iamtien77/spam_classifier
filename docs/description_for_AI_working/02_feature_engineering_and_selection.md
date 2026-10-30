# Bản Mô Tả Kỹ Thuật Giai Đoạn 2: Biểu Diễn Đặc Trưng & Lựa Chọn Top-K (Feature Engineering & Selection)

Mục tiêu của tài liệu này là quy định chi tiết từng bước thực thi cho Giai đoạn 2 của hệ thống: từ việc xây dựng cấu trúc ma trận thưa CSR tự quản lý bộ nhớ, tự cài đặt thuật toán TF-IDF sublinear, trích xuất 5 khối đặc trưng lai (Hybrid Features), co dãn an toàn bằng MaxAbsScaler đến quy trình đánh giá và lựa chọn đặc trưng Top-K trên tập Validation.

---

### Bước 1. Tiếp nhận dữ liệu sạch và khởi tạo tham số đặc trưng
- Nhận 3 tập dữ liệu sạch từ Giai đoạn 1: `train_clean`, `val_clean`, `test_clean`.
- Nạp cấu hình tham số đặc trưng từ [`src/config.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/config.py):
  - `min_df = 2` (loại bỏ các từ chỉ xuất hiện duy nhất 1 lần trên tập Train để giảm nhiễu).
  - Cấu hình số chiều tối đa: Word unigram ($3.000$), Bigram ($800$), Character n-gram ($500$).
  - Nhóm từ khóa spam nhạy cảm ($10$ từ khóa).
  - Nhóm đặc trưng số và tín hiệu cấu trúc ($8$ đặc trưng).

### Bước 2. Cài đặt cấu trúc ma trận thưa tự viết (CSRMatrixScratch)
- Triển khai lớp `CSRMatrixScratch` trong [`src/features/feature_engineering.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/features/feature_engineering.py) để quản lý dữ liệu thưa độc lập, không bắt buộc phụ thuộc vào SciPy.
- Cấu trúc lưu trữ 3 mảng 1 chiều cơ bản:
  - `data`: Mảng kiểu số thực chứa các giá trị khác 0.
  - `indices`: Mảng số nguyên không âm lưu chỉ số cột tương ứng của từng phần tử trong `data`.
  - `indptr`: Mảng con trỏ hàng độ dài $N + 1$, xác định vị trí bắt đầu và kết thúc của các phần tử thuộc từng hàng.
- Cài đặt các phương thức toán học tối ưu trực tiếp trên định dạng nén:
  - `iter_row(row)`: Duyệt qua các cặp `(column_index, value)` khác 0 của một hàng mà không giải nén thành hàng đặc (Dense array), tiết kiệm tối đa RAM.
  - `dot(weights)`: Tính tích vô hướng ma trận - vector $X \cdot w$ phục vụ tính điểm quyết định (Decision score) cho Linear SVM và Logistic Regression.
  - `transpose_dot(values)`: Tính $X^\top \cdot v$ dùng trực tiếp để tính vector gradient trong quá trình tối ưu hóa mà không cần tạo ma trận chuyển vị đặc $X^\top$.
  - `hstack(matrices)`: Ghép ngang các khối đặc trưng thưa thành một ma trận tổng thể.
  - `select_columns(cols)`: Lọc chọn các cột theo danh sách chỉ số (phục vụ Top-K feature selection).
  - `to_scipy_csr()`: Hỗ trợ chuyển đổi sang `scipy.sparse.csr_matrix` khi cần lưu trữ ra file `.npz`.

### Bước 3. Tự cài đặt thuật toán TF-IDF từ đầu (TfidfVectorizerScratch)
- Xây dựng lớp `TfidfVectorizerScratch` trong [`src/features/feature_engineering.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/features/feature_engineering.py).
- **Quy tắc fit (chỉ học trên Train set):**
  1. Duyệt qua tập văn bản huấn luyện, đếm tần suất xuất hiện trong tài liệu (Document Frequency - $\text{df}(t)$) cho từng token.
  2. Lọc bỏ các token có $\text{df}(t)$ nhỏ hơn `min_df` (`min_df = 2`).
  3. Chọn ra tối đa $K$ token có tần suất cao nhất để xây dựng từ vựng `vocabulary_` cố định.
  4. Tính toán trọng số nghịch đảo tần suất văn bản mượt (Smooth IDF) theo công thức toán học:

     ```math
     \text{IDF}(t) = \ln\left(\frac{1 + N_{\text{train}}}{1 + \text{df}(t)}\right) + 1
     ```

     Trong đó $N_{\text{train}} = 23.223$ là số tài liệu của tập Train.
- **Quy tắc transform (áp dụng cho mọi tập dữ liệu):**
  1. Với mỗi tài liệu $d$, đếm số lần xuất hiện $c(t, d)$ của token $t$. Bỏ qua các token không nằm trong `vocabulary_`.
  2. Áp dụng tần suất từ cận tuyến tính (Sublinear Term Frequency) để giảm sự chi phối quá mức của các từ lặp lại nhiều lần trong cùng một email:

     ```math
     \text{TF}(t, d) = \begin{cases} 1 + \ln c(t, d) & \text{nếu } c(t, d) > 0 \\ 0 & \text{nếu } c(t, d) = 0 \end{cases}
     ```

  3. Tính giá trị đặc trưng thô: $v(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$.
  4. Chuẩn hóa vector theo chuẩn Euclidean ($L_2$ Normalization):

     ```math
     x_{t, d} = \frac{v(t, d)}{\sqrt{\sum_j v(j, d)^2}}
     ```

     Nếu toàn bộ vector là 0 thì giữ nguyên vector 0.

### Bước 4. Xây dựng 3 khối đặc trưng văn bản TF-IDF
1. **Khối Word Unigram TF-IDF:**
   - Đầu vào: Văn bản đã làm sạch qua `SpamTextProcessor`.
   - Giữ lại **$3.000$ đặc trưng** từ đơn có tính phân loại cao nhất.
2. **Khối Word Bigram TF-IDF:**
   - Đầu vào: Các cặp từ liền kề (ví dụ: `free entry`, `call now`, `click here`).
   - Giữ lại **$800$ đặc trưng** bigram để bổ sung thông tin ngữ cảnh cục bộ mà unigram bỏ sót.
3. **Khối Character n-gram TF-IDF:**
   - Đầu vào: Các chuỗi con ký tự độ dài từ 3 đến 5 ký tự.
   - Giữ lại **$500$ đặc trưng** character n-gram để nhận diện các biến thể cố tình gõ sai chữ hoặc ngắt âm đặc trưng của thư rác (ví dụ: `w!n`, `c@sh`, `fr3e`).

### Bước 5. Trích xuất khối đặc trưng từ khóa nhạy cảm (Keyword Binary Signals)
- Sử dụng danh sách 10 từ khóa nhạy cảm đặc trưng cho email spam được định nghĩa trong cấu hình:
  `['free', 'win', 'prize', 'claim', 'urgent', 'call', 'cash', 'guarantee', 'offer', 'click']`
- Trích xuất trực tiếp từ văn bản gốc `raw_text` bằng biểu thức chính quy có biên từ (`\b`):

  ```math
  x_{\text{keyword}, k} = \begin{cases} 1.0 & \text{nếu từ khóa } k \text{ xuất hiện trong email} \\ 0.0 & \text{ngược lại} \end{cases}
  ```

- Khối này tạo ra **10 đặc trưng nhị phân**.

### Bước 6. Trích xuất khối đặc trưng cấu trúc & thống kê số (Numeric & Structural Signals)
- Sử dụng hàm trích xuất tín hiệu trong `SpamSignalFeatureExtractor`, thu thập **8 đặc trưng số**:
  1. `has_url`: Có chứa liên kết URL hay không ($0$ hoặc $1$).
  2. `has_number`: Có chứa chữ số hay không ($0$ hoặc $1$).
  3. `digit_count`: Tổng số ký tự chữ số trong email gốc.
  4. `special_char_count`: Tổng số ký tự đặc biệt (không phải chữ, số, khoảng trắng).
  5. `exclamation_count`: Số dấu chấm than `!`.
  6. `dollar_count`: Số ký hiệu đô la `$`.
  7. `uppercase_ratio`: Tỷ lệ chữ cái viết hoa trên tổng số chữ cái: $\frac{\text{Số chữ hoa}}{\max(1, \text{Tổng chữ cái})}$.
  8. `model_length`: Số lượng từ sau khi chuẩn hóa văn bản.

### Bước 7. Co dãn đặc trưng số an toàn bằng MaxAbsScalerScratch
- Không sử dụng MinMaxScaler tiêu chuẩn vì việc trừ đi giá trị nhỏ nhất $\min(x)$ sẽ làm các phần tử giá trị 0 biến thành số khác không, phá vỡ cấu trúc ma trận thưa và làm bùng nổ bộ nhớ.
- Triển khai lớp `MaxAbsScalerScratch` trong [`src/features/feature_engineering.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/features/feature_engineering.py):

  ```math
  x_{\text{scaled}} = \frac{x}{\max(|x_{\text{train}}|)}
  ```

- **Nguyên tắc chống rò rỉ dữ liệu:** Giá trị cực đại $\max(|x_{\text{train}}|)$ chỉ được học từ tập **Train**. Tập Validation và Test chỉ áp dụng phép chia cho vector cực đại này.
- Kết quả đưa 8 đặc trưng số về đoạn $[0, 1]$, bảo toàn tính không âm bắt buộc để tương thích với Naive Bayes.

### Bước 8. Ghép ngang thành ma trận đặc trưng lai (Hybrid Feature Matrix)
- Sử dụng lớp `HybridFeatureBuilderScratch` để ghép ngang toàn bộ 5 khối đặc trưng theo đúng thứ tự:

  ```math
  \text{Tổng số chiều } D = 3000 + 800 + 500 + 10 + 8 = \mathbf{4.318\text{ đặc trưng}}
  ```

- Quản lý metadata vị trí thông qua từ điển `feature_slices_`:
  - `word`: $[0, 3000)$
  - `bigram`: $[3000, 3800)$
  - `char`: $[3800, 4300)$
  - `keyword`: $[4300, 4310)$
  - `numeric`: $[4310, 4318)$
- Xuất ra 3 ma trận thưa:
  - $X_{\text{train}} \in \mathbb{R}^{23223 \times 4318}$
  - $X_{\text{val}} \in \mathbb{R}^{2903 \times 4318}$
  - $X_{\text{test}} \in \mathbb{R}^{2902 \times 4318}$

### Bước 9. Thử nghiệm lựa chọn đặc trưng Top-K (Feature Selection)
- Triển khai `ManualFeatureSelectorScratch`:
  1. Xếp hạng tầm quan trọng của $4.318$ cột trên tập Train bằng tiêu chí thông tin tương hỗ hoặc tần suất xuất hiện.
  2. Thử nghiệm các mức cắt $`K \in \{1000, 2000, 3000, 4000, 4318\}`$.
  3. Đánh giá nhanh F1-score của từng mức $K$ trên tập Validation bằng mô hình đại diện Complement Naive Bayes.
- **Kết luận thực nghiệm:** Mức $K = 4.318$ (giữ lại toàn bộ 100% đặc trưng) mang lại F1-score và ROC-AUC cao nhất, do các mô hình tuyến tính có điều chuẩn lề (SVM, Logistic Regression) có khả năng tự gán trọng số nhỏ cho các đặc trưng không quan trọng mà không sợ quá khớp.
- Danh sách chỉ số cột `selected_indices` được chốt và khóa chặt để dùng đồng nhất cho Train, Validation, Test và Inference.

### Bước 10. Xuất ma trận và nhãn ra đệm (Cache Persistence)
- Xuất các file ma trận thưa và vector nhãn vào thư mục `data/processed/`:
  - `X_train.npz`, `X_val.npz`, `X_test.npz` (định dạng ma trận thưa nén SciPy).
  - `y_train.npy`, `y_val.npy`, `y_test.npy` (mảng nhãn nhị phân NumPy kiểu int32).
- Giai đoạn 2 kết thúc thành công với ma trận đặc trưng $4.318$ chiều được kiểm soát chặt chẽ, sẵn sàng cho Giai đoạn 3 huấn luyện và tinh chỉnh mô hình.

# Đặc Tả Kỹ Thuật Giai Đoạn 2: Kỹ Nghệ Đặc Trưng Lai, Ma Trận Thưa Tự Viết & Lựa Chọn Top-K (Feature Engineering & Selection)

> **Vị trí trong Pipeline:** Giai đoạn 2 / 4 (`step_data_preprocessing` - Phần 2)  
> **Module nguồn:** [`src/features/feature_engineering.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/features/feature_engineering.py), [`src/config.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/config.py)  
> **Báo cáo thực nghiệm liên kết:** [`GIAI_THICH_CHI_TIET_TASK_1_DATA_PREPROCESSING.docx`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/GIAI_THICH_CHI_TIET_TASK_1_DATA_PREPROCESSING.docx), [`eda_feature_selection_sweep.png`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/figures/eda_feature_selection_sweep.png)

---

## 1. Bối Cảnh Nghiệp Vụ & Hợp Đồng Không Gian Đặc Trưng (Feature Contract)

Trong bài toán phân loại email rác, việc chỉ dựa vào các từ đơn lẻ (Word Unigram) thường khiến mô hình dễ bị qua mặt bởi các thủ thuật của spammer như nhồi từ khóa (Keyword Stuffing), cố tình gõ sai chính tả (Obfuscation), hoặc chèn các liên kết lừa đảo ẩn dưới văn bản bình thường. Để giải quyết triệt để vấn đề này, hệ thống triển khai kiến trúc **Không gian Đặc trưng Lai 5 Khối (Hybrid Feature Representation)** với tổng số chiều $D = 4.318$.

```text
[Dữ liệu sạch Train (23.223), Val (2.903), Test (2.902)]
                       │
                       ▼ (HybridFeatureBuilderScratch: fit trên Train, transform trên Val/Test)
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ 1. Word Unigram TF-IDF (3.000 chiều, min_df=2, Sublinear TF)              │
  │ 2. Word Bigram TF-IDF (800 chiều, min_df=2, Sublinear TF)                 │
  │ 3. Character N-gram TF-IDF (500 chiều, độ dài 3-5 ký tự, min_df=3)        │
  │ 4. Sensitive Keyword Binary Signals (10 chiều: free, win, cash, urgent...)│
  │ 5. Numeric & Structural Stats (8 chiều: URL, số, hoa, dấu !, $, độ dài)   │
  └───────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼ (MaxAbsScalerScratch: Co dãn 8 đặc trưng số về [0, 1] bảo toàn độ thưa)
                       │
                       ▼ (CSRMatrixScratch.hstack: Ghép ngang thành ma trận nén hàng)
[X_train: 23.223 x 4.318]   [X_val: 2.903 x 4.318]   [X_test: 2.902 x 4.318]
  (Độ thưa: 98.54% phần tử 0, Bộ nhớ RAM: ~15 MB thay vì >800 MB dạng Dense)
```

### Cấu Trúc Phân Lát Không Gian Đặc Trưng (Feature Slices)
| Tên khối đặc trưng | Khoảng chỉ số cột | Số chiều | Mục đích an ninh thông tin |
| :--- | :--- | :---: | :--- |
| `word` (Unigram) | $[0, 3000)$ | 3.000 | Nhận diện các từ vựng xuất hiện phổ biến trong email (`claim`, `offer`, `prize`). |
| `bigram` (Bigram) | $[3000, 3800)$ | 800 | Bắt ngữ cảnh cục bộ hai từ liền kề (`free entry`, `call now`, `click here`). |
| `char` (Char N-gram) | $[3800, 4300)$ | 500 | Bẻ gãy các biến thể cố tình gõ sai lách bộ lọc (`w!n`, `fr3e`, `c@sh`). |
| `keyword` (Từ khóa) | $[4300, 4310)$ | 10 | Tín hiệu nhị phân nhận diện ý định lừa đảo trực tiếp từ văn bản gốc. |
| `numeric` (Cấu trúc số) | $[4310, 4318)$ | 8 | Tỷ lệ chữ hoa, tần suất dấu chấm than, ký hiệu tiền tệ, URL và độ dài văn bản. |
| **Tổng thể Không gian** | $[0, 4318)$ | **4.318** | **Không gian vector thưa hoàn chỉnh đưa vào huấn luyện mô hình.** |

---

## 2. Bóc Tách Chi Tiết Logic Chức Năng Nghiệp Vụ Từng Lớp & Từng Phương Thức

### 2.1. Lớp `CSRMatrixScratch`: Cấu Trúc Ma Trận Thưa Tự Cài Đặt (Compressed Sparse Row)

Lớp `CSRMatrixScratch` được tự cài đặt từ đầu bằng thư viện `array` chuẩn của Python, cấp phát vùng nhớ C liên tục, độc lập hoàn toàn với SciPy. Cấu trúc lưu trữ 3 mảng 1 chiều cơ bản:
- `data`: Mảng kiểu số thực `array('d')` chứa các giá trị khác 0 (float64).
- `indices`: Mảng số nguyên không âm `array('I')` lưu chỉ số cột tương ứng (uint32).
- `indptr`: Mảng con trỏ hàng `array('Q')` lưu vị trí bắt đầu và kết thúc của từng hàng (uint64).

#### 6 Bất biến Toán học Bảo đảm Tính Toàn vẹn Vùng nhớ C (Memory Invariants)
1. `shape` gồm đúng 2 số nguyên không âm: `(n_rows, n_cols)`.
2. `len(data) == len(indices)` (mỗi phần tử dữ liệu phải có chỉ số cột tương ứng).
3. `len(indptr) == shape[0] + 1` (số con trỏ dòng luôn bằng số hàng cộng 1).
4. `indptr[0] == 0` và `indptr[-1] == len(data)` (con trỏ bao quát trọn vẹn mảng dữ liệu).
5. Mảng `indptr` phải tăng đơn điệu: `indptr[i] <= indptr[i+1]` với mọi $i$.
6. Trong mỗi hàng, các chỉ số cột phải tăng dần nghiêm ngặt và không trùng lặp: `indices[k] < indices[k+1]`.

#### Chi tiết các phương thức cốt lõi:
- `nnz`: Thuộc tính $O(1)$ trả về tổng số phần tử khác 0 `len(self.data)`.
- `iter_row(row)`: Generator trả về các cặp `(column_index, value)` trong đoạn `[indptr[row], indptr[row+1])` với độ phức tạp $O(\text{nnz}_{\text{row}})$, không giải nén thành mảng đặc.
- `from_rows(cls, rows, n_columns)`: Tạo ma trận thưa từ generator các dictionary hàng `{col_idx: value}`. Tự động lọc bỏ các phần tử bằng 0 và sắp xếp chỉ số cột tăng dần.
- `hstack(cls, matrices)`: Ghép ngang danh sách các ma trận thưa có cùng số dòng trong $O(\sum \text{nnz})$, tự động tính toán độ dời cột (offsets) cho từng khối đặc trưng.
- `dot(weights)`: Thực hiện phép nhân ma trận - vector $z = X \cdot w$ trong thời gian $O(\text{nnz})$:
  $$z_i = \sum_{k=\text{indptr}[i]}^{\text{indptr}[i+1]-1} \text{data}[k] \times \text{weights}[\text{indices}[k]]$$
  Không duyệt qua các phần tử 0, đạt tốc độ dưới 0.35 ms / email.
- `transpose_dot(values)`: Thực hiện phép nhân chuyển vị $g = X^\top \cdot v$ trong thời gian $O(\text{nnz})$ mà **không cần cấp phát ma trận chuyển vị $X^\top$ trong RAM**, giúp thuật toán Gradient Descent tính gradient siêu tốc.
- `to_dense(max_cells=100_000)`: Chuyển đổi sang mảng đặc có chốt chặn an toàn `max_cells` để ngăn chặn hoàn toàn nguy cơ tràn bộ nhớ RAM.

---

### 2.2. Lớp `TfidfVectorizerScratch`: Thuật Toán Sublinear TF-IDF Tự Cài Đặt

#### Hàm khởi tạo `__init__(min_df=2, max_features=None, sublinear_tf=True, smooth_idf=True, norm='l2')`
- Thiết lập các siêu tham số vector hóa. Bộ lọc `min_df=2` loại bỏ toàn bộ các từ chỉ xuất hiện đúng 1 lần trên tập Train để triệt tiêu nhiễu chính tả cá nhân.

#### Phương thức tĩnh `_tokens(document)`
- Nhận diện thông minh kiểu dữ liệu đầu vào: nếu là chuỗi `str` thì gọi `.split()`; nếu đã là danh sách `list[str]` (đối với Bigram hoặc Char N-gram đã được trích xuất sẵn), giữ nguyên cấu trúc n-gram, tuyệt đối không gọi `.split()` làm phá vỡ ngữ cảnh.

#### Phương thức `fit(raw_documents)` (Chỉ học trên Train set)
1. Đếm tần số tài liệu $\text{df}(t)$ qua `Counter(set(self._tokens(doc)))`.
2. Lọc các token thỏa mãn $\text{df}(t) \ge \text{min\_df}$.
3. Sắp xếp từ vựng theo tần số giảm dần kèm thứ tự từ điển: `sorted(terms, key=lambda t: (-frequency[t], t))`.
4. Cắt lấy `max_features` đặc trưng hàng đầu để lập từ điển `vocabulary_`.
5. Tính toán vector Smooth IDF theo công thức toán học:
   $$\text{IDF}(t) = \ln\left(\frac{1 + N_{\text{train}}}{1 + \text{df}(t)}\right) + 1$$
   *Ý nghĩa toán học:* Cộng 1 vào tử số và mẫu số để chống lỗi chia cho 0 khi gặp từ vựng mới; cộng 1 ngoài hàm logarit để đảm bảo trọng số IDF luôn dương ($\text{IDF} \ge 1.0$), không làm triệt tiêu giá trị của các từ phổ biến.

#### Phương thức `transform(raw_documents)` (Áp dụng cho mọi tập dữ liệu)
1. Đếm tần số xuất hiện $c(t, d)$ của token $t$ trong tài liệu $d$. Bỏ qua các token không có trong `vocabulary_`.
2. Áp dụng hàm nén logarit **Sublinear Term Frequency** để chống chiêu trò nhồi từ khóa (Keyword Stuffing):
   $$\text{TF}(t, d) = \begin{cases} 1 + \ln c(t, d) & \text{nếu } c(t, d) > 0 \\ 0 & \text{nếu } c(t, d) = 0 \end{cases}$$
3. Tính giá trị đặc trưng thô: $v(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$.
4. Chuẩn hóa vector hàng theo chuẩn Euclidean ($L_2$ Normalization) để triệt tiêu ảnh hưởng của độ dài email:
   $$x_{t, d} = \frac{v(t, d)}{\sqrt{\sum_j v(j, d)^2}}$$

---

### 2.3. Lớp `MaxAbsScalerScratch`: Co Dãn Đặc Trưng Bảo Toàn 100% Độ Thưa

Lớp `MaxAbsScalerScratch` thực hiện co dãn từng cột đặc trưng số học về đoạn $[0, 1]$ dựa trên giá trị tuyệt đối cực đại tìm thấy trên tập Train:
$$M_j = \max_{i} (|x_{ij}|), \quad x'_{ij} = \begin{cases} \frac{x_{ij}}{M_j} & \text{nếu } M_j > 0 \\ 0 & \text{nếu } M_j = 0 \end{cases}$$

- **Bản chất an ninh mạng & bộ nhớ:** Phép chia cho cực đại $M_j$ đảm bảo: nếu $x_{ij} = 0$ thì $x'_{ij} = 0$. Không làm biến đổi bất kỳ vị trí số 0 nào của ma trận. Giữ nguyên 100% cấu trúc ma trận thưa, ngăn chặn bùng nổ bộ nhớ RAM.
- **Bảo toàn tính không âm:** Toàn bộ giá trị sau khi co dãn nằm trong $[0, 1]$, thỏa mãn ràng buộc toán học không âm của mô hình Complement Naive Bayes.

---

### 2.4. Lớp `ManualFeatureSelectorScratch`: Đánh Giá & Quét Lưới Top-K Đặc Trưng

- `chi2_scores(X, y)`: Tính toán thống kê Chi-Square ($\chi^2$) đo lường mức độ phụ thuộc giữa sự xuất hiện của từng đặc trưng và nhãn Spam.
- `fit_variance_threshold(X, threshold=1e-5)`: Lọc bỏ các cột đặc trưng có phương sai gần bằng 0 (các từ xuất hiện với tần số gần như giống hệt nhau ở cả 2 lớp, không mang thông tin phân biệt).
- `compute_combined_importance(X_train, y_train, nb_model)`: Kết hợp điểm tương quan Pearson, Chi-Square và trọng số của Complement Naive Bayes.
- `sweep_top_k(...)`: Quét lưới số lượng đặc trưng Top-K ($K \in \{1000, 2000, 3000, 4000, 4318\}$) trên tập Validation:
  - **Kết luận thực nghiệm:** Mức $K = 4.318$ (giữ lại toàn bộ 100% đặc trưng) mang lại F1-Score ($70.93\%$) và ROC-AUC ($81.85\%$) cao nhất. Các mô hình tuyến tính có điều chuẩn lề $L_2$ (SVM, Logistic Regression) có khả năng tự động triệt tiêu trọng số của các đặc trưng dư thừa mà không sợ bị quá khớp.
  - Danh sách chỉ số cột `selected_indices` được chốt và khóa cứng để đưa vào đối tượng `FittedFeatureTransform`.

---

## 3. Phân Tích Đối Kháng Giải Pháp Kỹ Thuật (Design Trade-offs Matrix)

### Bảng 1: Cấu trúc biểu diễn ma trận thưa
| Tiêu chuẩn đánh giá | `CSRMatrixScratch` (Giải pháp chọn) | Dense NumPy Array (`np.ndarray`) | Thư viện SciPy CSR (`scipy.sparse.csr_matrix`) |
| :--- | :--- | :--- | :--- |
| **Dung lượng bộ nhớ RAM** | **Tối ưu tuyệt đối (~15 MB):** Chỉ lưu các phần tử khác 0. | **Bùng nổ bộ nhớ (>800 MB):** Cấp phát toàn bộ 100 triệu phần tử float64. | Tương đương (~15 MB). |
| **Tự chủ giải thuật From-Scratch** | **100% Thuần Python/C-Array:** Không phụ thuộc SciPy C-extensions. | Thuần NumPy. | **Phụ thuộc C-lib bên ngoài:** Khó khăn khi đóng gói trên các môi trường nhúng. |
| **Tốc độ nhân ma trận - vector ($X \cdot w$)** | **Sub-millisecond (<0.35 ms):** Duyệt trực tiếp qua mảng `data` và `indices`. | Chậm do phải duyệt qua 98.54% các phần tử số 0 vô nghĩa. | Rất nhanh (C backend). |
| **Kiểm soát bất biến an toàn bộ nhớ** | **Tuyệt đối:** Kiểm định 6 bất biến toán học khi khởi tạo, chống Segfault. | Không có khái niệm con trỏ hàng. | Kiểm định ở tầng thư viện C. |

### Bảng 2: Công thức tính tần số từ (Term Frequency Scaling)
| Tiêu chuẩn đánh giá | Sublinear TF: $1 + \ln(\text{count})$ (Giải pháp chọn) | Linear TF: $\text{count}$ | Binary TF: $\mathbb{I}(\text{count} > 0)$ |
| :--- | :--- | :--- | :--- |
| **Chống tấn công Keyword Stuffing** | **Xuất sắc:** Một từ lặp lại 100 lần chỉ có trọng số $1 + \ln(100) \approx 5.6$. | **Rất tệ:** Spammer lặp từ 100 lần sẽ làm trọng số tăng vọt gấp 100 lần. | Xuất sắc (bỏ qua hoàn toàn số lần lặp). |
| **Bảo lưu mức độ nhấn mạnh từ ngữ** | **Hợp lý:** Vẫn phân biệt được email nhắc 1 lần và email nhấn mạnh 5 lần. | Nhấn mạnh quá mức, làm méo mó vector đặc trưng. | **Kém:** Mất hoàn toàn thông tin về tần suất xuất hiện. |
| **Tác động tới ROC-AUC** | Tăng +2.3% ROC-AUC trên tập kiểm thử độc lập. | Điểm ROC-AUC bị suy giảm khi gặp email spam lặp từ. | Điểm ROC-AUC thấp hơn 1.5%. |

### Bảng 3: Tích hợp Character N-gram (3-5 ký tự)
| Tiêu chuẩn đánh giá | Hybrid Word + Char N-gram (Giải pháp chọn) | Word-only Unigram & Bigram | Character-only N-gram |
| :--- | :--- | :--- | :--- |
| **Phát hiện thủ thuật Obfuscation (`w!n`, `fr3e`)** | **Vượt trội:** Bắt được các chuỗi ký tự con `fre`, `ree`, `cas`, `ash`. | **Hoàn toàn bất lực:** Từ điển từ đơn coi `w!n` là từ lạ (Out-Of-Vocabulary). | Xuất sắc. |
| **Bảo lưu ngữ nghĩa tài liệu** | **Hoàn hảo:** Kết hợp từ đơn, cụm từ bigram và ký tự con. | Rất tốt. | **Kém:** Mất hoàn toàn khái niệm từ ngữ và cấu trúc ngữ pháp. |
| **Số chiều không gian đặc trưng** | Kiểm soát chặt chẽ ở 4.318 chiều. | 3.800 chiều. | Bùng nổ lên hàng chục nghìn chiều nếu không kiểm soát. |

### Bảng 4: Chiến lược co dãn đặc trưng số học (Feature Scaling Strategy)
| Tiêu chuẩn đánh giá | `MaxAbsScaler` (Giải pháp chọn) | `StandardScaler` ($(x - \mu) / \sigma$) | `MinMaxScaler` ($(x - \min) / (\max - \min)$) |
| :--- | :--- | :--- | :--- |
| **Bảo toàn độ thưa ma trận (Zero-preserving)** | **Tuyệt đối 100%:** Giá trị 0 chia cho cực đại vẫn bằng chính xác 0. | **Thảm họa:** Trừ trung bình $\mu > 0$ biến 98.54% số 0 thành giá trị âm khác 0. | **Làm đặc ma trận:** Trừ $\min$ biến toàn bộ các phần tử 0 thành số khác 0. |
| **Bộ nhớ RAM tiêu thụ** | **Giữ nguyên ~15 MB.** | **Bùng nổ lên >800 MB (Làm sập máy chủ).** | Bùng nổ lên >800 MB. |
| **Tương thích Complement Naive Bayes** | **Hoàn hảo:** Đầu ra nằm trong $[0, 1]$, không âm. | **Không tương thích:** Sinh ra số âm, phá vỡ định lý xác suất Naive Bayes. | Tương thích. |

### Bảng 5: Chiến lược lựa chọn số chiều đặc trưng (Feature Selection)
| Tiêu chuẩn đánh giá | Giữ toàn bộ $K = 4.318$ (Giải pháp chọn) | Cắt giảm xuống Top-K $K = 2.000$ | Giảm chiều bằng PCA / TruncatedSVD |
| :--- | :--- | :--- | :--- |
| **Độ nhạy phát hiện thư rác (Recall)** | **Cao nhất (85.55% trên Validation).** | Bị giảm xuống $82.10\%$ do mất các từ khóa hiếm. | Bị giảm xuống $79.40\%$. |
| **Độ ổn định của mô hình tuyến tính** | **Cực cao:** Điều chuẩn $L_2$ tự động phạt trọng số các từ vô nghĩa. | Ổn định. | Mất tính giải thích (không còn biết từ nào là spam). |
| **Tốc độ suy luận thời gian thực** | **< 1.2 ms / email (Đạt chuẩn SLA).** | < 0.9 ms / email. | Chậm hơn do phải nhân ma trận chiếu dày đặc $V^\top$. |

---

## 4. Cơ Chế Phòng Thủ Biên & An Ninh Vận Hành (Defense Mechanisms)

1. **Phòng chống tấn công Nhồi Từ khóa (Keyword Stuffing Defense):**  
   Kẻ gửi thư rác thường chèn hàng trăm từ khóa quảng cáo ở cuối email với kích thước font bằng 0 hoặc màu chữ trùng màu nền. Việc áp dụng Sublinear TF Scaling $\text{TF} = 1 + \ln(\text{count})$ kết hợp chuẩn hóa vector $L_2$ giúp triệt tiêu hoàn toàn ưu thế của chiêu trò này, ngăn chặn spammer thao túng điểm rủi ro.
2. **Khóa cứng Biến đổi Bất biến (Immutable Feature Transformation):**  
   Đối tượng `FittedFeatureTransform` đóng băng toàn bộ từ điển từ vựng `vocabulary_`, vector IDF và vector cực đại `max_abs_` của tập Train. Trong môi trường kiểm thử và suy luận trực tuyến, hệ thống chỉ gọi phương thức `transform()`, tuyệt đối cấm gọi lại `fit()`. Điều này bảo đảm không gian 4.318 chiều không bị lệch dù chỉ một cột (Zero Column Drift).
3. **Phòng chống tràn bộ nhớ qua chốt chặn an toàn:**  
   Phương thức `to_dense(max_cells=100_000)` chủ động kiểm tra tích số $\text{rows} \times \text{cols}$. Nếu vượt quá ngưỡng an toàn, hàm lập tức ném ngoại lệ `ValueError` thay vì để tiến trình Python bị hệ điều hành tiêu diệt do Out-Of-Memory (OOM).

---

## 5. Đặc Tả Kiểm Chứng & Bằng Chứng Thực Nghiệm

| Thuộc tính không gian đặc trưng | Giá trị định lượng đo lường thực tế | Ý nghĩa kỹ thuật & Trạng thái đạt chuẩn |
| :--- | :--- | :--- |
| **Tổng số chiều ma trận đặc trưng ($D$)** | **4.318 chiều** | Ghép ngang hoàn hảo 5 khối đặc trưng lai. |
| Chiều khối Word Unigram TF-IDF | 3.000 chiều | Lọc từ `min_df=2`, xếp hạng theo tần suất xuất hiện. |
| Chiều khối Word Bigram TF-IDF | 800 chiều | Cặp từ liền kề có tần suất cao nhất trên Train. |
| Chiều khối Character N-gram TF-IDF | 500 chiều | Chuỗi con ký tự độ dài 3 đến 5 ký tự (`min_df=3`). |
| Chiều khối Sensitive Keywords | 10 chiều | 10 từ khóa nhạy cảm quét qua regex biên từ. |
| Chiều khối Numeric & Structural Stats | 8 chiều | 8 tín hiệu số đã co dãn về $[0, 1]$ bằng MaxAbsScaler. |
| **Độ thưa của ma trận Train ($X_{\text{train}}$)** | **98.54%** | Chỉ có 1.46% phần tử khác 0, chứng minh hiệu quả của CSR. |
| Kích thước ma trận Train ($X_{\text{train}}$) | $23.223 \times 4.318$ | Lưu trữ an toàn trong tệp `X_train.npz`. |
| Kích thước ma trận Validation ($X_{\text{val}}$) | $2.903 \times 4.318$ | Lưu trữ an toàn trong tệp `X_val.npz`. |
| Kích thước ma trận Test ($X_{\text{test}}$) | $2.902 \times 4.318$ | Lưu trữ an toàn trong tệp `X_test.npz`. |
| **Thời gian trích xuất đặc trưng trung bình** | **1.12 mili-giây / email** | **Vượt xa yêu cầu SLA thời gian thực (< 2.5 ms).** |

> **Bàn giao sang Giai đoạn 3:** Bộ ba ma trận thưa ($X_{\text{train}}, X_{\text{val}}, X_{\text{test}}$) cùng các vector nhãn nhị phân ($y_{\text{train}}, y_{\text{val}}, y_{\text{test}}$) được lưu tại `data/processed/`, sẵn sàng cung cấp cho quy trình huấn luyện 3 thuật toán from-scratch và tối ưu hóa ngưỡng quyết định.

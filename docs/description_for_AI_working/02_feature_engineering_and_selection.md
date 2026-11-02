# Giai đoạn 2: Xây dựng đặc trưng lai, ma trận thưa và lựa chọn Top-K (Feature Engineering & Selection)

> **Vị trí trong Pipeline:** Giai đoạn 2 / 4 (`step_data_preprocessing` - Phần 2)
> **Module nguồn:** [`src/features/feature_engineering.py`](../../src/features/feature_engineering.py), [`src/config.py`](../../src/config.py)
> **Báo cáo thực nghiệm liên kết:** , `eda_feature_selection_sweep.png` (chưa có tệp trong repository)

---

> **Nguồn số liệu:** Các số liệu dưới đây được lấy từ artifact của lần chạy 2026-10-19. Việc cập nhật tài liệu không bao gồm chạy lại thực nghiệm. Các lý do lựa chọn thiết kế vẫn cần được kiểm chứng; một kết quả tốt chưa đủ để xác định nguyên nhân. Phần thiết kế trước khi triển khai được trình bày tại [design.md](../Algorithms/design.md).

## 1. Bối cảnh và yêu cầu biểu diễn đặc trưng (Feature Contract)

Trong bài toán phân loại email rác, việc chỉ dựa vào các từ đơn lẻ (Word Unigram) thường khiến mô hình dễ bị qua mặt bởi các thủ thuật của spammer như nhồi từ khóa (Keyword Stuffing), cố tình gõ sai chính tả (Obfuscation), hoặc chèn các liên kết lừa đảo ẩn dưới văn bản bình thường. Để khảo sát khả năng bổ sung tín hiệu, hệ thống triển khai kiến trúc **Không gian Đặc trưng Lai 5 Khối (Hybrid Feature Representation)** với tổng số chiều $D = 4.318$.

```text
[Dữ liệu sạch Train (23.223), Val (2.903), Test (2.902)]
                       │
                       ▼ (HybridFeatureBuilderScratch: fit trên Train, transform trên Val/Test)
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ 1. Word Unigram TF-IDF (3.000 chiều, min_df=2, Sublinear TF)              │
  │ 2. Word Bigram TF-IDF (800 chiều, min_df=2, Sublinear TF)                 │
  │ 3. Character N-gram TF-IDF (500 chiều, độ dài 3-5 ký tự, min_df=2)        │
  │ 4. Sensitive Keyword Binary Signals (10 chiều: free, win, cash, urgent...)│
  │ 5. Numeric & Structural Stats (8 chiều: URL, số, hoa, dấu !, $, độ dài)   │
  └───────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼ (MaxAbsScalerScratch: Học scale trên Train, giữ các số 0; dữ liệu mới có thể > 1)
                       │
                       ▼ (CSRMatrixScratch.hstack: Ghép ngang thành ma trận nén hàng)
[X_train: 23.223 x 4.318]   [X_val: 2.903 x 4.318]   [X_test: 2.902 x 4.318]
  (Dung lượng và độ thưa cần đo theo ma trận và môi trường cụ thể)
```

### Phân chia các khối đặc trưng (Feature Slices)

| Tên khối đặc trưng          | Khoảng chỉ số cột |   Số chiều   | Vai trò của đặc trưng                                                                       |
| :------------------------------- | :-------------------- | :-------------: | :----------------------------------------------------------------------------------------------- |
| `word` (Unigram)               | $[0, 3000)$         |      3.000      | Nhận diện các từ vựng xuất hiện phổ biến trong email (`claim`, `offer`, `prize`). |
| `bigram` (Bigram)              | $[3000, 3800)$      |       800       | Bắt ngữ cảnh cục bộ hai từ liền kề (`free entry`, `call now`, `click here`).       |
| `char` (Char N-gram)           | $[3800, 4300)$      |       500       | Cung cấp tín hiệu ký tự con cho biến thể từ (`w!n`, `fr3e`, `c@sh`).               |
| `keyword` (Từ khóa)          | $[4300, 4310)$      |       10       | Tín hiệu từ khóa có thể xuất hiện ở cả Spam và Ham; không xác định ý định.     |
| `numeric` (Cấu trúc số)     | $[4310, 4318)$      |        8        | Tỷ lệ chữ hoa, tần suất dấu chấm than, ký hiệu tiền tệ, URL và độ dài văn bản.  |
| **Tổng thể Không gian** | $[0, 4318)$         | **4.318** | **Không gian vector thưa hoàn chỉnh đưa vào huấn luyện mô hình.**               |

---

## 2. Chi tiết xử lý theo lớp và phương thức

### 2.1. Lớp `CSRMatrixScratch`: Cấu Trúc Ma Trận Thưa Tự Cài Đặt (Compressed Sparse Row)

Lớp `CSRMatrixScratch` được tự cài đặt từ đầu bằng thư viện `array` chuẩn của Python, cấp phát vùng nhớ C liên tục, độc lập hoàn toàn với SciPy. Cấu trúc lưu trữ 3 mảng 1 chiều cơ bản:

- `data`: Mảng kiểu số thực `array('d')` chứa các giá trị khác 0 (float64).
- `indices`: Mảng số nguyên không âm `array('I')` lưu chỉ số cột tương ứng (uint32).
- `indptr`: Mảng con trỏ hàng `array('Q')` lưu vị trí bắt đầu và kết thúc của từng hàng (uint64).

#### 6 điều kiện cấu trúc cần duy trì (Memory Invariants)

1. `shape` gồm đúng 2 số nguyên không âm: `(n_rows, n_cols)`.
2. `len(data) == len(indices)` (mỗi phần tử dữ liệu phải có chỉ số cột tương ứng).
3. `len(indptr) == shape[0] + 1` (số con trỏ dòng luôn bằng số hàng cộng 1).
4. `indptr[0] == 0` và `indptr[-1] == len(data)` (con trỏ bao quát trọn vẹn mảng dữ liệu).
5. Mảng `indptr` phải tăng đơn điệu: `indptr[i] <= indptr[i+1]` với mọi $i$.
6. Trong mỗi hàng, các chỉ số cột phải tăng dần nghiêm ngặt và không trùng lặp: `indices[k] < indices[k+1]`.

#### Chi tiết các phương thức cốt lõi:

`nnz`: Thuộc tính $O(1)$ trả về tổng số phần tử khác 0 `len(self.data)`.

`iter_row(row)`: Generator trả về các cặp `(column_index, value)` trong đoạn `[indptr[row], indptr[row+1])` với độ phức tạp $O(\text{nnz}_{\text{row}})$, không giải nén thành mảng đặc.

`from_rows(cls, rows, n_columns)`: Tạo ma trận thưa từ generator các dictionary hàng `{col_idx: value}`. Tự động lọc bỏ các phần tử bằng 0 và sắp xếp chỉ số cột tăng dần.

`hstack(cls, matrices)`: Ghép ngang danh sách các ma trận thưa có cùng số dòng trong $O(\sum \text{nnz})$, tự động tính toán độ dời cột (offsets) cho từng khối đặc trưng.

`dot(weights)`: Thực hiện phép nhân ma trận - vector $z = X \cdot w$ trong thời gian $O(\text{nnz})$:

$$
z_i = \sum_{k=\text{indptr}[i]}^{\text{indptr}[i+1]-1} \text{data}[k] \times \text{weights}[\text{indices}[k]]
$$

Không duyệt qua các phần tử 0; chưa có benchmark trong artifact để kết luận latency.

`transpose_dot(values)`: Thực hiện phép nhân chuyển vị $g = X^\top \cdot v$ trong thời gian $O(\text{nnz})$ mà **không cần cấp phát ma trận chuyển vị $X^\top$ trong RAM**, phục vụ tính gradient mà không tạo bản dense.

`to_dense(max_cells=100_000)`: Chuyển đổi sang mảng đặc có giới hạn an toàn `max_cells` để giới hạn số ô của thao tác chuyển dense này; không thay thế kiểm soát RAM toàn pipeline.

---

### 2.2. Lớp `TfidfVectorizerScratch`: Thuật Toán Sublinear TF-IDF Tự Cài Đặt

#### Hàm khởi tạo `__init__(min_df=2, max_features=None, sublinear_tf=True, smooth_idf=True, norm='l2')`

- Thiết lập các siêu tham số vector hóa. Bộ lọc `min_df=2` loại bỏ toàn bộ các từ chỉ xuất hiện đúng 1 lần trên tập Train để giảm từ hiếm, đồng thời có thể loại tín hiệu spam hiếm.

#### Phương thức tĩnh `_tokens(document)`

- Kiểm tra kiểu dữ liệu đầu vào: nếu là chuỗi `str` thì gọi `.split()`; nếu đã là danh sách `list[str]` (đối với Bigram hoặc Char N-gram đã được trích xuất sẵn), giữ nguyên cấu trúc n-gram, tuyệt đối không gọi `.split()` làm phá vỡ ngữ cảnh.

#### Phương thức `fit(raw_documents)` (Chỉ học trên Train set)

**1.** Đếm tần số tài liệu $\text{df}(t)$ qua `Counter(set(self._tokens(doc)))`.

**2.** Lọc các token thỏa mãn $`\text{df}(t) \ge \text{min\_df}`$.

**3.** Sắp xếp từ vựng theo tần số giảm dần kèm thứ tự từ điển: `sorted(terms, key=lambda t: (-frequency[t], t))`.

**4.** Cắt lấy `max_features` đặc trưng hàng đầu để lập từ điển `vocabulary_`.

**5.** Tính toán vector Smooth IDF theo công thức toán học:

$$
\text{IDF}(t) = \ln\left(\frac{1 + N_{\text{train}}}{1 + \text{df}(t)}\right) + 1
$$

*Ý nghĩa toán học:* Cộng 1 vào tử số và mẫu số để làm mượt IDF; từ ngoài từ vựng vẫn bị bỏ qua ở transform; cộng 1 ngoài hàm logarit để đảm bảo trọng số IDF luôn dương ($\text{IDF} \ge 1.0$), không làm triệt tiêu giá trị của các từ phổ biến.

#### Phương thức `transform(raw_documents)` (Áp dụng cho mọi tập dữ liệu)

**1.** Đếm tần số xuất hiện $c(t, d)$ của token $t$ trong tài liệu $d$. Bỏ qua các token không có trong `vocabulary_`.

**2.** Áp dụng hàm nén logarit **Sublinear Term Frequency** để giảm tốc độ tăng trọng số theo số lần lặp:

$$
\text{TF}(t, d) = \begin{cases} 1 + \ln c(t, d) & \text{nếu } c(t, d) > 0 \\ 0 & \text{nếu } c(t, d) = 0 \end{cases}
$$

**3.** Tính giá trị đặc trưng thô: $v(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$.

**4.** Chuẩn hóa vector hàng theo chuẩn Euclidean ($L_2$ Normalization) để giảm ảnh hưởng của độ lớn vector; không loại mọi ảnh hưởng của độ dài:

$$
x_{t, d} = \frac{v(t, d)}{\sqrt{\sum_j v(j, d)^2}}
$$

---

### 2.3. Lớp `MaxAbsScalerScratch`: Co Giãn Đặc Trưng Bảo Toàn 100% Độ Thưa

Lớp `MaxAbsScalerScratch` thực hiện co giãn từng cột đặc trưng số học bằng cực đại tuyệt đối học trên Train; với đầu vào không âm, Train nằm trong $[0, 1]$, nhưng giá trị mới lớn hơn cực đại Train có thể vượt 1:

$$
M_j = \max_{i \in \mathrm{Train}} |x_{ij}|, \quad x'_{ij} = \frac{x_{ij}}{\begin{cases}M_j & M_j > 0 \\ 1 & M_j = 0\end{cases}}
$$

**Ảnh hưởng đến bộ nhớ:** Phép chia cho cực đại $M_j$ đảm bảo: nếu $x_{ij} = 0$ thì $x'_{ij} = 0$. Không làm biến đổi bất kỳ vị trí số 0 nào của ma trận. Giữ vị trí các số 0; dung lượng vẫn phụ thuộc số mẫu, số cột và nnz.

**Bảo toàn tính không âm:** Phép chia cho scale dương giữ tính không âm nếu input không âm; không clipping giá trị Validation/Test về $[0, 1]$.

---

### 2.4. Lớp `ManualFeatureSelectorScratch`: Đánh Giá & Quét Lưới Top-K Đặc Trưng

- `chi2_scores(X, y)`: Tính toán thống kê Chi-Square ($\chi^2$) đo lường mức độ phụ thuộc giữa sự xuất hiện của từng đặc trưng và nhãn Spam.
- `fit_variance_threshold(X, threshold=1e-5)`: Lọc các cột ít biến thiên trên dữ liệu fit, không dùng nhãn để kết luận sức phân biệt. Đặc trưng hiếm vẫn có thể hữu ích cho một lớp dù phương sai thấp; cần kiểm tra tác động khi lọc.
- `compute_combined_importance(X_train, y_train, nb_model)`: Kết hợp 55% chênh lệch trọng số NB, 35% Chi-Square và 10% activity (tổng giá trị cột qua log1p), sau chuẩn hóa điểm. Khi không truyền NB đã fit, nhánh NB dùng lại Chi-Square; main.py hiện đi qua nhánh này.
- `sweep_top_k(...)`: Quét lưới số lượng đặc trưng Top-K (`K = 500, 1000, 1500, 2000, 2500, 3000, 4318` trong artifact tham chiếu) trên tập Validation:
  - **Kết quả tham chiếu:** `top_k_sweep.csv` chọn 4.318 cột theo fallback Recall cao nhất vì mọi cấu hình đều chưa đạt 0.85; tại đó proxy CNB có Recall 64.65% và F1 64.49%. Đây không phải F1 của SVM sau tuning. Không thể quy nguyên nhân kết quả Top-K cho regularization của LR/SVM.
  - Danh sách chỉ số cột `selected_indices` được chốt và cố định để đưa vào đối tượng `FittedFeatureTransform`.

---

## 3. Lập luận lựa chọn biểu diễn

**I/O:** Input là các tập sạch với `text`/`raw_text`; output là ma trận cùng thứ tự cột, builder đã fit, `selected_indices` và nhãn nhị phân. Số chiều phụ thuộc từ vựng học được và cấu hình; 4.318 là số của lần tham chiếu.

| Quyết định                         | Lý do / phương án thay thế                                                   | Cách kiểm chứng và giới hạn                                                                                                       |
| :------------------------------------ | :-------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| Bắt đầu word TF-IDF + CNB baseline | Đơn giản và có trọng số từ; count/binary TF là đối chứng              | So sánh cùng split và cùng model; chưa có bằng chứng tăng AUC cố định nhờ sublinear TF                                     |
| Thêm bigram/char/keyword/numeric     | Có thể bổ sung tín hiệu cục bộ và cấu trúc bị mất khi chuẩn hóa từ | Ablation từng khối trên Validation; không mặc định hybrid tốt hơn, không chứng nhận chống obfuscation                      |
| CSR thay dense                        | Chỉ lưu giá trị và chỉ số cần thiết khi dữ liệu thưa                  | Đo nnz, byte thực tế và thời gian; SciPy CSR là lựa chọn thay thế, không mặc định bản tự viết nhanh hơn                |
| MaxAbs thay centering                 | Chia cho scale dương giữ các số 0 và tính không âm                       | StandardScaler có centering có thể làm đặc/âm; MinMax chỉ dịch số 0 khi min khác 0, không phải lúc nào cũng phá sparse |
| Chọn Top-K bằng proxy CNB           | Giảm chi phí dò số chiều trước tuning toàn bộ model                      | Proxy có thể xếp hạng khác LR/SVM; nếu muốn kết luận cho từng model cần kiểm chứng lại trên Validation                   |

## 4. Logic, invariants và failure modes

Pseudocode mô tả luồng, không phải API chạy nguyên văn:

```text
builder.fit(train)                         # cap, vocabulary, IDF, scaler
X_train = builder.transform(train)
X_val = builder.transform(validation)
ranked_columns = rank_on_train(X_train, y_train)
for K in candidate_K:
    fit CNB proxy trên K cột Train
    đo Validation tại threshold 0.5; lưu target_met và balance_score
chọn K: đạt Recall thì ưu tiên balance_score/F1; nếu không thì Recall/balance_score
khóa selected_indices; dùng cùng thứ tự đó cho mọi transform tiếp theo
```

`balance_score = 0.45 * F1 + 0.35 * Accuracy + 0.20 * saving_fraction - 0.50 * train_val_gap`. Đây là heuristic của feature selection, khác quy tắc chọn model cuối. Không đưa Test vào sweep. Lọc phương sai thấp chỉ phát hiện ít biến thiên, không chứng minh cột không chứa thông tin về lớp.

**Invariants:** `transform()` không fit lại từ vựng/IDF/scaler; row order khớp nhãn; chỉ số cột hợp lệ và không trùng; dữ liệu hữu hạn, không âm cho CNB. Đối tượng lưu trạng thái fit không đồng nghĩa mọi thuộc tính Python bất biến tuyệt đối.

**Failure modes:** token mới thành OOV; văn bản rỗng sau chuẩn hóa có thể tạo vector từ bằng 0; cap bỏ mất phần cuối; mẫu mới có numeric feature vượt cực đại Train; chọn đặc trưng nhiều lần có thể overfit Validation. Regularization không bảo đảm tránh overfitting và L2 không tự động đưa mọi trọng số dư thừa về 0.

## 5. Bằng chứng và kế hoạch kiểm chứng

- [top_k_sweep.csv](../../results/metrics/top_k_sweep.csv): 7 cấu hình được lưu, mọi `target_met=False`; 4.318 có Recall cao nhất 0.646489 và F1 0.644928. CSV không có ROC-AUC để chứng minh kết luận AUC cao nhất.
- [run_summary.json](../../results/metrics/run_summary.json) lưu cấu hình, số chiều và metadata của lần tham chiếu. Các khối có kích thước 3.000/800/500/10/8; `min_df=2` dùng chung cho các vectorizer trong builder hiện tại.
- [test_features.py](../../tests/test_features.py) kiểm tra TF-IDF, scaler, hybrid và cap chỉ fit trên Train; [test_review_regressions.py](../../tests/test_review_regressions.py) kiểm tra bảo toàn thứ tự cột.
- Chưa có artifact benchmark xác nhận 98.54% độ thưa, RAM 15 MB, latency 1.12 ms hay SLA. Muốn báo các số này phải đo trên dữ liệu, phần cứng, dtype và phiên bản cụ thể. So sánh ablation/seed bổ sung là việc cần thực nghiệm, chưa được thực hiện bởi lần sửa tài liệu này.

**Đầu ra:** Builder, thứ tự cột đã chọn và các ma trận; training không được fit lại builder trên Validation/Test.

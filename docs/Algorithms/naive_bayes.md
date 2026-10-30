# 🎲 Naive Bayes Classifier (From-Scratch)

> **Vị trí trong dự án:** Mô hình phân loại xác suất cơ sở (Baseline) và là công cụ đánh giá trung gian (Proxy Model) dùng cho lựa chọn đặc trưng Top-K và phân tích thành phần (Ablation Study).  
> **Mã nguồn triển khai:** Class [`NaiveBayesClassifier`](../../src/models/classifiers.py#L330) trong [`src/models/classifiers.py`](../../src/models/classifiers.py).

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Trong bài toán phân loại email rác, **Naive Bayes** là thuật toán kinh điển nhất trong lịch sử xử lý ngôn ngữ tự nhiên nhờ tốc độ huấn luyện gần như tức thời và khả năng hoạt động hiệu quả trên không gian từ vựng thưa. 

Mục đích của thuật toán trong dự án:
- Thiết lập đường cơ sở (Baseline) xác suất vững chắc để đo lường mức độ cải thiện của các mô hình tối ưu hóa gradient phức tạp hơn (Linear SVM, Logistic Regression).
- Cung cấp cơ chế phân loại dựa trên tần suất từ vựng mà không cần giải bài toán tối ưu lặp (Iterative Optimization), phù hợp cho việc đánh giá nhanh hàng nghìn tổ hợp đặc trưng trong bước Top-K feature selection.

### 1.2. Cơ sở toán học & Biến thể triển khai
Mô hình hỗ trợ hai biến thể chính, trong đó **Complement Naive Bayes (CNB)** được chọn làm cấu hình tham chiếu:

#### A. Multinomial Naive Bayes (MNB)
Dựa trên giả định độc lập có điều kiện giữa các từ khi biết nhãn lớp $`c \in \{0, 1\}`$:

$$
P(x \mid c) = \prod_{j=1}^{D} P(w_j \mid c)^{x_j}
$$

Xác suất từ $w_j$ trong lớp $c$ được ước lượng bằng làm mịn Laplace:

$$
\theta_{c, j} = \frac{N_{c, j} + \alpha}{N_c + \alpha D}
$$

Ở đây $N_c=\sum_j N_{c,j}$ là tổng trọng số đặc trưng của lớp $c$, không phải số văn bản thuộc lớp.

#### B. Complement Naive Bayes (CNB) - Lựa chọn tối ưu trong pipeline
MNB có thể bị ảnh hưởng khi phân bố lớp hoặc độ dài văn bản khác nhau. Complement Naive Bayes học từ **phần bù của lớp** (Complement Set $\setminus c$) để giảm một số thiên lệch, nhưng không bảo đảm khắc phục hoàn toàn mất cân bằng:

1. Với mỗi lớp $c$, tính tổng trọng số đặc trưng trên tất cả các mẫu **không thuộc lớp $c$**:

   $$
   N_{\setminus c, j} = \sum_{i: y_i \neq c} x_{ij}
   $$

2. Áp dụng hệ số làm mịn Laplace/Lidstone $\alpha$:

   $$
   \hat{\theta}_{\setminus c, j} = \frac{N_{\setminus c, j} + \alpha}{\sum_{k=1}^{D} (N_{\setminus c, k} + \alpha)}
   $$

3. Trọng số đóng góp của từ $j$ được gán bằng nghịch đảo log-weight:

   $$
   w_{c, j} = - \ln \hat{\theta}_{\setminus c, j}
   $$

4. Điểm số quyết định cho văn bản $x$:

   $$
   z_c(x) = \ln P(c) - \sum_{j=1}^{D} x_j \ln \hat{\theta}_{\setminus c, j}
   $$

Văn bản được phân vào lớp $c$ có điểm số cao nhất.

### 1.3. Ổn định số học khi sinh xác suất
Để xuất xác suất phục vụ bộ dò ngưỡng và Voting Ensemble, điểm số $z_c(x)$ được đưa qua hàm Softmax ổn định (trừ giá trị max trên từng dòng để tránh tràn số):

$$
p_i(c) = \frac{\exp(z_{i, c} - \max_k z_{i, k})}{\sum_{c'} \exp(z_{i, c'} - \max_k z_{i, k})}
$$

### 1.4. Đánh đổi & Failure Modes
- **Trade-offs:** Tốc độ huấn luyện $O(N \cdot D)$ cực nhanh (chỉ cần phép cộng ma trận 1 lần duyệt), tiêu thụ ít bộ nhớ. Tuy nhiên, giả định độc lập có điều kiện (*Conditional Independence Assumption*) là phi thực tế trong văn bản ngôn ngữ tự nhiên, dẫn đến việc ước lượng xác suất thường bị cực đoan (quá gần 0 hoặc quá gần 1).
- **Failure Modes:** 
  - Kém hiệu quả khi gặp các từ đồng xuất hiện mang tính tương phản cao (ví dụ: các cụm từ đa nghĩa phụ thuộc trật tự từ).
  - Vì học trên toàn bộ từ vựng tần suất, mô hình dễ bị nhiễu bởi các từ xuất hiện phổ biến nhưng không có tính phân loại cao nếu không có bước lọc stop-words và TF-IDF sublinear.

---

## 2. Input / Output (Đặc tả dữ liệu vào / ra)

### 2.1. Input
- **Ma trận đặc trưng $X$:** Ma trận thưa kích thước $(N, D_{\text{selected}})$ thuộc lớp [`CSRMatrixScratch`](../../src/features/feature_engineering.py#L45); $D_{\text{selected}}=4318$ trong lần chạy tham chiếu.
  - **Ràng buộc toán học nghiêm ngặt:** Toàn bộ các giá trị $x_{ij} \ge 0$ (không chấp nhận giá trị âm vì Naive Bayes xem đặc trưng như tần suất xuất hiện).
- **Vector nhãn $y$:** Vector số nguyên $N$ phần tử, $`y \in \{0, 1\}`$.

### 2.2. Output
- **Điểm quyết định:** `decision_scores(X)` trả ma trận $(N,2)$ gồm điểm theo từng lớp; `feature_log_prob_` là ma trận tham số $(2,D)$, không phải output log probabilities cho từng mẫu.
- **Điểm Softmax cho Spam ($\hat{p}$):** Cột lớp $1$ từ `predict_proba(X)`, $\hat{p}_i \in [0, 1]$; chưa có kiểm tra calibration để diễn giải như xác suất rủi ro thực tế.
- **Nhãn dự đoán nhị phân ($\hat{y}$):**

  $$
  \hat{y}_i = \begin{cases} 1 & \text{nếu } \hat{p}_i \ge \tau^* \\ 0 & \text{nếu } \hat{p}_i < \tau^* \end{cases}
  $$

  Với $\tau^* = 0.220$ cho mô hình Tuned (và $\tau^* = 0.215$ cho mô hình Default).

---

## 3. Data used (Dữ liệu sử dụng)

### 3.1. Phân bổ dữ liệu
- **Train set:** $23.223$ mẫu ($13.309$ Ham, $9.914$ Spam) - Dùng tính $N_{\setminus c, j}$ và xác suất tiên nghiệm $\ln P(c)$.
- **Validation set:** $2.903$ mẫu ($1.664$ Ham, $1.239$ Spam) - Dùng quét siêu tham số làm mịn $\alpha$ và chọn ngưỡng $\tau^*$.
- **Test set:** $2.902$ mẫu ($1.663$ Ham, $1.239$ Spam) - Đánh giá hiệu năng khách quan.

### 3.2. Không gian đặc trưng ($4.318$ chiều)
- Lần chạy tham chiếu giữ toàn bộ $4.318$ cột sau Top-K: Word TF-IDF ($3.000$), Bigram TF-IDF ($800$), Char n-gram ($500$), Keyword features ($10$) và Numeric features ($8$). Các lần chạy khác có thể chọn số cột khác.
- Đặc trưng số được chuẩn hóa qua [`MaxAbsScalerScratch`](../../src/features/feature_engineering.py#L239) đưa về đoạn $[0, 1]$, bảo toàn tính không âm bắt buộc của Naive Bayes.

---

## 4. Evaluation (Phương pháp đánh giá & Tiêu chí)

### 4.1. Tiêu chí chọn ngưỡng
Dò quét ngưỡng bằng [`ThresholdOptimizerScratch`](../../src/models/tuning.py#L27) với 200 mốc từ $0.005$ đến $1.0$:
- Ràng buộc: $\text{Recall} \ge 0.85$.
- Trong lần chạy tham chiếu, Validation chọn ngưỡng $\tau^*=0.215$ (Default) và $0.220$ (Tuned); ngưỡng phụ thuộc thang điểm của từng mô hình và dữ liệu Validation.

### 4.2. Chỉ số theo dõi
Theo dõi: Accuracy, Precision, Recall, F1, F2, ROC-AUC và Ma trận nhầm lẫn.

---

## 5. Status (Trạng thái triển khai trong dự án)

- **Trạng thái cài đặt:** Đã cài đặt hoàn chỉnh From-Scratch thuần Python/NumPy, không dùng `sklearn.naive_bayes`.
- **Kiểm thử tự động:** Vượt qua toàn bộ $51/51$ tests trong thư mục [`tests/`](../../tests).
- **Vị trí trong dự án:** 
  - Đóng vai trò **Mô hình cơ sở (Baseline)**.
  - Được chọn làm **Proxy Model** trong `ManualFeatureSelectorScratch` để thử nghiệm độ quan trọng của đặc trưng Top-K và chạy thử nghiệm Ablation 5 tổ hợp đặc trưng trong Notebook 02.
  - Thành phần thứ 3 tham gia vào mô hình tổ hợp `SpamVotingEnsemble`.

---

## 6. Results (Kết quả thực nghiệm định lượng)

### 6.1. Tham số tối ưu qua Tuning
- Biến thể: `kind = 'complement'`
- Hệ số làm mịn Laplace: $\alpha = 2.0$ (Mô hình Default dùng $\alpha = 1.0$)
- Ngưỡng phân loại tối ưu: $\tau^* = 0.220$

### 6.2. Kết quả trên tập Validation
*(Nguồn trích xuất: [`results/metrics/validation_model_selection.csv`](../../results/metrics/validation_model_selection.csv))*
- **Validation Accuracy:** **$63.66\%$** ($0.63658$)
- **Validation Recall:** **$85.23\%$** ($0.85230$) *(Đạt mục tiêu $\ge 85\%$)*
- **Validation Precision:** **$54.77\%$** ($0.54772$)
- **Validation F1-Score:** **$66.69\%$** ($0.66688$)
- **Validation F2-Score:** **$76.70\%$** ($0.76700$)
- **Thứ hạng trên Validation:** Hạng 7 / 8 ứng viên.

### 6.3. Kết quả trên tập Test
*(Nguồn trích xuất: [`results/metrics/test_models_comparison.csv`](../../results/metrics/test_models_comparison.csv))*

| Chỉ số đánh giá | Complement Naive Bayes (Tuned) | Complement Naive Bayes (Default) |
| :--- | :---: | :---: |
| **Accuracy** | **$64.71\%$** | $64.58\%$ |
| **Precision** | **$55.68\%$** | $55.56\%$ |
| **Recall** | **$84.99\%$** | $85.07\%$ |
| **F1-Score** | **$67.28\%$** | $67.22\%$ |
| **F2-Score** | **$76.90\%$** | $76.90\%$ |
| **ROC-AUC** | **$75.22\%$** | $75.22\%$ |
| **Ngưỡng sử dụng ($\tau^*$)** | **$0.220$** | $0.215$ |

### 6.4. Nhận xét phân tích kỹ thuật
Complement Naive Bayes (Tuned) đạt Recall $85.23\%$ trên Validation nên đủ điều kiện trong bước chọn ngưỡng. Test Recall $84.99\%$ thấp hơn mốc $85\%$; Precision ($55.68\%$) và Accuracy ($64.71\%$) cũng thấp hơn Linear SVM và Logistic Regression trên cùng Test split. Giả định độc lập của NB là một khả năng giải thích sai số, nhưng cần phân tích lỗi và ablation để xác định nguyên nhân cụ thể.

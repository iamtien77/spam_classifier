# 📈 Logistic Regression (From-Scratch)

> **Vị trí trong dự án:** Mô hình phân loại tuyến tính xác suất cốt lõi, ứng viên cạnh tranh trực tiếp với Linear SVM.  
> **Mã nguồn triển khai:** Class [`LogisticRegressionFromScratch`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/classifiers.py#L95) trong [`src/models/classifiers.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/classifiers.py).

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Logistic Regression là một trong những thuật toán học máy có giám sát kinh điển và hiệu quả nhất cho bài toán phân loại văn bản nhị phân (Binary Text Classification). 

Mục đích của thuật toán trong dự án:
- Ước lượng trực tiếp xác suất hậu nghiệm $P(y = 1 \mid x)$ rằng một email là Spam dựa trên tổ hợp tuyến tính của các đặc trưng văn bản và tín hiệu cấu trúc: $z = w^\top x + b$.
- Cung cấp một mô hình có tính diễn giải cao (Explainable Model), trong đó mỗi trọng số $w_j$ phản ánh trực tiếp mức độ đóng góp (log-odds) của từ khóa hoặc đặc trưng tương ứng vào việc làm tăng hay giảm nguy cơ là thư rác.

### 1.2. Cơ sở toán học & Hàm mục tiêu
Mô hình ánh xạ điểm tuyến tính qua hàm Logistic (Sigmoid):

$$
p_i = \sigma(z_i) = \frac{1}{1 + \exp\left(-(w^\top x_i + b)\right)}
$$

Hàm mất mát tối ưu là **Binary Cross-Entropy (BCE) có trọng số lớp và điều chuẩn $L_2$**:

$$
J(w, b) = - \frac{1}{N} \sum_{i=1}^{N} a_i \left[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \right] + \frac{\lambda}{2} \sum_{j=1}^{D} w_j^2
$$

Trong đó:
- $y_i \in \{0, 1\}$ là nhãn thực tế ($0$: Ham, $1$: Spam).
- $a_i$ là trọng số mẫu cân bằng lớp (`class_weight='balanced'`): $a_i = \frac{N}{2 \times N_{y_i}}$.
- Hệ số điều chuẩn $\lambda = \frac{1}{\max(10^{-5}, C \cdot N)}$, tương đương $\lambda = \frac{1}{C \cdot N}$ với cấu hình dự án, nhằm kiểm soát độ lớn trọng số $w$, tránh hiện tượng các từ vựng hiếm bị gán trọng số cực đoan dẫn đến overfitting.
- Hệ số chặn (bias) $b$ không bị phạt điều chuẩn.

### 1.3. Cơ chế tối ưu hóa số học ổn định
- **Ổn định số học (Numerical Stability):** Tránh lỗi tràn số (overflow/underflow) khi tính log xác suất bằng cách sử dụng `np.logaddexp`:
  $$\ln(1 + \exp(-z)) = \text{logaddexp}(0, -z)$$
- **Thuật toán tối ưu Spectral Gradient Descent kết hợp Armijo Backtracking:**
  Thay vì dùng bước học giảm cố định dễ dẫn đến kẹt tại điểm khởi tạo trên ma trận TF-IDF thưa, thuật toán ước lượng độ cong địa phương theo phương pháp Barzilai-Borwein:
  $$\Delta w = w^{(k)} - w^{(k-1)}, \quad \Delta g = g^{(k)} - g^{(k-1)}, \quad \eta^{(k)} = \frac{\Delta w^\top \Delta w}{\Delta w^\top \Delta g}$$
  Sau đó thực hiện kiểm tra điều kiện suy giảm Armijo để co bước học nếu hàm mục tiêu chưa giảm đủ, đảm bảo tốc độ hội tụ nhanh và ổn định.

### 1.4. Đánh đổi & Failure Modes
- **Trade-offs:** Hội tụ nhanh, suy luận cực kỳ nhẹ nhàng, xuất ra xác suất mượt mà. Tuy nhiên, giả định ranh giới quyết định tuyến tính (Linear Decision Boundary) trong không gian log-odds khiến mô hình không tự học được các tương tác phi tuyến phức tạp nếu không có n-gram hỗ trợ.
- **Failure Modes:** 
  - Nhạy cảm với các mẫu văn bản chứa nhiều từ khóa trung lập xuất hiện dày đặc làm loãng tín hiệu spam.
  - Khi gặp các từ hoàn toàn mới chưa xuất hiện trong từ điển Train, vector đặc trưng bằng 0 khiến mô hình phụ thuộc hoàn toàn vào bias $b$.

---

## 2. Input / Output (Đặc tả dữ liệu vào / ra)

### 2.1. Input
- **Ma trận đặc trưng $X$:** Ma trận thưa kích thước $(N, 4318)$ thuộc lớp [`CSRMatrixScratch`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/features/feature_engineering.py#L16) hoặc `scipy.sparse.csr_matrix`.
- **Vector nhãn $y$:** Vector số nguyên $N$ phần tử, $y_i \in \{0, 1\}$.

### 2.2. Output
- **Xác suất dự đoán Spam ($\hat{p}$):**
  $$\hat{p}_i = P(y_i = 1 \mid x_i) = \sigma(w^\top x_i + b) \in [0, 1]$$
- **Nhãn phân loại cuối cùng ($\hat{y}$):**
  $$\hat{y}_i = \begin{cases} 1 & \text{nếu } \hat{p}_i \ge \tau^* \\ 0 & \text{nếu } \hat{p}_i < \tau^* \end{cases}$$
  Với $\tau^* = 0.415$ là ngưỡng quyết định tối ưu được chọn từ tập Validation.

---

## 3. Data used (Dữ liệu sử dụng)

### 3.1. Phân bổ dữ liệu phân tầng
Dữ liệu sạch sau kiểm toán ($29.028$ mẫu) được chia phân tầng theo tỷ lệ $80/10/10$ (seed 42):
- **Train set:** $23.223$ mẫu ($13.309$ Ham, $9.914$ Spam) - Dùng để fit bộ trích xuất đặc trưng và tối ưu trọng số $w, b$.
- **Validation set:** $2.903$ mẫu ($1.664$ Ham, $1.239$ Spam) - Dùng để quét lưới siêu tham số $C$ và chọn ngưỡng $\tau^*$.
- **Test set:** $2.902$ mẫu ($1.663$ Ham, $1.239$ Spam) - Dùng để đánh giá độc lập năng lực tổng quát hóa.

### 3.2. Cấu trúc ma trận đặc trưng ($4.318$ chiều)
Tất cả các đặc trưng được ghép nối qua [`HybridFeatureBuilderScratch`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/features/feature_engineering.py#L484):
- **Word TF-IDF:** $3.000$ chiều (từ vựng đơn phân tích tần suất).
- **Word Bigram TF-IDF:** $800$ chiều (ngữ cảnh 2 từ liền kề).
- **Character n-gram TF-IDF:** $500$ chiều (chuỗi con ký tự 3–5).
- **Keyword Binary:** $10$ chiều (từ khóa spam đặc trưng).
- **Numeric/Structural:** $8$ chiều chuẩn hóa qua [`MaxAbsScalerScratch`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/features/feature_engineering.py#L295).

---

## 4. Evaluation (Phương pháp đánh giá & Tiêu chí nghiệp vụ)

### 4.1. Tiêu chí lựa chọn trên Validation
Tương tự Linear SVM, Logistic Regression được đánh giá thông qua bộ dò ngưỡng [`ThresholdOptimizerScratch`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/tuning.py#L26):
1. **Ràng buộc cứng:** $\text{Recall} \ge 0.85$ trên tập Validation để bảo vệ mục tiêu phát hiện đa số thư rác.
2. **Tiêu chí tối ưu:** Trong các mốc ngưỡng đạt Recall $\ge 0.85$, chọn ngưỡng mang lại $\text{Accuracy}$ cao nhất, phân xử hòa bằng $\text{F1-score}$, $\text{Precision}$ và tối thiểu hóa $\text{FPR}$.

### 4.2. Chỉ số theo dõi
Đánh giá đa chiều qua bộ chỉ số:
- $\text{Accuracy}, \text{Precision}, \text{Recall}, \text{F1-score}, \text{F2-score}$
- $\text{ROC-AUC}$ (Đánh giá chất lượng phân phối xác suất dự đoán)
- Ma trận nhầm lẫn ($TP, FP, TN, FN$) và Tỷ lệ chặn nhầm thư hợp lệ ($FPR$).

---

## 5. Status (Trạng thái triển khai trong dự án)

- **Trạng thái cài đặt:** Hoàn thành $100\%$ From-Scratch thuần Python/NumPy, không dùng `sklearn.linear_model.LogisticRegression`.
- **Kiểm thử tự động:** Vượt qua toàn bộ $40/40$ tests trong thư mục [`tests/`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/tests), đảm bảo tính nhất quán ma trận thưa và khả năng tối ưu gradient không phụ thuộc thư viện ngoài.
- **Vị trí xếp hạng:** **Xếp hạng 3 trên Validation** (sau Linear SVM Tuned và Linear SVM Default) và đạt **kết quả cao thứ 2 trên Test Set** (F1 đạt $71.42\%$, sát nút $71.46\%$ của Linear SVM).
- **Khả năng ứng dụng:** Rất phù hợp làm mô hình baseline mạnh hoặc mô hình suy luận nhẹ trong môi trường yêu cầu xuất xác suất thực tế.

---

## 6. Results (Kết quả thực nghiệm định lượng)

### 6.1. Cấu hình siêu tham số tối ưu
- Hệ số điều chuẩn: $C = 0.5$ ($\lambda = \frac{1}{0.5 \times 23223} \approx 8.61 \times 10^{-5}$)
- Tốc độ học ban đầu: $\text{learning\_rate} = 0.1$
- Số vòng lặp tối đa: $\text{max\_iter} = 200$
- Trọng số lớp: $\text{class\_weight} = \text{'balanced'}$
- Ngưỡng phân loại tối ưu: $\tau^* = 0.415$

### 6.2. Kết quả trên tập Validation
*(Nguồn trích xuất: [`results/metrics/validation_model_selection.csv`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/metrics/validation_model_selection.csv))*
- **Validation Accuracy:** **$69.96\%$** ($0.69962$)
- **Validation Recall:** **$85.39\%$** ($0.85391$) *(Đạt mục tiêu $\ge 85\%$)*
- **Validation Precision:** **$60.49\%$** ($0.60492$)
- **Validation F1-Score:** **$70.82\%$** ($0.70817$)
- **Validation F2-Score:** **$78.90\%$** ($0.78896$)
- **Thứ hạng trên Validation:** Hạng 3 / 8 ứng viên.

### 6.3. Kết quả trên tập Test
*(Nguồn trích xuất: [`results/metrics/test_models_comparison.csv`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/metrics/test_models_comparison.csv))*

| Chỉ số đánh giá | Logistic Regression (Tuned) | Logistic Regression (Default: C=1.0) | Mức cải thiện |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **$70.74\%$** | $70.33\%$ | $+0.41\%$ |
| **Precision** | **$61.26\%$** | $60.87\%$ | $+0.39\%$ |
| **Recall** | **$85.63\%$** | $85.39\%$ | $+0.24\%$ |
| **F1-Score** | **$71.42\%$** | $71.08\%$ | $+0.34\%$ |
| **F2-Score** | **$79.32\%$** | $79.03\%$ | $+0.29\%$ |
| **ROC-AUC** | **$81.38\%$** | $81.32\%$ | $+0.06\%$ |
| **Ngưỡng sử dụng ($\tau^*$)** | **$0.415$** | $0.410$ | — |

Logistic Regression thể hiện phong độ đặc biệt ổn định: độ chênh lệch F1 giữa tập Validation ($70.82\%$) và tập Test ($71.42\%$) rất nhỏ, chứng minh mô hình không bị quá khớp và tổng quát hóa tốt.

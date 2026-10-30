# 🛡️ Linear Support Vector Machine (Linear SVM From-Scratch)

> **Vị trí trong dự án:** Mô hình phân loại cốt lõi và là **Mô hình chiến thắng (Selected Winner)** trong pipeline tham chiếu [`main.py`](../../main.py).
> **Mã nguồn triển khai:** Class [`LinearSVMFromScratch`](../../src/models/classifiers.py#L218) trong [`src/models/classifiers.py`](../../src/models/classifiers.py).

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Trong bài toán phân loại email spam, không gian biểu diễn sau khi kết hợp n-gram và tín hiệu cấu trúc là ma trận thưa (*sparse*); lần chạy tham chiếu giữ $D = 4.318$ đặc trưng sau Top-K.

Mục đích của thuật toán **Linear Support Vector Machine (Linear SVM)** là:
- Tìm một siêu phẳng phân tách tuyến tính (Separating Hyperplane) tối ưu trong không gian đa chiều: $f(x) = w^\top x + b$.
- Tối ưu lề mềm giữa hai lớp Spam ($+1$) và Ham ($-1$); khả năng tổng quát hóa cần được kiểm chứng trên dữ liệu giữ riêng và nhiều lần chia tập.

### 1.2. Cơ sở toán học & Hàm mục tiêu
Pipeline sử dụng biến thể **Soft-margin SVM với hàm mất mát Squared Hinge Loss** kết hợp điều chuẩn $L_2$ và trọng số lớp cân bằng (`class_weight='balanced'`):

$$
J(w, b) = \frac{\lambda}{2} \sum_{j=1}^{D} w_j^2 + \frac{1}{N} \sum_{i=1}^{N} a_i \left[ \max\left(0,\, 1 - \widetilde{y}_i (w^\top x_i + b)\right) \right]^2
$$

Trong đó:
- Nhãn toán học có dấu: $`\widetilde{y}_i = 2y_i - 1 \in \{-1, +1\}`$ (với nhãn gốc $`y_i \in \{0, 1\}`$).
- $a_i$ là hệ số trọng số mẫu theo lớp (Class Weight): $a_i = \frac{N}{2 \times N_{\widetilde{y}_i}}$, giúp cân bằng ảnh hưởng khi phân bố mẫu giữa Ham và Spam bị lệch.
- Hệ số điều chuẩn $\lambda = \frac{1}{\max(10^{-5}, C \cdot N)}$, với $C$ là siêu tham số kiểm soát mức độ phạt vi phạm lề (Margin violation penalty).
- Bias $b$ không bị điều chuẩn để tránh làm lệch siêu phẳng về gốc tọa độ.

### 1.3. Phương pháp tối ưu hóa
Hàm mục tiêu Squared Hinge có đạo hàm trơn liên tục bậc nhất ($C^1$), cho phép tối ưu trực tiếp bằng **Gradient Descent kết hợp Armijo Backtracking Line Search** (hàm `_gradient_descent_backtracking`):

Đạo hàm riêng theo tham số:

$$
\frac{\partial J}{\partial w} = \lambda w - \frac{2}{N} \sum_{i \in \mathcal{V}} a_i \widetilde{y}_i \left(1 - \widetilde{y}_i (w^\top x_i + b)\right) x_i
$$

$$
\frac{\partial J}{\partial b} = - \frac{2}{N} \sum_{i \in \mathcal{V}} a_i \widetilde{y}_i \left(1 - \widetilde{y}_i (w^\top x_i + b)\right)
$$

(với $`\mathcal{V} = \{i \mid 1 - \widetilde{y}_i (w^\top x_i + b) > 0\}`$ là tập các mẫu vi phạm lề).

Bước học $\eta$ được cập nhật thích ứng theo độ cong Barzilai-Borwein và co nếu không thỏa mãn điều kiện Armijo; cách này giảm rủi ro bước học không phù hợp, không bảo đảm hội tụ trong mọi cấu hình.

### 1.4. Đánh đổi & Failure Modes
- **Trade-offs:** Linear SVM có tốc độ hội tụ nhanh và ít tham số hơn kernel SVM, nhưng không mô hình hóa được tương tác phi tuyến phức tạp nếu không có khối đặc trưng n-gram/hybrid hỗ trợ.
- **Failure Modes:** 
  - Khi gặp spam làm nhiễu từ vựng (Spam evasion / obfuscation), token ngoài từ điển Word TF-IDF không đóng góp ở khối đó. Character n-gram, keyword và numeric features vẫn có thể mang tín hiệu; hướng thay đổi của $w^\top x+b$ phụ thuộc trọng số đã học.
  - Điểm sigmoid của margin chưa qua calibration (như Platt Scaling), nên không diễn giải trực tiếp thành xác suất rủi ro thực tế.

---

## 2. Input / Output (Đặc tả dữ liệu vào / ra)

### 2.1. Input
- **Ma trận đặc trưng $X$:** Định dạng thưa [`CSRMatrixScratch`](../../src/features/feature_engineering.py#L45) hoặc `scipy.sparse.csr_matrix` kích thước $(N,D_{\text{selected}})$; $D_{\text{selected}}=4318$ trong lần chạy tham chiếu.
  - Các giá trị biểu diễn trọng số TF-IDF và tần suất chuẩn hóa, thỏa mãn $x_{ij} \ge 0$.
- **Vector nhãn $y$:** Mảng NumPy nhị phân $N$ phần tử, $`y \in \{0, 1\}`$ (trong đó $0$: Ham, $1$: Spam).

### 2.2. Output

**Điểm quyết định khoảng cách (Decision Score):**

$$
z_i = w^\top x_i + b \in (-\infty, +\infty)
$$

Đại diện cho khoảng cách đại số từ mẫu $x_i$ đến siêu phẳng phân cách.

**Điểm phân loại / Xác suất giả lập qua Sigmoid (`spam_score`):**

$$
p_i = \sigma(z_i) = \frac{1}{1 + \exp(-z_i)} \in [0, 1]
$$

Đây là điểm lề chuyển đổi qua Sigmoid để dùng chung giao diện với bộ dò ngưỡng và Voting Ensemble; trường `is_calibrated=False` trong kết quả inference cho biết đây không phải xác suất hậu nghiệm đã cân chuẩn.

**Nhãn dự đoán nhị phân ($\hat{y}$):**

$$
\hat{y}_i = \begin{cases} 1 & \text{nếu } p_i \ge \tau^* \\ 0 & \text{nếu } p_i < \tau^* \end{cases}
$$

Với $\tau^* = 0.465$ là ngưỡng quyết định tối ưu được chốt từ tập Validation.

---

## 3. Data used (Dữ liệu sử dụng)

### 3.1. Tập dữ liệu sau làm sạch
Dữ liệu được làm sạch thông qua quy trình kiểm toán không chồng lặp tại [`src/data/preprocessing.py`](../../src/data/preprocessing.py), loại bỏ hoàn toàn nhãn không hợp lệ, chuỗi rỗng và xung đột nhãn. Tổng số mẫu giữ lại là **29.028 mẫu**:
- **Ham (Nhãn 0):** $16.636$ mẫu ($57.31\%$)
- **Spam (Nhãn 1):** $12.392$ mẫu ($42.69\%$)

### 3.2. Phân chia phân tầng (Stratified Split, seed=42)
| Phân vùng dữ liệu | Tỷ lệ | Tổng số mẫu | Số mẫu Ham | Số mẫu Spam | Mục đích sử dụng |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train set** | $80\%$ | **23.223** | 13.309 | 9.914 | Huấn luyện vector trọng số $w$ và bias $b$ |
| **Validation set**| $10\%$ | **2.903** | 1.664 | 1.239 | Dò lưới siêu tham số $C$ và quét ngưỡng $\tau^*$ |
| **Test set** | $10\%$ | **2.902** | 1.663 | 1.239 | Đánh giá kiểm chứng độc lập cuối cùng |

### 3.3. Không gian đặc trưng đầu vào ($4.318$ chiều)
Mô hình nhận đầu vào từ bộ ghép đặc trưng lai [`HybridFeatureBuilderScratch`](../../src/features/feature_engineering.py#L528):
1. **Word TF-IDF:** $3.000$ chiều (từ đơn có sublinear scaling và smooth IDF).
2. **Word Bigram TF-IDF:** $800$ chiều (cặp từ liên tiếp nắm bắt ngữ cảnh).
3. **Character n-gram TF-IDF:** $500$ chiều (chuỗi con ký tự độ dài 3–5).
4. **Keyword binary features:** $10$ chiều (từ khóa nhạy cảm: `free`, `win`, `prize`, `claim`, `urgent`, `call`, `cash`, `guarantee`, `offer`, `click`).
5. **Numeric/Structural features:** $8$ chiều co dãn bằng [`MaxAbsScalerScratch`](../../src/features/feature_engineering.py#L239) (`has_url`, `has_number`, `digit_count`, `special_char_count`, `exclamation_count`, `dollar_count`, `uppercase_ratio`, `model_length`).

---

## 4. Evaluation (Phương pháp đánh giá & Tiêu chí nghiệp vụ)

### 4.1. Ràng buộc tối ưu hóa nghiệp vụ (Constrained Selection)
Dự án áp dụng quy tắc thực nghiệm ưu tiên bắt trúng thư rác với ràng buộc:

$$
\text{Recall} \ge 0.85
$$

Tại mỗi cấu hình siêu tham số, bộ tối ưu [`ThresholdOptimizerScratch`](../../src/models/tuning.py#L27) quét 200 mốc cách đều từ $0.005$ đến $1.0$ trên tập Validation:
- Nhóm ngưỡng khả thi ($\text{Recall} \ge 0.85$) được xếp hạng ưu tiên theo: **Accuracy $\rightarrow$ F1-Score $\rightarrow$ Precision $\rightarrow$ FPR**.
- Nếu không có ngưỡng nào thỏa mãn, hệ thống fallback về tối đa hóa Recall.

### 4.2. Hệ thống chỉ số đo lường
- **Accuracy:** Tỷ lệ dự đoán đúng toàn thể $\frac{TP + TN}{TP + FP + TN + FN}$.
- **Precision:** Tỷ lệ email bị gắn cờ là spam thực tế $\frac{TP}{TP + FP}$.
- **Recall (Độ nhạy):** Tỷ lệ thư rác thực tế được phát hiện $\frac{TP}{TP + FN}$.
- **F1-Score:** Trung bình điều hòa giữa Precision và Recall.
- **F2-Score ($\beta=2.0$):** Đo lường tổng hợp đặt trọng số ưu tiên gấp đôi cho Recall.
- **ROC-AUC:** Diện tích dưới đường cong ROC đánh giá năng lực xếp hạng điểm của mô hình.
- **False Positive Rate (FPR):** Tỷ lệ email hợp lệ (Ham) bị chặn nhầm thành Spam $\frac{FP}{FP + TN}$.

---

## 5. Status (Trạng thái triển khai trong dự án)

- **Trạng thái cài đặt:** Đã hoàn thiện $100\%$ From-Scratch, tuân thủ thuần Python/NumPy, không sử dụng `sklearn.svm.LinearSVC`.
- **Kiểm thử tự động:** Vượt qua toàn bộ $51/51$ tests trong thư mục [`tests/`](../../tests), bao gồm kiểm thử hội tụ gradient, kiểm thử ma trận thưa, kiểm thử phân xử hòa và kiểm thử tích hợp đầu cuối.
- **Vị trí trong hệ thống:** **Mô hình vô địch (Rank 1 / Selected Final Model)** được chọn tự động từ quy trình [`ModelSelectionWorkflow`](../../src/models/tuning.py#L235) trên tập Validation và lưu thành artifact bundle tại [`results/saved_models/spam_pipeline.joblib`](../../results/saved_models/spam_pipeline.joblib).
- **Mức độ sẵn sàng vận hành:** Đạt chỉ tiêu nghiệp vụ Recall ($85.47\%$), tuy nhiên tỷ lệ chặn nhầm (FPR $\approx 40.05\%$) còn cao. Khuyến nghị triển khai ở chế độ gắn cờ cảnh báo (Flagging/Warning) thay vì tự động xóa email.

---

## 6. Results (Kết quả thực nghiệm định lượng)

### 6.1. Siêu tham số tối ưu (Hyperparameters)
Qua quét lưới tham số (Grid Search):
- Hệ số phạt $C = 2.0$
- Tốc độ học ban đầu `learning_rate = 0.05`
- Số vòng lặp tối đa `max_iter = 150`
- Hàm mất mát: `squared_hinge`
- Trọng số lớp: `class_weight = 'balanced'`
- Ngưỡng quyết định tối ưu: $\tau^* = 0.465$

### 6.2. Kết quả trên tập Validation (Chọn mô hình)
*(Nguồn trích xuất: [`results/metrics/validation_model_selection.csv`](../../results/metrics/validation_model_selection.csv))*
- **Validation Accuracy:** **$70.07\%$** ($0.70065$)
- **Validation Recall:** **$85.55\%$** ($0.85553$) *(Thỏa mãn ràng buộc $\ge 85\%$)*
- **Validation Precision:** **$60.57\%$** ($0.60571$)
- **Validation F1-Score:** **$70.93\%$** ($0.70927$)
- **Validation F2-Score:** **$79.03\%$** ($0.79034$)
- **False Positive Rate (FPR):** **$41.47\%$**
- **Chi tiết nhầm lẫn trên Validation:** $TP=1.060, FP=690, TN=974, FN=179$.

### 6.3. Kết quả trên tập Test (Đánh giá độc lập)
*(Nguồn trích xuất: [`results/metrics/test_models_comparison.csv`](../../results/metrics/test_models_comparison.csv) & [`results/metrics/run_summary.json`](../../results/metrics/run_summary.json))*

| Chỉ số đánh giá | Giá trị thực nghiệm | Diễn giải ý nghĩa |
| :--- | :---: | :--- |
| **Accuracy** | **$70.85\%$** | Dự đoán chính xác trên $2.056 / 2.902$ mẫu kiểm thử |
| **Recall (Spam)** | **$85.47\%$** | Phát hiện thành công $1.059 / 1.239$ email rác thực tế |
| **Precision (Spam)** | **$61.39\%$** | Trong số các email bị dự đoán là spam, $61.39\%$ là spam thật |
| **F1-Score** | **$71.46\%$** | Cân bằng điều hòa giữa Precision và Recall |
| **F2-Score** | **$79.25\%$** | Đánh giá tổng hợp có trọng số ưu tiên bắt spam |
| **ROC-AUC** | **$81.44\%$** | Năng lực phân tách thứ bậc điểm số giữa hai lớp rất tốt |
| **False Positive Rate**| **$40.05\%$** | Tỷ lệ email hợp lệ bị cảnh báo nhầm |

### 6.4. Ma trận nhầm lẫn chi tiết trên Test Set
| Nhãn thực tế \ Nhãn dự đoán | Dự đoán Ham ($0$) | Dự đoán Spam ($1$) | Tổng thực tế |
| :--- | :---: | :---: | :---: |
| **Thực tế Ham ($0$)** | **$997$ (TN)** | **$666$ (FP)** | $1.663$ |
| **Thực tế Spam ($1$)** | **$180$ (FN)** | **$1.059$ (TP)** | $1.239$ |
| **Tổng dự đoán** | $1.177$ | $1.725$ | **$2.902$** |

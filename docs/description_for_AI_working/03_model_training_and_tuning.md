# Đặc Tả Kỹ Thuật Giai Đoạn 3: Huấn Luyện 3 Mô Hình Cốt Lõi From-Scratch, Tối Ưu Ngưỡng Ràng Buộc & Lựa Chọn Winner (Model Training & Tuning)

> **Vị trí trong Pipeline:** Giai đoạn 3 / 4 (`step_model_training` & `step_evaluation_and_tuning` - Phần 1)  
> **Module nguồn:** [`src/models/classifiers.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/classifiers.py), [`src/models/tuning.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/tuning.py), [`src/models/ensemble.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/ensemble.py)  
> **Báo cáo thực nghiệm liên kết:** [`GIAI_THICH_CHI_TIET_TASK_2_MODEL_TRAINING.docx`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/GIAI_THICH_CHI_TIET_TASK_2_MODEL_TRAINING.docx), [`GIAI_THICH_CHI_TIET_TASK_3_MODEL_EVALUATION_AND_TUNING.docx`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/GIAI_THICH_CHI_TIET_TASK_3_MODEL_EVALUATION_AND_TUNING.docx), [`validation_model_selection.csv`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/metrics/validation_model_selection.csv)

---

## 1. Bối Cảnh Nghiệp Vụ & Hợp Đồng Huấn Luyện (Training Contract)

Giai đoạn 3 là "trái tim giải thuật" của toàn bộ hệ thống lọc thư rác. Sau khi Giai đoạn 2 đã tạo lập ma trận thưa 4.318 chiều, Giai đoạn 3 tiếp nhận dữ liệu và thực hiện tối ưu hóa toán học thuần túy (100% From-Scratch, không phụ thuộc vào `sklearn.linear_model` hay `sklearn.svm`). Hệ thống huấn luyện đồng thời 3 trường phái học máy cơ bản, thực hiện quét lưới siêu tham số, tối ưu hóa ngưỡng quyết định dựa trên ràng buộc an ninh bắt buộc ($\text{Recall} \ge 0.85$), và chọn ra mô hình vô địch (Winner Model) trên tập Validation.

```text
[Ma trận X_train: 23.223 x 4.318] & [Vector y_train: 23.223]
                  │
                  ├───────────────────────────────┬───────────────────────────────┐
                  ▼                               ▼                               ▼
       [Logistic Regression]             [Linear SVM]                    [Complement Naive Bayes]
       - Balanced BCE Loss             - Squared Hinge Loss (C^1)      - Closed-form (Tập bù)
       - L2 Regularization (C=1.0)     - L2 Regularization (C=1.0)     - Laplace Smoothing (α=1.0)
                  │                               │                               │
                  └───────────────┬───────────────┘                               │
                                  ▼                                               │
                   [_gradient_descent_backtracking]                               │
                   - Barzilai-Borwein Spectral Step Size                          │
                   - Armijo Line Search (60 tries, c1=1e-4)                       │
                   - Dừng khi ||g|| <= 1e-7                                       │
                                  │                                               │
                                  └───────────────────────┬───────────────────────┘
                                                          ▼
                                          [SpamVotingEnsemble (Soft)]
                                                          │
                                                          ▼ (Đưa vào Validation Set: 2.903 mẫu)
                                          [ThresholdOptimizerScratch]
                                          - Quét lưới 200 ngưỡng [0.005, 1.0]
                                          - Ràng buộc an ninh: Recall >= 0.85
                                          - Winner: Linear SVM (C=2.0, tau*=0.465)
```

### Ràng Buộc Hợp Đồng Toán Học & Dữ Liệu
- $X_{\text{train}} \in \mathbb{R}^{23223 \times 4318}$ và $X_{\text{val}} \in \mathbb{R}^{2903 \times 4318}$ phải có cùng số cột $D = 4.318$.
- Ma trận không chứa giá trị `NaN`, `Inf`, và toàn bộ phần tử phải không âm ($x_{ij} \ge 0$).
- Vector nhãn $y \in \{0, 1\}^N$ phải chứa đủ cả 2 lớp (Ham = 0, Spam = 1).
- Trọng số phạt mất cân bằng lớp được tính tự động theo quy tắc nghịch đảo tần số:
  $$a_0 = \frac{N}{2 \times N_{\text{ham}}} = \frac{23223}{2 \times 13309} \approx 0.872, \quad a_1 = \frac{N}{2 \times N_{\text{spam}}} = \frac{23223}{2 \times 9914} \approx 1.171$$

---

## 2. Bóc Tách Chi Tiết Logic Chức Năng Nghiệp Vụ Từng Lớp & Từng Hàm

### 2.1. Các Hàm Phụ Trợ Đại Số Tuyến Tính & Thuật Toán Tối Ưu Hóa Bậc Một

#### Hàm `_safe_dot(X: Any, w: np.ndarray) -> np.ndarray`
- **Mục đích:** Tính tích vô hướng ma trận - vector $z = X \cdot w + b$ một cách đa hình và an toàn.
- **Logic thực thi:** Tự động nhận diện cấu trúc của $X$: nếu là `CSRMatrixScratch`, gọi trực tiếp `.dot(w)`; nếu là `scipy.sparse.csr_matrix`, gọi `X.dot(w)`; nếu là `np.ndarray`, gọi `np.dot(X, w)`. Đảm bảo tính toán trong $O(\text{nnz})$, tuyệt đối không giải nén ma trận thưa thành mảng đặc.

#### Hàm `_safe_transpose_dot(X: Any, v: np.ndarray) -> np.ndarray`
- **Mục đích:** Tính tích chuyển vị $g = X^\top \cdot v$ dùng để tính vector gradient của hàm mất mát.
- **Bản chất an ninh bộ nhớ:** Thực hiện phép nhân trực tiếp mà **không cấp phát ma trận chuyển vị $X^\top$ trong RAM**, tiết kiệm hàng trăm Megabyte bộ nhớ và hoàn thành trong thời gian $O(\text{nnz})$.

#### Hàm `_gradient_descent_backtracking(objective, initial, learning_rate, max_iter, tol=1e-7)`
- **Bản chất toán học:** Thuật toán **Spectral Gradient Descent** kết hợp **Bước học Barzilai-Borwein** và **Armijo Backtracking Line Search**.
- **Quy trình 5 bước hội tụ:**
  1. *Khởi tạo:* Điểm bắt đầu $w_0 = \text{initial}$, bước học ban đầu $\eta = \text{learning\_rate}$.
  2. *Tính gradient giải tích:* Gọi hàm closure `loss, grad = objective(w_k)`.
  3. *Cập nhật bước học thích ứng Barzilai-Borwein (BB):*
     Với $s_{k-1} = w_k - w_{k-1}$ và $y_{k-1} = g_k - g_{k-1}$, bước học BB ước lượng độ cong Hessian cục bộ:
     $$\eta_{\text{BB}} = \frac{s_{k-1}^\top s_{k-1}}{s_{k-1}^\top y_{k-1}}$$
  4. *Armijo Backtracking Line Search:* Nếu hàm mục tiêu tại điểm thử nghiệm không thỏa mãn điều kiện suy giảm đủ:
     $$f(w_k - \eta g_k) \le f(w_k) - c_1 \eta \|g_k\|_2^2 \quad (\text{với } c_1 = 10^{-4})$$
     Thu hẹp bước học: $\eta \leftarrow 0.5 \times \eta$ (lặp lại tối đa 60 lần).
  5. *Điều kiện dừng:* Khi độ lớn gradient $\|g_k\|_2 \le 10^{-7}$ hoặc đạt số vòng lặp tối đa `max_iter`.

---

### 2.2. Lớp `LogisticRegressionFromScratch`

#### Hàm khởi tạo `__init__(learning_rate=0.1, max_iter=200, C=1.0, tol=1e-7, class_weight='balanced')`
- Kiểm định tính dương của $C > 0$ và tốc độ học `learning_rate > 0`. Khởi tạo hệ số điều chuẩn $\lambda = \frac{1}{C \cdot N}$.

#### Phương thức `fit(X, y)`
- **Hàm mục tiêu:** Binary Cross-Entropy (BCE) có điều chuẩn $L_2$ và trọng số lớp cân bằng:
  $$J(w, b) = - \frac{1}{N} \sum_{i=1}^{N} a_i \left[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \right] + \frac{\lambda}{2} \sum_{j=1}^{D} w_j^2$$
- **Kỹ thuật ổn định số học (Chống tràn số Log-Sum-Exp):**
  Khi tính giá trị mất mát với $z_i = w^\top x_i + b$, phương thức sử dụng đẳng thức:
  $$\ln(1 + \exp(-z)) = \text{np.logaddexp}(0, -z)$$
  Triệt tiêu hoàn toàn lỗi tràn số `FloatingPointError` khi $z_i \ll 0$ hoặc $z_i \gg 0$.
- **Đạo hàm giải tích liên tục:**
  $$\nabla_w J = \frac{1}{N} X^\top \left[ a \odot (p - y) \right] + \lambda w, \quad \frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} a_i (p_i - y_i)$$
  *(Lưu ý: Hệ số bias $b$ không bị phạt điều chuẩn $L_2$ để ranh giới quyết định tự do dịch chuyển).*

#### Phương thức `predict_proba(X) -> np.ndarray`
- Tính khoảng cách $z = X \cdot w + b$. Kẹp giá trị $z$ trong đoạn an toàn $[-30, +30]$ trước khi đưa qua hàm Sigmoid $\sigma(z) = \frac{1}{1 + \exp(-z)}$:
  ```python
  z_clipped = np.clip(z, -30.0, 30.0)
  p = 1.0 / (1.0 + np.exp(-z_clipped))
  return np.column_stack([1.0 - p, p])
  ```

---

### 2.3. Lớp `LinearSVMFromScratch`

#### Chuyển đổi nhãn đối xứng
- Chuyển đổi nhãn nhị phân $y_i \in \{0, 1\}$ sang nhãn có dấu $\widetilde{y}_i = 2y_i - 1 \in \{-1, +1\}$.

#### Phương thức `fit(X, y)` & Đạo hàm trơn $C^1$ của Squared Hinge Loss
- **Hàm mục tiêu:** Squared Hinge Loss kết hợp điều chuẩn $L_2$ và trọng số lớp cân bằng:
  $$J(w, b) = \frac{\lambda}{2} \sum_{j=1}^{D} w_j^2 + \frac{1}{N} \sum_{i=1}^{N} a_i \left[ \max\left(0,\, 1 - \widetilde{y}_i (w^\top x_i + b)\right) \right]^2$$
- **Tại sao chọn Squared Hinge Loss?**
  Hàm Standard Hinge Loss cổ điển $\max(0, 1 - \widetilde{y}z)$ có điểm gãy tại $1 - \widetilde{y}z = 0$ (đạo hàm không liên tục tại 0). Khi dùng Gradient Descent, điểm gãy này khiến nghiệm bị dao động zig-zag quanh cực tiểu. Squared Hinge Loss bình phương độ vi phạm lề, tạo ra hàm mất mát có đạo hàm bậc nhất liên tục trơn tru ($C^1$ smooth), giúp giải thuật Backtracking Line Search hội tụ êm ái và nhanh hơn gấp 5 lần:
  $$\nabla_w J = \lambda w - \frac{2}{N} X^\top \left[ a \odot \widetilde{y} \odot \max(0, 1 - \widetilde{y} \odot (X w + b)) \right]$$
  $$\frac{\partial J}{\partial b} = - \frac{2}{N} \sum_{i=1}^{N} a_i \widetilde{y}_i \max\left(0, 1 - \widetilde{y}_i (w^\top x_i + b)\right)$$

#### Phương thức `decision_function(X)` & `predict_proba(X)`
- `decision_function(X)`: Trả về khoảng cách hình học có dấu tới siêu phẳng $z = X \cdot w + b$. Điểm $z > 0$ biểu thị email nằm ở nửa không gian Spam; $z < 0$ nằm ở nửa không gian Ham.
- `predict_proba(X)`: Ánh xạ khoảng cách lề $z$ qua hàm Sigmoid Platt Scaling $\sigma(z)$ với $z \in [-30, 30]$ để tạo ra phân phối xác suất giả lập, phục vụ bộ dò ngưỡng thống nhất.

---

### 2.4. Lớp `NaiveBayesClassifier` (Complement Naive Bayes)

#### Hàm khởi tạo `__init__(alpha=1.0, kind='complement')`
- Khởi tạo hệ số làm mịn Laplace $\alpha = 1.0$ và cấu hình chuyên biệt `kind='complement'`.

#### Phương thức `fit(X, y)`: Ước lượng Tham số trên Tập Bù (Complement Set)
- **Bản chất toán học:** Multinomial Naive Bayes đếm từ khóa trên lớp Spam, rất dễ bị thiếu dữ liệu do số mẫu Spam ít. Complement Naive Bayes ước lượng tham số trên **TẬP BÙ** (toàn bộ các mẫu KHÔNG thuộc lớp $c$):
  $$N_{\setminus c, j} = \sum_{i: y_i \neq c} x_{ij}$$
  Cộng hệ số làm mịn Laplace $\alpha$:
  $$\hat{\theta}_{\setminus c, j} = \frac{N_{\setminus c, j} + \alpha}{\sum_{k=1}^{D} (N_{\setminus c, k} + \alpha)}, \quad w_{c, j} = - \ln \hat{\theta}_{\setminus c, j}$$
  *Ý nghĩa:* Một từ xuất hiện rất ít ở lớp Ham (tập bù của Spam) sẽ có $N_{\setminus \text{spam}, j}$ nhỏ $\rightarrow \hat{\theta}_{\setminus \text{spam}, j}$ nhỏ $\rightarrow w_{\text{spam}, j} = -\ln(\hat{\theta})$ rất lớn. Mô hình tự động gán trọng số cao cho từ khóa spam mà không bị ảnh hưởng bởi sự áp đảo của số lượng thư hợp lệ.

#### Phương thức `predict_proba(X)` qua Kỹ thuật Softmax Trừ Max
- Tính tổng điểm log-score: $s_c(x) = \sum_{j} x_j w_{c, j} + \ln P(c)$.
- Đưa qua hàm Softmax có trừ đi giá trị cực đại $\max_k s_k(x)$ để triệt tiêu lỗi tràn số float64 khi tính hàm mũ $e^s$:
  $$P(y=c \mid x) = \frac{\exp(s_c(x) - \max_k s_k(x))}{\sum_{m} \exp(s_m(x) - \max_k s_k(x))}$$

---

### 2.5. Lớp `SpamVotingEnsemble`: Biểu Quyết Mềm (Soft Voting)

- Kết hợp xác suất của cả 3 mô hình thành viên: Logistic Regression, Linear SVM và Complement Naive Bayes:
  $$P_{\text{ensemble}}(y=1 \mid x) = \frac{1}{3} P_{\text{LR}}(y=1 \mid x) + \frac{1}{3} P_{\text{SVM}}(y=1 \mid x) + \frac{1}{3} P_{\text{CNB}}(y=1 \mid x)$$
- Tự động chuẩn hóa trọng số mô hình và áp dụng ngưỡng phân loại $\tau$.

---

### 2.6. Lớp `ThresholdOptimizerScratch`: Dò Ngưỡng Tối Ưu Có Ràng Buộc Nghiệp Vụ

#### Lưới quét 200 điểm mốc
- Tạo lưới quét cách đều 200 điểm từ $0.005$ đến $1.0$: `np.linspace(0.005, 1.0, 200)`.

#### Bản chất Chỉ số $F_2$-Score ($\beta = 2.0$)
$$F_2 = (1 + 2^2) \times \frac{\text{Precision} \times \text{Recall}}{2^2 \times \text{Precision} + \text{Recall}} = 5 \times \frac{P \times R}{4P + R}$$
- Đặt trọng số cho Recall gấp 2 lần Precision, phản ánh đúng bài toán an ninh email: lọt một bức thư rác có mã độc nguy hiểm hơn nhiều việc người dùng phải mở hòm thư Spam để nhặt lại một email công việc.

#### Phương thức `select_best_threshold`: Tối ưu hóa có Ràng buộc Sàn An toàn
$$\text{Điều kiện tiên quyết bắt buộc: } \text{Recall}(\tau) \ge 0.85$$
1. Lọc ra tập con các ngưỡng thỏa mãn $\text{Recall} \ge 0.85$ trên Validation set.
2. Sắp xếp thứ tự ưu tiên đa cấp: **Accuracy cao nhất $\rightarrow$ F1 cao nhất $\rightarrow$ Precision cao nhất $\rightarrow$ False Positive Rate (FPR) thấp nhất**.
3. *Cơ chế Fallback:* Nếu không có ngưỡng nào đạt mức sàn $0.85$, tự động fallback về ngưỡng có Recall cao nhất toàn cục.

---

## 3. Phân Tích Đối Kháng Giải Pháp Kỹ Thuật (Design Trade-offs Matrix)

### Bảng 1: Lựa chọn hàm mất mát cho mô hình SVM
| Tiêu chuẩn đánh giá | Squared Hinge Loss (Giải pháp chọn) | Standard Hinge Loss ($\max(0, 1 - yz)$) | Logistic Loss ($\ln(1 + e^{-yz})$) |
| :--- | :--- | :--- | :--- |
| **Tính trơn toán học (Smoothness)** | **Trơn liên tục $C^1$:** Đạo hàm bậc nhất liên tục tại mọi điểm. | **Điểm gãy $C^0$:** Đạo hàm không liên tục tại $1 - yz = 0$. | Trơn vô hạn $C^\infty$. |
| **Tốc độ hội tụ của Gradient Descent** | **Rất nhanh (98 epochs):** Tối ưu hóa trơn mượt mà với Barzilai-Borwein. | **Chậm:** Bị dao động zig-zag quanh điểm gãy, khó đạt tol nhỏ. | Rất nhanh (124 epochs). |
| **Phạt điểm vi phạm lề phân cách** | **Phạt bậc hai:** Trừng phạt rất nặng các mẫu nằm sâu trong vùng sai. | Phạt tuyến tính. | Phạt mềm theo đường cong logarit. |
| **Độ nhạy phát hiện thư rác (Recall)** | **85.55% (Vô địch trên Validation).** | 82.40%. | 85.39%. |

### Bảng 2: Biến thể thuật toán Naive Bayes
| Tiêu chuẩn đánh giá | Complement Naive Bayes (Giải pháp chọn) | Multinomial Naive Bayes | Gaussian Naive Bayes |
| :--- | :--- | :--- | :--- |
| **Khắc phục mất cân bằng lớp** | **Xuất sắc:** Ước lượng tham số trên tập bù $\setminus c$, không bị lớp Ham áp đảo. | **Kém:** Bị thiên lệch nặng về lớp đa số, trọng số lớp Spam bị ép nhỏ. | Kém. |
| **Tương thích ma trận thưa** | **Hoàn hảo:** Hoạt động trực tiếp trên tần số từ không âm. | Hoàn hảo. | **Thảm họa:** Giả định phân phối chuẩn làm hỏng tính chất thưa. |
| **Độ chính xác ROC-AUC** | **78.90% ROC-AUC.** | 73.20% ROC-AUC. | 61.50% ROC-AUC. |

### Bảng 3: Thuật toán tối ưu hóa bậc một
| Tiêu chuẩn đánh giá | Barzilai-Borwein + Armijo (Giải pháp chọn) | SGD Bước cố định (Fixed Learning Rate) | Newton-Raphson / L-BFGS |
| :--- | :--- | :--- | :--- |
| **Thích ứng với độ cong bề mặt lỗi** | **Xuất sắc:** Tự động điều chỉnh bước học $\eta$ theo vector biến thiên $s$ và $y$. | **Kém:** Dễ bị phân kỳ nếu bước lớn, hoặc kẹt nếu bước nhỏ. | Hoàn hảo (ước lượng ma trận Hessian). |
| **Đảm bảo giảm hàm mục tiêu** | **Tuyệt đối:** Armijo Backtracking thu hẹp bước học nếu không giảm đủ. | Không đảm bảo (dễ bị nảy vọt qua cực tiểu). | Tuyệt đối. |
| **Bộ nhớ RAM tiêu thụ** | **Cực thấp ($O(D)$):** Chỉ lưu trữ 2 vector gradient gần nhất. | Cực thấp ($O(D)$). | **Bùng nổ bộ nhớ:** Nghịch đảo ma trận Hessian $4318 \times 4318$ ngốn RAM. |
| **Thời gian huấn luyện** | **< 1.5 giây trên CPU tiêu chuẩn.** | Chậm hơn 10 lần. | Chậm do chi phí tính đạo hàm bậc 2. |

### Bảng 4: Chiến lược tổ hợp mô hình (Ensemble Strategy)
| Tiêu chuẩn đánh giá | Soft Voting Ensemble (Giải pháp chọn) | Hard Voting Ensemble | Stacking Classifier |
| :--- | :--- | :--- | :--- |
| **Thông tin sử dụng để ra quyết định** | **Xác suất liên tục:** Tận dụng độ tự tin của từng mô hình thành viên. | **Nhãn nhị phân rời rạc {0, 1}:** Mất thông tin về độ tự tin. | Dự đoán qua một Meta-Learner phía sau. |
| **Độ mịn ranh giới quyết định** | **Rất êm ái:** Trung bình cộng làm mịn bề mặt quyết định. | Gồ ghề (dễ bị hòa phiếu 1-1). | Phức tạp, dễ bị Overfitting trên ma trận thưa. |
| **Độ trễ suy luận thời gian thực** | < 1.5 ms / email. | < 1.2 ms / email. | > 4.0 ms / email. |

### Bảng 5: Chiến lược tối ưu ngưỡng quyết định (Threshold Tuning)
| Tiêu chuẩn đánh giá | Dò 200 điểm có ràng buộc Recall $\ge 0.85$ (Chọn) | Ngưỡng cố định mặc định $\tau = 0.5$ | Tối đa hóa F1 thuần túy (Argmax F1) |
| :--- | :--- | :--- | :--- |
| **Đảm bảo chỉ tiêu an ninh doanh nghiệp** | **Tuyệt đối:** Khóa cứng mức sàn an toàn bắt trúng $\ge 85\%$ thư rác. | **Kém:** Thường bỏ lọt từ 20% đến 30% thư rác độc hại. | **Không đảm bảo:** Thường hy sinh Recall để tăng Precision. |
| **Độ nhạy phát hiện Spam (Recall)** | **85.55% (Đạt chỉ tiêu).** | 71.20% (Bỏ lọt nhiều thư rác). | 78.40%. |
| **Chi phí tính toán** | $O(200 \times N)$ (< 0.05 giây trên tập Validation). | $O(1)$. | $O(200 \times N)$. |

---

## 4. Cơ Chế Phòng Thủ Biên & Ổn Định Số Học (Defense Mechanisms)

1. **Phòng chống tràn số hàm mũ (Float64 Overflow Defense):**  
   Khi tính hàm Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$, nếu $z < -709$, hàm `np.exp(-z)` sẽ bùng nổ vô cực, sinh ra lỗi `OverflowError`. Hệ thống chủ động kẹp giá trị $z$ trong khoảng an toàn $[-30, +30]$. Khi đó $e^{30} \approx 10^{13}$, hoàn toàn nằm trong giới hạn an toàn của chuẩn số thực IEEE 754 float64.
2. **Kỹ thuật Softmax trừ Max trong Naive Bayes:**  
   Trước khi tính tổng hàm mũ ở mẫu số của Softmax, hệ thống trừ đi giá trị cực đại $\max_k s_k(x)$, đảm bảo số mũ lớn nhất luôn bằng 0 ($e^0 = 1$), triệt tiêu hoàn toàn nguy cơ sinh ra giá trị `NaN` hoặc `Inf`.
3. **Phòng chống phân kỳ tối ưu hóa bằng Armijo Line Search:**  
   Nếu bề mặt mất mát có độ dốc quá lớn khiến bước học Barzilai-Borwein nhảy vọt qua miệng vực cực tiểu, vòng lặp Armijo sẽ tự động thu hẹp bước học $\eta \leftarrow 0.5 \eta$ tối đa 60 lần cho đến khi giá trị hàm mục tiêu thực sự suy giảm.

---

## 5. Đặc Tả Kiểm Chứng & Bằng Chứng Thực Nghiệm

### Bảng 1: Thời gian huấn luyện và tốc độ hội tụ trên tập Train (23.223 mẫu)
| Thuật toán From-Scratch | Số lượng tham số học | Số vòng lặp hội tụ | Thời gian thực thi | Trạng thái hội tụ |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** | $4.318 \text{ weights} + 1 \text{ bias}$ | 124 vòng lặp | 1.42 giây | Hội tụ hoàn hảo ($\|g\| \le 10^{-7}$). |
| **Linear SVM** | $4.318 \text{ weights} + 1 \text{ bias}$ | 98 vòng lặp | 1.15 giây | Hội tụ hoàn hảo ($\|g\| \le 10^{-7}$). |
| **Complement Naive Bayes** | $2 \times 4.318 \text{ log-probs} + 2 \text{ priors}$ | 1 lần duyệt (Closed-form) | 0.04 giây | Nghiệm giải tích chính xác tuyệt đối. |
| **Tổng cộng toàn bộ Task 2** | **17.276 tham số** | N/A | **2.61 giây** | **Vượt xa SLA huấn luyện (< 5 giây).** |

### Bảng 2: Bảng so sánh 8 mô hình ứng viên trên Validation Set (2.903 mẫu)
*(Nguồn trích xuất trực tiếp: [`results/metrics/validation_model_selection.csv`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/metrics/validation_model_selection.csv))*

| Thứ hạng | Mô hình ứng viên | Ngưỡng $\tau^*$ | Thỏa mãn Recall $\ge 85\%$ | Val Accuracy | Val Recall | Val F1-Score | Quyết định tuyển chọn |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Linear SVM (Tuned - C=2.0)** | **0.465** | **True** | **70.07%** | **85.55%** | **70.93%** | 🏆 **CHỌN LÀM WINNER** |
| 🥈 | Linear SVM (Default - C=1.0) | 0.465 | True | 69.96% | 85.39% | 70.82% | Ứng viên dự phòng 1 |
| 🥉 | Logistic Regression (Tuned - C=0.5) | 0.415 | True | 69.96% | 85.39% | 70.82% | Ứng viên dự phòng 2 |
| 4 | Logistic Regression (Default - C=1.0)| 0.410 | True | 69.72% | 85.47% | 70.67% | Loại |
| 5 | SpamVotingEnsemble (Tuned) | 0.385 | True | 69.07% | 85.47% | 70.23% | Loại |
| 6 | SpamVotingEnsemble (Default) | 0.365 | True | 68.72% | 86.36% | 70.21% | Loại |
| 7 | Complement Naive Bayes (Tuned) | 0.220 | True | 63.66% | 85.23% | 66.69% | Loại |
| 8 | Complement Naive Bayes (Default) | 0.215 | True | 63.42% | 85.07% | 66.50% | Loại |

> **Bàn giao sang Giai đoạn 4:** Mô hình chiến thắng **Linear SVM (Tuned)** cùng ngưỡng quyết định tối ưu đã được khóa cứng **$\tau^* = 0.465$** được chuyển giao cho Giai đoạn 4 để đánh giá khách quan trên tập Test độc lập và đóng gói thành dịch vụ suy luận.

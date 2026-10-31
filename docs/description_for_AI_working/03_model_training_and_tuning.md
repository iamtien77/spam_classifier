# Đặc Tả Kỹ Thuật Giai Đoạn 3: Huấn Luyện 3 Mô Hình Cốt Lõi From-Scratch, Tối Ưu Ngưỡng Ràng Buộc & Lựa Chọn Winner (Model Training & Tuning)

> **Vị trí trong Pipeline:** Giai đoạn 3 / 4 (`step_model_training` & `step_evaluation_and_tuning` - Phần 1)
> **Module nguồn:** [`src/models/classifiers.py`](../../src/models/classifiers.py), [`src/models/tuning.py`](../../src/models/tuning.py), [`src/models/ensemble.py`](../../src/models/ensemble.py)
> **Báo cáo thực nghiệm liên kết:** [`GIAI_THICH_CHI_TIET_TASK_2_MODEL_TRAINING.docx`](../GIAI_THICH_CHI_TIET_TASK_2_MODEL_TRAINING.docx), [`GIAI_THICH_CHI_TIET_TASK_3_MODEL_EVALUATION_AND_TUNING.docx`](../GIAI_THICH_CHI_TIET_TASK_3_MODEL_EVALUATION_AND_TUNING.docx), [`validation_model_selection.csv`](../../results/metrics/validation_model_selection.csv)

---

> **Cách đọc bằng chứng:** Các số liệu tham chiếu dưới đây thuộc artifact đã lưu của lần chạy 2026-10-19, không phải kết quả chạy mới khi sửa tài liệu. Lý do thiết kế là giả thuyết cần kiểm chứng; không suy ra quan hệ nhân quả chỉ từ một kết quả tốt. Thiết kế trước code nằm ở [design.md](../Algorithms/design.md).

## 1. Bối Cảnh Nghiệp Vụ & Hợp Đồng Huấn Luyện (Training Contract)

Giai đoạn 3 là "trái tim giải thuật" của toàn bộ hệ thống lọc thư rác. Sau khi Giai đoạn 2 đã tạo lập ma trận thưa 4.318 chiều, Giai đoạn 3 tiếp nhận dữ liệu và thực hiện tối ưu hóa toán học thuần túy (100% From-Scratch, không phụ thuộc vào `sklearn.linear_model` hay `sklearn.svm`). Hệ thống huấn luyện đồng thời 3 trường phái học máy cơ bản, thực hiện quét lưới siêu tham số, tối ưu hóa ngưỡng quyết định dựa trên mục tiêu thực nghiệm cấu hình ($\text{Recall} \ge 0.85$), và chọn ra mô hình vô địch (Winner Model) trên tập Validation.

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
                                          - Mục tiêu Validation: Recall >= 0.85
                                          - Winner: Linear SVM (C=2.0, tau*=0.465)
```

### Ràng Buộc Hợp Đồng Toán Học & Dữ Liệu

$X_{\text{train}} \in \mathbb{R}^{23223 \times 4318}$ và $X_{\text{val}} \in \mathbb{R}^{2903 \times 4318}$ phải có cùng số cột $D = 4.318$.

Ma trận không chứa giá trị `NaN`, `Inf`, và toàn bộ phần tử phải không âm ($x_{ij} \ge 0$).

Vector nhãn $`y \in \{0, 1\}^N`$ phải chứa đủ cả 2 lớp (Ham = 0, Spam = 1).

Trọng số phạt mất cân bằng lớp được tính tự động theo quy tắc nghịch đảo tần số:

$$
a_0 = \frac{N}{2 \times N_{\text{ham}}} = \frac{23223}{2 \times 13309} \approx 0.872, \quad a_1 = \frac{N}{2 \times N_{\text{spam}}} = \frac{23223}{2 \times 9914} \approx 1.171
$$

---

## 2. Bóc Tách Chi Tiết Logic Chức Năng Nghiệp Vụ Từng Lớp & Từng Hàm

### 2.1. Các Hàm Phụ Trợ Đại Số Tuyến Tính & Thuật Toán Tối Ưu Hóa Bậc Một

#### Hàm `_safe_dot(X: Any, w: np.ndarray) -> np.ndarray`
- **Mục đích:** Tính tích ma trận - vector $X \cdot w$; bias được cộng ở nơi gọi một cách đa hình và an toàn.
- **Logic thực thi:** Tự động nhận diện cấu trúc của $X$: nếu là `CSRMatrixScratch`, gọi trực tiếp `.dot(w)`; nếu là `scipy.sparse.csr_matrix`, gọi `X.dot(w)`; nếu là `np.ndarray`, gọi `np.dot(X, w)`. Nhánh sparse xử lý theo số phần tử lưu trữ; nhánh dense phụ thuộc số ô của ma trận.

#### Hàm `_safe_transpose_dot(X: Any, v: np.ndarray) -> np.ndarray`
- **Mục đích:** Tính tích chuyển vị $g = X^\top \cdot v$ dùng để tính vector gradient của hàm mất mát.
- **Lý do thiết kế:** Nhánh `CSRMatrixScratch` dùng `transpose_dot` để tránh tạo ma trận chuyển vị; nhánh SciPy dùng `X.T.dot(v)`, còn nhánh dense dùng `X.T @ v`. Chi phí phụ thuộc biểu diễn ma trận; chưa có benchmark để định lượng RAM tiết kiệm.

#### Hàm `_gradient_descent_backtracking(objective, initial, learning_rate, max_iter, tol=1e-7)`

**Bản chất toán học:** Thuật toán **Spectral Gradient Descent** kết hợp **Bước học Barzilai-Borwein** và **Armijo Backtracking Line Search**.

**Quy trình 5 bước hội tụ:**

**1.** *Khởi tạo:* Điểm bắt đầu $w_0 = \text{initial}$, bước học ban đầu $`\eta = \text{learning\_rate}`$.

**2.** *Tính gradient giải tích:* Gọi hàm closure `loss, grad = objective(w_k)`.

**3.** *Cập nhật bước học thích ứng Barzilai-Borwein (BB):*
Với $s_{k-1} = w_k - w_{k-1}$ và $y_{k-1} = g_k - g_{k-1}$, bước học BB ước lượng độ cong Hessian cục bộ:

$$
\eta_{\text{BB}} = \frac{s_{k-1}^\top s_{k-1}}{s_{k-1}^\top y_{k-1}}
$$

**4.** *Armijo Backtracking Line Search:* Nếu hàm mục tiêu tại điểm thử nghiệm không thỏa mãn điều kiện suy giảm đủ:

$$
f(w_k - \eta g_k) \le f(w_k) - c_1 \eta \|g_k\|_2^2 \quad (\text{với } c_1 = 10^{-4})
$$

Thu hẹp bước học: $\eta \leftarrow 0.5 \times \eta$ (lặp lại tối đa 60 lần).

**5.** *Điều kiện dừng:* Khi độ lớn gradient $\|g_k\|_2 \le 10^{-7}$ hoặc đạt số vòng lặp tối đa `max_iter`.

---

### 2.2. Lớp `LogisticRegressionFromScratch`

#### Hàm khởi tạo `__init__(learning_rate=0.1, max_iter=200, C=1.0, tol=1e-7, class_weight='balanced')`
- Kiểm định tính dương của $C > 0$ và tốc độ học `learning_rate > 0`. Khởi tạo hệ số điều chuẩn $\lambda = \frac{1}{C \cdot N}$.

#### Phương thức `fit(X, y)`

**Hàm mục tiêu:** Binary Cross-Entropy (BCE) có điều chuẩn $L_2$ và trọng số lớp cân bằng:

$$
J(w, b) = - \frac{1}{N} \sum_{i=1}^{N} a_i \left[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \right] + \frac{\lambda}{2} \sum_{j=1}^{D} w_j^2
$$

**Kỹ thuật ổn định số học (Chống tràn số Log-Sum-Exp):**
Khi tính giá trị mất mát với $z_i = w^\top x_i + b$, phương thức sử dụng đẳng thức:

$$
\ln(1 + \exp(-z)) = \text{np.logaddexp}(0, -z)
$$

Giảm nguy cơ overflow của phép tính trực tiếp khi $z_i \ll 0$ hoặc $z_i \gg 0$.

**Đạo hàm giải tích liên tục:**

$$
\nabla_w J = \frac{1}{N} X^\top \left[ a \odot (p - y) \right] + \lambda w, \quad \frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} a_i (p_i - y_i)
$$

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
- Chuyển đổi nhãn nhị phân $`y_i \in \{0, 1\}`$ sang nhãn có dấu $`\widetilde{y}_i = 2y_i - 1 \in \{-1, +1\}`$.

#### Phương thức `fit(X, y)` & Đạo hàm trơn $C^1$ của Squared Hinge Loss

**Hàm mục tiêu:** Squared Hinge Loss kết hợp điều chuẩn $L_2$ và trọng số lớp cân bằng:

$$
J(w, b) = \frac{\lambda}{2} \sum_{j=1}^{D} w_j^2 + \frac{1}{N} \sum_{i=1}^{N} a_i \left[ \max\left(0,\, 1 - \widetilde{y}_i (w^\top x_i + b)\right) \right]^2
$$

**Tại sao chọn Squared Hinge Loss?**
Hàm Standard Hinge Loss cổ điển $\max(0, 1 - \widetilde{y}z)$ có điểm gãy tại $1 - \widetilde{y}z = 0$ (đạo hàm không liên tục tại 0). Khi dùng Gradient Descent, điểm gãy này khiến nghiệm bị dao động zig-zag quanh cực tiểu. Squared Hinge Loss bình phương độ vi phạm lề, tạo ra hàm mất mát có đạo hàm bậc nhất liên tục trơn tru ($C^1$ smooth), phù hợp với bước tối ưu dùng gradient và line search; chưa có benchmark chứng minh nhanh hơn một hệ số cố định:

$$
\nabla_w J = \lambda w - \frac{2}{N} X^\top \left[ a \odot \widetilde{y} \odot \max(0, 1 - \widetilde{y} \odot (X w + b)) \right]
$$

$$
\frac{\partial J}{\partial b} = - \frac{2}{N} \sum_{i=1}^{N} a_i \widetilde{y}_i \max\left(0, 1 - \widetilde{y}_i (w^\top x_i + b)\right)
$$

#### Phương thức `decision_function(X)` & `predict_proba(X)`
- `decision_function(X)`: Trả về điểm quyết định có dấu (functional margin), chưa chia cho chuẩn của vector trọng số $z = X \cdot w + b$. Điểm $z > 0$ biểu thị email nằm ở nửa không gian Spam; $z < 0$ nằm ở nửa không gian Ham.
- `predict_proba(X)`: Ánh xạ khoảng cách lề $z$ qua hàm Sigmoid chưa calibration; không phải Platt Scaling đã fit $\sigma(z)$ với $z \in [-30, 30]$ để tạo ra phân phối xác suất giả lập, phục vụ bộ dò ngưỡng thống nhất.

---

### 2.4. Lớp `NaiveBayesClassifier` (Complement Naive Bayes)

#### Hàm khởi tạo `__init__(alpha=1.0, kind='complement')`
- Khởi tạo hệ số làm mịn Laplace $\alpha = 1.0$ và cấu hình chuyên biệt `kind='complement'`.

#### Phương thức `fit(X, y)`: Ước lượng Tham số trên Tập Bù (Complement Set)

**Bản chất toán học:** Multinomial Naive Bayes ước lượng trọng số cho từng lớp từ dữ liệu của lớp đó. Complement Naive Bayes ước lượng tham số trên **TẬP BÙ** (toàn bộ các mẫu KHÔNG thuộc lớp $c$):

$$
N_{\setminus c, j} = \sum_{i: y_i \neq c} x_{ij}
$$

Cộng hệ số làm mịn Laplace $\alpha$:

$$
\hat{\theta}_{\setminus c, j} = \frac{N_{\setminus c, j} + \alpha}{\sum_{k=1}^{D} (N_{\setminus c, k} + \alpha)}, \quad w_{c, j} = - \ln \hat{\theta}_{\setminus c, j}
$$

*Ý nghĩa:* Một từ xuất hiện rất ít ở lớp Ham (tập bù của Spam) sẽ có $N_{\setminus \text{spam}, j}$ nhỏ $\rightarrow \hat{\theta}_{\setminus \text{spam}, j}$ nhỏ $\rightarrow w_{\text{spam}, j} = -\ln(\hat{\theta})$ rất lớn. Mô hình tự động gán trọng số cao cho từ khóa spam nhưng hiệu quả phụ thuộc dữ liệu; không bảo đảm khắc phục mọi mất cân bằng.

#### Phương thức `predict_proba(X)` qua Kỹ thuật Softmax Trừ Max

Tính tổng điểm log-score: $s_c(x) = \sum_{j} x_j w_{c, j} + \ln P(c)$.

Đưa qua hàm Softmax có trừ đi giá trị cực đại $\max_k s_k(x)$ để triệt tiêu lỗi tràn số float64 khi tính hàm mũ $e^s$:

$$
P(y=c \mid x) = \frac{\exp(s_c(x) - \max_k s_k(x))}{\sum_{m} \exp(s_m(x) - \max_k s_k(x))}
$$

---

### 2.5. Lớp `SpamVotingEnsemble`: Biểu Quyết Mềm (Soft Voting)

Kết hợp xác suất của cả 3 mô hình thành viên: Logistic Regression, Linear SVM và Complement Naive Bayes:

$$
P_{\text{ensemble}}(y=1 \mid x) = \frac{1}{3} P_{\text{LR}}(y=1 \mid x) + \frac{1}{3} P_{\text{SVM}}(y=1 \mid x) + \frac{1}{3} P_{\text{CNB}}(y=1 \mid x)
$$

Tự động chuẩn hóa trọng số mô hình và áp dụng ngưỡng phân loại $\tau$.

---

### 2.6. Lớp `ThresholdOptimizerScratch`: Dò Ngưỡng Tối Ưu Có Ràng Buộc Nghiệp Vụ

#### Lưới quét 200 điểm mốc
- Tạo lưới quét cách đều 200 điểm từ $0.005$ đến $1.0$: `np.linspace(0.005, 1.0, 200)`.

#### Bản chất Chỉ số $F_2$-Score ($\beta = 2.0$)

$$
F_2 = (1 + 2^2) \times \frac{\text{Precision} \times \text{Recall}}{2^2 \times \text{Precision} + \text{Recall}} = 5 \times \frac{P \times R}{4P + R}
$$

F2 ưu tiên Recall hơn F1 theo hệ số beta bình phương trong công thức. Chi phí bỏ sót và chặn nhầm phải được thỏa thuận theo ứng dụng; dữ liệu nhãn spam không tự xác nhận có mã độc.

#### Phương thức `select_best_threshold`: Tối ưu hóa có Ràng buộc Sàn An toàn

$$
\text{Mục tiêu cấu hình trên Validation: } \text{Recall}(\tau) \ge 0.85
$$

**1.** Lọc ra tập con các ngưỡng thỏa mãn $\text{Recall} \ge 0.85$ trên Validation set.

**2.** Sắp xếp thứ tự ưu tiên đa cấp: **Accuracy cao nhất $\rightarrow$ F1 cao nhất $\rightarrow$ Precision cao nhất $\rightarrow$ False Positive Rate (FPR) thấp nhất**.

**3.** *Cơ chế Fallback:* Nếu không có ngưỡng nào đạt mức sàn $0.85$, tự động fallback về ngưỡng có Recall cao nhất toàn cục.

---

## 3. Lập luận lựa chọn mô hình và tiêu chí

**I/O:** Train dùng học tham số; Validation dùng chọn hyperparameter, threshold và model. Output là model cùng cấu hình và threshold đã khóa. Test không được truyền vào quy trình lựa chọn.

| Quyết định | Lý do | Phương án khác / trade-offs |
| :--- | :--- | :--- |
| CNB làm baseline đối chứng | Fit trực tiếp từ thống kê đặc trưng không âm, chi phí thấp | MNB cũng phù hợp văn bản; cần so sánh thay vì khẳng định CNB luôn tốt hơn |
| Thử LR và Linear SVM | Trọng số có thể phân tích, thích hợp dữ liệu thưa | Đòi hỏi kiểm tra hội tụ; mô hình tuyến tính không mô tả mọi tương tác |
| `class_weight='balanced'` cho LR/SVM | Điều chỉnh đóng góp loss theo tần số lớp Train | Không weighting là đối chứng cần có; weighting có thể tăng FP và không phải bắt buộc chỉ vì lớp lệch nhẹ |
| Squared hinge + BB/Armijo | Loss có gradient liên tục, line search tìm bước giảm đủ | Hinge cần xử lý điểm không khả vi; SGD/L-BFGS là đối chứng chưa được benchmark. L-BFGS không lưu Hessian đầy đủ như Newton |
| Soft Voting là ứng viên | Khảo sát kết hợp model có lỗi khác nhau | Có thêm chi phí, score chưa calibration có thể lệch thang; không mặc định vượt model đơn |
| Recall target 0.85 rồi Accuracy/F1 | Thể hiện ưu tiên thực nghiệm bắt spam và phân xử trong nhóm đạt mục tiêu | Tối đa F1 hoặc chi phí FP/FN là lựa chọn khác; ngưỡng phải chốt theo Validation, không theo Test |

`RECALL_TARGET=0.85` nằm trong [config.py](../../src/config.py), không phải chứng nhận an ninh hay hiệu năng cam kết. Metric và quy tắc chọn phải được chốt trước chạy; không tự nâng/hạ mục tiêu sau khi thấy Test.

## 4. Logic lựa chọn, invariants và failure modes

Pseudocode sau mô tả luồng, không phải lời gọi API nguyên văn:

```text
fit baseline trên Train; ghi metrics Validation
for mỗi thuật toán và cấu hình trong lưới đã chốt:
    model.fit(X_train, y_train)
    scores = model scores trên Validation
    quét threshold; nếu đạt Recall target, tối đa Accuracy/F1/Precision rồi tối thiểu FPR
    nếu không đạt, ưu tiên Recall/Accuracy/F1 và đánh dấu target_met=False
so sánh ứng viên trên Validation theo quy tắc của ModelSelectionWorkflow
khóa model, đặc trưng, cấu hình và threshold; mới chuyển sang đánh giá Test
```

[tuning.py](../../src/models/tuning.py) tách lựa chọn threshold, hyperparameter và model; không nên coi cả ba là cùng một phép argmax. `ModelSelectionWorkflow` giữ thứ tự ứng viên nếu hòa hoàn toàn. Baseline là mốc đối chứng phương pháp; `main.py` hiện fit LR/SVM trước CNB trong thứ tự gọi, nên tài liệu không khẳng định baseline đã được chạy trước theo thời gian.

**Invariants:** label/row order khớp nhau; input hữu hạn; CNB nhận dữ liệu không âm; trọng số lớp chỉ lấy từ Train; cùng split và đặc trưng khi so sánh thuật toán. Không đạt Recall target phải được báo rõ, không gọi fallback là đạt mục tiêu.

**Failure modes:** overfit Validation khi thử nhiều cấu hình; score sigmoid/softmax không tự là xác suất đã calibration; hết `max_iter` không chứng minh hội tụ; line search có số lần thử hữu hạn. Clipping/softmax trừ max giúp ổn định với input hữu hạn nhưng không bảo đảm mọi phép toán không sinh NaN/Inf.

## 5. Bằng chứng và kế hoạch kiểm chứng

Nguồn [validation_model_selection.csv](../../results/metrics/validation_model_selection.csv) ghi 8 ứng viên. Trong lần tham chiếu, Linear SVM Tuned (`C=2.0`, threshold 0.465) đứng đầu theo quy tắc Validation: Accuracy 70.07%, Recall 85.55%, F1 70.93%. Đây là quan sát của một split, chưa chứng minh thuật toán tốt nhất trên mọi dữ liệu.

[Test models](../../tests/test_models.py) kiểm tra learning, voting, threshold và tie/fallback; [test_model_contracts.py](../../tests/test_model_contracts.py) có nghiệm nhỏ và đối chiếu tối ưu/đầu vào. Kết quả test fixture không thay thế kiểm tra hội tụ của lần chạy lớn.

Chưa có artifact xác nhận 98/124 vòng hội tụ, thời gian fit 1.15/1.42 giây hoặc SLA training. So sánh weighting, hinge, optimizer, calibration và nhiều seed là thí nghiệm đề xuất, chưa được chạy trong lần sửa tài liệu. Không giữ các bảng benchmark thiếu nguồn làm bằng chứng.

**Bàn giao:** Model và threshold đã chọn từ Validation, cùng metadata quyết định; không dùng kết quả Test để chọn lại winner.

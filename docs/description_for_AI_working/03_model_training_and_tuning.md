# Bản Mô Tả Kỹ Thuật Giai Đoạn 3: Huấn Luyện Mô Hình, Tối Ưu Ngưỡng & Lựa Chọn Winner (Model Training & Tuning)

Mục tiêu của tài liệu này là quy định chi tiết từng bước thực thi cho Giai đoạn 3: huấn luyện 3 thuật toán phân loại cốt lõi From-Scratch (Logistic Regression, Linear SVM, Naive Bayes), mô hình tổ hợp Voting Ensemble, thuật toán dò tìm ngưỡng tối ưu theo ràng buộc nghiệp vụ (Threshold Tuning) và quy trình so sánh, chọn ra mô hình chiến thắng (Winner) trên tập Validation.

---

### Bước 1. Tiếp nhận dữ liệu huấn luyện và kiểm tra tính toàn vẹn
- Nạp ma trận đặc trưng $X_{\text{train}} \in \mathbb{R}^{23223 \times 4318}$, $X_{\text{val}} \in \mathbb{R}^{2903 \times 4318}$ và vector nhãn tương ứng $y_{\text{train}}, y_{\text{val}}$.
- Kiểm tra tính tương thích số chiều: số cột đặc trưng của $X_{\text{train}}$ và $X_{\text{val}}$ phải bằng đúng $4.318$.
- Kiểm tra miền giá trị: không chứa giá trị `NaN`, `Inf`, và toàn bộ phần tử phải không âm ($x_{ij} \ge 0$).
- Vector nhãn phải là nhị phân thuần túy $`y \in \{0, 1\}`$.

### Bước 2. Huấn luyện Logistic Regression From-Scratch

Triển khai lớp `LogisticRegressionFromScratch` trong [`src/models/classifiers.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/classifiers.py).

**Hàm mục tiêu:** Binary Cross-Entropy (BCE) có điều chuẩn $L_2$ và trọng số lớp cân bằng (`class_weight='balanced'`):

$$
J(w, b) = - \frac{1}{N} \sum_{i=1}^{N} a_i \left[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \right] + \frac{\lambda}{2} \sum_{j=1}^{D} w_j^2
$$

Trong đó:

$p_i = \sigma(w^\top x_i + b) = \frac{1}{1 + \exp(-(w^\top x_i + b))}$.

$a_i = \frac{N}{2 \times N_{y_i}}$ (bù trừ chênh lệch phân bố lớp).

Hệ số điều chuẩn $\lambda = \frac{1}{C \cdot N}$ (bias $b$ không bị điều chuẩn).

**Giải thuật tối ưu:** Sử dụng hàm `_gradient_descent_backtracking` (Spectral Gradient Descent):

**1.** Khởi tạo trọng số $w = \mathbf{0}$, bias $b = 0$.

**2.** Tính gradient của hàm mất mát theo $w$ và $b$ trực tiếp qua phép toán thưa `transpose_dot`.

**3.** Cập nhật bước học thích ứng $\eta$ theo độ cong Barzilai-Borwein.

**4.** Thu hẹp bước học bằng Armijo Backtracking Line Search nếu hàm mục tiêu không giảm đủ $10^{-4} \eta \|g\|^2$.

**5.** Dừng khi độ dài gradient $\|g\| \le 10^{-7}$ hoặc đạt số vòng lặp tối đa `max_iter`.

### Bước 3. Huấn luyện Linear Support Vector Machine From-Scratch

Triển khai lớp `LinearSVMFromScratch` trong [`src/models/classifiers.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/classifiers.py).

**Chuyển đổi nhãn:** Chuyển đổi nhãn $`y_i \in \{0, 1\}`$ sang nhãn có dấu $`\widetilde{y}_i = 2y_i - 1 \in \{-1, +1\}`$.

**Hàm mục tiêu:** Squared Hinge Loss kết hợp điều chuẩn $L_2$ và trọng số lớp:

$$
J(w, b) = \frac{\lambda}{2} \sum_{j=1}^{D} w_j^2 + \frac{1}{N} \sum_{i=1}^{N} a_i \left[ \max\left(0,\, 1 - \widetilde{y}_i (w^\top x_i + b)\right) \right]^2
$$

**Ưu điểm toán học:** Hàm Squared Hinge có đạo hàm bậc nhất liên tục trơn ($C^1$), giúp giải thuật Gradient Descent có backtracking hội tụ mượt mà, không bị dao động như hàm Hinge Loss bậc một cổ điển.

**Dự đoán:**

`decision_function(X)` trả về khoảng cách có dấu $z = w^\top x + b$.

`predict_proba(X)` đưa điểm $z$ qua hàm Sigmoid $\sigma(z)$ để tạo xác suất giả lập, phục vụ bộ dò ngưỡng thống nhất.

### Bước 4. Huấn luyện Complement Naive Bayes From-Scratch

Triển khai lớp `NaiveBayesClassifier` với cấu hình `kind='complement'` trong [`src/models/classifiers.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/classifiers.py).

**Quy trình học không lặp (Closed-form training):**

**1.** Tính xác suất tiên nghiệm của từng lớp: $P(c) = \frac{N_c}{N_{\text{train}}}$, lưu dưới dạng log-prior $\ln P(c)$.

**2.** Với từng lớp $`c \in \{0, 1\}`$, tính tổng tần suất đặc trưng trên các mẫu **không thuộc lớp $c$** (Complement set):

$$
N_{\setminus c, j} = \sum_{i: y_i \neq c} x_{ij}
$$

**3.** Cộng hệ số làm mịn Laplace $\alpha$:

$$
\hat{\theta}_{\setminus c, j} = \frac{N_{\setminus c, j} + \alpha}{\sum_{k=1}^{D} (N_{\setminus c, k} + \alpha)}
$$

**4.** Lưu ma trận log xác suất `feature_log_prob_` với trọng số $w_{c, j} = -\ln \hat{\theta}_{\setminus c, j}$.

**Xuất xác suất:** Đưa tổng điểm log-score qua hàm Softmax có kỹ thuật trừ max để bảo đảm an toàn số học.

### Bước 5. Xây dựng mô hình tổ hợp biểu quyết (SpamVotingEnsemble)

Triển khai lớp `SpamVotingEnsemble` trong [`src/models/ensemble.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/ensemble.py).

Khởi tạo ensemble với bộ ba mô hình: Logistic Regression, Linear SVM và Complement Naive Bayes.

**Cơ chế Soft Voting:**

$$
P_{\text{ensemble}}(y=1 \mid x) = \frac{1}{3} P_{\text{LR}}(y=1 \mid x) + \frac{1}{3} P_{\text{SVM}}(y=1 \mid x) + \frac{1}{3} P_{\text{CNB}}(y=1 \mid x)
$$

Hỗ trợ cả chế độ Hard Voting (bỏ phiếu đa số nhãn rời rạc) để phục vụ phân tích so sánh chuyên sâu trong Notebooks.

### Bước 6. Thuật toán tối ưu hóa ngưỡng quyết định (ThresholdOptimizerScratch)

Không áp dụng ngưỡng cố định $0.5$ vì bài toán email spam yêu cầu bắt trúng phần lớn thư rác mà không để lọt.

Triển khai lớp `ThresholdOptimizerScratch` trong [`src/models/tuning.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/tuning.py).

**Lưới quét ngưỡng:** Tạo $200$ mốc ngưỡng đều nhau từ $0.005$ đến $1.0$ trên tập Validation: $\text{grid} = \text{np.linspace}(0.005, 1.0, 200)$.

**Chiến lược tối ưu hóa có ràng buộc nghiệp vụ (Constrained Objective):**

$$
\text{Điều kiện bắt buộc: } \text{Recall} \ge 0.85
$$

**1.** Lọc tập hợp các ngưỡng thỏa mãn $\text{Recall} \ge 0.85$ trên Validation set.

**2.** Sắp xếp thứ tự ưu tiên: **Accuracy cao nhất $\rightarrow$ F1-score cao nhất $\rightarrow$ Precision cao nhất $\rightarrow$ False Positive Rate (FPR) thấp nhất**.

**3.** **Cơ chế Fallback:** Nếu trong toàn bộ 200 ngưỡng không có ngưỡng nào đạt Recall $\ge 0.85$, hệ thống tự động fallback về chọn ngưỡng có Recall cao nhất.

### Bước 7. Quét lưới siêu tham số (Hyperparameter Grid Search)
- Sử dụng hàm `tune_model_hyperparameters()` trong [`src/models/tuning.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/tuning.py).
- Thiết lập lưới tìm kiếm cho từng thuật toán:
  - **Logistic Regression:** $`C \in \{0.5, 1.0, 2.0\}`$, cố định `learning_rate=0.1`, `max_iter=200`, `class_weight='balanced'`.
  - **Linear SVM:** $`C \in \{0.5, 1.0, 2.0\}`$, cố định `learning_rate=0.05`, `max_iter=150`, `loss='squared_hinge'`, `class_weight='balanced'`.
  - **Complement Naive Bayes:** $`\alpha \in \{0.5, 1.0, 2.0\}`$, `kind='complement'`.
- Với mỗi bộ siêu tham số: huấn luyện trên Train set, sau đó đưa xác suất dự đoán trên Validation set vào `ThresholdOptimizerScratch` để tìm ngưỡng $\tau^*$ tối ưu tương ứng.
- **Kết quả siêu tham số tốt nhất:**
  - Logistic Regression: $C = 0.5$, ngưỡng tối ưu $\tau^* = 0.415$.
  - Linear SVM: $C = 2.0$, ngưỡng tối ưu $\tau^* = 0.465$.
  - Complement Naive Bayes: $\alpha = 2.0$, ngưỡng tối ưu $\tau^* = 0.220$.

### Bước 8. Quy trình so sánh và xếp hạng 8 ứng viên (Model Selection Workflow)
- Triển khai `ModelSelectionWorkflow` trong [`src/models/tuning.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/tuning.py).
- So sánh công bằng 8 mô hình ứng viên trên cùng tập Validation set ($2.903$ mẫu):
  1. `Linear SVM (Tuned)`: Accuracy **$70.07\%$**, Recall **$85.55\%$**, F1 **$70.93\%$** (Hạng 1).
  2. `Linear SVM (Default)`: Accuracy $69.96\%$, Recall $85.39\%$, F1 $70.82\%$ (Hạng 2).
  3. `Logistic Regression (Tuned)`: Accuracy $69.96\%$, Recall $85.39\%$, F1 $70.82\%$ (Hạng 3).
  4. `Logistic Regression (Default)`: Accuracy $69.72\%$, Recall $85.47\%$, F1 $70.67\%$ (Hạng 4).
  5. `SpamVotingEnsemble (Tuned)`: Accuracy $69.07\%$, Recall $85.47\%$, F1 $70.23\%$ (Hạng 5).
  6. `SpamVotingEnsemble (Default)`: Accuracy $68.72\%$, Recall $86.36\%$, F1 $70.21\%$ (Hạng 6).
  7. `Complement Naive Bayes (Tuned)`: Accuracy $63.66\%$, Recall $85.23\%$, F1 $66.69\%$ (Hạng 7).
  8. `Complement Naive Bayes (Default)`: Accuracy $63.42\%$, Recall $85.07\%$, F1 $66.50\%$ (Hạng 8).

### Bước 9. Chốt mô hình vô địch (Winner Selection) & Xuất Báo Cáo Ngưỡng
- Căn cứ theo tiêu chí đạt Recall $\ge 0.85$ và có Validation Accuracy cao nhất: **Linear SVM (Tuned)** đạt $70.07\%$, vượt qua toàn bộ các ứng viên còn lại.
- Hệ thống chính thức chọn Linear SVM (Tuned) làm **Selected Final Model** với ngưỡng quyết định chốt là **$\tau^* = 0.465$**.
- Xuất toàn bộ bảng kết quả so sánh ứng viên ra file [`results/metrics/validation_model_selection.csv`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/metrics/validation_model_selection.csv).
- Xuất bảng quét 200 mốc ngưỡng đánh đổi (Threshold Tradeoffs) của mô hình winner ra [`results/metrics/validation_threshold_tradeoffs.csv`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/metrics/validation_threshold_tradeoffs.csv) và vẽ biểu đồ đường cong độ nhạy ngưỡng tại [`results/figures/threshold_sensitivity.png`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/figures/threshold_sensitivity.png).
- Giai đoạn 3 kết thúc thành công với mô hình chiến thắng và ngưỡng tối ưu sẵn sàng chuyển giao cho Giai đoạn 4 đánh giá độc lập và đóng gói triển khai.

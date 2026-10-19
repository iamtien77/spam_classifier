# 🤝 Voting Ensemble (From-Scratch)

> **Vị trí trong dự án:** Phương pháp học kết hợp (Ensemble Learning) đa mô hình, đánh giá khả năng cải thiện độ ổn định và giảm phương sai dự đoán.  
> **Mã nguồn triển khai:** Class [`SpamVotingEnsemble`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/ensemble.py#L25) trong [`src/models/ensemble.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/ensemble.py).

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Trong học máy ứng dụng, một mô hình đơn lẻ có thể gặp phải các điểm mù (Blind spots) hoặc sai số cục bộ do đặc thù hàm mất mát và không gian giả thuyết của nó. 

Mục đích của phương pháp **Voting Ensemble** trong dự án:
- Tổng hợp sức mạnh phân loại từ ba họ thuật toán có nguyên lý hoạt động hoàn toàn khác nhau:
  1. **Linear SVM:** Tối ưu hóa siêu phẳng lề cực đại dựa trên các điểm hỗ trợ (Support Vectors).
  2. **Logistic Regression:** Ước lượng xác suất log-odds toàn cục trên toàn bộ mẫu dữ liệu.
  3. **Complement Naive Bayes:** Phân tích tần suất từ vựng trên tập phần bù của lớp.
- Giảm phương sai sai số (Variance Reduction), hạn chế rủi ro phụ thuộc vào một kiến trúc phân loại đơn lẻ.

### 1.2. Cơ chế hoạt động & Toán học kết hợp
Lớp `SpamVotingEnsemble` hỗ trợ hai cơ chế kết hợp:

#### A. Soft Voting (Bỏ phiếu mềm - Mặc định trong pipeline tham chiếu)
Mô hình tính trung bình cộng có trọng số của các phân phối xác suất dự đoán từ $M$ mô hình thành phần:

$$
P_{\text{ensemble}}(y = 1 \mid x) = \sum_{m=1}^{M} w_m \cdot P_m(y = 1 \mid x)
$$

Với $\sum_{m=1}^{M} w_m = 1$. Trong cấu hình mặc định của pipeline tham chiếu, trọng số được phân bổ đồng đều ($w_m = \frac{1}{3}$). Sau khi có xác suất tổng hợp, hệ thống áp dụng ngưỡng tối ưu $\tau^*$ được chọn từ tập Validation để ra quyết định nhị phân.

#### B. Hard Voting (Bỏ phiếu cứng theo đa số)
Mỗi mô hình đưa ra dự đoán nhãn rời rạc $\hat{y}_m \in \{0, 1\}$. Nhãn cuối cùng là nhãn nhận được đa số phiếu bầu:

$$
\hat{y}_{\text{ensemble}} = \operatorname{mode}\left( \{\hat{y}_1, \hat{y}_2, \dots, \hat{y}_M\} \right)
$$

### 1.3. Đánh đổi & Failure Modes
- **Trade-offs:** Soft Voting tận dụng được mức độ tin cậy (Confidence score) của từng mô hình, giúp đường biên xác suất mượt mà hơn. Tuy nhiên, thời gian huấn luyện và suy luận tăng gấp 3 lần do phải thực thi toàn bộ 3 mô hình nền tảng.
- **Failure Modes:** 
  - **Hiệu ứng kéo tụt (Dragging Down Effect):** Mô hình Ensemble chỉ đạt hiệu quả tối ưu khi các mô hình thành phần có sức mạnh tương đương. Khi một mô hình thành phần có độ chính xác yếu hơn đáng kể (ở đây là Complement Naive Bayes với Accuracy $64.71\%$), việc lấy trung bình đồng đều làm suy giảm độ chính xác của hai mô hình mạnh (Linear SVM $70.85\%$ và Logistic Regression $70.74\%$).

---

## 2. Input / Output (Đặc tả dữ liệu vào / ra)

### 2.1. Input
- **Danh sách mô hình thành phần (Estimators):** Bộ ba mô hình [`LogisticRegressionFromScratch`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/classifiers.py#L142), [`LinearSVMFromScratch`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/classifiers.py#L218) và [`NaiveBayesClassifier`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/classifiers.py#L330).
- **Ma trận đặc trưng $X$:** Kích thước $(N, 4318)$ thuộc lớp [`CSRMatrixScratch`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/features/feature_engineering.py#L45) hoặc ma trận thưa chuẩn.
- **Nhãn mục tiêu $y$:** Vector nhãn nhị phân $\{0, 1\}$.

### 2.2. Output
- **Xác suất trung bình Soft Voting ($\hat{p}_{\text{soft}}$):** Vector $N$ phần tử trong đoạn $[0, 1]$.
- **Nhãn phân loại cuối cùng ($\hat{y}$):**
  $$\hat{y}_i = \begin{cases} 1 & \text{nếu } \hat{p}_{\text{soft}, i} \ge \tau^* \\ 0 & \text{nếu } \hat{p}_{\text{soft}, i} < \tau^* \end{cases}$$
  Với $\tau^* = 0.385$ cho nhánh Tuned và $\tau^* = 0.365$ cho nhánh Default.

---

## 3. Data used (Dữ liệu sử dụng)

- **Train set:** $23.223$ mẫu ($13.309$ Ham, $9.914$ Spam) - Dùng huấn luyện độc lập cho cả 3 mô hình thành phần.
- **Validation set:** $2.903$ mẫu ($1.664$ Ham, $1.239$ Spam) - Dùng tổng hợp xác suất và quét lưới ngưỡng $\tau^*$.
- **Test set:** $2.902$ mẫu ($1.663$ Ham, $1.239$ Spam) - Dùng đánh giá kiểm chứng độc lập.
- **Không gian đặc trưng:** Thống nhất $4.318$ đặc trưng lai chuẩn hóa.

---

## 4. Evaluation (Phương pháp đánh giá)

- Tối ưu hóa ngưỡng qua [`ThresholdOptimizerScratch`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/tuning.py#L27) với điều kiện tiên quyết $\text{Recall} \ge 0.85$.
- Do xác suất tổng hợp là trung bình của SVM, LR và CNB, phân phối xác suất có xu hướng co cụm về trung tâm, dẫn đến việc ngưỡng tối ưu $\tau^*$ rơi vào khoảng $0.365 - 0.385$.

---

## 5. Status (Trạng thái triển khai trong dự án)

- **Trạng thái cài đặt:** Hoàn thành $100\%$ From-Scratch thuần Python/NumPy trong module `src/models/ensemble.py`.
- **Kiểm thử tự động:** Vượt qua $51/51$ tests trong thư mục [`tests/`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/tests).
- **Vị trí xếp hạng:** Xếp hạng 5 (nhánh Tuned) và hạng 6 (nhánh Default) trên tập Validation.
- **Kết luận kiến trúc:** Mô hình Ensemble được đưa vào quy trình so sánh công bằng như mọi ứng viên độc lập khác. Vì không đạt điểm Validation Accuracy cao hơn Linear SVM, Ensemble không được chọn làm mô hình cuối cùng để triển khai, đảm bảo tính thực nghiệm khách quan và tiết kiệm tài nguyên tính toán.

---

## 6. Results (Kết quả thực nghiệm định lượng)

### 6.1. Kết quả trên tập Validation
*(Nguồn trích xuất: [`results/metrics/validation_model_selection.csv`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/metrics/validation_model_selection.csv))*

| Cấu hình Ensemble | Ngưỡng ($\tau^*$) | Accuracy | Precision | Recall | F1-Score | F2-Score | Thứ hạng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SpamVotingEnsemble (Tuned)** | **$0.385$** | **$69.07\%$** | $59.59\%$ | $85.47\%$ | **$70.23\%$** | $78.64\%$ | **5 / 8** |
| **SpamVotingEnsemble (Default)**| **$0.365$** | **$68.72\%$** | $59.15\%$ | $86.36\%$ | **$70.21\%$** | $79.08\%$ | **6 / 8** |

### 6.2. Kết quả trên tập Test
*(Nguồn trích xuất: [`results/metrics/test_models_comparison.csv`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/metrics/test_models_comparison.csv))*

| Chỉ số đánh giá | Voting Ensemble (Tuned) | Voting Ensemble (Default) | So sánh với Linear SVM (Winner) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **$70.23\%$** | $69.47\%$ | $-0.62\%$ |
| **Precision** | **$60.83\%$** | $59.97\%$ | $-0.56\%$ |
| **Recall** | **$84.99\%$** | $85.71\%$ | $-0.48\%$ |
| **F1-Score** | **$70.91\%$** | $70.56\%$ | $-0.55\%$ |
| **F2-Score** | **$78.73\%$** | $78.94\%$ | $-0.52\%$ |
| **ROC-AUC** | **$80.26\%$** | $79.98\%$ | $-1.18\%$ |

### 6.3. Bài học thực nghiệm rút ra
Kết quả thực nghiệm cho thấy kết hợp Ensemble không phải lúc nào cũng mang lại hiệu năng cao hơn mô hình đơn lẻ tốt nhất. Trong bài toán này, Linear SVM tự thân đã tạo ra mặt phân cách lề cực đại vượt trội trên không gian đặc trưng thưa; việc hòa trộn xác suất của Naive Bayes đã làm giảm độ sắc bén của ranh giới quyết định. Do đó, quyết định chọn Linear SVM làm mô hình triển khai duy nhất là hoàn toàn có cơ sở khoa học và tối ưu về mặt chi phí vận hành.

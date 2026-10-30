# 🤝 Voting Ensemble (From-Scratch)

> **Vị trí trong dự án:** Phương pháp học kết hợp (Ensemble Learning) đa mô hình, dùng để kiểm tra khả năng cải thiện kết quả so với từng mô hình đơn lẻ.
> **Mã nguồn triển khai:** Class [`SpamVotingEnsemble`](../../src/models/ensemble.py#L25) trong [`src/models/ensemble.py`](../../src/models/ensemble.py).

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Trong học máy ứng dụng, một mô hình đơn lẻ có thể gặp phải các điểm mù (Blind spots) hoặc sai số cục bộ do đặc thù hàm mất mát và không gian giả thuyết của nó. 

Mục đích của phương pháp **Voting Ensemble** trong dự án:
- Tổng hợp sức mạnh phân loại từ ba họ thuật toán có nguyên lý hoạt động hoàn toàn khác nhau:
  1. **Linear SVM:** Tối ưu hóa siêu phẳng lề cực đại dựa trên các điểm hỗ trợ (Support Vectors).
  2. **Logistic Regression:** Ước lượng xác suất log-odds toàn cục trên toàn bộ mẫu dữ liệu.
  3. **Complement Naive Bayes:** Phân tích tần suất từ vựng trên tập phần bù của lớp.
- Kiểm tra liệu việc kết hợp các mô hình có giảm sai số trên tập Validation hay không; hiệu quả giảm phương sai cần được xác minh qua nhiều lần chia dữ liệu.

### 1.2. Cơ chế hoạt động & Toán học kết hợp
Lớp `SpamVotingEnsemble` hỗ trợ hai cơ chế kết hợp:

#### A. Soft Voting (Bỏ phiếu mềm - Mặc định trong pipeline tham chiếu)
Mô hình tính trung bình có trọng số của điểm số trong $[0,1]$ từ $M$ mô hình thành phần:

$$
s_{\text{ensemble}}(x) = \sum_{m=1}^{M} w_m \cdot s_m(x)
$$

Với $\sum_{m=1}^{M} w_m = 1$ sau chuẩn hóa. Trong pipeline tham chiếu, nhánh **Default** đặt trọng số thô LR:SVM:CNB = $1:1:1.2$ (sau chuẩn hóa là $1/3.2:1/3.2:1.2/3.2$); nhánh **Tuned** dùng trọng số đều $1:1:1$. Điểm SVM là sigmoid của margin chưa được calibration, nên điểm tổng hợp cũng không được diễn giải là xác suất hậu nghiệm đã calibration. Hệ thống chọn ngưỡng $\tau^*$ trên Validation để phân loại.

#### B. Hard Voting (Bỏ phiếu cứng theo đa số)
Mỗi mô hình đưa ra dự đoán nhãn rời rạc $`\hat{y}_m \in \{0, 1\}`$. Nhãn cuối cùng là nhãn nhận được đa số phiếu bầu:

$$
\hat{y}_{\text{ensemble}} = \operatorname{mode}\left( \{\hat{y}_1, \hat{y}_2, \dots, \hat{y}_M\} \right)
$$

### 1.3. Đánh đổi & Failure Modes
- **Trade-offs:** Soft Voting sử dụng điểm số của ba mô hình; thang điểm chưa calibration có thể khiến một thành viên ảnh hưởng quá mức. Chi phí suy luận gồm ba lần dự đoán; chi phí huấn luyện tăng nếu phải fit cả ba mô hình.
- **Failure Modes:** 
  - **Hiệu ứng kéo tụt (Dragging Down Effect):** CNB có Test Accuracy $64.71\%$, thấp hơn Linear SVM ($70.85\%$) và LR ($70.74\%$); điểm Ensemble cũng thấp hơn winner trong lần chạy tham chiếu. Chưa có ablation về trọng số/calibration để kết luận riêng CNB là nguyên nhân.

---

## 2. Input / Output (Đặc tả dữ liệu vào / ra)

### 2.1. Input
- **Danh sách mô hình thành phần (Estimators):** Bộ ba mô hình [`LogisticRegressionFromScratch`](../../src/models/classifiers.py#L142), [`LinearSVMFromScratch`](../../src/models/classifiers.py#L218) và [`NaiveBayesClassifier`](../../src/models/classifiers.py#L330).
- **Ma trận đặc trưng $X$:** Kích thước $(N, D_{\text{selected}})$ thuộc lớp [`CSRMatrixScratch`](../../src/features/feature_engineering.py#L45) hoặc ma trận thưa chuẩn; $D_{\text{selected}}=4318$ trong lần chạy tham chiếu.
- **Nhãn mục tiêu $y$:** Vector nhãn nhị phân $`\{0, 1\}`$.

### 2.2. Output
- **Điểm Soft Voting ($\hat{p}_{\text{soft}}$):** Vector $N$ phần tử trong đoạn $[0, 1]$; tên biến giữ theo API, giá trị chưa calibration.
- **Nhãn phân loại cuối cùng ($\hat{y}$):**

  ```math
  \hat{y}_i = \begin{cases} 1 & \text{nếu } \hat{p}_{\text{soft}, i} \ge \tau^* \\ 0 & \text{nếu } \hat{p}_{\text{soft}, i} < \tau^* \end{cases}
  ```

  Với $\tau^* = 0.385$ cho nhánh Tuned và $\tau^* = 0.365$ cho nhánh Default.

---

## 3. Data used (Dữ liệu sử dụng)

- **Train set:** $23.223$ mẫu ($13.309$ Ham, $9.914$ Spam) - Dùng huấn luyện độc lập cho cả 3 mô hình thành phần.
- **Validation set:** $2.903$ mẫu ($1.664$ Ham, $1.239$ Spam) - Dùng tổng hợp điểm số và quét lưới ngưỡng $\tau^*$.
- **Test set:** $2.902$ mẫu ($1.663$ Ham, $1.239$ Spam) - Dùng đánh giá kiểm chứng độc lập.
- **Không gian đặc trưng:** $4.318$ cột trong lần chạy tham chiếu; số cột của lần chạy khác phụ thuộc builder và Top-K.

---

## 4. Evaluation (Phương pháp đánh giá)

- Tối ưu hóa ngưỡng qua [`ThresholdOptimizerScratch`](../../src/models/tuning.py#L27) với điều kiện tiên quyết $\text{Recall} \ge 0.85$.
- Trong lần chạy tham chiếu, ngưỡng chọn trên Validation là $0.365$ (Default) và $0.385$ (Tuned); các giá trị này phụ thuộc thang điểm và trọng số từng nhánh.

---

## 5. Status (Trạng thái triển khai trong dự án)

- **Trạng thái cài đặt:** Hoàn thành $100\%$ From-Scratch thuần Python/NumPy trong module `src/models/ensemble.py`.
- **Kiểm thử tự động:** Vượt qua $51/51$ tests trong thư mục [`tests/`](../../tests).
- **Vị trí xếp hạng:** Xếp hạng 5 (nhánh Tuned) và hạng 6 (nhánh Default) trên tập Validation.
- **Kết luận kiến trúc:** Mô hình Ensemble được đưa vào quy trình so sánh công bằng như mọi ứng viên độc lập khác. Vì không đạt điểm Validation Accuracy cao hơn Linear SVM, Ensemble không được chọn làm mô hình cuối cùng để triển khai, đảm bảo tính thực nghiệm khách quan và tiết kiệm tài nguyên tính toán.

---

## 6. Results (Kết quả thực nghiệm định lượng)

### 6.1. Kết quả trên tập Validation
*(Nguồn trích xuất: [`results/metrics/validation_model_selection.csv`](../../results/metrics/validation_model_selection.csv))*

| Cấu hình Ensemble | Ngưỡng ($\tau^*$) | Accuracy | Precision | Recall | F1-Score | F2-Score | Thứ hạng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SpamVotingEnsemble (Tuned)** | **$0.385$** | **$69.07\%$** | $59.59\%$ | $85.47\%$ | **$70.23\%$** | $78.64\%$ | **5 / 8** |
| **SpamVotingEnsemble (Default)**| **$0.365$** | **$68.72\%$** | $59.15\%$ | $86.36\%$ | **$70.21\%$** | $79.08\%$ | **6 / 8** |

### 6.2. Kết quả trên tập Test
*(Nguồn trích xuất: [`results/metrics/test_models_comparison.csv`](../../results/metrics/test_models_comparison.csv))*

| Chỉ số đánh giá | Voting Ensemble (Tuned) | Voting Ensemble (Default) | So sánh với Linear SVM (Winner) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **$70.23\%$** | $69.47\%$ | $-0.62\%$ |
| **Precision** | **$60.83\%$** | $59.97\%$ | $-0.56\%$ |
| **Recall** | **$84.99\%$** | $85.71\%$ | $-0.48\%$ |
| **F1-Score** | **$70.91\%$** | $70.56\%$ | $-0.55\%$ |
| **F2-Score** | **$78.73\%$** | $78.94\%$ | $-0.52\%$ |
| **ROC-AUC** | **$80.26\%$** | $79.98\%$ | $-1.18\%$ |

### 6.3. Bài học thực nghiệm rút ra
Kết quả trên split tham chiếu cho thấy cả hai nhánh Ensemble có Validation Accuracy thấp hơn Linear SVM (Tuned), nên quy tắc chọn mô hình đã chọn SVM. Test Recall của Ensemble (Tuned) là $84.99\%$, thấp hơn mốc $85\%$ dù Validation Recall đạt $85.47\%$. Để xác định nguyên nhân chênh lệch hoặc kết luận về độ ổn định, cần thêm ablation trọng số, calibration và đánh giá qua nhiều split; Test không được dùng để chọn lại mô hình.

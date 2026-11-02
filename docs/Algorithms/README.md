# Các thuật toán học máy trong dự án

Thư mục này trình bày các thuật toán phân loại email spam được sử dụng trong dự án **Spam Classifier (Practice 1: Classifying Spam Emails)**.

Xem [design.md](./design.md) để nắm thiết kế theo **5W1H**, luồng Train → Validation → Test → suy luận và **cấu trúc thư mục của toàn dự án**. Tài liệu thiết kế trình bày cách tổ chức hệ thống; các trang bên dưới tập trung vào từng thuật toán.

Tất cả các thuật toán đều được **tự xây dựng từ đầu (From-Scratch)** bằng Python và NumPy/SciPy, tuân thủ nguyên tắc không sử dụng mô hình học máy hộp đen có sẵn từ scikit-learn.

---

## 4 giai đoạn của pipeline

Phần phân tích theo 4 khía cạnh giúp làm rõ lý do lựa chọn và giới hạn của từng phương án. Bộ 4 tài liệu dưới đây trình bày cách xử lý ở từng giai đoạn cùng chức năng của các lớp và hàm:

| Giai đoạn | Tài liệu chi tiết | Module mã nguồn chính | Nội dung chính |
| :---: | :--- | :--- | :--- |
| **Giai đoạn 1** | 📥 [01_data_preprocessing_and_audit.md](../description_for_AI_working/01_data_preprocessing_and_audit.md) | `loader.py`, `preprocessing.py` | Xử lý HTML theo sự kiện, loại bản ghi trùng và các nhóm mâu thuẫn nhãn trên toàn bộ dữ liệu, chia phân tầng Hamilton 80/10/10 để tránh rò rỉ dữ liệu. |
| **Giai đoạn 2** | ⚙️ [02_feature_engineering_and_selection.md](../description_for_AI_working/02_feature_engineering_and_selection.md) | `feature_engineering.py` | Cấu trúc ma trận thưa nén hàng `CSRMatrixScratch`, Sublinear TF-IDF, Char n-gram 3-5, `MaxAbsScalerScratch` bảo toàn độ thưa, Top-K 4.318 chiều. |
| **Giai đoạn 3** | 🧠 [03_model_training_and_tuning.md](../description_for_AI_working/03_model_training_and_tuning.md) | `classifiers.py`, `tuning.py`, `ensemble.py` | 3 mô hình cốt lõi From-Scratch (Balanced BCE, Squared Hinge $C^1$, Complement NB), Barzilai-Borwein + Armijo Line Search, dò ngưỡng sàn Recall $\ge 85\%$. |
| **Giai đoạn 4** | 🚀 [04_evaluation_and_deployment.md](../description_for_AI_working/04_evaluation_and_deployment.md) | `metrics.py`, `error_analysis.py`, `persistence.py`, `predictor.py` | Kiểm chứng độc lập Test (Mann-Whitney U ROC-AUC), phân tích 846 trường hợp dự đoán sai, Joblib Atomic Bundle, dịch vụ suy luận `SpamInferenceService` (< 2.5 ms SLA, XAI). |

---

## 1. Danh Mục Tài Liệu Kỹ Thuật

Mỗi tài liệu gồm 6 phần: mục đích, dữ liệu đầu vào và kết quả đầu ra, dữ liệu sử dụng, phương pháp đánh giá, trạng thái triển khai và kết quả thực nghiệm:

| Tài liệu | Thuật toán | Vai trò trong hệ thống | Trạng thái / Thứ hạng |
| :--- | :--- | :--- | :---: |
| 🛡️ [linear_svm.md](./linear_svm.md) | **Linear Support Vector Machine** | Tìm siêu phẳng phân cách lề cực đại với Squared Hinge Loss | **Mô hình được chọn để triển khai** |
| 📈 [logistic_regression.md](./logistic_regression.md) | **Logistic Regression** | Phân loại xác suất nhị phân với Binary Cross-Entropy & Armijo | 🥈 **Hạng 2 trên Test (F1: 71.42%)** |
| 🎲 [naive_bayes.md](./naive_bayes.md) | **Complement & Multinomial Naive Bayes** | Đường cơ sở xác suất và Proxy Model cho Top-K Feature Selection | 🥉 **Baseline thực nghiệm** |
| 🤝 [voting_ensemble.md](./voting_ensemble.md) | **Spam Voting Ensemble** | Kết hợp Soft/Hard Voting và so sánh với mô hình đơn lẻ | 🏅 **Mô hình nghiên cứu thực nghiệm** |

---

## 2. So sánh và lựa chọn mô hình trên tập Validation

Quá trình lựa chọn mô hình được thực hiện trên tập **Validation** ($2.903$ mẫu) với ràng buộc nghiệp vụ: $\text{Recall} \ge 0.85$, sau đó ưu tiên **Accuracy $\rightarrow$ F1-Score $\rightarrow$ Precision $\rightarrow$ FPR**:

*(Nguồn số liệu: [`results/metrics/validation_model_selection.csv`](../../results/metrics/validation_model_selection.csv))*

| Thứ hạng | Mô hình ứng viên | Nhánh tham số | Ngưỡng ($\tau^*$) | Thỏa mãn Recall $\ge 85\%$ | Validation Accuracy | Validation Recall | Validation F1 | Đạt chuẩn chọn |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Linear SVM (Tuned)** | Grid Search ($C=2.0$) | **$0.465$** | **True** | **$70.07\%$** | **$85.55\%$** | **$70.93\%$** | **Được chọn** |
| 2 | Linear SVM (Default) | Cố định ($C=1.0$) | $0.465$ | True | $69.96\%$ | $85.39\%$ | $70.82\%$ | Không |
| 3 | Logistic Regression (Tuned) | Grid Search ($C=0.5$) | $0.415$ | True | $69.96\%$ | $85.39\%$ | $70.82\%$ | Không |
| 4 | Logistic Regression (Default)| Cố định ($C=1.0$) | $0.410$ | True | $69.72\%$ | $85.47\%$ | $70.67\%$ | Không |
| 5 | SpamVotingEnsemble (Tuned) | Soft Voting | $0.385$ | True | $69.07\%$ | $85.47\%$ | $70.23\%$ | Không |
| 6 | SpamVotingEnsemble (Default)| Soft Voting | $0.365$ | True | $68.72\%$ | $86.36\%$ | $70.21\%$ | Không |
| 7 | Complement NB (Tuned) | Grid Search ($\alpha=2.0$)| $0.220$ | True | $63.66\%$ | $85.23\%$ | $66.69\%$ | Không |
| 8 | Complement NB (Default) | Cố định ($\alpha=1.0$) | $0.215$ | True | $63.42\%$ | $85.07\%$ | $66.50\%$ | Không |

---

## 3. Kết quả đánh giá trên tập Test

Đánh giá khách quan trên tập **Test** ($2.902$ mẫu) tại các ngưỡng tối ưu đã được chốt từ Validation set:

*(Nguồn số liệu: [`results/metrics/test_models_comparison.csv`](../../results/metrics/test_models_comparison.csv))*

| Mô hình | Accuracy | Precision | Recall | F1-Score | F2-Score | ROC-AUC | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Tuned)** | **$70.85\%$** | **$61.39\%$** | $85.47\%$ | **$71.46\%$** | $79.25\%$ | **$81.44\%$** | 🏆 **Mô hình triển khai** |
| **Logistic Regression (Tuned)** | $70.74\%$ | $61.26\%$ | **$85.63\%$** | $71.42\%$ | **$79.32\%$** | $81.38\%$ | Ứng viên mạnh thứ 2 |
| **Linear SVM (Default)** | $70.74\%$ | $61.31\%$ | $85.31\%$ | $71.35\%$ | $79.12\%$ | $81.44\%$ | Đối chứng |
| **Logistic Regression (Default)**| $70.33\%$ | $60.87\%$ | $85.39\%$ | $71.08\%$ | $79.03\%$ | $81.32\%$ | Đối chứng |
| **SpamVotingEnsemble (Tuned)** | $70.23\%$ | $60.83\%$ | $84.99\%$ | $70.91\%$ | $78.73\%$ | $80.26\%$ | Thực nghiệm |
| **SpamVotingEnsemble (Default)**| $69.47\%$ | $59.97\%$ | $85.71\%$ | $70.56\%$ | $78.94\%$ | $79.98\%$ | Thực nghiệm |
| **Complement NB (Tuned)** | $64.71\%$ | $55.68\%$ | $84.99\%$ | $67.28\%$ | $76.89\%$ | $75.22\%$ | Baseline |
| **Complement NB (Default)** | $64.58\%$ | $55.56\%$ | $85.07\%$ | $67.22\%$ | $76.90\%$ | $75.22\%$ | Baseline |

---

## 4. Nhận xét và lựa chọn kỹ thuật

1. **Hiệu năng của mô hình tuyến tính:** Trong lần chạy tham chiếu, Linear SVM và Logistic Regression có Accuracy và F1 cao hơn Complement Naive Bayes trên cùng Validation/Test split; $4.318$ là số cột được giữ sau Top-K của lần chạy này.
2. **Lý do chọn Linear SVM:** Linear SVM (Tuned) có Validation Accuracy cao nhất ($70.07\%$) trong nhóm đạt Validation Recall $\ge 85\%$. Test Accuracy $70.85\%$ và Recall $85.47\%$ chỉ được dùng để đánh giá sau khi đã chốt winner; một split chưa đủ để kết luận khả năng chống nhiễu.
3. **Giới hạn khi sử dụng:** Mặc dù mô hình đạt Recall cao ($85.47\%$), tỷ lệ chặn nhầm thư hợp lệ vẫn ở mức $40.05\%$. Vì vậy, pipeline này được sử dụng trong phạm vi thực nghiệm học tập và phù hợp để **cảnh báo / đưa thư vào hòm thư rác tạm thời** thay vì tự động xóa vĩnh viễn.

---

## 5. Quy Trình Chọn Ngưỡng, Tham Số Và Mô Hình

Đây là bài toán **binary text classification** với nhãn Spam = 1. `RECALL_TARGET = 0.85` là mục tiêu thực nghiệm của dự án trên **Validation**, không phải mức bảo đảm cho Test hay dữ liệu thực tế. Baseline là Complement Naive Bayes (`alpha=1.0`); Linear SVM, Logistic Regression và Voting Ensemble được so sánh với baseline bằng cùng split và bộ chỉ số Accuracy, Precision, Recall, F1, F2, ROC-AUC, FPR và confusion matrix.

1. **Train:** Fit từ vựng, IDF, scaler, xếp hạng đặc trưng và trọng số mô hình. `HybridFeatureBuilderScratch` tạo tối đa số cột theo cấu hình; `ManualFeatureSelectorScratch` chọn Top-K trên Train/Validation. Lần chạy tham chiếu có $4.318$ cột trước và sau chọn lọc.
2. **Tuning trên Validation:** Với mỗi cấu hình hyperparameter, [`ThresholdOptimizerScratch`](../../src/models/tuning.py#L27) quét 200 ngưỡng cách đều từ $0.005$ đến $1.0$. Trong nhóm đạt Recall $\ge 0.85$, chọn Accuracy cao nhất, rồi phân xử bằng F1, Precision và FPR thấp hơn. Nếu không ngưỡng nào đạt, ưu tiên Recall, sau đó Accuracy và F1.
3. **Chọn cấu hình và winner:** [`tune_model_hyperparameters`](../../src/models/tuning.py#L159) so sánh các cấu hình theo điều kiện đạt Recall, rồi Accuracy/F1. [`ModelSelectionWorkflow`](../../src/models/tuning.py#L235) quét ngưỡng cho cả nhánh Default, Tuned và Ensemble, xếp hạng theo điều kiện đạt Recall, điểm mục tiêu, Accuracy và F1. Thứ tự ứng viên ban đầu được giữ nếu hòa hoàn toàn.
4. **Test:** Khóa cấu hình, winner và ngưỡng từ Validation trước khi tính chỉ số trên Test. Vì vậy Test Recall $84.99\%$ của CNB Tuned và Voting Tuned là thấp hơn $85\%$, dù cả hai đạt mục tiêu trên Validation. Không chọn lại model/ngưỡng dựa trên Test.

Các điểm sigmoid của SVM và điểm Voting chưa được calibration; ngay cả điểm LR/CNB cũng chưa có báo cáo calibration trong lần chạy này. Dùng chúng để xếp hạng và chọn ngưỡng, không diễn giải trực tiếp thành xác suất rủi ro thực tế. Dữ liệu hiện là SMS ngắn, chưa kiểm chứng trên email có metadata/HTML; đánh giá nhiều split và kiểm tra calibration là bước tiếp theo nếu muốn kết luận về độ ổn định hoặc vận hành.

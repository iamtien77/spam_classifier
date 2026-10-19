# 📚 Tổng Hợp Các Thuật Toán Học Máy Trong Dự Án (Algorithms Overview)

Thư mục này chứa toàn bộ tài liệu kỹ thuật chuyên sâu về các thuật toán phân loại email spam được triển khai trong dự án **Spam Classifier (Practice 1: Classifying Spam Emails)**.

Tất cả các thuật toán đều được **tự xây dựng từ đầu (From-Scratch)** bằng Python và NumPy/SciPy, tuân thủ nguyên tắc không sử dụng mô hình học máy hộp đen có sẵn từ scikit-learn.

---

## 1. Danh Mục Tài Liệu Kỹ Thuật

Mỗi thuật toán được trình bày độc lập theo cấu trúc chuẩn mực gồm 6 phần: **Purpose**, **Input / Output**, **Data used**, **Evaluation**, **Status** và **Results**:

| Tài liệu | Thuật toán | Vai trò trong hệ thống | Trạng thái / Thứ hạng |
| :--- | :--- | :--- | :---: |
| 🛡️ [linear_svm.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/Algorithms/linear_svm.md) | **Linear Support Vector Machine** | Tìm siêu phẳng phân cách lề cực đại với Squared Hinge Loss | 🥇 **Winner (Mô hình triển khai)** |
| 📈 [logistic_regression.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/Algorithms/logistic_regression.md) | **Logistic Regression** | Phân loại xác suất nhị phân với Binary Cross-Entropy & Armijo | 🥈 **Hạng 2 trên Test (F1: 71.42%)** |
| 🎲 [naive_bayes.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/Algorithms/naive_bayes.md) | **Complement & Multinomial Naive Bayes** | Đường cơ sở xác suất và Proxy Model cho Top-K Feature Selection | 🥉 **Baseline thực nghiệm** |
| 🤝 [voting_ensemble.md](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/docs/Algorithms/voting_ensemble.md) | **Spam Voting Ensemble** | Kết hợp Soft/Hard Voting đa mô hình để kiểm soát phương sai | 🏅 **Mô hình nghiên cứu thực nghiệm** |

---

## 2. Bảng So Sánh Đối Đầu Trên Tập Validation (Lựa Chọn Mô Hình)

Quá trình lựa chọn mô hình được thực hiện trên tập **Validation** ($2.903$ mẫu) với ràng buộc nghiệp vụ: $\text{Recall} \ge 0.85$, sau đó ưu tiên **Accuracy $\rightarrow$ F1-Score $\rightarrow$ Precision $\rightarrow$ FPR**:

*(Nguồn trích xuất: [`results/metrics/validation_model_selection.csv`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/metrics/validation_model_selection.csv))*

| Thứ hạng | Mô hình ứng viên | Nhánh tham số | Ngưỡng ($\tau^*$) | Thỏa mãn Recall $\ge 85\%$ | Validation Accuracy | Validation Recall | Validation F1 | Đạt chuẩn chọn |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Linear SVM (Tuned)** | Grid Search ($C=2.0$) | **$0.465$** | **True** | **$70.07\%$** | **$85.55\%$** | **$70.93\%$** | 🏆 **CHỌN LÀM WINNER** |
| 2 | Linear SVM (Default) | Cố định ($C=1.0$) | $0.465$ | True | $69.96\%$ | $85.39\%$ | $70.82\%$ | Không |
| 3 | Logistic Regression (Tuned) | Grid Search ($C=0.5$) | $0.415$ | True | $69.96\%$ | $85.39\%$ | $70.82\%$ | Không |
| 4 | Logistic Regression (Default)| Cố định ($C=1.0$) | $0.410$ | True | $69.72\%$ | $85.47\%$ | $70.67\%$ | Không |
| 5 | SpamVotingEnsemble (Tuned) | Soft Voting | $0.385$ | True | $69.07\%$ | $85.47\%$ | $70.23\%$ | Không |
| 6 | SpamVotingEnsemble (Default)| Soft Voting | $0.365$ | True | $68.72\%$ | $86.36\%$ | $70.21\%$ | Không |
| 7 | Complement NB (Tuned) | Grid Search ($\alpha=2.0$)| $0.220$ | True | $63.66\%$ | $85.23\%$ | $66.69\%$ | Không |
| 8 | Complement NB (Default) | Cố định ($\alpha=1.0$) | $0.215$ | True | $63.42\%$ | $85.07\%$ | $66.50\%$ | Không |

---

## 3. Bảng So Sánh Độc Lập Trên Tập Test (Final Evaluation)

Đánh giá khách quan trên tập **Test** ($2.902$ mẫu) tại các ngưỡng tối ưu đã được chốt từ Validation set:

*(Nguồn trích xuất: [`results/metrics/test_models_comparison.csv`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/metrics/test_models_comparison.csv))*

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

## 4. Tổng Kết Kiến Trúc & Quyết Định Kỹ Thuật

1. **Hiệu năng của mô hình tuyến tính:** Cả Linear SVM và Logistic Regression đều hoạt động rất tốt trên không gian đặc trưng thưa $4.318$ chiều, vượt trội hơn mô hình xác suất Naive Bayes xấp xỉ $+6\%$ về Accuracy và $+4\%$ về F1-Score.
2. **Lý do chọn Linear SVM:** Linear SVM (Tuned) mang lại Accuracy cao nhất ($70.07\%$ trên Validation và $70.85\%$ trên Test) trong khi vẫn bảo toàn mục tiêu bắt trúng trên $85\%$ email rác. Ranh giới lề cực đại (Maximum Margin) giúp mô hình chống chịu tốt trước các biến thể văn bản spam gây nhiễu.
3. **Thực tế vận hành:** Mặc dù mô hình đạt Recall cao ($85.47\%$), tỷ lệ chặn nhầm thư hợp lệ vẫn ở mức $40.05\%$. Vì vậy, pipeline này được định vị là hệ thống thực nghiệm học tập và phù hợp để **cảnh báo / đưa thư vào hòm thư rác tạm thời** thay vì tự động xóa vĩnh viễn.

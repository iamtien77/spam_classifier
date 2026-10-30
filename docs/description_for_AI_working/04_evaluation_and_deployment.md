# Bản Mô Tả Kỹ Thuật Giai Đoạn 4: Đánh Giá Độc Lập, Phân Tích Lỗi & Triển Khai Suy Luận (Evaluation & Deployment)

Mục tiêu của tài liệu này là quy định chi tiết từng bước thực thi cho Giai đoạn 4 của hệ thống: từ việc đánh giá khách quan mô hình chiến thắng trên tập Test độc lập, tính toán ma trận nhầm lẫn và chỉ số phân loại, trực quan hóa biểu đồ, phân tích mẫu lỗi (Error Analysis), đóng gói artifact bundle joblib đến việc cung cấp dịch vụ suy luận (Inference Service) cho email mới.

---

### Bước 1. Tiếp nhận mô hình chiến thắng và dữ liệu kiểm thử độc lập
- Nhận mô hình chiến thắng từ Giai đoạn 3: **Linear SVM (Tuned)** cùng ngưỡng quyết định tối ưu đã khóa: **$\tau^* = 0.465$**.
- Nhận ma trận đặc trưng tập Test: $X_{\text{test}} \in \mathbb{R}^{2902 \times 4318}$ và vector nhãn $y_{\text{test}}$ ($1.663$ Ham, $1.239$ Spam).
- **Quy tắc kiểm thử bất biến:** Tập Test tuyệt đối không được tham gia vào bất kỳ khâu huấn luyện, chọn siêu tham số hay tinh chỉnh ngưỡng nào trước đó. Ngưỡng $\tau^* = 0.465$ được giữ nguyên vẹn để bảo đảm tính khách quan 100%.

### Bước 2. Thực hiện dự đoán trên tập Test
- Gọi hàm tính xác suất / điểm sigmoid xấp xỉ trên tập Test:

  ```math
  p_i = \sigma(w^\top x_{\text{test}, i} + b) = \frac{1}{1 + \exp(-(w^\top x_{\text{test}, i} + b))}
  ```

- Áp dụng ngưỡng phân loại $\tau^* = 0.465$ để xác định nhãn nhị phân:

  ```math
  \hat{y}_i = \begin{cases} 1 & \text{nếu } p_i \ge 0.465 \\ 0 & \text{nếu } p_i < 0.465 \end{cases}
  ```

### Bước 3. Tính toán ma trận nhầm lẫn (Confusion Matrix)
- Sử dụng hàm `compute_confusion_matrix()` trong lớp `SpamModelEvaluator` ([`src/evaluation/metrics.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/evaluation/metrics.py)).
- Quy ước trục ma trận: **Hàng là Nhãn thực tế ($y$), Cột là Nhãn dự đoán ($\hat{y}$)**.
- Ghi nhận 4 giá trị định lượng trên $2.902$ mẫu kiểm thử:
  - **True Negatives ($TN$):** **$997$ mẫu** (Email hợp lệ được phân loại đúng là Ham).
  - **False Positives ($FP$):** **$666$ mẫu** (Email hợp lệ bị phân loại nhầm thành Spam).
  - **False Negatives ($FN$):** **$180$ mẫu** (Email rác bị bỏ sót, gán nhầm thành Ham).
  - **True Positives ($TP$):** **$1.059$ mẫu** (Email rác được phát hiện chính xác là Spam).

### Bước 4. Tính toán bộ chỉ số đánh giá toàn diện (Performance Metrics)
- Tính toán đầy đủ các chỉ số theo công thức chuẩn:
  1. **Accuracy (Độ chính xác toàn thể):**

     ```math
     \text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN} = \frac{1059 + 997}{2902} = \mathbf{70,85\%} \quad (0.70848)
     ```

  2. **Precision (Độ chuẩn xác trên lớp Spam):**

     ```math
     \text{Precision} = \frac{TP}{TP + FP} = \frac{1059}{1059 + 666} = \mathbf{61,39\%} \quad (0.61391)
     ```

  3. **Recall / Sensitivity (Độ nhạy phát hiện Spam):**

     ```math
     \text{Recall} = \frac{TP}{TP + FN} = \frac{1059}{1059 + 180} = \mathbf{85,47\%} \quad (0.85472)
     ```

     *(Thỏa mãn chỉ tiêu nghiệp vụ đặt ra ban đầu $\text{Recall} \ge 0.85$)*.
  4. **F1-Score (Trung bình điều hòa giữa Precision và Recall):**

     ```math
     \text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \mathbf{71,46\%} \quad (0.71457)
     ```

  5. **F2-Score ($\beta = 2.0$ - Đo lường ưu tiên bắt spam):**

     ```math
     \text{F2} = (1 + 2^2) \times \frac{\text{Precision} \times \text{Recall}}{2^2 \times \text{Precision} + \text{Recall}} = \mathbf{79,25\%} \quad (0.79255)
     ```

  6. **Specificity (Tỷ lệ nhận diện đúng email hợp lệ):**

     ```math
     \text{Specificity} = \frac{TN}{TN + FP} = \frac{997}{997 + 666} = \mathbf{59,95\%} \quad (0.59952)
     ```

  7. **False Positive Rate ($FPR$ - Tỷ lệ chặn nhầm):**

     ```math
     \text{FPR} = \frac{FP}{FP + TN} = \frac{666}{1663} = \mathbf{40,05\%} \quad (0.40048)
     ```

  8. **ROC-AUC (Diện tích dưới đường cong ROC):**
     - Tự cài đặt trong `compute_roc_auc()` dựa trên thống kê xếp hạng Wilcoxon-Mann-Whitney có xử lý điểm số đồng hạng (Ties handling):

         ```math
         \text{ROC-AUC} = \mathbf{81,44\%} \quad (0.81436)
         ```

     - Chứng minh năng lực phân tách thứ bậc điểm giữa hai lớp của Linear SVM là rất tốt.

### Bước 5. Đối chiếu kết quả Test của tất cả các mô hình ứng viên
- Nạp toàn bộ kết quả kiểm thử của 8 mô hình ứng viên vào bảng [`results/metrics/test_models_comparison.csv`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/metrics/test_models_comparison.csv):
  - Linear SVM (Tuned): Accuracy **$70.85\%$**, F1 **$71.46\%$**, ROC-AUC **$81.44\%$** (Vô địch).
  - Logistic Regression (Tuned): Accuracy $70.74\%$, F1 $71.42\%$, ROC-AUC $81.38\%$ (Sát nút).
  - SpamVotingEnsemble (Tuned): Accuracy $70.23\%$, F1 $70.91\%$, ROC-AUC $80.26\%$.
  - Complement Naive Bayes (Tuned): Accuracy $64.71\%$, F1 $67.28\%$, ROC-AUC $75.22\%$.
- Khẳng định quyết định lựa chọn Linear SVM là chuẩn xác cả trên Validation và Test.

### Bước 6. Trích xuất và phân tích mẫu phân loại sai (Error Analysis)
- Sử dụng module [`src/evaluation/error_analysis.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/evaluation/error_analysis.py).
- Lọc ra toàn bộ **846 mẫu dự đoán sai** trên tập Test ($666$ FP và $180$ FN).
- Xuất danh sách chi tiết kèm `source_id`, nhãn thực tế, nhãn dự đoán, xác suất và nội dung văn bản ra file [`results/metrics/misclassified_samples.csv`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/metrics/misclassified_samples.csv).
- **Đánh giá bản chất lỗi:**
  - **Nhóm False Positive (Ham bị chặn nhầm):** Thường là các email công việc chứa nhiều con số (báo cáo tài chính, số liệu kỹ thuật), link họp online hoặc chứa các từ ngữ nhấn mạnh khẩn cấp (`urgent`, `call`, `offer`).
  - **Nhóm False Negative (Spam bị lọt lưới):** Thường là các email rác rất ngắn, nội dung mơ hồ, hoặc cố tình làm nhiễu từ ngữ để tránh các từ khóa spam điển hình.

### Bước 7. Trực quan hóa và lưu trữ biểu đồ đánh giá
- Sử dụng module [`src/evaluation/visualization.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/evaluation/visualization.py).
- Vẽ và lưu các biểu đồ chất lượng cao (300 DPI) vào thư mục `results/figures/`:
  - `test_confusion_matrix.png`: Ma trận nhầm lẫn trực quan hóa bằng Seaborn heatmap.
  - `roc_curve.png`: Đường cong ROC biểu diễn đánh đổi giữa TPR và FPR qua mọi ngưỡng.
  - `precision_recall_curve.png`: Đường cong PR thể hiện tương quan giữa Precision và Recall.
  - `test_models_comparison.png`: Biểu đồ cột so sánh Accuracy, F1 và Recall giữa các mô hình.

### Bước 8. Đóng gói mô hình và bộ biến đổi (Model Persistence)
- Sử dụng module [`src/models/persistence.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/models/persistence.py).
- Đóng gói toàn bộ đối tượng phục vụ suy luận vào một Bundle Dictionary duy nhất:
  ```python
  bundle = {
      "model": winner,                       # Mô hình phân loại tối ưu (LinearSVMFromScratch)
      "run_metadata": run_metadata,          # SHA-256 fingerprint của dữ liệu thô và 21 file mã nguồn
      "text_processor": text_processor,      # Bộ tiền xử lý và chuẩn hóa văn bản
      "feature_builder": builder,            # HybridFeatureBuilderScratch đã fit (TF-IDF, Scaler, vocab)
      "selected_indices": selected_indices,  # Danh sách 4.318 cột đặc trưng đã chọn
      "best_threshold": 0.465,               # Ngưỡng quyết định đã chốt trên Validation
      "metrics_val": best_th_info,           # Các chỉ số và quy tắc chọn trên Validation
      "metrics_test": final_metrics          # Các chỉ số Test tham chiếu
  }
  ```
- Lưu bundle bằng thư viện `joblib` (mức nén 3) vào [`results/saved_models/spam_pipeline.joblib`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/saved_models/spam_pipeline.joblib).

### Bước 9. Kiểm toán toàn vẹn mã nguồn (Source Integrity Audit)
- Tự động tính toán mã băm SHA-256 cho toàn bộ 21 file mã nguồn chính trong `src/` và `main.py`.
- Lưu thông tin thực nghiệm, chỉ số Test, ma trận nhầm lẫn và bảng mã băm vào file [`results/metrics/run_summary.json`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/results/metrics/run_summary.json) để đảm bảo khả năng tái lập và kiểm chứng gian lận mã nguồn.

### Bước 10. Triển khai dịch vụ suy luận email mới (SpamInferenceService)
- Triển khai lớp `SpamInferenceService` trong [`src/deployment/predictor.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/deployment/predictor.py).
- **Quy trình nạp và dự đoán:**
  1. `load_service(bundle_path)`: Nạp file joblib bundle lên bộ nhớ.
  2. `predict_single(text)`:
     - Làm sạch văn bản email bằng `SpamTextProcessor` có sẵn trong bundle.
     - Biến đổi thành vector đặc trưng thưa bằng `feature_builder.transform()`.
     - Lọc cột bằng `selected_indices`.
     - Tính điểm phân loại spam bằng `model.predict_proba()`.
     - So sánh với `threshold = 0.465` và xuất kết quả: `prediction` (`ham` hoặc `spam`), `spam_score` (kèm alias `spam_probability`), `is_calibrated=False`, `risk_level`, `threshold_used`, `detected_signals`.
  3. `predict_batch(texts)`: Dự đoán hàng loạt và trả về bảng `pd.DataFrame`.
- **Thực thi kiểm chứng Demo:** Chạy dự đoán trực tiếp trên 5 mẫu email mẫu ở cuối chương trình `main.py` để chứng minh hệ thống hoạt động hoàn hảo từ đầu cuối (End-to-End).

### Bước 11. Chạy thực nghiệm phát triển đa seed độc lập (Development Experiments Runner)
- Triển khai module độc lập [`src/evaluation/experiments.py`](file:///e:/Hoc_May_Tren_truong/lab_1%20spam_classifier/src/evaluation/experiments.py) (`run_development_experiments()`).
- Cho phép chạy khảo sát kiểm toán trên tập phát triển (Train & Validation) qua nhiều seed ngẫu nhiên (`seeds=(17, 42, 73)`), xuất manifest metadata và split JSON.
- **Nguyên tắc bất biến:** Tuyệt đối bảo lưu tập Test độc lập (`reserved_test_source_ids`), không dùng cho bất kỳ lượt chạy chẩn đoán phát triển nào để phòng chống rò rỉ dữ liệu (Data Leakage).

# Giai đoạn 4: Lập luận đánh giá, phân tích lỗi và đóng gói suy luận

> **Nguồn số liệu:** Các số liệu dưới đây được lấy từ artifact của lần chạy 2026-10-19. Việc cập nhật tài liệu không bao gồm chạy lại thực nghiệm. Các lý do lựa chọn thiết kế vẫn cần được kiểm chứng; một kết quả tốt chưa đủ để xác định nguyên nhân. Phần thiết kế trước khi triển khai được trình bày tại [design.md](../Algorithms/design.md).

## 1. Mục tiêu và hợp đồng đầu vào / đầu ra

Đầu vào gồm tập Test được giữ riêng, nhãn Test, model và threshold đã chốt trên Validation, cùng trạng thái của bộ biến đổi sau khi fit. Đầu ra gồm metrics, confusion matrix, các mẫu FP/FN có `source_id`, báo cáo về giới hạn của kết quả và bundle để suy luận trên văn bản mới.

[main.py](../../main.py) đánh giá các ứng viên với threshold đã chọn riêng trên Validation và giữ cờ `selected_on_validation`. Báo kết quả Test của nhiều ứng viên không cho phép chọn lại winner từ Test. Dữ liệu là tin nhắn; chưa có kiểm chứng cho gateway email hay SLA production.

## 2. Logic các module và lý do thiết kế

### 2.1. Metrics và confusion matrix

[metrics.py](../../src/evaluation/metrics.py) quy ước hàng là nhãn thật, cột là nhãn dự đoán, với ma trận `[[TN, FP], [FN, TP]]`. `safe_divide` xử lý mẫu số 0 theo giá trị mặc định; cần báo rõ khi một metric không có đủ dữ liệu để diễn giải.

Accuracy dễ che lỗi lớp thiểu số nên phải đọc cùng Recall, Precision, F1/F2, FPR và confusion matrix. FPR đo tỷ lệ Ham bị chặn; Recall đo tỷ lệ Spam bị phát hiện. Hai đại lượng có mẫu số khác nhau.

ROC-AUC dùng thứ hạng trung bình khi score bằng nhau:

$$
\mathrm{AUC}=\frac{\sum_{i\in\mathrm{Pos}}R_i-N_{\mathrm{pos}}(N_{\mathrm{pos}}+1)/2}{N_{\mathrm{pos}}N_{\mathrm{neg}}}
$$

Cách tính này tương ứng xác suất score của một mẫu dương lớn hơn mẫu âm cộng nửa xác suất hòa. Tích phân trên ROC dùng đầy đủ các điểm ngưỡng và xử lý ties đúng có thể cho cùng AUC; không mặc định phương pháp hình thang kém chính xác. ROC-AUC không đo calibration và không thay thế chất lượng tại threshold đã chọn.

### 2.2. Phân tích lỗi

[error_analysis.py](../../src/evaluation/error_analysis.py) nhận bảng Test và các vector dự đoán, tách FP/FN, rồi gắn mô tả bằng heuristic từ khóa, chữ hoa, dấu câu hoặc độ dài. Các mô tả `suspected_causes` là gợi ý để xem xét, chưa đủ để xác định nguyên nhân.

846 lỗi tham chiếu gồm 666 FP và 180 FN. Không đủ căn cứ để khẳng định phần lớn FP là email tài chính, hoặc FN là tin nhắn lừa đảo cực ngắn, nếu chưa tổng hợp và thẩm định mẫu. Không đổi nhãn chỉ vì model dự đoán khác nhãn gốc. Khi chia sẻ mẫu lỗi, cần bỏ thông tin cá nhân không cần thiết.

### 2.3. Lưu và nạp bundle

[persistence.py](../../src/models/persistence.py) dùng `joblib.dump(..., compress=3)` để lưu model, builder, selected columns, threshold và metadata do caller cung cấp. Gom một bundle giảm nguy cơ ghép nhầm model với từ vựng; chưa có cơ chế ghi atomic hoặc tự xác minh chữ ký trong hàm này.

`load_spam_pipeline` kiểm tra đường dẫn tồn tại và kết quả nạp là dictionary. [SpamInferenceService.load_service](../../src/deployment/predictor.py) mới kiểm tra model, feature_builder và threshold hữu hạn thuộc [0, 1]; threshold thiếu dùng mặc định 0.5. `FittedFeatureTransform` kiểm tra selected indices khi chọn ma trận.

Joblib cần class/module và môi trường tương thích; chỉ nạp artifact từ nguồn tin cậy. Hash dữ liệu/mã nguồn trong `experiment_metadata` giúp đối chiếu provenance, không tự xác thực người phát hành hoặc chứng minh artifact chưa bị thay đổi. `archive_run_outputs` sao chép kết quả cũ trước lần chạy mới, giữ dấu COMPLETE khi sao chép thành công.

### 2.4. Suy luận và ý nghĩa score

Pseudocode mô tả quy trình, không phải API nguyên văn:

```text
kiểm tra văn bản là chuỗi không rỗng
nạp bundle và kiểm tra các thành phần
X = fitted_feature_transform.transform(văn bản mới)
score = model score trên X; kiểm tra hữu hạn và thuộc [0, 1]
label = score >= threshold đã lưu
trả label, spam_score, threshold_used, is_calibrated=False và detected_signals
```

`predict_single` dùng cùng phép biến đổi/thứ tự cột đã fit, không fit lại. `spam_probability` là alias tương thích; trường `score_kind='uncalibrated_score'` và `is_calibrated=False` mới mô tả đúng giới hạn. Sigmoid của SVM không phải Platt Scaling đã được học.

`_extract_signals` liệt kê dấu hiệu từ khóa/URL/dấu câu/tiền tệ/chữ hoa độc lập với model. Đây không phải attribution trọng số hay bằng chứng mẫu có mã độc. `predict_batch` hiện gọi `predict_single` theo từng phần tử; không được mô tả là vector hóa toàn bộ batch với throughput đã đo.

## 3. Phương án thay thế và trade-offs

| Quyết định | Lý do | Phương án khác / điều kiện kiểm chứng |
| :--- | :--- | :--- |
| Khóa trước khi xem Test | Tách model selection khỏi ước lượng cuối | Nếu dùng Test để cải tiến, cần holdout mới hoặc kế hoạch đánh giá độc lập |
| Báo nhiều metrics và lỗi FP/FN | Thấy đánh đổi bắt spam/chặn Ham | Metric chi phí FP/FN cần trọng số từ người dùng; không tự gán chi phí nghiệp vụ |
| Joblib bundle | Phù hợp class Python tự viết, lưu cùng trạng thái transform | File rời cần cơ chế đồng bộ; ONNX cần đánh giá khả năng chuyển đổi, chưa có benchmark so sánh |
| Service trong tiến trình | Phù hợp demo local, ít thành phần triển khai | REST/serverless cần đo overhead và nhu cầu vận hành; chưa có kết luận latency |
| Heuristic signals | Dễ đọc để hỗ trợ rà mẫu | Muốn giải thích quyết định cần đối chiếu đóng góp đặc trưng; SHAP/LIME là phương án nghiên cứu chưa triển khai |

## 4. Invariants, failure modes và kế hoạch đánh giá

**Invariants:** row order khớp nhãn/score/source_id; tổng confusion matrix bằng số mẫu; threshold và selected indices giữ nguyên sau lựa chọn; dự đoán sau save/load phải khớp trong sai số số học cho phép. Không báo mục tiêu Recall đạt trên Test như một bảo đảm cho dữ liệu tương lai.

**Failure modes:** Test đã bị quan sát nhiều lần; lệch miền/ngôn ngữ; score chưa calibration; phiên bản module khác lúc lưu; mẫu mới vượt miền numeric Train; heuristic bị hiểu nhầm là nguyên nhân. Không có monitoring drift hoặc ngưỡng FPR production được chứng minh trong lần tham chiếu.

[Test evaluation](../../tests/test_evaluation.py) kiểm tra counts, ties/AUC và đồ thị; [test deployment](../../tests/test_deployment.py) kiểm tra bundle và dự đoán; [test pipeline](../../tests/test_pipeline.py) kiểm tra tích hợp bằng fixture tạm; [test review regressions](../../tests/test_review_regressions.py) kiểm tra archive, thứ tự cột, alias score và Test tách khỏi thực nghiệm phát triển.

Các đánh giá bổ sung chưa thực hiện ở lần sửa này: nhiều split/holdout mới; calibration; latency có warm-up, batch size, phần cứng và percentile; thẩm định nhóm lỗi; chênh lệch lỗi giữa các nhóm nếu có metadata hợp lệ và đủ mẫu. Không suy đoán thuộc tính nhạy cảm từ văn bản để tạo nhóm.

## 5. Bằng chứng tham chiếu và giới hạn kết luận

Nguồn: [test_models_comparison.csv](../../results/metrics/test_models_comparison.csv), [run_summary.json](../../results/metrics/run_summary.json), [misclassified_samples.csv](../../results/metrics/misclassified_samples.csv).

| Chỉ số Linear SVM Tuned trên 2.902 mẫu Test | Giá trị |
| :--- | ---: |
| Accuracy | 70.85% |
| Precision | 61.39% |
| Recall | 85.47% |
| F1 | 71.46% |
| F2 | 79.25% |
| ROC-AUC | 81.44% |
| TN / FP / FN / TP | 997 / 666 / 180 / 1.059 |
| Specificity / FPR (từ confusion matrix) | 59.95% / 40.05% |

FPR 40.05% nghĩa là khoảng 4/10 Ham trong Test bị chặn nhầm tại threshold 0.465. Không gọi đây là kiểm soát tốt hoặc đủ an toàn để tự động xóa thư. Winner đã được chọn trên Validation; Test chỉ mô tả kết quả và hạn chế của quyết định đó.

Chưa có artifact kiểm chứng các score demo 0.9842/0.9615, latency 2.40 ms, throughput 1.250 email/giây, tỷ lệ nén 75% hoặc các SLA từng được ghi trong tài liệu. Các số thiếu bằng chứng đã được bỏ khỏi bảng kết quả. Không có thí nghiệm mới, calibration hay chứng nhận production phát sinh từ việc cập nhật Markdown này.

**Đầu ra:** Bundle, cấu hình và báo cáo cùng lần chạy để demo suy luận; giới hạn chất lượng và môi trường cần được đọc cùng kết quả.

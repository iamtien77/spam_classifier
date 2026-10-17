# Phân loại email Spam / Ham

Dự án cho bài “Practice 1: Classifying Spam Emails”. Có tiền xử lý, TF-IDF, ba mô hình LR/SVM/Naive Bayes, đánh giá Test, lưu/nạp pipeline và dự đoán văn bản mới.

## Đối chiếu đề bài

| Nội dung | Triển khai |
|---|---|
| Làm sạch stop words, dấu câu, HTML | `src/data/preprocessing.py` |
| Đặc trưng số và chia tập | TF-IDF từ, bigram, char n-gram; keywords; đặc trưng cấu trúc; Train/Validation/Test phân tầng |
| Huấn luyện | Logistic Regression, Linear SVM, Complement Naive Bayes |
| Đánh giá | Accuracy, Precision, Recall, F1, F2, ROC-AUC, confusion matrix, phân tích FP/FN |
| Dự đoán email mới | `SpamInferenceService`, pipeline joblib và demo trong `main.py` |
| Thực nghiệm mở rộng | Ablation đặc trưng trong notebook 02, Top-K, Grid Search, Voting Ensemble |

Sender, recipient và subject là ví dụ đặc trưng trong PDF; dữ liệu hiện tại chỉ cung cấp nhãn và nội dung. Random forest/boosting là ví dụ ensemble; dự án dùng voting. PDF không yêu cầu web UI, Recall ≥ 0,85 hoặc tự cài thuật toán; đây là các lựa chọn bổ sung của dự án.

## Dữ liệu và phương pháp

Dữ liệu thô: 43,849 dòng; loại 2,195 dòng nhãn sai, 2,192 dòng text rỗng, 1,480 dòng thuộc 738 nhóm mâu thuẫn nhãn và 8,954 bản lặp cùng nhãn. Còn 29,028 mẫu: 16,636 Ham, 12,392 Spam. Train/Validation/Test: 23,223/2,903/2,902.

Khóa trùng chuẩn hóa Unicode và khoảng trắng, giữ hoa/thường và dấu câu. Nhóm cùng nội dung nhưng khác nhãn bị loại toàn nhóm. Các số lượng loại trừ ở trên không chồng lặp. Chỉ fit từ vựng, IDF, capping và scaler trên Train; chọn Top-K trên Validation. Không có metadata địa chỉ người gửi/người nhận riêng trong pipeline hiện tại.

Logistic Regression tối ưu binary cross-entropy có L2; Linear SVM tối ưu squared hinge có L2. Cả hai tự cài đặt gradient descent với bước spectral và Armijo backtracking; lambda = 1/(C × N). Naive Bayes hỗ trợ multinomial và complement. Không dùng lớp mô hình hoặc metrics của sklearn. Sigmoid trên điểm SVM chưa được calibration; không gọi đó là Platt scaling và không coi điểm này là xác suất rủi ro đã hiệu chỉnh.

Chỉ dùng Validation để chọn tham số, Top-K, ngưỡng và mô hình: ưu tiên ứng viên đạt Recall ≥ 0,85, sau đó tối đa Accuracy, phân xử bằng F1. Nếu không đạt Recall mục tiêu, ưu tiên Recall rồi Accuracy/F1. Đây là lựa chọn của dự án; PDF không đặt Recall tối thiểu. F1/F2 là chỉ số báo cáo, không phải mục tiêu chính.

TF-IDF dùng sublinear TF, smooth IDF và L2 normalization. Số chiều trước Top-K: **4,318**, sau Top-K: **4,318**. CSR giúp tránh giải nén toàn bộ ma trận; không có cam kết tiết kiệm RAM cố định hoặc không bao giờ tràn RAM.

## Kết quả chạy tham chiếu

Mô hình: **Linear SVM (Tuned)**. Ngưỡng chọn trên Validation: **0.465**. Test giữ nguyên mô hình và ngưỡng này.

| Chỉ số Test | Giá trị |
|---|---:|
| Accuracy | 70.85% |
| Precision (Spam) | 61.39% |
| Recall (Spam) | 85.47% |
| F1 | 71.46% |
| F2 | 79.25% |
| ROC-AUC | 81.44% |
| False-positive rate | 40.05% |

Test có TN=997, FP=666, FN=180, TP=1059; 40.05% email hợp lệ bị chặn nhầm. Hiệu năng đã cải thiện nhưng chưa đủ để tự động xóa/chặn email thực tế. Kết quả chỉ phản ánh một split cố định của dữ liệu hiện tại; chưa có tập ngoài, cross-validation hoặc khoảng tin cậy. Test đã được quan sát trong vòng phát triển này nên cần một tập ngoài mới nếu muốn tuyên bố chất lượng tổng quát độc lập. Không loại mẫu dựa trên việc mô hình dự đoán sai và không đổi nhãn để làm tăng điểm.

Kết quả cũ trước sửa: Accuracy 66,20%, Precision 56,89%, Recall 85,96%, F1 68,47%, FP=807. Đây là mốc so sánh của phiên bản trước trên cùng split, không phải một thí nghiệm đánh giá độc lập mới.

## Chạy và kiểm chứng

Python 3.10 trở lên:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py
```

Bộ kiểm thử đã chạy trong lần sửa này: 36 tests pass, gồm hồi quy hội tụ trên tín hiệu TF-IDF thưa và tích hợp train → chọn trên Validation → lưu/nạp → inference. Tests không thay thế việc đo chất lượng trên dữ liệu thật.

Đầu ra trong `results/`: `metrics/data_audit.json`, `top_k_sweep.csv`, `validation_model_selection.csv`, `test_models_comparison.csv`, `misclassified_samples.csv`, `run_summary.json`; biểu đồ trong `figures/`; model trong `saved_models/spam_pipeline.joblib`. `run_summary.json` lưu hash mã nguồn của lần chạy.

```python
from src.deployment.predictor import SpamInferenceService
service = SpamInferenceService(bundle_path="results/saved_models/spam_pipeline.joblib").load_service()
print(service.predict_single("Please review the meeting notes."))
```

## Notebooks và báo cáo

main.py là thí nghiệm tham chiếu dùng cho báo cáo này. Năm notebooks là chuỗi thực nghiệm riêng, có baseline, lưới tham số và ensemble khác; số liệu có thể khác main.py. Không trộn bảng, hình hoặc model giữa hai thí nghiệm. Notebooks có run_id và hash cache để kiểm tra nguồn dữ liệu. Xem notebooks/README.md và notebooks/results/metrics/test_models_comparison.csv.

Chạy notebook 01 → 05 trong Jupyter bằng Restart & Run All, lưu notebook sau khi chạy thành công.

Năm báo cáo Word đã được cập nhật theo kết quả `main.py`. Bản Word gốc được giữ tại `docs/original_reports_before_fix.zip`. Khi chạy lại thí nghiệm, cập nhật số liệu README và báo cáo theo `results/metrics/run_summary.json` và `data_audit.json`.

`src/` chứa mã dùng chung; `main.py` điều phối thí nghiệm tham chiếu; `tests/` chứa kiểm thử; `notebooks/` chứa thực nghiệm; `docs/` chứa đề và báo cáo. Dữ liệu, kết quả, model và docs hiện được bỏ qua theo `.gitignore`, cần đóng gói chúng riêng khi nộp bài.

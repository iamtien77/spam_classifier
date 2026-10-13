# Task 2 – Bàn giao phần huấn luyện mô hình

## Phần đã hoàn thiện

- `src/models/classifiers.py`: Logistic Regression, Linear SVM, Multinomial Naive Bayes và Complement Naive Bayes; hỗ trợ ma trận SciPy sparse và `CSRMatrixScratch` của Task 1.
- `src/models/tuning.py`: Grid Search, tính chỉ số ở từng ngưỡng, quét 200 ngưỡng, điều kiện Recall mục tiêu và chọn mô hình trên Validation.
- `src/models/ensemble.py`: Soft Voting, Hard Voting, Random Forest và Gradient Boosting.
- `src/models/persistence.py`: hàm lưu/nạp bundle bằng joblib.
- `tests/test_models.py`: 5 bài kiểm thử cho mô hình và threshold optimizer.
- `notebooks/03_model_baseline_and_training.ipynb`: notebook baseline đã chạy.
- `notebooks/04_tuning_and_ensemble.ipynb`: notebook tuning, threshold, voting, tree experiments và chọn mô hình đã chạy.
- `results/saved_models/spam_pipeline.joblib`: artifact Task 2 đã tạo, bao gồm mô hình chọn cuối, threshold và metadata Validation.

## Kết quả thực nghiệm trên tập Validation hiện có

Mô hình được chọn trong lần chạy notebook này là `soft_voting_ensemble`, với threshold `0.465`. Các chỉ số ghi trong artifact:

- Accuracy: 0.5749
- Precision: 0.5010
- Recall: 0.9637
- F1: 0.6593
- F-beta (beta=2): 0.8135
- ROC-AUC: 0.6880

Recall vượt điều kiện mục tiêu 0.85. Đây là kết quả trên Validation, không phải kết quả Test cuối cùng.

## Lưu ý bàn giao cho Task 3

Repository ban đầu không có artifact đã lưu của `text_processor` và `feature_builder` sau khi Task 1 fit. Việc fit lại `HybridFeatureBuilderScratch` trên toàn bộ dữ liệu trong môi trường kiểm thử mất quá nhiều thời gian và không được hoàn tất trong lần chạy này. Vì vậy, `spam_pipeline.joblib` hiện lưu mô hình, threshold và metadata, **chưa bao gồm** bộ xử lý/vectơ hóa đặc trưng để nhận email thô trực tiếp. Task 3 cần nhận hoặc tái tạo chính xác bộ `text_processor`/`feature_builder` đã fit của Task 1 trước khi triển khai dự đoán email mới.

Ngoài ra, nhãn Spam trong `y_val.npy` hiện chiếm khoảng 42.68%, khác với tỷ lệ khoảng 13.4% nêu trong tài liệu tổng quan. Nhóm nên kiểm tra xem các file processed có được tạo từ cùng phiên bản dữ liệu và quy trình lấy mẫu hay không.

## Kiểm thử

Lệnh `python -m pytest -q` chạy thành công: **22 passed**. Hai notebook Task 2 cũng đã được thực thi thành công và lưu output để tiện xem kết quả.

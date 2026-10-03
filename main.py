"""
main.py - Entry point, điều phối toàn bộ workflow
===================================================
Thứ tự chạy:
  1. src.data.loader          → Tải + khám phá dữ liệu
  2. src.data.preprocessing   → Tiền xử lý (clean, split, scale, select features)
  3. src.features             → Tạo features mới, PCA
  4. src.evaluation.visual... → EDA
  5. src.models.classifiers   → Huấn luyện 3 mô hình (LR, SVM, NB)
  6. src.evaluation.metrics   → Đánh giá (accuracy, precision, recall, F1)
  7. src.models.tuning        → Tối ưu hyperparameter (GridSearchCV)
  8. src.models.ensemble      → Ensemble (Random Forest, Gradient Boosting, Voting)
  9. src.evaluation.visual... → Biểu đồ so sánh
  10. src.models.persistence  → Lưu model tốt nhất
  11. src.deployment.pred...  → Demo phân loại email mới
"""

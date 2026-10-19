Phiên bản v5. Trực quan hóa và xuất biểu đồ đánh giá thực nghiệm (Visualization)
Mục tiêu của bản này là ứng dụng module visualization.py để tự động vẽ và lưu trữ toàn bộ các biểu đồ phân tích đánh giá chất lượng cao (300 DPI) vào thư mục results/figures/.

Bước 1. Vẽ biểu đồ nhiệt ma trận nhầm lẫn (Confusion Matrix Heatmap)
Hàm plot_confusion_matrix():
- Sử dụng thư viện Seaborn vẽ heatmap biểu diễn 4 ô TN (997), FP (666), FN (180), TP (1059).
- Chú thích nhãn trục rõ ràng: Trục tung là Actual Label (Ham / Spam), Trục hoành là Predicted Label (Ham / Spam).
- Lưu biểu đồ ra file results/figures/test_confusion_matrix.png.

Bước 2. Vẽ đường cong Receiver Operating Characteristic (ROC Curve)
Hàm plot_roc_curve():
- Quét qua các mốc ngưỡng xác suất để tính cặp tọa độ (FPR, TPR).
- Vẽ đường cong ROC thể hiện năng lực phân loại qua mọi ngưỡng, kèm đường phân cách ngẫu nhiên (đường chéo y = x).
- Chú thích diện tích dưới đường cong: ROC-AUC = 0.8144.
- Đánh dấu mốc ngưỡng vận hành tối ưu tau* = 0.465 trên đường cong.
- Lưu biểu đồ ra file results/figures/roc_curve.png.

Bước 3. Vẽ đường cong Precision-Recall (PR Curve)
Hàm plot_precision_recall_curve():
- Tính toán và vẽ tương quan giữa Precision và Recall qua các mốc ngưỡng.
- Vẽ đường tham chiếu baseline (tỷ lệ lớp Spam = 42,69%).
- Đánh dấu điểm hoạt động của mô hình tại ngưỡng tau* = 0.465 (Precision = 61,39%, Recall = 85,47%).
- Lưu biểu đồ ra file results/figures/precision_recall_curve.png.

Bước 4. Vẽ biểu đồ so sánh các mô hình (Model Comparison Bar Chart)
Hàm plot_models_comparison():
- Vẽ biểu đồ cột nhóm thể hiện các chỉ số Accuracy, Precision, Recall, F1-score và ROC-AUC của các mô hình ứng viên.
- Thể hiện trực quan sự vượt trội của Linear SVM (Tuned) và Logistic Regression (Tuned) so với Complement Naive Bayes.
- Lưu biểu đồ ra file results/figures/test_models_comparison.png.
Tất cả các biểu đồ được hoàn thành và xuất bản tự động phục vụ báo cáo khoa học.

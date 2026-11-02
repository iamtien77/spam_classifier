Phiên bản v3. Tự cài đặt và đánh giá chỉ số ROC-AUC bằng thống kê thứ bậc Wilcoxon
Phiên bản này tự xây dựng hàm compute_roc_auc() từ đầu dựa trên thống kê xếp hạng Wilcoxon-Mann-Whitney có xử lý điểm số đồng hạng (ties handling), đánh giá năng lực phân biệt thứ bậc của mô hình mà không phụ thuộc vào ngưỡng.

Bước 1. Xác định ý nghĩa của chỉ số ROC-AUC trong bài toán lọc thư rác
ROC-AUC đo lường xác suất một email Spam được mô hình gán điểm số cao hơn một email Ham được chọn ngẫu nhiên.
Chỉ số này đánh giá chất lượng toàn cục của hàm điểm mô hình trên mọi ngưỡng phân loại khả dĩ.

Bước 2. Thuật toán tự cài đặt compute_roc_auc()
Triển khai trong src/evaluation/metrics.py:
1. Nhận vector nhãn y_true in {0, 1} và vector điểm số y_score.
2. Kiểm tra điều kiện dữ liệu: y_true phải chứa đủ cả hai lớp 0 và 1, độ dài hai vector phải bằng nhau.
3. Ghép cặp và sắp xếp tăng dần theo y_score.
4. Gán thứ hạng (ranks) từ 1 đến N. Nếu có các điểm số bằng nhau (ties), gán thứ hạng bằng trung bình cộng thứ hạng của nhóm hòa điểm.
5. Tính tổng thứ hạng của các mẫu thuộc lớp dương (Spam): sum_ranks_pos = sum(ranks[y_true == 1]).
6. Áp dụng công thức thống kê Wilcoxon-Mann-Whitney:
   U = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.
   ROC-AUC = U / (n_pos * n_neg).

Bước 3. Kết quả đánh giá ROC-AUC trên tập Test
- Linear SVM (Tuned): ROC-AUC = 81,44% (0.81436).
- Logistic Regression (Tuned): ROC-AUC = 81,38% (0.81381).
- SpamVotingEnsemble (Tuned): ROC-AUC = 80,26% (0.80258).
- Complement Naive Bayes (Tuned): ROC-AUC = 75,22% (0.75219).
Kết quả ROC-AUC vượt 81% khẳng định mô hình Linear SVM có khả năng phân biệt hai lớp qua thứ hạng điểm số tốt.

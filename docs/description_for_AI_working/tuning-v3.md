Phiên bản v3. Thiết lập hàm mục tiêu ràng buộc nghiệp vụ (Recall >= 0.85) và cơ chế Fallback
Phiên bản này cài đặt chiến lược lựa chọn ngưỡng tối ưu dựa trên ràng buộc nghiệp vụ: ưu tiên bắt trúng ít nhất 85% thư rác, sau đó tối đa hóa độ chính xác và xử lý an toàn các trường hợp hòa điểm hoặc không đạt mục tiêu.

Bước 1. Xác định ràng buộc nghiệp vụ (Business Constraint)
Dự án thiết lập cấu hình: recall_target = 0.85.
Mục tiêu: Đảm bảo bộ lọc phát hiện được phần lớn thư rác để bảo vệ hòm thư người dùng.
Phương thức select_for_recall_target(threshold_df, recall_target):
Lọc các mốc ngưỡng thỏa mãn điều kiện: threshold_df['recall'] >= 0.85.

Bước 2. Quy tắc xếp hạng ngưỡng khả thi (Feasible Thresholds)
Trong tập các ngưỡng đạt Recall >= 0.85, sắp xếp theo thứ tự ưu tiên đa tầng:
1. Accuracy cao nhất (tối ưu tỷ lệ dự đoán đúng toàn thể).
2. F1-score cao nhất (nếu Accuracy bằng nhau, ưu tiên cân bằng Precision và Recall).
3. Precision cao nhất (nếu F1 bằng nhau, ưu tiên giảm thư rác chặn nhầm).
4. FPR thấp nhất (nếu Precision bằng nhau, ưu tiên tỷ lệ chặn nhầm thấp nhất).
Lấy ngưỡng đứng đầu danh sách làm best_threshold.

Bước 3. Xây dựng cơ chế Fallback an toàn (Khi không đạt Target)
Nếu không có mốc ngưỡng nào trong 200 mốc đạt Recall >= 0.85:
Không để chương trình báo lỗi crash hoặc dừng lại.
Tự động chuyển sang chế độ Fallback: Sắp xếp theo thứ tự Recall cao nhất -> Accuracy cao nhất -> F1 cao nhất.
Ghi nhận cờ target_met = False và lưu lý do vào trường selection_rule.
Cơ chế này bảo đảm bộ dò ngưỡng luôn đưa ra quyết định tối ưu trong mọi tình huống dữ liệu.

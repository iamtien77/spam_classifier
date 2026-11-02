Phiên bản v1. Cài đặt cấu trúc ma trận thưa tự viết (CSRMatrixScratch)
Phiên bản này tự xây dựng lớp ma trận thưa Compressed Sparse Row (CSR) độc lập với SciPy, giúp hệ thống biểu diễn và tính toán trên hàng nghìn chiều đặc trưng văn bản mà không bị cạn kiệt bộ nhớ RAM.

Bước 1. Xác định sự cần thiết của ma trận thưa trong Text Classification
Mỗi email chỉ chứa vài chục đến vài trăm từ khác nhau trong khi từ điển có hơn 4.000 từ.
Nếu lưu trữ dưới dạng mảng đặc (Dense NumPy Array), ma trận sẽ chứa hơn 98% phần tử là số 0, tiêu tốn hàng gigabyte RAM vô ích.
Cần tự cài đặt cấu trúc CSR chỉ lưu trữ các giá trị khác 0 theo từng hàng.

Bước 2. Thiết kế 3 mảng cốt lõi của CSRMatrixScratch
Khởi tạo lớp CSRMatrixScratch trong src/features/feature_engineering.py.
Quản lý dữ liệu bằng 3 mảng số học tiêu chuẩn:
- data: Lưu toàn bộ các giá trị khác 0 (kiểu số thực double).
- indices: Lưu chỉ số cột tương ứng của từng phần tử trong data (kiểu số nguyên unsigned int).
- indptr: Mảng con trỏ hàng kích thước (số hàng + 1), chỉ ra đoạn dữ liệu của hàng i nằm từ indptr[i] đến indptr[i+1].
Xác thực điều kiện cấu trúc CSR: con trỏ tăng dần, chỉ số cột trong mỗi hàng tăng dần và không có giá trị NaN.

Bước 3. Cài đặt các phép toán tối ưu trực tiếp trên ma trận thưa
Phương thức iter_row(row): Cho phép duyệt qua các cặp (column_index, value) khác 0 mà không giải nén thành hàng đặc.
Phương thức dot(weights): Tính tích vô hướng X @ w để dự đoán điểm số tuyến tính trong Linear SVM và Logistic Regression.
Phương thức transpose_dot(values): Tính X.T @ v phục vụ tính vector đạo hàm riêng (gradient) trong quá trình huấn luyện mà không cần tạo ma trận chuyển vị X.T.
Phương thức hstack(matrices): Ghép ngang nhiều ma trận thưa theo thứ tự khối.
Phương thức select_columns(cols): Lọc chọn các cột theo danh sách chỉ số phục vụ Top-K selection.
Phương thức to_scipy_csr(): Hỗ trợ chuyển đổi sang scipy.sparse.csr_matrix khi cần lưu trữ ra đĩa.

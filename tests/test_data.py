"""
test_data.py - Unit test logic cho module data
Dựa trên kiến trúc dữ liệu từ Classification_email_spam.ipynb
"""

def test_load_raw_data():
    """
    LOGIC KIỂM THỬ NẠP DỮ LIỆU THÔ:
    - Kiểm tra nạp thành công file spam.csv.
    - Kiểm tra cấu trúc DataFrame trả về có chứa đúng 2 cột cốt lõi: 'label' và 'text'.
    - Đảm bảo các nhãn chỉ nằm trong tập {'ham', 'spam'}.
    """
    pass


def test_stratified_split_preserves_distribution():
    """
    LOGIC KIỂM THỬ CHIA PHÂN TẦNG:
    - Kiểm tra tỷ lệ mẫu Train / Val / Test xấp xỉ 80% / 10% / 10%.
    - Kiểm tra tỷ lệ nhãn Spam/Ham trong cả 3 tập con tương đồng với tỷ lệ gốc trong toàn bộ dữ liệu.
    - Đảm bảo không có mẫu nào bị trùng lặp giữa 3 tập (tập chỉ số hoàn toàn rời nhau).
    """
    pass


def test_clean_email_split():
    """
    LOGIC KIỂM THỬ LÀM SẠCH CHẤT LƯỢNG MẪU:
    - Kiểm tra loại bỏ thành công các dòng có giá trị text là NaN hoặc chuỗi rỗng.
    - Kiểm tra nhãn được chuyển đổi chính xác thành số nguyên {0, 1}.
    """
    pass

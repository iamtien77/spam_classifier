"""
test_data.py - Unit test logic cho module data
Kiểm thử toàn diện không dùng scikit-learn, tương thích unittest và pytest.
"""
import unittest
import tempfile
import csv
import sys
from pathlib import Path

# Đảm bảo import được src khi chạy độc lập
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_raw_data, explore_raw_data
from src.data.preprocessing import (
    clean_email_split,
    stratified_split_dataframe,
    stratified_split_records,
    audit_data_quality,
    SpamTextProcessor,
)


def create_synthetic_csv(path: Path) -> None:
    """Tạo file CSV giả lập với cấu hình nhãn và dữ liệu đa dạng."""
    rows = [
        ("ham", "Hello, can we meet for lunch today?"),
        ("spam", "WINNER! You have won a cash prize of $1000! Call 123 now."),
        ("ham", "Please review the attached project document."),
        ("spam", "FREE entry to win a brand new car! Claim now."),
        ("ham", "Are you available for a quick sync this afternoon?"),
        ("invalid", "This row has an invalid label and should be removed."),
        ("ham", "   "),  # Empty text
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["v1", "v2"])
        writer.writerows(rows)


def test_load_raw_data():
    """
    LOGIC KIỂM THỬ NẠP DỮ LIỆU THÔ:
    - Kiểm tra nạp thành công file CSV giả lập.
    - Kiểm tra cấu trúc bản ghi trả về có chứa đúng các cột cốt lõi: 'label', 'text', 'raw_text'.
    - Đảm bảo explore_raw_data thống kê đúng số dòng và nhãn.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        create_synthetic_csv(tmp_path)
        records = load_raw_data(tmp_path)
        assert len(records) == 7, f"Kỳ vọng 7 bản ghi thô, nhận được {len(records)}"
        for r in records:
            assert "source_id" in r
            assert "label" in r
            assert "text" in r
            assert "raw_text" in r
            assert isinstance(r["text"], str)

        stats = explore_raw_data(records)
        assert stats["n_rows"] == 7
        assert stats["invalid_label_count"] >= 1
    finally:
        if tmp_path.is_file():
            tmp_path.unlink()


def test_stratified_split_preserves_distribution():
    """
    LOGIC KIỂM THỬ CHIA PHÂN TẦNG:
    - Kiểm tra tỷ lệ mẫu Train / Val / Test xấp xỉ 80% / 10% / 10%.
    - Kiểm tra tỷ lệ nhãn Spam/Ham trong cả 3 tập con tương đồng với tỷ lệ gốc trong toàn bộ dữ liệu.
    - Đảm bảo không có mẫu nào bị trùng lặp giữa 3 tập (tập chỉ số hoàn toàn rời nhau).
    """
    # Tạo 100 mẫu giả lập: 85 ham, 15 spam
    records = []
    for i in range(85):
        records.append({
            "source_id": i + 1,
            "label": "ham",
            "text": f"Normal message number {i}",
            "raw_text": f"Normal message number {i}",
        })
    for i in range(15):
        records.append({
            "source_id": 85 + i + 1,
            "label": "spam",
            "text": f"Spam winner cash offer number {i}",
            "raw_text": f"Spam winner cash offer number {i}",
        })

    train, val, test = stratified_split_records(
        records, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, random_state=42
    )

    # 1. Bảo toàn tổng số mẫu và tỷ lệ xấp xỉ 80% / 10% / 10%
    assert len(train) == 80, f"Kỳ vọng 80 mẫu train, nhận {len(train)}"
    assert abs(len(val) - 10) <= 1, f"Kỳ vọng val xấp xỉ 10 mẫu (+-1), nhận {len(val)}"
    assert abs(len(test) - 10) <= 1, f"Kỳ vọng test xấp xỉ 10 mẫu (+-1), nhận {len(test)}"
    assert len(train) + len(val) + len(test) == 100, "Tổng số mẫu phải được bảo toàn chính xác 100"

    # 2. Rời rạc hoàn toàn giữa 3 tập (Không rò rỉ dữ liệu)
    ids_train = {r["source_id"] for r in train}
    ids_val = {r["source_id"] for r in val}
    ids_test = {r["source_id"] for r in test}
    assert ids_train.isdisjoint(ids_val), "Tập train và val bị giao nhau!"
    assert ids_train.isdisjoint(ids_test), "Tập train và test bị giao nhau!"
    assert ids_val.isdisjoint(ids_test), "Tập val và test bị giao nhau!"

    # 3. Bảo toàn phân phối nhãn
    train_spam = sum(1 for r in train if r["label"] == "spam")
    val_spam = sum(1 for r in val if r["label"] == "spam")
    test_spam = sum(1 for r in test if r["label"] == "spam")

    # 15 spam chia 80/10/10: 12 train, 1 hoặc 2 ở val/test
    assert train_spam == 12, f"Kỳ vọng 12 spam trong train, nhận {train_spam}"
    assert val_spam + test_spam == 3, f"Kỳ vọng tổng 3 spam trong val+test, nhận {val_spam + test_spam}"


def test_clean_email_split():
    """
    LOGIC KIỂM THỬ LÀM SẠCH CHẤT LƯỢNG MẪU:
    - Kiểm tra loại bỏ thành công các dòng có giá trị text là rỗng/chỉ chứa khoảng trắng.
    - Loại bỏ nhãn bất hợp lệ và nhóm mâu thuẫn nhãn.
    - Kiểm tra nhãn được chuyển đổi chính xác thành số nguyên {0, 1} trong prepare_feature_frame.
    """
    dirty_records = [
        {"source_id": 1, "label": "ham", "text": "Valid ham email 1"},
        {"source_id": 2, "label": "spam", "text": "Win free cash now"},
        {"source_id": 3, "label": "invalid_label", "text": "Some text"},
        {"source_id": 4, "label": "ham", "text": "   "},
        {"source_id": 5, "label": "ham", "text": "Valid ham email 1"},  # duplicate
        {"source_id": 6, "label": "spam", "text": "Conflict message"},
        {"source_id": 7, "label": "ham", "text": "Conflict message"},   # conflict
    ]

    quality = audit_data_quality(dirty_records)
    assert quality["invalid_label_rows"] == 1
    assert quality["empty_text_rows"] == 1
    assert quality["duplicate_rows_removed"] == 1
    assert quality["conflicting_groups"] == 1

    cleaned = clean_email_split(dirty_records)
    assert len(cleaned) == 2, f"Kỳ vọng 2 bản ghi hợp lệ sau làm sạch, nhận {len(cleaned)}"
    assert {r["source_id"] for r in cleaned} == {1, 2}

    # Kiểm tra chuyển đổi nhãn thành y {0, 1}
    processor = SpamTextProcessor()
    prepared = processor.prepare_feature_frame(cleaned)
    assert prepared[0]["y"] == 0, "Nhãn ham phải ánh xạ về 0"
    assert prepared[1]["y"] == 1, "Nhãn spam phải ánh xạ về 1"


class TestDataModule(unittest.TestCase):
    def test_load_raw_data_case(self):
        test_load_raw_data()

    def test_stratified_split_case(self):
        test_stratified_split_preserves_distribution()

    def test_clean_email_split_case(self):
        test_clean_email_split()


if __name__ == "__main__":
    unittest.main()

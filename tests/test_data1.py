from pathlib import Path

import pytest

from src.data.loader import load_raw_data
from src.data.preprocessing import (
    clean_email_split,
    stratified_split_dataframe,
)


def make_toy_data():
    """40 bản ghi giả lập, cân bằng nhãn và không trùng nội dung."""
    rows = []
    for i in range(20):
        rows.append({"label": "ham", "text": f"normal personal message {i}"})
    for i in range(20):
        rows.append({"label": "spam", "text": f"claim free prize offer {i}"})
    return rows


def _labels(rows):
    return [row["label"] for row in rows]


def _texts(rows):
    return [row["text"] for row in rows]


def test_load_raw_data():
    """Kiểm tra loader đọc được CSV thật và giữ cấu trúc bản ghi."""
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "raw" / "spam.csv"

    assert csv_path.is_file(), f"Không tìm thấy dữ liệu đầu vào: {csv_path}"

    rows = load_raw_data(csv_path)

    assert isinstance(rows, list)
    assert len(rows) > 0
    assert all(isinstance(row, dict) for row in rows)
    assert all("label" in row for row in rows)
    assert all(("text" in row) or ("raw_text" in row) for row in rows)


def test_stratified_split_preserves_distribution():
    """Kiểm tra đủ dữ liệu, giữ tỷ lệ nhãn và không rò rỉ nội dung giữa các tập."""
    data = make_toy_data()

    train, val, test = stratified_split_dataframe(
        data,
        label_col="label",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=42,
    )

    # Mỗi bản ghi phải thuộc đúng một tập.
    assert len(train) + len(val) + len(test) == len(data)

    train_texts = set(_texts(train))
    val_texts = set(_texts(val))
    test_texts = set(_texts(test))

    # Không có nội dung trùng giữa Train, Validation và Test.
    assert train_texts.isdisjoint(val_texts)
    assert train_texts.isdisjoint(test_texts)
    assert val_texts.isdisjoint(test_texts)

    # Mỗi tập đều có đủ hai lớp; toy data được tạo cân bằng nên mỗi lớp
    # phải có số lượng bằng nhau trong từng tập với các tỷ lệ đã chọn.
    for split in (train, val, test):
        assert set(_labels(split)) == {"ham", "spam"}
        assert _labels(split).count("ham") == _labels(split).count("spam")

    # Tổng số mẫu của từng nhãn sau khi chia phải được bảo toàn.
    for label in ("ham", "spam"):
        original_count = sum(row["label"] == label for row in data)
        split_count = sum(
            row["label"] == label for split in (train, val, test) for row in split
        )
        assert split_count == original_count


def test_clean_email_split():
    """Kiểm tra giữ email hợp lệ và loại dòng rỗng/nhãn không hợp lệ."""
    data = [
        {"label": "ham", "text": "hello friend"},
        {"label": "spam", "text": "win a free prize"},
        {"label": "ham", "text": ""},
        {"label": "ham", "text": "   "},
        {"label": "unknown", "text": "message with invalid label"},
    ]

    cleaned = clean_email_split(
        data,
        text_col="text",
        label_col="label",
    )

    assert isinstance(cleaned, list)
    assert len(cleaned) == 2
    assert {row["text"] for row in cleaned} == {
        "hello friend",
        "win a free prize",
    }
    assert all(row["label"] in {"ham", "spam"} for row in cleaned)
    assert all(isinstance(row["text"], str) and row["text"].strip() for row in cleaned)

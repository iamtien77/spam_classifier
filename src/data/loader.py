"""Đọc CSV bằng thư viện chuẩn. Bảng dữ liệu là list[dict], không phải DataFrame."""
import csv
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Optional


def resolve_data_path(custom_path: Optional[Path] = None) -> Path:
    """Đường dẫn tùy chỉnh sai phải báo lỗi, không âm thầm dùng dữ liệu khác."""
    if custom_path is not None:
        path = Path(custom_path)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu: {path}")
        return path
    root = Path(__file__).resolve().parents[2]
    for path in (root / 'data/raw/spam.csv', root / 'data/spam.csv',
                 Path('data/raw/spam.csv'), Path('../data/raw/spam.csv')):
        if path.is_file():
            return path
    raise FileNotFoundError('Không tìm thấy spam.csv. Hãy truyền đường dẫn cụ thể.')


def load_raw_data(data_path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Giữ cả dòng lỗi để EDA đếm được; lọc chất lượng ở clean_email_split.

    raw_text không bị sửa. source_id là số thứ tự bản ghi CSV, dùng kiểm tra
    chia tập và đối chiếu nguồn (không phải số dòng vật lý nếu có xuống dòng).
    """
    path = resolve_data_path(data_path)
    # CSV dự án có nội dung dài hơn giới hạn mặc định 128 KiB.
    # Chặn ở giới hạn C long 32 bit để tương thích cả Windows.
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(max(csv.field_size_limit(), min(path.stat().st_size, 2**31 - 1)))
    try:
        for encoding in ('utf-8-sig', 'latin-1'):
            try:
                with path.open(encoding=encoding, newline='') as stream:
                    reader = csv.DictReader(stream)
                    fields = reader.fieldnames or []
                    lookup = {field.strip().lower(): field for field in fields}
                    label_field = next((lookup[k] for k in ('label', 'category', 'v1') if k in lookup), None)
                    text_field = next((lookup[k] for k in ('text', 'message', 'v2', 'body') if k in lookup), None)
                    if label_field is None or text_field is None:
                        raise ValueError('CSV cần cột label/text, Category/Message hoặc v1/v2.')
                    records = []
                    for source_id, row in enumerate(reader, 1):
                        text = row.get(text_field) or ''
                        records.append({'source_id': source_id,
                                        'label': (row.get(label_field) or '').strip().lower(),
                                        'text': text, 'raw_text': text})
                    return records
            except UnicodeDecodeError:
                continue
        raise ValueError('Không thể đọc mã hóa của CSV.')
    finally:
        # Khôi phục cấu hình toàn cục để không ảnh hưởng module CSV khác.
        csv.field_size_limit(previous_limit)


def explore_raw_data(df: list[dict[str, Any]]) -> dict[str, Any]:
    """Thống kê trực tiếp từ dữ liệu, không gán trước tỷ lệ Ham/Spam."""
    counts = Counter(row.get('label', '') for row in df)
    valid = sum(counts.get(label, 0) for label in ('ham', 'spam'))
    lengths = [len(str(row.get('text') or '')) for row in df]
    words = [len(str(row.get('text') or '').split()) for row in df]
    def describe(values):
        return {'min': min(values), 'max': max(values), 'mean': mean(values),
                'median': median(values)} if values else {'min': 0, 'max': 0, 'mean': 0, 'median': 0}
    return {'n_rows': len(df), 'n_columns': len(df[0]) if df else 0,
            'missing_values': {key: sum(not str(row.get(key) or '').strip() for row in df)
                               for key in ('label', 'text')},
            'label_counts': dict(counts),
            'label_percentages': {label: 100 * counts.get(label, 0) / valid if valid else 0
                                  for label in ('ham', 'spam')},
            'invalid_label_count': len(df) - valid,
            'character_length': describe(lengths), 'word_length': describe(words)}

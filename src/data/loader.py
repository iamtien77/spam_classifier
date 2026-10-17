"""Đọc CSV bằng thư viện chuẩn và pandas. Trả về DataFrame (hoặc list[dict] nếu yêu cầu)."""
import csv
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Optional, Union
import pandas as pd


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


def load_raw_data(data_path: Optional[Path] = None, as_dataframe: bool = True) -> Union[pd.DataFrame, list[dict[str, Any]]]:
    """Giữ nhãn sai/text rỗng để EDA; lỗi cấu trúc CSV báo ValueError.

    raw_text không bị sửa. source_id là số thứ tự bản ghi CSV, dùng kiểm tra
    chia tập và đối chiếu nguồn. Mặc định trả về pandas.DataFrame.
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
                    reader = csv.DictReader(stream, strict=True)
                    fields = reader.fieldnames or []
                    normalized_fields = [field.strip().lower() for field in fields]
                    if len(set(normalized_fields)) != len(normalized_fields):
                        raise ValueError('CSV có tên cột trùng sau khi chuẩn hóa hoa/thường và khoảng trắng.')
                    lookup = dict(zip(normalized_fields, fields))
                    label_field = next((lookup[k] for k in ('label', 'category', 'v1') if k in lookup), None)
                    text_field = next((lookup[k] for k in ('text', 'message', 'v2', 'body') if k in lookup), None)
                    if label_field is None or text_field is None:
                        raise ValueError('CSV cần cột label/text, Category/Message hoặc v1/v2.')
                    records = []
                    for source_id, row in enumerate(reader, 1):
                        if None in row or any(value is None for value in row.values()):
                            raise ValueError(
                                f'CSV sai số trường tại bản ghi {source_id} '
                                f'(kết thúc ở dòng {reader.line_num}); cần {len(fields)} trường.'
                            )
                        text = row.get(text_field) or ''
                        records.append({'source_id': source_id,
                                        'label': (row.get(label_field) or '').strip().lower(),
                                        'text': text, 'raw_text': text})
                    if as_dataframe:
                        return pd.DataFrame(records, columns=['source_id', 'label', 'text', 'raw_text'])
                    return records
            except UnicodeDecodeError:
                continue
            except csv.Error as exc:
                raise ValueError(f'CSV không hợp lệ gần dòng {reader.line_num}: {exc}') from exc
        raise ValueError('Không thể đọc mã hóa của CSV.')
    finally:
        # Khôi phục cấu hình toàn cục để không ảnh hưởng module CSV khác.
        csv.field_size_limit(previous_limit)


def explore_raw_data(records: Optional[Any] = None, *, df: Optional[Any] = None) -> dict[str, Any]:
    """Thống kê trực tiếp từ dữ liệu, không gán trước tỷ lệ Ham/Spam."""
    data = records if records is not None else df
    if data is None:
        raise ValueError("Cần truyền danh sách bản ghi records hoặc df.")
    if hasattr(data, "to_dict"):
        data = data.to_dict("records")
    counts = Counter(row.get('label', '') for row in data)
    valid = sum(counts.get(label, 0) for label in ('ham', 'spam'))
    lengths = [len(str(row.get('text') or '')) for row in data]
    words = [len(str(row.get('text') or '').split()) for row in data]
    def describe(values):
        return {'min': min(values), 'max': max(values), 'mean': mean(values),
                'median': median(values)} if values else {'min': 0, 'max': 0, 'mean': 0, 'median': 0}
    return {'n_rows': len(data), 'n_columns': len(data[0]) if data else 0,
            'missing_values': {key: sum(not str(row.get(key) or '').strip() for row in data)
                                for key in ('label', 'text')},
            'label_counts': dict(counts),
            'label_percentages': {label: 100 * counts.get(label, 0) / valid if valid else 0
                                  for label in ('ham', 'spam')},
            'invalid_label_count': len(data) - valid,
            'character_length': describe(lengths), 'word_length': describe(words)}

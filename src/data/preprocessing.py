"""Tiền xử lý thuần Python. Các hàm tên dataframe nhận/trả list[dict].

Quy trình: kiểm tra chất lượng toàn bộ dữ liệu -> chia tập -> chuẩn hóa text.
Không học bất kỳ từ vựng, IDF hoặc ngưỡng độ dài nào từ Validation/Test.
"""
import html
from html.parser import HTMLParser
import math
import random
import re
import unicodedata
from typing import Any, Optional

# Danh sách cố định trong mã nguồn: không tải tài nguyên NLP bên ngoài.
# Giữ 'not', 'no', 'never' vì phủ định có thể thay đổi nghĩa văn bản.
STOP_WORDS = frozenset('a an the and or but if then than as at by for from in into of on to with is am are was were be been being it its this that these those i me my we us our you your he him his she her they them their do does did have has had can could would should will shall may might must also very just so such there here about up down out over under again once during before after all any both each few more most other some own same too s t'.split())
URL_PATTERN = re.compile(r'(?:https?://|www\.)[^\s<>]+', re.IGNORECASE)
# Bắt tiền tệ trước khi NFKD hoặc loại dấu câu làm mất ký hiệu £/€/$.
MONEY_PATTERN = re.compile(r'[$£€¥]\s*\d+(?:[.,]\d+)*|\b\d+(?:[.,]\d+)*\s*(?:usd|gbp|eur|dollars?|pounds?)\b', re.IGNORECASE)
NUMBER_PATTERN = re.compile(r'\b\d+(?:[.,]\d+)*\b')



# Chỉ loại thẻ HTML đã biết; giữ đoạn ngoặc nhọn của văn bản thường.
# Không decode &lt; trước khi parse: văn bản escaped không phải markup.
HTML_TAGS = frozenset("html head body title meta link style script div span p br hr b i u strong em a img table thead tbody tfoot tr td th ul ol li dl dt dd h1 h2 h3 h4 h5 h6 blockquote pre code section article header footer nav main aside form input button label select option textarea font center small big sub sup s strike del ins video audio source iframe noscript wbr address figure figcaption details summary".split())


class _EmailHTMLTextParser(HTMLParser):
    """Bỏ markup đã biết, giữ nội dung và thẻ lạ để tránh mất tín hiệu email.

    Đây là bước trích văn bản ML, không phải bộ lọc bảo mật HTML.
    """
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        self.parts.append(' ' if tag in HTML_TAGS else self.get_starttag_text())

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        self.parts.append(' ' if tag in HTML_TAGS else f'</{tag}>')


def _extract_html_text(text):
    parser = _EmailHTMLTextParser()
    parser.feed(text)
    parser.close()
    return html.unescape(''.join(parser.parts))


def _normalise_label(value: Any) -> str:
    return str(value or '').strip().lower()


def _content_key(text: str) -> str:
    """Khóa trùng chỉ chuẩn hóa Unicode và khoảng trắng; giữ hoa/thường, dấu câu.

    Không xóa token/số để tránh gộp nhầm các thông điệp khác nhau.
    """
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def audit_data_quality(df, text_col='text', label_col='label') -> dict[str, int]:
    """Phương án: đếm lỗi, nhóm mâu thuẫn và bản lặp cùng nhãn riêng biệt."""
    # Ưu tiên đếm nhãn sai trước, rồi text rỗng: mỗi dòng chỉ vào một nhóm lỗi.
    # Tổng nhóm loại + retained_rows phải bằng input_rows.
    groups = {}
    invalid_label = empty_text = 0
    for row in df:
        label = _normalise_label(row.get(label_col))
        text = row.get(text_col)
        if label not in ('ham', 'spam'):
            invalid_label += 1
        elif not isinstance(text, str) or not text.strip():
            empty_text += 1
        else:
            groups.setdefault(_content_key(text), []).append(label)
    conflicts = [labels for labels in groups.values() if len(set(labels)) > 1]
    duplicate_rows = sum(len(labels) - 1 for labels in groups.values() if len(set(labels)) == 1)
    return {'input_rows': len(df), 'invalid_label_rows': invalid_label,
            'empty_text_rows': empty_text, 'conflicting_groups': len(conflicts),
            'conflicting_rows': sum(map(len, conflicts)),
            'duplicate_rows_removed': duplicate_rows,
            'retained_rows': sum(len(set(labels)) == 1 for labels in groups.values())}


def clean_email_split(df, text_col='text', label_col='label') -> list[dict[str, Any]]:
    """Lọc chất lượng, không thay đổi dữ liệu đầu vào.

    QUAN TRỌNG: gọi trên TOÀN BỘ dữ liệu trước chia tập để loại cả nhóm
    mâu thuẫn và trùng liên tập. Tên hàm giữ để nhóm dễ đối chiếu khung cũ.
    Nhãn vẫn là ham/spam; nhãn số y được tạo ở prepare_feature_frame.
    """
    groups = {}
    for row in df:
        label = _normalise_label(row.get(label_col))
        text = row.get(text_col)
        if label not in ('ham', 'spam') or not isinstance(text, str) or not text.strip():
            continue
        key = _content_key(text)
        groups.setdefault(key, []).append((label, row))
    # dict giữ thứ tự xuất hiện: nhóm cùng nhãn giữ đúng bản đầu tiên.
    # Không dùng majority vote cho nhóm mâu thuẫn vì lặp dữ liệu có thể lệch nhãn.
    cleaned = []
    for group in groups.values():
        if len({label for label, _ in group}) != 1:
            continue  # Không tự đoán nhãn đúng cho nhóm mâu thuẫn.
        label, original = group[0]
        row = dict(original)
        row[label_col] = label
        row.setdefault('raw_text', row[text_col])
        cleaned.append(row)
    return cleaned


def stratified_split_dataframe(df, label_col='label', train_ratio=0.80,
                               val_ratio=0.10, test_ratio=0.10, random_state=42):
    """Chia phân tầng tự viết bằng Random; trả train, val, test là list[dict].

    Làm tròn bằng phần dư lớn nhất, nên tổng số mẫu được bảo toàn. Với lớp
    quá nhỏ, không thể bảo đảm lớp xuất hiện ở mọi tập; báo lỗi rõ ràng.
    """
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(not math.isfinite(r) or r < 0 for r in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError('Tỷ lệ chia phải không âm, hữu hạn và có tổng bằng 1.')
    if not df:
        raise ValueError('Dữ liệu chia tập không được rỗng.')
    groups = {}
    content_keys = set()
    for row in df:
        label = _normalise_label(row.get(label_col))
        if label not in ('ham', 'spam'):
            raise ValueError('Hãy lọc nhãn sai bằng clean_email_split trước khi chia.')
        if 'text' in row:
            key = _content_key(str(row['text']))
            if key in content_keys:
                raise ValueError('Nội dung trùng: hãy gọi clean_email_split trên toàn bộ dữ liệu trước.')
            content_keys.add(key)
        groups.setdefault(label, []).append(dict(row))
    if set(groups) != {'ham', 'spam'}:
        raise ValueError('Dữ liệu phân loại cần đủ hai nhãn ham và spam trước khi chia tập.')
    rng = random.Random(random_state)
    splits = [[], [], []]
    for label in sorted(groups):
        rows = groups[label]
        rng.shuffle(rows)
        exact = [len(rows) * r for r in ratios]
        sizes = [math.floor(n) for n in exact]
        order = sorted(range(3), key=lambda i: (-(exact[i] - sizes[i]), i))
        for i in order[:len(rows) - sum(sizes)]:
            sizes[i] += 1
        if any(r > 0 and n == 0 for r, n in zip(ratios, sizes)):
            raise ValueError(f'Lớp {label} quá ít mẫu để xuất hiện trong mỗi tập.')
        start = 0
        for split, size in zip(splits, sizes):
            split.extend(rows[start:start + size])
            start += size
    for split in splits:
        rng.shuffle(split)
    return tuple(splits)


def clean_text_basic(text: str) -> str:
    """Thay token đặc biệt, giữ chữ Unicode; không âm thầm bỏ ngôn ngữ khác.

    NFKD rồi ghép NFC giữ dấu, thay vì encode ASCII làm mất nội dung.
    Stop words cố định tiếng Anh phù hợp dữ liệu dự án; chưa có bộ riêng
    cho tiếng Việt, do đó không tuyên bố hỗ trợ tốt mọi ngôn ngữ.
    """
    if not isinstance(text, str):
        raise TypeError('Nội dung văn bản phải là chuỗi.')
    text = _extract_html_text(text)
    text = URL_PATTERN.sub(' urltoken ', text)
    text = MONEY_PATTERN.sub(' moneytoken ', text)
    text = NUMBER_PATTERN.sub(' numbertoken ', text)
    text = unicodedata.normalize('NFC', unicodedata.normalize('NFKD', text)).lower()
    text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)
    return ' '.join(token for token in text.split() if token not in STOP_WORDS)


class SpamTextProcessor:
    """Bộ xử lý không trạng thái học; ngưỡng cắt độ dài do builder fit trên Train."""
    def __init__(self, config: Optional[Any] = None):
        self.config = config

    def normalize_text(self, text: str) -> str:
        return clean_text_basic(text)

    def tokenize_words(self, text: str) -> list[str]:
        return text.split()

    def extract_word_ngrams(self, tokens: list[str], n: int = 2) -> list[str]:
        if not isinstance(n, int) or n < 1:
            raise ValueError('Bậc n-gram phải là số nguyên dương.')
        return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def extract_char_ngrams(self, text: str, min_n: int = 3, max_n: int = 5) -> list[str]:
        if min_n < 1 or max_n < min_n:
            raise ValueError('Khoảng character n-gram không hợp lệ.')
        padded = ' ' + text + ' '
        return [padded[i:i + n] for n in range(min_n, max_n + 1)
                for i in range(len(padded) - n + 1)]

    def extract_structural_features(self, raw_text: str) -> dict[str, float]:
        letters = sum(char.isalpha() for char in raw_text)
        return {'has_url': float(bool(URL_PATTERN.search(raw_text))),
                'has_number': float(any(char.isdigit() for char in raw_text)),
                'digit_count': float(sum(char.isdigit() for char in raw_text)),
                'special_char_count': float(sum(not char.isalnum() and not char.isspace() for char in raw_text)),
                'exclamation_count': float(raw_text.count('!')),
                'dollar_count': float(raw_text.count('$')),
                'question_count': float(raw_text.count('?')),
                'uppercase_ratio': sum(char.isupper() for char in raw_text) / letters if letters else 0.0,
                'raw_length': float(len(raw_text)), 'raw_word_count': float(len(raw_text.split()))}

    def prepare_feature_frame(self, df, cap_word_limit=None, use_length_cap=True):
        """Nhận cả văn bản mới không có label; không tự học ngưỡng từ đầu vào."""
        if cap_word_limit is not None and (not isinstance(cap_word_limit, int) or cap_word_limit < 1):
            raise ValueError('Ngưỡng số từ phải là số nguyên dương.')
        prepared = []
        for original in df:
            row = dict(original)
            raw_text = row.get('raw_text', row.get('text'))
            if not isinstance(raw_text, str):
                raise TypeError('Mỗi bản ghi cần raw_text hoặc text kiểu chuỗi.')
            model_text = self.normalize_text(raw_text)
            if use_length_cap and cap_word_limit is not None:
                model_text = ' '.join(model_text.split()[:cap_word_limit])
            row.update(self.extract_structural_features(raw_text))
            row.update(raw_text=raw_text, model_text=model_text,
                       model_length=float(len(model_text.split())))
            if 'label' in row:
                label = _normalise_label(row['label'])
                if label not in ('ham', 'spam'):
                    raise ValueError('Nhãn phải là ham hoặc spam.')
                row['label'] = label
                row['y'] = int(label == 'spam')
            prepared.append(row)
        return prepared

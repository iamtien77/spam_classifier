"""TF-IDF, scaler, CSR và hybrid tự viết, chỉ dùng thư viện chuẩn Python.

CSR lưu các phần tử khác 0 theo hàng, không tạo bảng đặc trưng đặc lớn.
Task 2 dùng iter_row(), dot() và transpose_dot() để huấn luyện mô hình.
"""
import math
from array import array
from collections import Counter
from copy import deepcopy
from typing import Any, Optional
import re
from src.config import SpamExperimentConfig
from src.data.preprocessing import SpamTextProcessor


class CSRMatrixScratch:
    """Ma trận Compressed Sparse Row độc lập với SciPy.

    data: giá trị; indices: chỉ số cột; indptr[i:i+2]: đoạn dữ liệu hàng i.
    Ví dụ [[0, 2], [3, 0]] -> data=[2,3], indices=[1,0], indptr=[0,1,2].
    Không hỗ trợ API NumPy/SciPy; phép toán có giao diện riêng bên dưới.
    """
    def __init__(self, data, indices, indptr, shape):
        if len(shape) != 2 or any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError('Kích thước ma trận phải gồm hai số nguyên không âm.')
        if (len(data) != len(indices) or len(indptr) != shape[0] + 1 or
                not indptr or indptr[0] != 0 or indptr[-1] != len(data) or
                any(a > b for a, b in zip(indptr, indptr[1:]))):
            raise ValueError('Cấu trúc CSR không hợp lệ.')
        if any(not isinstance(i, int) or not 0 <= i < shape[1] for i in indices):
            raise ValueError('Chỉ số cột CSR không hợp lệ.')
        if any(not isinstance(i, int) or not 0 <= i <= len(data) for i in indptr):
            raise ValueError('Con trỏ hàng CSR không hợp lệ.')
        if any(not math.isfinite(v) for v in data):
            raise ValueError('CSR không chấp nhận NaN hoặc vô cực.')
        for row in range(shape[0]):
            columns = indices[indptr[row]:indptr[row + 1]]
            if any(a >= b for a, b in zip(columns, columns[1:])):
                raise ValueError('Các cột trong mỗi hàng phải tăng dần, không trùng.')
        self.data = array('d', data)
        self.indices = array('I', indices)
        self.indptr = array('Q', indptr)
        self.shape = tuple(shape)

    @property
    def nnz(self):
        return len(self.data)

    def __len__(self):
        return self.shape[0]

    def iter_row(self, row):
        """Duyệt (column_index, value) khác 0, không giải nén thành hàng đặc."""
        if not 0 <= row < self.shape[0]:
            raise IndexError('Chỉ số hàng ngoài ma trận.')
        for pos in range(self.indptr[row], self.indptr[row + 1]):
            yield self.indices[pos], self.data[pos]

    @classmethod
    def from_rows(cls, rows, n_columns):
        data, indices, indptr = array('d'), array('I'), array('Q', [0])
        for row in rows:
            for col, value in sorted(row.items()):
                if value != 0:
                    indices.append(col)
                    data.append(value)
            indptr.append(len(data))
        return cls(data, indices, indptr, (len(indptr) - 1, n_columns))

    @classmethod
    def hstack(cls, matrices):
        """Ghép ngang theo thứ tự khối; không chuyển sang ma trận đặc."""
        if not matrices:
            raise ValueError('Cần ít nhất một ma trận để ghép.')
        n_rows = matrices[0].shape[0]
        if any(matrix.shape[0] != n_rows for matrix in matrices):
            raise ValueError('Các khối đặc trưng phải có cùng số hàng.')
        offsets, width = [], 0
        for matrix in matrices:
            offsets.append(width)
            width += matrix.shape[1]
        data, indices, indptr = array('d'), array('I'), array('Q', [0])
        for row in range(n_rows):
            for matrix, offset in zip(matrices, offsets):
                for col, value in matrix.iter_row(row):
                    indices.append(offset + col)
                    data.append(value)
            indptr.append(len(data))
        return cls(data, indices, indptr, (n_rows, width))

    def dot(self, weights):
        """X @ w -> list[float], hữu ích cho LR/SVM; bias cộng ở mô hình."""
        if len(weights) != self.shape[1]:
            raise ValueError('Vector trọng số sai số chiều.')
        return [sum(value * weights[col] for col, value in self.iter_row(row))
                for row in range(self.shape[0])]

    def transpose_dot(self, values):
        """X.T @ v -> list[float], dùng tính gradient không cần tạo X.T."""
        if len(values) != self.shape[0]:
            raise ValueError('Vector hệ số phải bằng số hàng.')
        result = [0.0] * self.shape[1]
        for row, coefficient in enumerate(values):
            for col, value in self.iter_row(row):
                result[col] += value * coefficient
        return result

    def select_columns(self, columns):
        """Chọn cột theo thứ tự yêu cầu, phục vụ thử nghiệm các tổ hợp."""
        columns = list(columns)
        if len(set(columns)) != len(columns) or any(not 0 <= col < self.shape[1] for col in columns):
            raise ValueError('Danh sách cột không hợp lệ hoặc bị trùng.')
        mapping = {old: new for new, old in enumerate(columns)}
        return self.from_rows(({mapping[col]: value for col, value in self.iter_row(row)
                                if col in mapping} for row in range(self.shape[0])), len(columns))

    def to_dense(self, max_cells=100_000):
        """Chỉ dùng cho ví dụ nhỏ/trực quan; chặn giải nén dữ liệu lớn."""
        if self.shape[0] * self.shape[1] > max_cells:
            raise ValueError('Ma trận quá lớn để chuyển đặc; hãy dùng iter_row/dot.')
        result = [[0.0] * self.shape[1] for _ in range(self.shape[0])]
        for row in range(self.shape[0]):
            for col, value in self.iter_row(row):
                result[row][col] = value
        return result


class TfidfVectorizerScratch:
    """DF đếm số tài liệu có token, không phải tổng số lần token xuất hiện.

    Ngoài chuỗi, nhận danh sách token để giữ nguyên bigram/char n-gram có
    khoảng trắng. Không nối n-gram rồi split, vì sẽ phá vỡ đơn vị đặc trưng.
    """
    def __init__(self, min_df=2, max_features=None, sublinear_tf=True,
                 smooth_idf=True, norm='l2'):
        if not isinstance(min_df, int) or min_df < 1:
            raise ValueError('min_df phải là số nguyên dương.')
        if max_features is not None and (not isinstance(max_features, int) or max_features < 1):
            raise ValueError('max_features phải là số nguyên dương hoặc None.')
        if norm not in ('l2', None):
            raise ValueError('norm chỉ hỗ trợ l2 hoặc None.')
        self.min_df, self.max_features = min_df, max_features
        self.sublinear_tf, self.smooth_idf, self.norm = sublinear_tf, smooth_idf, norm
        self.vocabulary_ = {}
        self.idf_diag_ = None

    @staticmethod
    def _tokens(document):
        return document.split() if isinstance(document, str) else document

    def fit(self, raw_documents):
        frequency, n_documents = Counter(), 0
        for document in raw_documents:
            frequency.update(set(self._tokens(document)))
            n_documents += 1
        if n_documents == 0:
            raise ValueError('Không thể fit TF-IDF trên tập rỗng.')
        terms = sorted((term for term, count in frequency.items() if count >= self.min_df),
                       key=lambda term: (-frequency[term], term))
        if self.max_features is not None:
            terms = terms[:self.max_features]
        self.vocabulary_ = {term: col for col, term in enumerate(terms)}
        # Từ điển rỗng là hợp lệ khi tất cả từ hiếm/rỗng; transform trả N x 0.
        self.idf_diag_ = [math.log((1 + n_documents) / (1 + frequency[term])) + 1
                          if self.smooth_idf else math.log(n_documents / frequency[term]) + 1
                          for term in terms]
        return self

    def transform(self, raw_documents):
        if self.idf_diag_ is None:
            raise ValueError('TF-IDF chưa được fit trên Train.')
        def rows():
            for document in raw_documents:
                values = {}
                for term, count in Counter(self._tokens(document)).items():
                    if term in self.vocabulary_:
                        col = self.vocabulary_[term]
                        tf = 1 + math.log(count) if self.sublinear_tf else count
                        values[col] = tf * self.idf_diag_[col]
                if self.norm == 'l2':
                    length = math.sqrt(sum(value * value for value in values.values()))
                    if length:
                        values = {col: value / length for col, value in values.items()}
                yield values
        return CSRMatrixScratch.from_rows(rows(), len(self.vocabulary_))

    def fit_transform(self, raw_documents):
        documents = list(raw_documents)  # Giữ lại nếu đầu vào là generator dùng một lần.
        return self.fit(documents).transform(documents)


class MaxAbsScalerScratch:
    """Fit cực đại trên Train; dữ liệu mới có thể vượt 1, không cắt giá trị."""
    def __init__(self):
        self.max_abs_ = None

    def fit(self, X):
        if isinstance(X, CSRMatrixScratch):
            if X.shape[0] == 0:
                raise ValueError('Không thể fit scaler trên tập rỗng.')
            maximum = [0.0] * X.shape[1]
            for col, value in zip(X.indices, X.data):
                maximum[col] = max(maximum[col], abs(value))
        else:
            if not X:
                raise ValueError('Không thể fit scaler trên tập rỗng.')
            maximum = [0.0] * len(X[0])
            for row in X:
                if len(row) != len(maximum):
                    raise ValueError('Các hàng số phải có cùng độ dài.')
                for col, value in enumerate(row):
                    if not math.isfinite(value):
                        raise ValueError('Đặc trưng số phải hữu hạn.')
                    maximum[col] = max(maximum[col], abs(value))
        self.max_abs_ = [value if value else 1.0 for value in maximum]
        return self

    def transform(self, X):
        if self.max_abs_ is None:
            raise ValueError('Scaler chưa được fit trên Train.')
        if isinstance(X, CSRMatrixScratch):
            if X.shape[1] != len(self.max_abs_):
                raise ValueError('Số cột không khớp scaler.')
            return CSRMatrixScratch([value / self.max_abs_[col] for col, value in zip(X.indices, X.data)],
                                    X.indices, X.indptr, X.shape)
        transformed = []
        for row in X:
            if len(row) != len(self.max_abs_) or any(not math.isfinite(value) for value in row):
                raise ValueError('Dữ liệu số không hợp lệ hoặc sai số chiều.')
            transformed.append([value / scale for value, scale in zip(row, self.max_abs_)])
        return transformed

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class SpamSignalFeatureExtractor:
    def __init__(self, config=None):
        self.config = config or SpamExperimentConfig()

    def build_keyword_matrix(self, texts, patterns=None):
        patterns = self.config.keyword_patterns if patterns is None else patterns
        compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns.values()]
        return CSRMatrixScratch.from_rows(
            ({col: 1.0 for col, pattern in enumerate(compiled) if pattern.search(text)} for text in texts),
            len(compiled))

    def extract_numeric_features(self, df, scaler=None, fit=True):
        rows = []
        for record in df:
            values = []
            for name, mode in self.config.numeric_feature_config:
                value = float(record[name])
                if value < 0 or not math.isfinite(value):
                    raise ValueError(f'Đặc trưng {name} phải không âm và hữu hạn.')
                if mode == 'log_minmax':
                    value = math.log1p(value)
                elif mode not in ('minmax', 'binary'):
                    raise ValueError(f'Chế độ đặc trưng không hỗ trợ: {mode}')
                # Tên mode giữ theo config cũ; scaler thực tế là MaxAbs, không MinMax.
                values.append(value)
            rows.append(values)
        if fit:
            scaler = MaxAbsScalerScratch().fit(rows)
        elif scaler is None:
            raise ValueError('Transform cần scaler đã fit trên Train.')
        # Với đầu vào rỗng, vẫn giữ số chiều đã fit.
        scaled = scaler.transform(rows)
        matrix = CSRMatrixScratch.from_rows(({i: value for i, value in enumerate(row)} for row in scaled),
                                             len(scaler.max_abs_))
        return matrix, scaler


class HybridFeatureBuilderScratch:
    """Một fit duy nhất trên Train; mọi transform giữ nguyên thứ tự cột.

    feature_slices_ ghi (start, stop) cho từng khối để mô hình/phân tích lỗi
    xác định nguồn đặc trưng. Tất cả đặc trưng không âm, dùng được cho NB.
    """
    def __init__(self, config: Any, text_processor: Any, signal_extractor: SpamSignalFeatureExtractor):
        self.config, self.text_processor, self.signal_extractor = config, text_processor, signal_extractor
        self.word_vectorizer = self.bigram_vectorizer = self.char_vectorizer = None
        self.numeric_scaler = None
        self.feature_names_, self.feature_slices_ = [], {}
        self.cap_word_limit_ = None
        self.params_ = None
        self.fitted_config_ = None
        self.fitted_signal_extractor_ = None
        self.is_fitted_ = False

    def _prepare(self, records):
        return self.text_processor.prepare_feature_frame(records, self.cap_word_limit_,
                                                         self.params_['use_length_cap'])

    def _documents(self, records, kind):
        for row in records:
            text = row['model_text']
            if kind == 'word':
                yield text
            elif kind == 'bigram':
                yield self.text_processor.extract_word_ngrams(text.split(), 2)
            else:
                yield self.text_processor.extract_char_ngrams(text, *self.fitted_config_.char_ngram_range)

    def fit(self, train_df):
        # Một fit mới bị lỗi không được để transform dùng trạng thái học dở.
        self.is_fitted_ = False
        if not train_df:
            raise ValueError('Train không được rỗng.')
        # Chụp cấu hình tại fit để đổi config bên ngoài không làm lệch cột
        # khi transform hoặc dự đoán sau này. deepcopy thuộc thư viện chuẩn.
        self.fitted_config_ = deepcopy(self.config)
        self.fitted_signal_extractor_ = SpamSignalFeatureExtractor(deepcopy(self.signal_extractor.config))
        if self.fitted_config_.text_column_for_model != 'model_text':
            raise ValueError('Task 1 chỉ hỗ trợ text_column_for_model= model_text do processor tạo.')
        self.params_ = self.fitted_config_.completed_params()
        # Không âm thầm bỏ qua cấu hình count/log_count trong khung cũ.
        # Pipeline bàn giao này dùng đúng TF-IDF sublinear theo Task 1.
        if self.params_['text_feature_mode'] != 'log_count':
            raise ValueError('Pipeline Task 1 dùng log_count (TF-IDF sublinear); không hỗ trợ chế độ khác.')
        self.cap_word_limit_ = None
        if self.params_['use_length_cap'] and self.params_['cap_quantile'] is not None:
            q = self.params_['cap_quantile']
            if not 0 < q <= 1:
                raise ValueError('cap_quantile phải thuộc (0, 1].')
            initial = self.text_processor.prepare_feature_frame(train_df, use_length_cap=False)
            lengths = sorted(int(row['model_length']) for row in initial)
            self.cap_word_limit_ = max(1, lengths[max(0, math.ceil(q * len(lengths)) - 1)])
        prepared = self._prepare(train_df)
        self.feature_names_, self.feature_slices_ = [], {}
        self.word_vectorizer = self.bigram_vectorizer = self.char_vectorizer = None
        self.numeric_scaler = None
        blocks = [('word', 'word_vectorizer', self.params_['max_unigram_features'], True),
                  ('bigram', 'bigram_vectorizer', self.params_['max_bigram_features'], self.params_['use_bigram']),
                  ('char', 'char_vectorizer', self.params_['max_char_features'], self.params_['use_char_ngram'])]
        for name, attr, maximum, enabled in blocks:
            if not enabled:
                continue
            vectorizer = TfidfVectorizerScratch(min_df=self.params_['min_df'], max_features=maximum)
            vectorizer.fit(self._documents(prepared, name))
            setattr(self, attr, vectorizer)
            self._register(name, [f'{name}:{term}' for term in vectorizer.vocabulary_])
        if self.params_['use_keyword_features']:
            self._register('keyword', list(self.fitted_signal_extractor_.config.keyword_patterns))
        if self.params_['use_numeric_features']:
            _, self.numeric_scaler = self.fitted_signal_extractor_.extract_numeric_features(prepared)
            self._register('numeric', [name for name, _ in self.fitted_signal_extractor_.config.numeric_feature_config])
        self.is_fitted_ = True
        return self

    def _register(self, name, names):
        start = len(self.feature_names_)
        self.feature_names_.extend(names)
        self.feature_slices_[name] = (start, len(self.feature_names_))

    @staticmethod
    def _weight(matrix, weight):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Trọng số khối phải hữu hạn và không âm (để tương thích NB).')
        if weight == 0:
            return CSRMatrixScratch.from_rows(({} for _ in range(matrix.shape[0])), matrix.shape[1])
        return CSRMatrixScratch([value * weight for value in matrix.data], matrix.indices, matrix.indptr, matrix.shape)

    def transform(self, df):
        if not self.is_fitted_:
            raise ValueError('Builder chưa được fit trên Train.')
        prepared = self._prepare(df)
        matrices = []
        for name, vectorizer in [('word', self.word_vectorizer), ('bigram', self.bigram_vectorizer),
                                 ('char', self.char_vectorizer)]:
            if vectorizer is not None:
                matrices.append(vectorizer.transform(self._documents(prepared, name)))
        if self.params_['use_keyword_features']:
            matrix = self.fitted_signal_extractor_.build_keyword_matrix(row['raw_text'] for row in prepared)
            matrices.append(self._weight(matrix, self.params_['keyword_group_weight']))
        if self.params_['use_numeric_features']:
            matrix, _ = self.fitted_signal_extractor_.extract_numeric_features(prepared, self.numeric_scaler, fit=False)
            matrices.append(self._weight(matrix, self.params_['numeric_group_weight']))
        return CSRMatrixScratch.hstack(matrices)

    def fit_transform(self, train_df):
        return self.fit(train_df).transform(train_df)


def experiment_feature_combinations(train_df, val_df):
    """Trả bốn cặp ma trận; chưa đánh giá F1 vì mô hình do Task 2 phụ trách.

    Tổ hợp 2 chỉ thêm character FREQUENCY (!, $), khác char n-grams.
    Fit một builder trên Train rồi chọn cột: IDF/scaler giống nhau giữa nhánh.
    """
    config = SpamExperimentConfig()
    processor = SpamTextProcessor(config)
    builder = HybridFeatureBuilderScratch(config, processor, SpamSignalFeatureExtractor(config))
    X_train, X_val = builder.fit_transform(train_df), builder.transform(val_df)
    def columns(name):
        start, stop = builder.feature_slices_.get(name, (0, 0))
        return list(range(start, stop))
    word = columns('word')
    char_frequency = [i for i, name in enumerate(builder.feature_names_)
                      if name in ('exclamation_count', 'dollar_count')]
    combinations = {'word_tfidf': word, 'word_char_frequency': word + char_frequency,
                    'word_keywords': word + columns('keyword'),
                    'hybrid': list(range(X_train.shape[1]))}
    return {name: (X_train.select_columns(cols), X_val.select_columns(cols))
            for name, cols in combinations.items()}

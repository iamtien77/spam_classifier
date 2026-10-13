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

    def to_scipy_csr(self):
        """Chuyển đổi sang scipy.sparse.csr_matrix để tương thích hoàn toàn với Task 2."""
        try:
            from scipy.sparse import csr_matrix
            import numpy as np
            return csr_matrix(
                (np.array(self.data, dtype=np.float64),
                 np.array(self.indices, dtype=np.int32),
                 np.array(self.indptr, dtype=np.int64)),
                shape=self.shape
            )
        except ImportError as err:
            raise ImportError(
                "Chức năng to_scipy_csr() yêu cầu thư viện scipy. "
                "Hãy kích hoạt môi trường ảo .venv (ví dụ: .venv\\Scripts\\Activate.ps1) "
                "hoặc chạy bằng .venv/Scripts/python.exe."
            ) from err


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
    """Fit cực đại trên Train; dữ liệu mới có thể vượt 1, không cắt giá trị.
    Hỗ trợ CSRMatrixScratch, scipy.sparse.csr_matrix, numpy.ndarray và list[list[float]].
    """
    def __init__(self):
        self.max_abs_ = None

    def fit(self, X):
        if isinstance(X, CSRMatrixScratch):
            if X.shape[0] == 0:
                raise ValueError('Không thể fit scaler trên tập rỗng.')
            maximum = [0.0] * X.shape[1]
            for col, value in zip(X.indices, X.data):
                maximum[col] = max(maximum[col], abs(value))
            self.max_abs_ = [value if value else 1.0 for value in maximum]
        elif hasattr(X, "tocsr") or (hasattr(X, "shape") and len(getattr(X, "shape", ())) == 2):
            import numpy as np
            if hasattr(X, "toarray"):
                # Scipy sparse
                X_csr = X.tocsr()
                if X_csr.shape[0] == 0:
                    raise ValueError('Không thể fit scaler trên tập rỗng.')
                max_abs = np.asarray(np.abs(X_csr).max(axis=0).toarray()).ravel()
            else:
                values = np.asarray(X, dtype=float)
                if values.shape[0] == 0:
                    raise ValueError('Không thể fit scaler trên tập rỗng.')
                max_abs = np.max(np.abs(values), axis=0)
            max_abs[max_abs == 0.0] = 1.0
            self.max_abs_ = max_abs.tolist()
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
        if hasattr(X, "tocsr") or (hasattr(X, "shape") and len(getattr(X, "shape", ())) == 2):
            import numpy as np
            scales = np.asarray(self.max_abs_, dtype=float)
            if hasattr(X, "toarray"):
                from scipy import sparse
                X_csr = X.tocsr().astype(float)
                inv_scales = 1.0 / scales
                diag_scaler = sparse.diags(inv_scales)
                return X_csr.dot(diag_scaler)
            values = np.asarray(X, dtype=float)
            return values / scales
        transformed = []
        for row in X:
            if len(row) != len(self.max_abs_) or any(not math.isfinite(value) for value in row):
                raise ValueError('Dữ liệu số không hợp lệ hoặc sai số chiều.')
            transformed.append([value / scale for value, scale in zip(row, self.max_abs_)])
        return transformed

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class ManualFeatureSelectorScratch:
    """Bộ chọn lọc đặc trưng tự xây dựng từ đầu (From-Scratch).

    Nguyên lý:
    - Variance Threshold: Loại bỏ các đặc trưng gần như hằng số (phương sai thấp).
    - Chi-Square Test (chi2_scores): Đo lường mức độ phụ thuộc thống kê giữa đặc trưng và nhãn.
    - Combined Importance: Kết hợp chi2, chênh lệch trọng số Naive Bayes, và Document Activity.
    - Top-K Sweeping: Thử nghiệm dải số lượng đặc trưng trên Validation set để tìm điểm cân bằng
      tối ưu giữa hiệu năng (F1/Recall), tiết kiệm tài nguyên (Resource saving) và chống Overfitting.
    """

    @staticmethod
    def _ensure_scipy_csr(matrix):
        try:
            from scipy import sparse
            if hasattr(matrix, "to_scipy_csr"):
                return matrix.to_scipy_csr()
            if sparse.issparse(matrix):
                return matrix.tocsr()
            return sparse.csr_matrix(matrix, dtype=float)
        except ImportError:
            return matrix

    def chi2_scores(self, X, y):
        """Tính điểm kiểm định Chi-Square giữa ma trận đặc trưng không âm X và nhãn y."""
        import numpy as np
        X_csr = self._ensure_scipy_csr(X)
        y_arr = np.asarray(y, dtype=int).ravel()
        classes = np.sort(np.unique(y_arr))

        if X_csr.shape[0] != y_arr.shape[0]:
            raise ValueError("X và y phải có cùng số lượng mẫu để tính Chi-Square.")
        if hasattr(X_csr, "data") and np.any(X_csr.data < 0):
            raise ValueError("chi2_scores đòi hỏi đặc trưng không âm (non-negative).")

        feature_sum = np.asarray(X_csr.sum(axis=0)).ravel().astype(np.float64)
        class_counts = np.array([(y_arr == cls).sum() for cls in classes], dtype=np.float64)
        total_samples = max(float(y_arr.size), 1.0)

        observed = np.vstack([np.asarray(X_csr[y_arr == cls].sum(axis=0)).ravel().astype(np.float64) for cls in classes])
        expected = np.outer(class_counts / total_samples, feature_sum)

        with np.errstate(divide="ignore", invalid="ignore"):
            diff = observed - expected
            chi2_matrix = np.where(expected > 1e-12, (diff ** 2) / expected, 0.0)

        scores = np.where(np.isfinite(chi2_matrix), chi2_matrix, 0.0).sum(axis=0).astype(np.float64)
        return scores

    def fit_variance_threshold(self, X, threshold=1e-5):
        """Lọc phương sai cho ma trận đặc trưng thưa."""
        import numpy as np
        X_csr = self._ensure_scipy_csr(X)
        feature_mean = np.asarray(X_csr.mean(axis=0)).ravel().astype(np.float64)
        feature_mean_sq = np.asarray(X_csr.power(2).mean(axis=0)).ravel().astype(np.float64)
        variances = np.maximum(feature_mean_sq - feature_mean ** 2, 0.0)
        mask = variances > float(threshold)
        if not mask.any() and variances.size > 0:
            mask[int(np.argmax(variances))] = True
        return mask, variances

    @staticmethod
    def normalize_score(values):
        import numpy as np
        values = np.asarray(values, dtype=np.float64)
        min_v = np.nanmin(values) if values.size > 0 else 0.0
        max_v = np.nanmax(values) if values.size > 0 else 1.0
        denom = max(max_v - min_v, 1e-12)
        return (values - min_v) / denom

    def compute_combined_importance(self, X_train, y_train, nb_model=None):
        """Tính điểm quan trọng kết hợp: 55% Naive Bayes + 35% Chi2 + 10% Activity."""
        import numpy as np
        X_csr = self._ensure_scipy_csr(X_train)
        n_features = X_csr.shape[1]

        # 1. Điểm Chi-Square
        chi2_raw = self.chi2_scores(X_csr, y_train)
        chi2_norm = self.normalize_score(chi2_raw)

        # 2. Điểm chênh lệch Naive Bayes (nếu có model fit)
        if nb_model is not None and hasattr(nb_model, "feature_log_prob_") and nb_model.feature_log_prob_.shape[1] == n_features:
            classes = getattr(nb_model, "classes_", np.array([0, 1]))
            spam_idx = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
            ham_idx = int(np.where(classes == 0)[0][0]) if 0 in classes else 0
            nb_raw = np.abs(nb_model.feature_log_prob_[spam_idx] - nb_model.feature_log_prob_[ham_idx]).astype(np.float64)
            nb_norm = self.normalize_score(nb_raw)
        else:
            nb_norm = chi2_norm

        # 3. Tần suất xuất hiện (Document Activity)
        activity_raw = np.asarray(X_csr.sum(axis=0)).ravel().astype(np.float64)
        activity_norm = self.normalize_score(np.log1p(np.clip(activity_raw, 0, None)))

        combined = 0.55 * nb_norm + 0.35 * chi2_norm + 0.10 * activity_norm
        ranked_indices = np.argsort(-combined)
        return combined, ranked_indices

    def sweep_top_k(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        ranked_indices,
        candidate_k_list=None,
        eval_model_fn=None,
        recall_target=0.85,
    ):
        """Quét dải Top-K đặc trưng trên Validation set để chọn điểm cân bằng tối ưu."""
        import numpy as np
        import pandas as pd

        X_train_csr = self._ensure_scipy_csr(X_train)
        X_val_csr = self._ensure_scipy_csr(X_val)
        full_p = X_train_csr.shape[1]
        if candidate_k_list is None:
            candidate_k_list = [500, 1000, 1500, 2000, 2500, 3000, min(4000, full_p), full_p]
        candidate_k_list = sorted({min(int(k), full_p) for k in candidate_k_list if k > 0})

        sweep_records = []
        for top_k in candidate_k_list:
            selected_idx = ranked_indices[:top_k]
            X_tr_sub = X_train_csr[:, selected_idx]
            X_va_sub = X_val_csr[:, selected_idx]

            # Đánh giá bằng eval_model_fn (nếu có truyền) hoặc mặc định Complement Naive Bayes
            if eval_model_fn is not None:
                metrics = eval_model_fn(X_tr_sub, y_train, X_va_sub, y_val)
            else:
                from src.models.classifiers import NaiveBayesClassifier
                m = NaiveBayesClassifier(alpha=1.0, kind="complement")
                m.fit(X_tr_sub, y_train)
                val_preds = m.predict(X_va_sub, threshold=0.5)
                val_probs = m.predict_proba(X_va_sub)[:, 1] if hasattr(m, "predict_proba") else None
                from src.evaluation.metrics import SpamModelEvaluator
                evaluator = SpamModelEvaluator()
                metrics = evaluator.compute_all_metrics(y_val, val_preds, y_prob=val_probs, beta=2.0)
                # Tính train accuracy để đo overfitting gap
                train_preds = m.predict(X_tr_sub, threshold=0.5)
                metrics["train_accuracy"] = float(np.mean(np.asarray(y_train) == train_preds))

            val_acc = float(metrics.get("accuracy", 0.0))
            val_f1 = float(metrics.get("f1", 0.0))
            val_rec = float(metrics.get("recall", 0.0))
            train_acc = float(metrics.get("train_accuracy", val_acc))
            overfit_gap = max(0.0, train_acc - val_acc)
            saving_pct = float((full_p - top_k) / full_p * 100.0)
            target_met = bool(val_rec >= recall_target)

            # Điểm cân bằng (Balance Score): Ưu tiên F1, Accuracy, tiết kiệm số chiều, và phạt overfit
            balance_score = val_f1 * 0.45 + val_acc * 0.35 + (saving_pct / 100.0) * 0.20 - overfit_gap * 0.50

            sweep_records.append({
                "top_k": top_k,
                "dimension_saving_pct": saving_pct,
                "val_accuracy": val_acc,
                "val_precision": float(metrics.get("precision", 0.0)),
                "val_recall": val_rec,
                "val_f1": val_f1,
                "train_val_gap": overfit_gap,
                "target_met": target_met,
                "balance_score": balance_score,
                "selected_indices": selected_idx,
            })

        df_sweep = pd.DataFrame(sweep_records)
        # Chọn ứng viên tốt nhất: ưu tiên đạt target_met, sau đó balance_score cao nhất
        met_df = df_sweep[df_sweep["target_met"] == True]
        if not met_df.empty:
            best_row = met_df.sort_values(by=["balance_score", "val_f1"], ascending=False).iloc[0]
        else:
            best_row = df_sweep.sort_values(by=["val_recall", "balance_score"], ascending=False).iloc[0]

        return best_row.to_dict(), df_sweep



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
        if hasattr(records, "to_dict"):
            records = records.to_dict("records")
        return self.text_processor.prepare_feature_frame(records, self.cap_word_limit_,
                                                         self.params_['use_length_cap'],
                                                         as_dataframe=False)

    def _documents(self, records, kind):
        if hasattr(records, "to_dict"):
            records = records.to_dict("records")
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
        if train_df is None or len(train_df) == 0:
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
    """Trả năm cặp ma trận; chưa đánh giá F1 vì mô hình do Task 2 phụ trách.

    Tổ hợp 2 chỉ thêm character FREQUENCY (!, $), khác char n-grams.
    Tổ hợp 3 thêm character N-GRAMS (chuỗi con ký tự 3-5).
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
    combinations = {'word_tfidf': word,
                    'word_char_frequency': word + char_frequency,
                    'word_char_ngrams': word + columns('char'),
                    'word_keywords': word + columns('keyword'),
                    'hybrid': list(range(X_train.shape[1]))}
    return {name: (X_train.select_columns(cols), X_val.select_columns(cols))
            for name, cols in combinations.items()}


def export_processed_data(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    builder: Optional[HybridFeatureBuilderScratch] = None,
    output_dir: Optional[Any] = None,
) -> dict[str, Any]:
    """Lưu dữ liệu phân tầng sạch và các ma trận đặc trưng vào data/processed/.

    Đảm bảo tính lưu trữ đệm (Cache Persistence) để Task 2 nạp nhanh không cần parse lại CSV 58MB.
    """
    import csv
    from pathlib import Path
    from src.config import PROCESSED_DATA_DIR

    target_dir = Path(output_dir) if output_dir is not None else PROCESSED_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    exported_files: dict[str, Any] = {}
    for name, split_records in [('train', train_records), ('val', val_records), ('test', test_records)]:
        csv_path = target_dir / f'{name}_clean.csv'
        if split_records is not None and len(split_records) > 0:
            if hasattr(split_records, "to_csv"):
                cols = [c for c in ['source_id', 'label', 'text', 'raw_text'] if c in split_records.columns]
                split_records[cols].to_csv(csv_path, index=False)
            else:
                fieldnames = ['source_id', 'label', 'text', 'raw_text']
                with csv_path.open('w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                    for row in split_records:
                        writer.writerow(row)
            exported_files[f'{name}_csv'] = str(csv_path)

    if builder is not None and builder.is_fitted_:
        try:
            import numpy as np
            from scipy.sparse import save_npz

            X_train_sp = builder.transform(train_records).to_scipy_csr()
            X_val_sp = builder.transform(val_records).to_scipy_csr()
            X_test_sp = builder.transform(test_records).to_scipy_csr()

            def _get_labels(data):
                if hasattr(data, "to_dict"):
                    return np.array([int(r.get('label') == 'spam') for r in data.to_dict('records')], dtype=np.int32)
                return np.array([int(r.get('label') == 'spam') for r in data], dtype=np.int32)

            y_train = _get_labels(train_records)
            y_val = _get_labels(val_records)
            y_test = _get_labels(test_records)

            save_npz(target_dir / 'X_train.npz', X_train_sp)
            save_npz(target_dir / 'X_val.npz', X_val_sp)
            save_npz(target_dir / 'X_test.npz', X_test_sp)

            np.save(target_dir / 'y_train.npy', y_train)
            np.save(target_dir / 'y_val.npy', y_val)
            np.save(target_dir / 'y_test.npy', y_test)

            exported_files['matrices_saved'] = True
        except ImportError:
            exported_files['matrices_saved'] = False

    return exported_files

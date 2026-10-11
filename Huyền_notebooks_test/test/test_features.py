import math

import pytest

from src.config import SpamExperimentConfig
from src.data.preprocessing import SpamTextProcessor
from src.features.feature_engineering import (
    HybridFeatureBuilderScratch,
    MaxAbsScalerScratch,
    SpamSignalFeatureExtractor,
    TfidfVectorizerScratch,
)


def _row_l2_norm(matrix, row_index):
    return math.sqrt(sum(value * value for _, value in matrix.iter_row(row_index)))


def _toy_messages():
    return [
        {"label": "ham", "text": "hello friend how are you"},
        {"label": "spam", "text": "claim your free prize now"},
        {"label": "ham", "text": "are you coming home today"},
        {"label": "spam", "text": "free cash prize click now"},
    ]


def test_tfidf_vectorizer_scratch():
    """Kiểm tra shape, chuẩn hóa L2, giá trị hữu hạn và xử lý tài liệu rỗng."""
    texts = [
        "hello hello world",
        "hello spam",
        "free spam",
    ]
    vectorizer = TfidfVectorizerScratch(min_df=1, norm="l2")
    matrix = vectorizer.fit_transform(texts)

    assert matrix.shape[0] == len(texts)
    assert matrix.shape[1] == len(vectorizer.vocabulary_)
    assert matrix.shape[1] > 0

    # Mỗi hàng có đặc trưng khác 0 phải có L2 norm xấp xỉ 1.
    for row_index in range(matrix.shape[0]):
        norm = _row_l2_norm(matrix, row_index)
        if norm > 0:
            assert norm == pytest.approx(1.0, abs=1e-9)

    assert all(math.isfinite(value) for value in matrix.data)
    assert all(value >= 0 for value in matrix.data)


def test_tfidf_transform_does_not_leak_validation_vocabulary():
    """Từ chỉ xuất hiện ở Validation không được thêm vào vocabulary khi transform."""
    train_texts = ["hello friend", "free prize", "hello again"]
    vectorizer = TfidfVectorizerScratch(min_df=1)
    vectorizer.fit(train_texts)

    original_vocabulary = dict(vectorizer.vocabulary_)
    validation_matrix = vectorizer.transform(["uniquevalidationtoken"])

    assert validation_matrix.shape == (1, len(original_vocabulary))
    assert vectorizer.vocabulary_ == original_vocabulary
    assert "uniquevalidationtoken" not in vectorizer.vocabulary_
    # Tài liệu chỉ chứa từ chưa thấy sẽ tạo hàng thưa rỗng, không lỗi.
    assert validation_matrix.nnz == 0


def test_max_abs_scaler_scratch():
    """Kiểm tra MaxAbsScaler trên Train nằm trong [-1, 1], kể cả cột toàn số 0."""
    train = [
        [-4.0, 0.0, 2.0],
        [2.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
    ]
    scaler = MaxAbsScalerScratch()
    scaled = scaler.fit_transform(train)

    assert len(scaled) == len(train)
    assert all(len(row) == 3 for row in scaled)
    assert all(math.isfinite(value) for row in scaled for value in row)
    assert all(-1.0 <= value <= 1.0 for row in scaled for value in row)

    # Cực đại tuyệt đối của mỗi cột không bằng 0 thì phải được scale về 1.
    assert max(abs(row[0]) for row in scaled) == pytest.approx(1.0)
    assert all(row[1] == 0.0 for row in scaled)
    assert max(abs(row[2]) for row in scaled) == pytest.approx(1.0)

    # Scaler không cắt dữ liệu mới: giá trị vượt phạm vi Train có thể > 1.
    transformed_new = scaler.transform([[8.0, 0.0, -6.0]])
    assert transformed_new[0][0] == pytest.approx(2.0)
    assert transformed_new[0][2] == pytest.approx(-3.0)


def test_max_abs_scaler_rejects_invalid_input():
    scaler = MaxAbsScalerScratch()
    with pytest.raises(ValueError):
        scaler.fit([])
    with pytest.raises(ValueError):
        scaler.fit([[1.0, 2.0], [3.0]])
    with pytest.raises(ValueError):
        scaler.fit([[1.0, float("nan")]])


def test_hybrid_feature_builder():
    """Kiểm tra ghép đặc trưng lai, giữ thứ tự cột và chỉ fit trên Train."""
    config = SpamExperimentConfig()
    config.min_df = 1
    config.max_unigram_features = 100
    config.max_bigram_features = 100
    config.max_char_features = 100
    config.char_ngram_range = (3, 4)
    config.default_tuning_params.update({
        "min_df": 1,
        "max_unigram_features": 100,
        "use_bigram": True,
        "max_bigram_features": 100,
        "use_char_ngram": True,
        "max_char_features": 100,
        "use_keyword_features": True,
        "use_numeric_features": True,
        "use_length_cap": False,
        "text_feature_mode": "log_count",
    })

    processor = SpamTextProcessor(config)
    extractor = SpamSignalFeatureExtractor(config)
    builder = HybridFeatureBuilderScratch(config, processor, extractor)

    train = _toy_messages()
    validation = [
        {"text": "uniquevalidationtoken free prize!!! https://example.com"},
        {"text": "hello friend 123"},
    ]

    train_matrix = builder.fit_transform(train)
    validation_matrix = builder.transform(validation)

    assert builder.is_fitted_
    assert train_matrix.shape[0] == len(train)
    assert validation_matrix.shape[0] == len(validation)
    assert train_matrix.shape[1] > 0
    assert validation_matrix.shape[1] == train_matrix.shape[1]
    assert len(builder.feature_names_) == train_matrix.shape[1]

    # Các khối đặc trưng phải có khoảng cột hợp lệ và không chồng lấn.
    slices = builder.feature_slices_
    assert "word" in slices
    assert "keyword" in slices
    assert "numeric" in slices
    ordered_slices = sorted(slices.values())
    for start, stop in ordered_slices:
        assert 0 <= start <= stop <= train_matrix.shape[1]
    for (_, previous_stop), (next_start, _) in zip(ordered_slices, ordered_slices[1:]):
        assert previous_stop == next_start

    # Ma trận sparse không được chứa NaN/vô cực; đặc trưng được thiết kế không âm.
    assert all(math.isfinite(value) for value in train_matrix.data)
    assert all(math.isfinite(value) for value in validation_matrix.data)
    assert all(value >= 0 for value in train_matrix.data)
    assert all(value >= 0 for value in validation_matrix.data)

    # Transform Validation không được thay đổi vocabulary đã học từ Train.
    vocabulary_before = dict(builder.word_vectorizer.vocabulary_)
    builder.transform([{"text": "anotheruniquevalidationtoken"}])
    assert builder.word_vectorizer.vocabulary_ == vocabulary_before


def test_hybrid_builder_requires_fit_before_transform():
    config = SpamExperimentConfig()
    processor = SpamTextProcessor(config)
    extractor = SpamSignalFeatureExtractor(config)
    builder = HybridFeatureBuilderScratch(config, processor, extractor)

    with pytest.raises(ValueError):
        builder.transform([{"text": "hello world"}])

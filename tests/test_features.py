"""
test_features.py - Unit test logic cho module features
Kiểm thử toàn diện không dùng scikit-learn, tương thích unittest và pytest.
"""
import unittest
import math
import sys
import tempfile
from pathlib import Path

# Đảm bảo import được src khi chạy độc lập
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SpamExperimentConfig
from src.data.preprocessing import SpamTextProcessor
from src.features.feature_engineering import (
    CSRMatrixScratch,
    TfidfVectorizerScratch,
    MaxAbsScalerScratch,
    SpamSignalFeatureExtractor,
    HybridFeatureBuilderScratch,
    experiment_feature_combinations,
    export_processed_data,
)


def test_tfidf_vectorizer_scratch():
    """
    LOGIC KIỂM THỬ VECTOR HÓA TF-IDF TỰ CÀI ĐẶT:
    - Kiểm tra fit tạo từ điển vocabulary chính xác từ danh sách văn bản mẫu.
    - Kiểm tra ma trận trả về từ transform là CSRMatrixScratch và có thể chuyển sang scipy CSR.
    - Đảm bảo mỗi dòng văn bản được chuẩn hóa L2 norm (tổng bình phương xấp xỉ 1.0).
    """
    docs = [
        "free cash win prize",
        "hello friend can we meet",
        "win free prize today",
        "hello lunch meeting today",
    ]

    vectorizer = TfidfVectorizerScratch(min_df=1, sublinear_tf=True, smooth_idf=True, norm="l2")
    vectorizer.fit(docs)

    assert len(vectorizer.vocabulary_) > 0, "Từ điển không được rỗng"
    assert "free" in vectorizer.vocabulary_
    assert "prize" in vectorizer.vocabulary_

    csr = vectorizer.transform(docs)
    assert isinstance(csr, CSRMatrixScratch)
    assert csr.shape == (len(docs), len(vectorizer.vocabulary_))

    # Kiểm tra chuẩn hóa L2 norm: tổng bình phương mỗi hàng != 0 xấp xỉ 1.0
    for row_idx in range(csr.shape[0]):
        row_values = [val for _, val in csr.iter_row(row_idx)]
        if row_values:
            norm_sq = sum(v * v for v in row_values)
            assert math.isclose(norm_sq, 1.0, rel_tol=1e-5), f"Hàng {row_idx} L2 norm khác 1.0: {norm_sq}"

    # Kiểm tra chuyển đổi sang scipy CSR (nếu môi trường có thư viện scipy)
    try:
        sp_csr = csr.to_scipy_csr()
        assert sp_csr.shape == csr.shape
        assert sp_csr.nnz == csr.nnz
    except ImportError:
        pass


def test_max_abs_scaler_scratch():
    """
    LOGIC KIỂM THỬ BỘ SCALE ĐỘ LỚN CỰC ĐẠI:
    - Kiểm tra giá trị cực đại sau scale nằm chính xác trong khoảng [-1.0, 1.0].
    - Đảm bảo các phần tử bằng 0 vẫn giữ nguyên bằng 0 (bảo toàn hoàn toàn tính thưa).
    """
    X = [
        [0.0, 10.0, -5.0],
        [0.0, 5.0, 2.5],
        [0.0, 0.0, -10.0],
    ]

    scaler = MaxAbsScalerScratch().fit(X)
    assert scaler.max_abs_ == [1.0, 10.0, 10.0], f"Kỳ vọng max_abs [1.0, 10.0, 10.0], nhận {scaler.max_abs_}"

    X_scaled = scaler.transform(X)

    # 1. Cột 0 toàn 0 phải giữ nguyên 0
    for row in X_scaled:
        assert row[0] == 0.0, "Cột 0 phải bảo toàn giá trị 0"

    # 2. Toàn bộ giá trị phải nằm trong [-1.0, 1.0]
    for row in X_scaled:
        for val in row:
            assert -1.0 <= val <= 1.0, f"Giá trị vượt khoảng [-1, 1]: {val}"

    # 3. Kiểm tra các giá trị cực đại được chuẩn hóa
    assert X_scaled[0][1] == 1.0  # 10 / 10
    assert X_scaled[2][2] == -1.0  # -10 / 10

    # 4. Kiểm tra với CSRMatrixScratch
    csr_in = CSRMatrixScratch.from_rows([{1: 10.0, 2: -5.0}, {1: 5.0, 2: 2.5}], 3)
    csr_scaler = MaxAbsScalerScratch().fit(csr_in)
    csr_out = csr_scaler.transform(csr_in)
    assert isinstance(csr_out, CSRMatrixScratch)
    for col, val in csr_out.iter_row(0):
        assert -1.0 <= val <= 1.0


def test_hybrid_feature_builder():
    """
    LOGIC KIỂM THỬ GHÉP NỐI MA TRẬN ĐẶC TRƯNG LAI:
    - Kiểm tra ma trận kết hợp chứa đủ số cột = N_words + N_chars + N_keywords + N_numerics.
    - Đảm bảo số lượng dòng đúng bằng số lượng mẫu ban đầu.
    - Đảm bảo experiment_feature_combinations trả đủ 5 tổ hợp đặc trưng.
    """
    train_records = [
        {"source_id": 1, "label": "ham", "text": "Can we meet at the office today?", "raw_text": "Can we meet at the office today?"},
        {"source_id": 2, "label": "spam", "text": "FREE entry to win cash $500! Call now!", "raw_text": "FREE entry to win cash $500! Call now!"},
        {"source_id": 3, "label": "ham", "text": "Attached is the project report.", "raw_text": "Attached is the project report."},
        {"source_id": 4, "label": "spam", "text": "Claim your prize money immediately!", "raw_text": "Claim your prize money immediately!"},
    ]
    val_records = [
        {"source_id": 5, "label": "spam", "text": "Win free cash prizes!", "raw_text": "Win free cash prizes!"},
        {"source_id": 6, "label": "ham", "text": "See you at the meeting.", "raw_text": "See you at the meeting."},
    ]

    config = SpamExperimentConfig(min_df=1)
    processor = SpamTextProcessor(config)
    extractor = SpamSignalFeatureExtractor(config)
    builder = HybridFeatureBuilderScratch(config, processor, extractor)

    X_train = builder.fit_transform(train_records)
    X_val = builder.transform(val_records)

    # Kiểm tra số lượng dòng
    assert X_train.shape[0] == len(train_records)
    assert X_val.shape[0] == len(val_records)

    # Kiểm tra số lượng cột đồng nhất
    assert X_train.shape[1] == X_val.shape[1] == len(builder.feature_names_)

    # Kiểm tra các khối slice đặc trưng
    assert "word" in builder.feature_slices_
    assert "keyword" in builder.feature_slices_
    assert "numeric" in builder.feature_slices_

    # Kiểm tra tính tương thích SciPy CSR (nếu môi trường có scipy)
    try:
        sp_train = X_train.to_scipy_csr()
        assert sp_train.shape == X_train.shape
    except ImportError:
        pass

    # Kiểm tra 5 tổ hợp đặc trưng
    combos = experiment_feature_combinations(train_records, val_records)
    assert set(combos.keys()) == {"word_tfidf", "word_char_frequency", "word_char_ngrams", "word_keywords", "hybrid"}
    for name, (xt, xv) in combos.items():
        assert xt.shape[0] == len(train_records)
        assert xv.shape[0] == len(val_records)
        assert xt.shape[1] == xv.shape[1]

    # Kiểm tra export_processed_data vào thư mục tạm
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = export_processed_data(train_records, val_records, val_records, builder, output_dir=tmp_dir)
        assert "train_csv" in res
        assert "val_csv" in res
        assert "test_csv" in res
        assert Path(res["train_csv"]).is_file()


class TestFeaturesModule(unittest.TestCase):
    def test_tfidf_case(self):
        test_tfidf_vectorizer_scratch()

    def test_max_abs_scaler_case(self):
        test_max_abs_scaler_scratch()

    def test_hybrid_builder_case(self):
        test_hybrid_feature_builder()


if __name__ == "__main__":
    unittest.main()

import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.features.feature_engineering import CSRMatrixScratch
from src.models.classifiers import (
    LinearSVMFromScratch,
    LogisticRegressionFromScratch,
    NaiveBayesClassifier,
    get_baseline_models,
)
from src.models.ensemble import SpamVotingEnsemble
from src.models.tuning import ThresholdOptimizerScratch, tune_model_hyperparameters


def test_logistic_regression_fit_predict():
    """Kiểm thử thuật toán Logistic Regression From-Scratch."""
    X_dense = np.array([
        [2.0, 1.5, 0.2],
        [0.1, 0.2, 0.1],
        [2.5, 2.0, 0.8],
        [0.2, 0.1, 0.0],
        [1.8, 2.2, 0.9],
        [0.0, 0.3, 0.1],
    ], dtype=float)
    y_toy = np.array([1, 0, 1, 0, 1, 0], dtype=float)

    model = LogisticRegressionFromScratch(C=1.0, learning_rate=0.2, max_iter=200, class_weight="balanced")
    model.fit(X_dense, y_toy)

    assert model.weights_ is not None
    assert len(model.weights_) == 3

    proba = model.predict_proba(X_dense)
    assert proba.shape == (6, 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)

    preds = model.predict(X_dense, threshold=0.5)
    assert len(preds) == 6
    assert np.array_equal(preds, y_toy)

    # Kiểm tra với cấu trúc ma trận thưa CSRMatrixScratch
    X_csr = CSRMatrixScratch.from_rows(
        [{0: row[0], 1: row[1], 2: row[2]} for row in X_dense],
        n_columns=3,
    )
    model_csr = LogisticRegressionFromScratch(C=1.0, learning_rate=0.2, max_iter=200)
    model_csr.fit(X_csr, y_toy)
    preds_csr = model_csr.predict(X_csr)
    assert np.array_equal(preds_csr, y_toy)


def test_linear_svm_from_scratch():
    """Kiểm thử thuật toán Linear SVM From-Scratch."""
    X_dense = np.array([
        [3.0, 3.0],
        [4.0, 3.0],
        [3.5, 4.0],
        [-2.0, -2.0],
        [-3.0, -1.0],
        [-2.5, -3.0],
    ], dtype=float)
    y_toy = np.array([1, 1, 1, 0, 0, 0], dtype=float)

    svm = LinearSVMFromScratch(C=1.0, learning_rate=0.1, max_iter=150, class_weight="balanced")
    svm.fit(X_dense, y_toy)

    assert svm.weights_ is not None
    assert len(svm.weights_) == 2

    scores = svm.decision_function(X_dense)
    assert len(scores) == 6

    # Điểm số cho lớp 1 phải lớn hơn lớp 0
    assert np.mean(scores[:3]) > np.mean(scores[3:])

    preds = svm.predict(X_dense, threshold=0.0)
    assert np.array_equal(preds, y_toy)

    # Kiểm tra với ma trận thưa CSRMatrixScratch
    X_csr = CSRMatrixScratch.from_rows(
        [{0: row[0], 1: row[1]} for row in X_dense],
        n_columns=2,
    )
    svm_csr = LinearSVMFromScratch(C=1.0, learning_rate=0.1, max_iter=150)
    svm_csr.fit(X_csr, y_toy)
    assert np.array_equal(svm_csr.predict(X_csr, threshold=0.0), y_toy)


def test_naive_bayes_fit_predict():
    """Kiểm thử thuật toán Naive Bayes (Multinomial & Complement)."""
    X_dense = np.array([
        [4.0, 3.0, 0.0, 0.0],
        [3.0, 4.0, 0.5, 0.0],
        [0.0, 0.0, 3.0, 4.0],
        [0.0, 0.5, 4.0, 3.0],
    ], dtype=float)
    y_toy = np.array([1, 1, 0, 0], dtype=int)

    # 1. Test Complement Naive Bayes
    nb_comp = NaiveBayesClassifier(alpha=1.0, kind="complement")
    nb_comp.fit(X_dense, y_toy)
    assert nb_comp.feature_log_prob_.shape == (2, 4)
    probs_comp = nb_comp.predict_proba(X_dense)
    assert probs_comp.shape == (4, 2)
    assert np.allclose(probs_comp.sum(axis=1), 1.0)
    preds_comp = nb_comp.predict(X_dense)
    assert np.array_equal(preds_comp, y_toy)

    # 2. Test Multinomial Naive Bayes
    nb_multi = NaiveBayesClassifier(alpha=1.0, kind="multinomial")
    nb_multi.fit(X_dense, y_toy)
    assert nb_multi.feature_log_prob_.shape == (2, 4)
    probs_multi = nb_multi.predict_proba(X_dense)
    assert np.allclose(probs_multi.sum(axis=1), 1.0)
    preds_multi = nb_multi.predict(X_dense)
    assert np.array_equal(preds_multi, y_toy)


def test_spam_voting_ensemble():
    """Kiểm thử mô hình kết hợp Voting Ensemble."""
    X_dense = np.array([
        [3.0, 2.0, 0.1],
        [0.1, 0.2, 0.0],
        [2.5, 3.0, 0.2],
        [0.0, 0.1, 0.0],
    ], dtype=float)
    y_toy = np.array([1, 0, 1, 0], dtype=float)

    lr = LogisticRegressionFromScratch(C=1.0, learning_rate=0.1, max_iter=100)
    svm = LinearSVMFromScratch(C=1.0, learning_rate=0.1, max_iter=100)
    nb = NaiveBayesClassifier(alpha=1.0, kind="complement")

    ensemble_soft = SpamVotingEnsemble(
        models={"lr": lr, "svm": svm, "nb": nb},
        weights={"lr": 1.0, "svm": 1.0, "nb": 1.0},
        voting="soft",
    )
    ensemble_soft.fit(X_dense, y_toy)
    probs = ensemble_soft.predict_proba(X_dense)
    assert probs.shape == (4, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    preds_soft = ensemble_soft.predict(X_dense)
    assert np.array_equal(preds_soft, y_toy)

    ensemble_hard = SpamVotingEnsemble(
        models={"lr": lr, "svm": svm, "nb": nb},
        voting="hard",
    )
    ensemble_hard.fit(X_dense, y_toy)
    preds_hard = ensemble_hard.predict(X_dense)
    assert len(preds_hard) == 4
    assert np.array_equal(preds_hard, y_toy)


def test_threshold_optimizer_scratch():
    """Kiểm thử giải thuật tối ưu hóa ngưỡng quyết định."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    spam_probs = np.array([0.95, 0.88, 0.75, 0.65, 0.45, 0.35, 0.25, 0.15, 0.05, 0.01])

    optimizer = ThresholdOptimizerScratch(recall_target=0.85, beta=2.0, n_points=50)
    scan_df = optimizer.scan_thresholds(y_true, spam_probs)

    assert len(scan_df) == 50
    assert "target_met" in scan_df.columns
    assert "f_beta" in scan_df.columns

    best = optimizer.select_best_threshold(scan_df)
    assert best["recall"] >= 0.85
    assert best["f1"] > 0.8


class TestModelsModule(unittest.TestCase):
    def test_lr_case(self):
        test_logistic_regression_fit_predict()

    def test_svm_case(self):
        test_linear_svm_from_scratch()

    def test_nb_case(self):
        test_naive_bayes_fit_predict()

    def test_ensemble_case(self):
        test_spam_voting_ensemble()

    def test_threshold_optimizer_case(self):
        test_threshold_optimizer_scratch()


if __name__ == "__main__":
    unittest.main()


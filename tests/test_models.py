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
    def test_threshold_invalid_inputs(self):
        optimizer = ThresholdOptimizerScratch()
        for labels, scores in [([], []), ([0, 1], [.8]), ([0, .5], [.2, .8]),
                               ([0, 1], [.2, np.nan]), ([0, 1], [.2, 1.1])]:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                optimizer.metrics_table(labels, scores)
        for grid in ([], [np.nan], [-.1], [1.1], [[.5]]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                optimizer.metrics_table([0, 1], [.2, .8], thresholds=grid)
            with self.subTest(default_grid=grid), self.assertRaises(ValueError):
                ThresholdOptimizerScratch(default_grid=grid)
        for kwargs in ({'recall_target': np.nan}, {'recall_target': 1.1},
                       {'beta': 0}, {'beta': np.inf}, {'n_points': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ThresholdOptimizerScratch(**kwargs)

    def test_tuning_ranks_ties_and_fallback(self):
        from src.models.tuning import tune_model_hyperparameters

        class Candidate:
            def __init__(self, predictions):
                self.predictions = predictions

            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.array(self.predictions)

        cases = [
            ([0, 0, 1, 1], [[0, 0, 0, 1], [1, 0, 1, 1]], .5),
            # Recall must outrank accuracy even when its improvement is tiny.
            ([1] * 1000 + [0] * 1000,
             [[1] * 500 + [0] * 1500, [1] * 501 + [0] * 499 + [1] * 1000], .85),
        ]
        for labels, candidates, target in cases:
            with self.subTest(target=target):
                _, params, _ = tune_model_hyperparameters(
                    'toy', Candidate, {'predictions': candidates}, None, labels,
                    None, labels, recall_target=target,
                )
                self.assertTrue(params['predictions'] is candidates[1])

    def test_sparse_linear_models_learn_small_tfidf_signals(self):
        from scipy.sparse import csr_matrix

        X = csr_matrix(np.eye(20)[np.arange(200) % 20] * .1)
        y = (np.arange(200) % 20 >= 10).astype(int)
        for model_class in (LogisticRegressionFromScratch, LinearSVMFromScratch):
            with self.subTest(model=model_class.__name__):
                model = model_class(C=1000).fit(X, y)
                probabilities = model.predict_proba(X)[:, 1]
                self.assertGreater(probabilities[y == 1].mean(), .7)
                self.assertLess(probabilities[y == 0].mean(), .3)
                self.assertTrue(np.isfinite(model.loss_history_).all())
                self.assertLess(model.loss_history_[-1], model.loss_history_[0])

    def test_hard_voting_scores_match_decisions(self):
        class FixedModel:
            def __init__(self, score):
                self.score = score

            def predict_proba(self, X):
                return np.array([[1 - self.score, self.score]])

            def predict(self, X):
                return np.array([int(self.score >= .5)])

        ensemble = SpamVotingEnsemble(
            {"a": FixedModel(.51), "b": FixedModel(.51), "c": FixedModel(.01)}, voting="hard",
        )
        self.assertAlmostEqual(ensemble.predict_proba(None)[0, 1], 2 / 3)
        self.assertEqual(ensemble.predict(None)[0], 1)

    def test_hyperparameters_evaluated_with_tuned_threshold(self):
        from src.models.tuning import tune_model_hyperparameters

        class Candidate:
            def __init__(self, variant):
                self.variant = variant

            def fit(self, X, y):
                return self

            def predict_proba(self, X):
                scores = np.array([.5, .5, .5, .5] if self.variant == 0 else [.1, .2, .3, .4])
                return np.column_stack([1 - scores, scores])

            def predict(self, X):
                return (self.predict_proba(X)[:, 1] >= .5).astype(int)

        model, params, score = tune_model_hyperparameters(
            "toy", Candidate, {"variant": [0, 1]}, None, [0, 0, 1, 1], None, [0, 0, 1, 1],
            threshold_optimizer=ThresholdOptimizerScratch(),
        )
        self.assertEqual(params, {"variant": 1})
        self.assertEqual(score, 1.0)

    def test_validation_selects_single_model_over_ensemble(self):
        from src.models.tuning import ModelSelectionWorkflow

        class FixedModel:
            def __init__(self, scores):
                self.scores = np.asarray(scores)

            def predict_proba(self, X):
                return np.column_stack([1 - self.scores, self.scores])

        models = {"ensemble": FixedModel([.5, .5, .5, .5]), "single": FixedModel([.1, .2, .8, .9])}
        name, info, comparison, _ = ModelSelectionWorkflow().select_candidates(models, None, [0, 0, 1, 1])
        self.assertEqual(name, "single")
        self.assertTrue(info["target_met"])
        self.assertEqual(comparison.iloc[0]["validation_accuracy"], 1.0)

    def test_validation_fallback_and_tie(self):
        from src.models.tuning import ModelSelectionWorkflow

        class FixedModel:
            def __init__(self, scores):
                self.scores = np.asarray(scores)

            def predict_proba(self, X):
                return np.column_stack([1 - self.scores, self.scores])

        optimizer = ThresholdOptimizerScratch(default_grid=np.array([.5]))
        models = {"low_recall": FixedModel([.1, .1, .1, .9]), "higher_recall": FixedModel([.9, .1, .9, .9])}
        name, info, _, _ = ModelSelectionWorkflow().select_candidates(models, None, [0, 1, 1, 1], optimizer)
        self.assertEqual(name, "higher_recall")
        self.assertFalse(info["target_met"])
        models["same"] = models["higher_recall"]
        self.assertEqual(ModelSelectionWorkflow().select_candidates(models, None, [0, 1, 1, 1], optimizer)[0], name)

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


"""Unit tests for custom spam-classification models and threshold optimization."""
import numpy as np
from scipy.sparse import csr_matrix

from src.models.classifiers import LogisticRegressionFromScratch, LinearSVMFromScratch, NaiveBayesClassifier
from src.models.ensemble import SpamVotingEnsemble
from src.models.tuning import ThresholdOptimizerScratch


def _toy_data():
    X = csr_matrix(np.array([
        [1, 0, 0], [0.8, 0.1, 0], [0.9, 0, 0.1],
        [0, 1, 1], [0, 0.8, 0.9], [0.1, 0.9, 0.8],
    ], dtype=float))
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def test_logistic_regression_fit_predict():
    X, y = _toy_data()
    model = LogisticRegressionFromScratch(max_iter=100, learning_rate=0.2).fit(X, y)
    proba = model.predict_proba(X)
    pred = model.predict(X)
    assert proba.shape == (len(y), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(pred)).issubset({0, 1})
    assert np.mean(pred == y) >= 0.8


def test_linear_svm_from_scratch():
    X, y = _toy_data()
    model = LinearSVMFromScratch(max_iter=40, learning_rate=0.1).fit(X, y)
    assert model.weights_.shape == (X.shape[1],)
    assert model.decision_function(X).shape == (len(y),)
    assert set(np.unique(model.predict(X))).issubset({0, 1})
    assert model.predict_proba(X).shape == (len(y), 2)


def test_naive_bayes_fit_predict():
    X, y = _toy_data()
    for kind in ("multinomial", "complement"):
        model = NaiveBayesClassifier(alpha=1.0, kind=kind).fit(X, y)
        proba = model.predict_proba(X)
        assert model.feature_log_prob_.shape == (2, X.shape[1])
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert set(np.unique(model.predict(X))).issubset({0, 1})


def test_spam_voting_ensemble():
    X, y = _toy_data()
    models = {
        "lr": LogisticRegressionFromScratch(max_iter=40),
        "svm": LinearSVMFromScratch(max_iter=20),
        "nb": NaiveBayesClassifier(kind="multinomial"),
    }
    ensemble = SpamVotingEnsemble(models, voting="soft").fit(X, y)
    probs = ensemble.predict_proba(X)
    assert probs.shape == (len(y), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert set(np.unique(ensemble.predict(X))).issubset({0, 1})
    hard = SpamVotingEnsemble(models, voting="hard")
    assert set(np.unique(hard.predict(X))).issubset({0, 1})


def test_threshold_optimizer_scratch():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probs = np.array([0.02, 0.1, 0.3, 0.4, 0.55, 0.7, 0.8, 0.95])
    optimizer = ThresholdOptimizerScratch(recall_target=0.85, beta=2.0)
    scan = optimizer.scan_thresholds(y, probs)
    best = optimizer.select_best_threshold(scan)
    assert len(scan) == 200
    assert 0.005 <= best["best_threshold"] <= 1.0
    assert best["recall"] >= 0.85 or best["target_met"] is False
    valid = scan[scan["target_met"]]
    if not valid.empty:
        assert np.isclose(best["f_beta"], valid.sort_values(["f_beta", "f1", "accuracy", "precision"], ascending=False).iloc[0]["f_beta"])

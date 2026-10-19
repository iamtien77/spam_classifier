"""Kiểm thử hồi quy và đối chiếu nghiệm số chuẩn (numerical oracle) qua giao diện public của các bộ phân loại."""
import unittest

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit

from src.features.feature_engineering import CSRMatrixScratch
from src.models.classifiers import (
    LogisticRegressionFromScratch, LinearSVMFromScratch, NaiveBayesClassifier,
)


class TestClassifierContracts(unittest.TestCase):
    model_classes = (LogisticRegressionFromScratch, LinearSVMFromScratch, NaiveBayesClassifier)

    def test_invalid_training_data_rejected(self):
        cases = [
            (np.ones((3, 2)), [1]),  # Kiểm tra lỗi phát tán mảng ngầm định (silent broadcasting bug) trước đây.
            (np.ones((2, 2)), [[0], [1]]),
            (np.ones((2, 2)), [0, .5]),
            (np.ones((2, 2)), [0, np.nan]),
            (np.ones((2, 2)), [1, 1]),
            (np.empty((0, 2)), []),
            (np.empty((2, 0)), [0, 1]),
            (np.ones(2), [0, 1]),
            (np.array([[np.inf], [1]]), [0, 1]),
            (csr_matrix([[0], [np.nan]]), [0, 1]),
        ]
        for cls in self.model_classes:
            for X, y in cases:
                with self.subTest(model=cls.__name__, shape=X.shape, y=y):
                    with self.assertRaises(ValueError):
                        cls().fit(X, y)

    def test_invalid_parameters_rejected(self):
        for cls in (LogisticRegressionFromScratch, LinearSVMFromScratch):
            for params in ({"C": 0}, {"C": np.nan}, {"learning_rate": np.inf},
                           {"max_iter": 1.5}, {"max_iter": 0}, {"class_weight": "typo"}):
                with self.subTest(model=cls.__name__, params=params), self.assertRaises(ValueError):
                    cls(**params).fit([[0], [1]], [0, 1])
        for params in ({"alpha": 0}, {"alpha": np.nan}, {"kind": "typo"}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                NaiveBayesClassifier(**params).fit([[0], [1]], [0, 1])

    def test_prediction_contract_and_sparse_parity(self):
        dense = np.array([[0., 2.], [1., 3.], [4., 0.], [3., 1.]])
        scratch = CSRMatrixScratch.from_rows([dict(enumerate(row)) for row in dense], 2)
        for cls in self.model_classes:
            with self.subTest(model=cls.__name__):
                with self.assertRaises(RuntimeError):
                    cls().predict_proba(dense)
                baseline = cls().fit(dense, [0, 0, 1, 1])
                expected = baseline.predict_proba(dense)
                for matrix in (csr_matrix(dense), scratch):
                    model = cls().fit(matrix, [0, 0, 1, 1])
                    np.testing.assert_allclose(model.predict_proba(matrix), expected, atol=1e-6)
                with self.assertRaises(ValueError):
                    baseline.predict_proba(np.ones((1, 3)))
                with self.assertRaises(ValueError):
                    baseline.predict(dense, threshold=np.nan)

    def test_nb_nonnegative_and_hand_worked_likelihoods(self):
        # Tần suất lớp 0 là [2, 0] -> làm mượt thành [3/4, 1/4]; lớp 1 đảo ngược lại.
        for kind in ("multinomial", "complement"):
            model = NaiveBayesClassifier(alpha=1, kind=kind).fit([[2, 0], [0, 2]], [0, 1])
            np.testing.assert_allclose(model.predict_proba([[1, 0], [0, 1]]),
                                       [[.75, .25], [.25, .75]], atol=1e-12)
            with self.assertRaises(ValueError):
                model.predict_proba([[-1, 1]])
        with self.assertRaises(ValueError):
            NaiveBayesClassifier().fit([[-1, 0], [0, 1]], [0, 1])

    def test_linear_models_against_finite_difference_optimizer(self):
        # Đạo hàm số học độc lập qua BFGS, không dùng gradient trong mã nguồn production.
        X = np.array([[-1., .2], [-.4, 1.], [.5, -.3], [1.2, .8], [.2, .1]])
        y = np.array([0, 0, 1, 1, 0])
        for cls in (LogisticRegressionFromScratch, LinearSVMFromScratch):
            with self.subTest(model=cls.__name__):
                def loss(theta):
                    margins = X @ theta[:-1] + theta[-1]
                    regularization = np.dot(theta[:-1], theta[:-1]) / (2 * len(y))
                    if cls is LogisticRegressionFromScratch:
                        return regularization + np.mean(np.logaddexp(0, margins) - y * margins)
                    return regularization + np.mean(np.maximum(0, 1 - (2 * y - 1) * margins) ** 2)

                reference = minimize(loss, np.zeros(3), method="BFGS", options={"gtol": 1e-7})
                self.assertLess(np.linalg.norm(reference.jac), 1e-5)
                model = cls(C=1, class_weight=None, max_iter=2000).fit(X, y)
                np.testing.assert_allclose(model.predict_proba(X)[:, 1],
                                           expit(X @ reference.x[:-1] + reference.x[-1]), atol=2e-4)

    def test_hinge_and_squared_hinge_known_optima(self):
        # Với X=[-1, 1], C=1: nghiệm tối ưu của loss hinge là w=1; squared hinge là w=4/5.
        for loss, expected in (("hinge", 1.), ("squared_hinge", .8)):
            model = LinearSVMFromScratch(loss=loss, class_weight=None, max_iter=2000)
            model.fit([[-1], [1]], [0, 1])
            np.testing.assert_allclose(model.decision_function([[-1], [1]]),
                                       [-expected, expected], atol=1e-4)


if __name__ == "__main__":
    unittest.main()

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.evaluation.metrics import SpamModelEvaluator
from src.evaluation.error_analysis import extract_misclassified_samples, profile_error_patterns


def test_confusion_matrix_manual():
    """Kiểm tra đếm ma trận nhầm lẫn thủ công."""
    evaluator = SpamModelEvaluator()
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])

    cm = evaluator.compute_confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)
    assert cm[0, 0] == 3  # TN: y=0, pred=0
    assert cm[0, 1] == 1  # FP: y=0, pred=1
    assert cm[1, 0] == 1  # FN: y=1, pred=0
    assert cm[1, 1] == 3  # TP: y=1, pred=1
    assert int(np.sum(cm)) == len(y_true)


def test_fbeta_score_calculation():
    """Kiểm tra tính điểm F-Beta (Beta=2.0)."""
    evaluator = SpamModelEvaluator()
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0])

    metrics = evaluator.compute_all_metrics(y_true, y_pred, beta=2.0)
    assert "f_beta" in metrics
    assert "f1" in metrics
    assert 0.0 <= metrics["f_beta"] <= 1.0

    # P = 2/2 = 1.0, R = 2/3 = 0.6667
    # Với beta=2.0, Recall được coi trọng hơn Precision -> F2 < F1
    assert metrics["f_beta"] < metrics["f1"]


def test_trapezoidal_roc_auc():
    """Kiểm tra tính ROC-AUC."""
    evaluator = SpamModelEvaluator()
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])

    metrics = evaluator.compute_all_metrics(y_true, y_pred=[1, 1, 0, 0], y_prob=y_prob)
    assert metrics["roc_auc"] == 1.0


def test_error_analysis_extraction():
    """Kiểm tra bóc tách mẫu phân loại sai FP và FN."""
    test_df = pd.DataFrame({
        "text": [
            "Normal work email about meeting",
            "Win free money claim prize now",
            "Hey free tickets call me",
            "Secret urgent notification",
        ]
    })
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])  # Mẫu 2 bị FP, Mẫu 3 bị FN
    y_prob = np.array([0.1, 0.95, 0.8, 0.2])

    fp_df, fn_df = extract_misclassified_samples(test_df, y_true, y_pred, y_prob)
    assert len(fp_df) == 1
    assert len(fn_df) == 1
    assert fp_df.iloc[0]["error_type"] == "False Positive (Chặn nhầm)"
    assert fn_df.iloc[0]["error_type"] == "False Negative (Lọt lưới)"

    profile_df = profile_error_patterns(fp_df, fn_df)
    assert len(profile_df) == 2
    assert "suspected_cause" in profile_df.columns


class TestEvaluationModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')

    def test_roc_geometry_and_exact_threshold_marker(self):
        from src.evaluation.visualization import plot_roc_and_pr_curves, plt

        for labels, scores, threshold in (
            ([0, 1], [.501, .502], .5015),
            ([0, 1, 0, 1], [.1, .5, .5, .9], .5),
            ([0, 1], [0., 1.], 1.),
        ):
            with self.subTest(scores=scores):
                fig, axes = plt.subplots(1, 2)
                try:
                    with patch.object(plt, 'subplots', return_value=(fig, axes)):
                        plot_roc_and_pr_curves(labels, scores, best_threshold=threshold)
                    metrics = SpamModelEvaluator().compute_all_metrics(
                        labels, (np.asarray(scores) >= threshold).astype(int), scores)
                    x, y = axes[0].lines[0].get_data()
                    area = np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2)
                    self.assertAlmostEqual(area, metrics['roc_auc'])
                    self.assertTrue(np.all(np.diff(x) >= 0))
                    self.assertTrue(np.all(np.diff(y) >= 0))
                    np.testing.assert_allclose(axes[0].collections[0].get_offsets()[0],
                                               [metrics['fpr'], metrics['recall']])
                    np.testing.assert_allclose(axes[1].collections[0].get_offsets()[0],
                                               [metrics['recall'], metrics['precision']])
                finally:
                    plt.close(fig)

    def test_length_histogram_includes_long_emails(self):
        from src.evaluation.visualization import plot_text_length_comparison, plt

        frame = pd.DataFrame({'label': ['ham', 'spam'],
                              'text': ['word ' * 151, 'word ' * 3000]})
        fig, axes = plt.subplots(1, 2)
        try:
            with patch.object(plt, 'subplots', return_value=(fig, axes)), patch.object(plt, 'show'):
                plot_text_length_comparison(frame)
            for bars in axes[1].containers:
                heights = np.array([bar.get_height() for bar in bars])
                self.assertTrue(np.all(np.isfinite(heights)))
                self.assertAlmostEqual(sum(bar.get_height() * bar.get_width() for bar in bars), 1.)
            self.assertAlmostEqual(max(bar.get_x() + bar.get_width() for bar in axes[1].patches), 3000)
        finally:
            plt.close(fig)

    def test_url_plots_match_pipeline(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from src.data.preprocessing import SpamTextProcessor
        from src.evaluation.visualization import plot_spam_signals_comparison, plot_feature_correlation_heatmap
        frame = pd.DataFrame({'label': ['ham', 'ham', 'spam', 'spam'],
                              'text': ['contact@example.com', 'plain text',
                                       'https://example.com', 'www.example.org']})
        urls = [SpamTextProcessor().extract_structural_features(t)['has_url'] for t in frame.text]
        try:
            with patch('matplotlib.pyplot.show'):
                plot_spam_signals_comparison(frame)
            axis = plt.gcf().axes[0]
            self.assertEqual([axis.containers[0][3].get_height(), axis.containers[1][3].get_height()],
                             [np.mean(urls[:2]) * 100, np.mean(urls[2:]) * 100])
            with patch('matplotlib.pyplot.show'):
                plot_feature_correlation_heatmap(frame)
            axis = plt.gcf().axes[0]
            labels = [label.get_text() for label in axis.get_xticklabels()]
            correlation = axis.images[0].get_array()
            self.assertAlmostEqual(float(correlation[labels.index('Chứa URL'), labels.index('Nhãn là Spam')]),
                                   float(np.corrcoef(urls, [0, 0, 1, 1])[0, 1]))
        finally:
            plt.close('all')

    def test_plot_auc_matches_metrics_for_ties_and_single_class(self):
        from src.evaluation.visualization import plot_roc_and_pr_curves, plt

        for labels, scores, expected in (
            ([0, 1], [.5, .5], "0.5000"),
            ([1, 0], [.5, .5], "0.5000"),
            ([0, 1, 0, 1], [.1, .5, .5, .9], "0.8750"),
            ([0, 0], [.2, .8], "N/A"),
        ):
            with self.subTest(labels=labels):
                fig, axes = plt.subplots(1, 2)
                try:
                    with patch.object(plt, "subplots", return_value=(fig, axes)):
                        plot_roc_and_pr_curves(labels, scores)
                    self.assertEqual(axes[0].lines[0].get_label(), f"ROC (AUC = {expected})")
                finally:
                    plt.close(fig)

    def test_plot_recall_target_is_vertical(self):
        from src.evaluation.visualization import plot_roc_and_pr_curves, plt

        fig, axes = plt.subplots(1, 2)
        try:
            with patch.object(plt, "subplots", return_value=(fig, axes)):
                plot_roc_and_pr_curves([0, 1], [.2, .8])
            target = next(line for line in axes[1].lines if line.get_label() == "Mục tiêu Recall >= 0.85")
            np.testing.assert_allclose(target.get_xdata(), [.85, .85])
            np.testing.assert_allclose(target.get_ydata(), [0, 1])
        finally:
            plt.close(fig)

    def test_auc_ties_and_permutations(self):
        evaluator = SpamModelEvaluator()
        for labels, scores, expected in (
            ([0, 1], [0.5, 0.5], 0.5),
            ([0, 1, 0, 1], [0.1, 0.5, 0.5, 0.9], 0.875),
        ):
            for order in (np.arange(len(labels)), np.arange(len(labels))[::-1]):
                y = np.asarray(labels)[order]
                p = np.asarray(scores)[order]
                self.assertAlmostEqual(evaluator.compute_all_metrics(y, (p >= .5).astype(int), p)["roc_auc"], expected)

    def test_invalid_evaluation_inputs(self):
        evaluator = SpamModelEvaluator()
        for truth, predicted, scores in (
            ([0, 1], [1], None), ([], [], None),
            ([0, 2], [0, 1], None), ([0, 1], [0, .5], None),
            ([0, 1], [0, 1], [.5]),
            ([0, 1], [0, 1], [0, float("nan")]),
        ):
            with self.subTest(truth=truth, predicted=predicted, scores=scores), self.assertRaises(ValueError):
                evaluator.compute_all_metrics(truth, predicted, scores)

    def test_cm_case(self):
        test_confusion_matrix_manual()

    def test_fbeta_case(self):
        test_fbeta_score_calculation()

    def test_roc_auc_case(self):
        test_trapezoidal_roc_auc()

    def test_error_case(self):
        test_error_analysis_extraction()


if __name__ == "__main__":
    unittest.main()


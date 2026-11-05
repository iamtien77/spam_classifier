"""Kiểm thử hành vi cho các sửa đổi sau review; không ghi các artifact production."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SpamExperimentConfig
from src.deployment.predictor import SpamInferenceService
from src.evaluation.experiments import run_development_experiments
from src.evaluation.error_analysis import extract_misclassified_samples, profile_error_patterns
from src.features.feature_engineering import FittedFeatureTransform
from src.models.classifiers import LinearSVMFromScratch
from src.models.ensemble import SpamVotingEnsemble
from src.models.tuning import ThresholdOptimizerScratch
from src.models.persistence import archive_run_outputs
from tests.test_deployment import _create_toy_pipeline_bundle


class TestReviewRegressions(unittest.TestCase):
    def test_error_csv_clears_previous_errors_and_preserves_schema(self):
        records = pd.DataFrame({"source_id": [1, 2], "text": ["meeting", "free cash"]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics" / "errors.csv"
            fp, fn = extract_misclassified_samples(records, [0, 1], [1, 0], [.9, .1])
            previous = profile_error_patterns(fp, fn, path)
            self.assertEqual(len(pd.read_csv(path)), 2)
            fp, fn = extract_misclassified_samples(records, [0, 1], [0, 1], [.1, .9])
            current = profile_error_patterns(fp, fn, path)
            saved = pd.read_csv(path)
            self.assertTrue(saved.empty)
            self.assertEqual(list(current.columns), list(previous.columns))
            self.assertEqual(list(saved.columns), list(previous.columns))
            new_path = Path(tmp) / "fresh" / "errors.csv"
            profile_error_patterns(fp, fn, new_path)
            self.assertTrue(pd.read_csv(new_path).empty)

    def test_ensemble_rejects_invalid_voting(self):
        for voting in ("sof", "", None, 1):
            with self.subTest(voting=voting), self.assertRaises(ValueError):
                SpamVotingEnsemble({"member": object()}, voting=voting)
        for voting in ("SOFT", "HARD"):
            model = SpamVotingEnsemble({"member": object()}, voting=voting)
            self.assertEqual(model.voting, voting.lower())
            model.voting = "sof"
            for method in (model.predict, model.predict_proba):
                with self.assertRaises(ValueError):
                    method(np.zeros((1, 1)))

    def test_svm_respects_gradient_tolerance(self):
        X = np.array([[-1.], [-2.], [1.], [2.]])
        y = np.array([0, 0, 1, 1])
        # Initial gradient norm is 3: tol=4 must stop before the first update.
        loose = LinearSVMFromScratch(tol=4).fit(X, y)
        strict = LinearSVMFromScratch(tol=1e-7).fit(X, y)
        self.assertEqual(loose.n_iter_, 0)
        np.testing.assert_array_equal(loose.decision_function(X), np.zeros(4))
        self.assertGreater(strict.n_iter_, 0)
        np.testing.assert_array_equal(strict.predict(X), y)

    def test_archive_preserves_previous_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "metrics"
            metrics.mkdir()
            original = metrics / "summary.json"
            original.write_text('{"accuracy": 0.7}', encoding="utf-8")
            archived = archive_run_outputs([metrics], root / "archive")
            original.write_text('{"accuracy": 0.8}', encoding="utf-8")
            self.assertEqual((archived / "metrics" / "summary.json").read_text(), '{"accuracy": 0.7}')
            self.assertTrue((archived / "COMPLETE").is_file())
            self.assertNotEqual(archived, archive_run_outputs([metrics], root / "archive"))

    def test_score_alias_and_threshold_interpretation(self):
        service = SpamInferenceService(pipeline_bundle=_create_toy_pipeline_bundle())
        result = service.predict_single("Win free cash now")
        self.assertEqual(result["spam_score"], result["spam_probability"])
        self.assertFalse(result["is_calibrated"])
        self.assertEqual(result["risk_level"], "Cao" if result["is_spam"] else "Thấp")
        for text in (None, "", "  ", 123):
            with self.assertRaises(ValueError):
                service.predict_single(text)

    def test_fitted_transform_preserves_column_order(self):
        bundle = _create_toy_pipeline_bundle()
        builder = bundle["feature_builder"]
        records = [{"text": "Win free cash"}, {"text": "Meeting report"}]
        full = builder.transform(records).to_scipy_csr()
        transform = FittedFeatureTransform(builder, [2, 0])
        np.testing.assert_allclose(transform.transform(records).toarray(), full[:, [2, 0]].toarray())
        for columns in ([0, 0], [-1], [0.5], [], [[0]]):
            with self.assertRaises(ValueError):
                FittedFeatureTransform(builder, columns).transform(records)

    def test_metrics_known_counts_and_f2_independent_of_beta(self):
        info = ThresholdOptimizerScratch(beta=1).compute_metrics_at_threshold(
            [0, 0, 1, 1, 1], [.1, .9, .8, .7, .2], .5)
        self.assertEqual((info["tn"], info["fp"], info["fn"], info["tp"]), (1, 1, 1, 2))
        self.assertAlmostEqual(info["accuracy"], .6)
        info = ThresholdOptimizerScratch(beta=1).compute_metrics_at_threshold(
            [0, 0, 1, 1], [.9, .8, .7, .6], .5)
        self.assertAlmostEqual(info["f_beta"], 2 / 3)
        self.assertAlmostEqual(info["f2"], 5 / 6)

    def test_development_experiments_exclude_reserved_test(self):
        rows = pd.DataFrame([
            {"source_id": 2 * i + label, "label": "spam" if label else "ham",
             "text": ("free cash prize " if label else "team meeting report ") + str(i)}
            for i in range(20) for label in (0, 1)
        ])
        config = SpamExperimentConfig(min_df=1, max_unigram_features=12,
                                      max_bigram_features=5, max_char_features=5)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "experiment"
            frame = run_development_experiments(rows, output, seeds=(17, 42), config=config)
            self.assertEqual(len(frame), 16)
            manifest = json.loads((output / "manifest.json").read_text())
            reserved = set(manifest["reserved_test_source_ids"])
            self.assertTrue(reserved)
            for seed in (17, 42):
                split = json.loads((output / f"split_{seed}.json").read_text())
                self.assertTrue(reserved.isdisjoint(split["train"] + split["validation"]))
                self.assertTrue(set(split["train"]).isdisjoint(split["validation"]))
            self.assertTrue((output / "development_summary.csv").is_file())
            with self.assertRaises(FileExistsError):
                run_development_experiments(rows, output, seeds=(17,), config=config)

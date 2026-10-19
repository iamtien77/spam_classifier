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
from src.features.feature_engineering import FittedFeatureTransform
from src.models.tuning import ThresholdOptimizerScratch
from src.models.persistence import archive_run_outputs
from tests.test_deployment import _create_toy_pipeline_bundle


class TestReviewRegressions(unittest.TestCase):
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

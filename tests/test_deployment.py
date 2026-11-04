import sys
from pathlib import Path
import unittest
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.config import SpamExperimentConfig
from src.data.preprocessing import SpamTextProcessor
from src.features.feature_engineering import HybridFeatureBuilderScratch, SpamSignalFeatureExtractor
from src.models.classifiers import LogisticRegressionFromScratch
from src.deployment.predictor import SpamInferenceService
from src.models.persistence import save_spam_pipeline


def _create_toy_pipeline_bundle():
    config = SpamExperimentConfig()
    text_processor = SpamTextProcessor(config=config)
    signal_extractor = SpamSignalFeatureExtractor(config=config)
    builder = HybridFeatureBuilderScratch(
        config=config,
        text_processor=text_processor,
        signal_extractor=signal_extractor,
    )

    toy_train = [
        {"text": "Win free cash prize claim now", "label": "spam"},
        {"text": "Urgent lottery winner get your money", "label": "spam"},
        {"text": "Hello let us meet tomorrow for coffee", "label": "ham"},
        {"text": "Please review the project report attached", "label": "ham"},
    ]
    X_train = builder.fit_transform(toy_train)
    y_train = np.array([1, 1, 0, 0])

    model = LogisticRegressionFromScratch(C=1.0, learning_rate=0.2, max_iter=100)
    model.fit(X_train, y_train)

    return {
        "model": model,
        "text_processor": text_processor,
        "feature_builder": builder,
        "best_threshold": 0.45,
    }


def test_inference_service_single_prediction():
    """Kiểm thử phân loại email đơn lẻ."""
    bundle = _create_toy_pipeline_bundle()
    service = SpamInferenceService(pipeline_bundle=bundle).load_service()

    # 1. Email rõ ràng là spam
    res_spam = service.predict_single("Win a free cash prize now! Click here!")
    assert "prediction" in res_spam
    assert "is_spam" in res_spam
    assert "spam_probability" in res_spam
    assert "threshold_used" in res_spam
    assert "risk_level" in res_spam
    assert "detected_signals" in res_spam
    assert 0.0 <= res_spam["spam_probability"] <= 1.0
    assert len(res_spam["detected_signals"]) > 0

    # 2. Email công việc bình thường
    res_ham = service.predict_single("Hi team, please find the meeting notes attached.")
    assert res_ham["prediction"] == "NOT SPAM / HAM"
    assert res_ham["is_spam"] is False


def test_inference_service_batch_prediction():
    """Kiểm thử phân loại email hàng loạt."""
    bundle = _create_toy_pipeline_bundle()
    service = SpamInferenceService(pipeline_bundle=bundle).load_service()

    emails = [
        "Win free money now!",
        "Can we schedule a meeting tomorrow?",
        "Urgent! Claim your lottery award!",
    ]
    df_res = service.predict_batch(emails)
    assert isinstance(df_res, pd.DataFrame)
    assert len(df_res) == 3
    assert "prediction" in df_res.columns
    assert "spam_probability" in df_res.columns


class TestDeploymentModule(unittest.TestCase):
    def test_persistence_defaults_match_service_outside_project_root(self):
        import os
        from unittest.mock import patch
        from src.models.persistence import load_spam_pipeline

        original_cwd = Path.cwd()
        bundle = {'sentinel': 'pipeline'}
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                service_path = SpamInferenceService().bundle_path
                with patch('src.models.persistence.joblib.dump') as dump, patch.object(Path, 'mkdir'):
                    saved_path = save_spam_pipeline(bundle)
                self.assertEqual(saved_path.resolve(), service_path.resolve())
                dump.assert_called_once_with(bundle, service_path, compress=3)
                with patch.object(Path, 'exists', return_value=True), \
                        patch('src.models.persistence.joblib.load', return_value=bundle) as load:
                    self.assertEqual(load_spam_pipeline(), bundle)
                load.assert_called_once_with(service_path)
            finally:
                os.chdir(original_cwd)

    def test_default_bundle_path(self):
        from unittest.mock import patch
        from src.config import SAVED_MODELS_DIR
        bundle = _create_toy_pipeline_bundle()
        with patch('src.deployment.predictor.load_spam_pipeline', return_value=bundle) as load:
            SpamInferenceService().load_service()
            load.assert_called_once_with(SAVED_MODELS_DIR / 'spam_pipeline.joblib')
        self.assertEqual(SpamInferenceService(bundle_path='custom.joblib').bundle_path,
                         Path('custom.joblib'))

    def test_selected_columns_roundtrip(self):
        for full_width in (True, False):
            with self.subTest(full_width=full_width), tempfile.TemporaryDirectory() as tmp:
                bundle = _create_toy_pipeline_bundle()
                text = "Win free cash prize claim now"
                matrix = bundle["feature_builder"].transform([{"text": text}]).to_scipy_csr()
                indices = np.arange(matrix.shape[1])[::-1]
                if not full_width:
                    indices = indices[:max(1, len(indices) // 2)]
                bundle["selected_indices"] = indices
                # Train a model in exactly the selected order, including full-width permutations.
                train = bundle["feature_builder"].transform([
                    {"text": text}, {"text": "Please review the project report attached"}
                ]).to_scipy_csr()[:, indices]
                bundle["model"].fit(train, np.array([1, 0]))
                expected = round(float(bundle["model"].predict_proba(matrix[:, indices])[0, 1]), 4)
                direct = SpamInferenceService(pipeline_bundle=bundle).predict_single(text)
                path = save_spam_pipeline(bundle, Path(tmp) / "pipeline.joblib")
                restored = SpamInferenceService(bundle_path=path).predict_single(text)
                self.assertEqual(direct["spam_probability"], expected)
                self.assertEqual(restored, direct)

    def test_invalid_bundle(self):
        for threshold in (float("nan"), -0.1, 1.1):
            bundle = _create_toy_pipeline_bundle()
            bundle["best_threshold"] = threshold
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                SpamInferenceService(pipeline_bundle=bundle).load_service()
        bundle.pop("feature_builder")
        with self.assertRaises(ValueError):
            SpamInferenceService(pipeline_bundle=bundle).load_service()

    def test_invalid_selected_indices(self):
        for indices in ([0, 0], [-1], [1000000], [.5], [[0]], []):
            with self.subTest(indices=indices):
                bundle = _create_toy_pipeline_bundle()
                bundle["selected_indices"] = indices
                with self.assertRaises(ValueError):
                    SpamInferenceService(pipeline_bundle=bundle).predict_single("Win free cash")

    def test_single_case(self):
        test_inference_service_single_prediction()

    def test_batch_case(self):
        test_inference_service_batch_prediction()


if __name__ == "__main__":
    unittest.main()


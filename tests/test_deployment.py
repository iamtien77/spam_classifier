import sys
from pathlib import Path
import unittest

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
    def test_single_case(self):
        test_inference_service_single_prediction()

    def test_batch_case(self):
        test_inference_service_batch_prediction()


if __name__ == "__main__":
    unittest.main()


"""Kiểm thử tích hợp luồng Train–Validation–Test và artifacts trong thư mục tạm."""
import csv
import io
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout

import main as pipeline
from src.config import SpamExperimentConfig
from src.data.loader import load_raw_data
from src.deployment.predictor import SpamInferenceService


class TestPipeline(unittest.TestCase):
    def test_training_selection_and_deployment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "input.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["label", "text"])
                for i in range(40):
                    writer.writerow(["ham", f"Hello team project report meeting schedule reference {i}"])
                    writer.writerow(["spam", f"WIN free cash prize claim urgent offer code {i}!"])
            config = SpamExperimentConfig(
                train_ratio=.6, val_ratio=.2, test_ratio=.2, random_state=17,
                max_unigram_features=40, max_bigram_features=20, max_char_features=10,
                default_tuning_params={"cap_quantile": .5},
            )
            logger = logging.getLogger("pipeline-test")
            # Redirect only I/O; run the actual feature, tuning, evaluation and persistence logic.
            with patch.object(pipeline, "load_raw_data", side_effect=lambda: load_raw_data(path)), \
                    patch.object(pipeline, "FIGURES_DIR", root), \
                    patch.object(pipeline, "METRICS_DIR", root), \
                    patch.object(pipeline, "SAVED_MODELS_DIR", root):
                data = pipeline.step_data_preprocessing.__wrapped__(logger, config)
                train, val, test, X_train, X_val, X_test, y_train, y_val, y_test, processor, builder, indices = data
                self.assertEqual((len(train), len(val), len(test)), (48, 16, 16))
                models = pipeline.step_model_training.__wrapped__(logger, X_train, y_train, X_val, y_val, config)
                self.assertEqual(len(models), 6)
                winner, threshold, info, metrics, summary = pipeline.step_evaluation_and_tuning.__wrapped__(
                    logger, models, X_val, y_val, X_test, y_test, test, config,
                )
                self.assertTrue((root / "validation_model_selection.csv").is_file())
                with redirect_stdout(io.StringIO()):
                    pipeline.step_deployment_and_demo.__wrapped__(logger, winner, processor, builder, indices, threshold, info, metrics)
                service = SpamInferenceService(bundle_path=root / "spam_pipeline.joblib").load_service()
                for i, text in enumerate(test["text"]):
                    expected = round(float(winner.predict_proba(X_test[i:i + 1])[0, 1]), 4)
                    self.assertEqual(service.predict_single(text)["spam_probability"], expected)
                self.assertEqual(service.feature_builder.cap_word_limit_, builder.cap_word_limit_)

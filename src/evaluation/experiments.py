"""Các lượt chạy baseline/ablation chỉ dùng trong phát triển. Tuyệt đối không đánh giá trên tập Test dự phòng (reserved Test).

Run explicitly: python -m src.evaluation.experiments --data data/raw/spam.csv
Đầu ra là các chẩn đoán phát triển, không phải là ước lượng tổng quát hóa độc lập.
"""
import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import SpamExperimentConfig
from src.data.loader import load_raw_data
from src.data.preprocessing import clean_email_split, stratified_split_dataframe, SpamTextProcessor
from src.features.feature_engineering import HybridFeatureBuilderScratch, SpamSignalFeatureExtractor
from src.models.classifiers import LogisticRegressionFromScratch, NaiveBayesClassifier
from src.models.persistence import experiment_metadata
from src.models.tuning import ThresholdOptimizerScratch


def run_development_experiments(raw, output_dir, seeds=(17, 42, 73), config=None):
    config = config or SpamExperimentConfig()
    if config.train_ratio <= 0 or config.val_ratio <= 0 or config.test_ratio <= 0:
        raise ValueError("Development experiments require positive Train/Validation/Test ratios.")
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least one seed, without duplicates.")
    # Giữ nguyên các phần tử thuộc tập Test ban đầu thay vì gộp chúng vào tập huấn luyện.
    cleaned = clean_email_split(raw)
    train, validation, reserved = stratified_split_dataframe(
        cleaned, train_ratio=config.train_ratio, val_ratio=config.val_ratio,
        test_ratio=config.test_ratio, random_state=config.random_state,
    )
    development = pd.concat([train, validation], ignore_index=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = experiment_metadata(config, raw)
    metadata.update({"seeds": list(seeds), "purpose": "development_only",
                     "independent_holdout": False,
                     "reserved_test_source_ids": reserved["source_id"].tolist()})
    (output_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    results, scans = [], []
    for seed in seeds:
        fit_rows, val_rows, _ = stratified_split_dataframe(
            development, train_ratio=config.train_ratio / (config.train_ratio + config.val_ratio),
            val_ratio=config.val_ratio / (config.train_ratio + config.val_ratio),
            test_ratio=0, random_state=seed,
        )
        # Ghi nhận danh sách ID phần tử chính xác để kiểm toán lượt chạy mà không làm lộ nội dung văn bản.
        split = {"train": fit_rows["source_id"].tolist(), "validation": val_rows["source_id"].tolist()}
        (output_dir / f"split_{seed}.json").write_text(json.dumps(split), encoding="utf-8")
        y_train = (fit_rows["label"] == "spam").to_numpy(dtype=int)
        y_val = (val_rows["label"] == "spam").to_numpy(dtype=int)
        for feature_mode in ("word_tfidf", "hybrid"):
            params = dict(config.default_tuning_params)
            if feature_mode == "word_tfidf":
                params.update(use_bigram=False, use_char_ngram=False,
                              use_keyword_features=False, use_numeric_features=False)
            run_config = replace(config, random_state=seed, default_tuning_params=params)
            builder = HybridFeatureBuilderScratch(run_config, SpamTextProcessor(run_config),
                                                  SpamSignalFeatureExtractor(run_config))
            X_train = builder.fit_transform(fit_rows).to_scipy_csr()
            X_val = builder.transform(val_rows).to_scipy_csr()
            models = {"LR": LogisticRegressionFromScratch(),
                      "NB": NaiveBayesClassifier(kind="complement")}
            optimizer = ThresholdOptimizerScratch(recall_target=config.recall_target)
            for name, model in models.items():
                model.fit(X_train, y_train)
                scores = model.predict_proba(X_val)[:, 1]
                scan = optimizer.metrics_table(y_val, scores)
                scans.append(scan.assign(seed=seed, features=feature_mode, model=name))
                choices = {"default": optimizer.compute_metrics_at_threshold(y_val, scores, .5),
                           "validation_selected": optimizer.select_for_recall_target(scan)}
                for policy, info in choices.items():
                    results.append({"seed": seed, "features": feature_mode, "model": name,
                                    "policy": policy, "n_features": X_train.shape[1], **info})
    frame = pd.DataFrame(results)
    frame.to_csv(output_dir / "development_metrics.csv", index=False)
    pd.concat(scans, ignore_index=True).to_csv(output_dir / "threshold_tradeoffs.csv", index=False)
    columns = ["accuracy", "precision", "recall", "f1", "fpr"]
    frame.groupby(["features", "model", "policy"])[columns].agg(["mean", "std", "count"]).to_csv(
        output_dir / "development_summary.csv")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    args = parser.parse_args()
    output = args.output or Path("results/experiments") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_development_experiments(load_raw_data(args.data), output, args.seeds)
    print(f"Development diagnostics saved to {output}; no independent holdout evaluated.")


if __name__ == "__main__":
    main()

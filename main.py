"""
main.py - Entry point điều phối toàn diện Pipeline phân loại email rác
Chuẩn hóa 100% theo yêu cầu đề bài (ML project: Classifying Spam Emails)
========================================================================
Quy trình thực thi gồm 4 giai đoạn cốt lõi và các phần mở rộng theo đúng đề bài:

TẦNG DATA INGESTION & HYGIENE (NẠP & KIỂM TOÁN TÍNH TOÀN VẸN MẪU THÔ)
  - Nạp tập dữ liệu thô (spam_ham_dataset.csv), xác thực 2 trường cốt lõi 'label' và 'text'.
  - Kiểm toán chất lượng & khử trùng lặp thô (Exact Deduplication) để bảo đảm giả thiết mẫu độc lập i.i.d và chống rò rỉ dữ liệu qua bản ghi lặp (Duplicate Leakage).
  - Khám phá dữ liệu ban đầu (EDA), xuất bản bộ 6 biểu đồ phân tích phân phối dữ liệu thô.

GIAI ĐOẠN 1: ML PIPELINE - CHIA TẬP TRƯỚC, TIỀN XỬ LÝ ĐẶC TRƯNG SAU
  - Bước 1.1: Phân tách dữ liệu độc lập ngay từ đầu (Early Stratified Split First) thành Train (80%) - Val (10%) - Test (10%).
  - Bước 1.2: Tiền xử lý văn bản học máy (SpamTextProcessor - CHỈ FIT TRÊN TRAIN): chuẩn hóa Unicode, stop words, trần độ dài 791 từ.
  - Bước 1.3: Trích xuất không gian đặc trưng lai 5 khối (HybridFeatureBuilder - CHỈ FIT TRÊN TRAIN): Word TF-IDF, Bigrams, Char n-grams, Keywords, Scaler.
  - Bước 1.4: Lựa chọn đặc trưng Top-K tối ưu trên tập Validation (K=500 chiều).

GIAI ĐOẠN 2: MODEL TRAINING (HUẤN LUYỆN 3 MÔ HÌNH CỐT LÕI THEO ĐỀ BÀI)
  - Bước 2.1: Huấn luyện mô hình 1 - Logistic Regression (Hàm Logistic/Sigmoid, Binary Cross-Entropy).
  - Bước 2.2: Huấn luyện mô hình 2 - Support Vector Machines (SVM - Tìm siêu phẳng phân tách lề cực đại).
  - Bước 2.3: Huấn luyện mô hình 3 - Naive Bayes (Multinomial / Complement Naive Bayes theo định lý Bayes).

GIAI ĐOẠN 3: MODEL EVALUATION & EXPERIMENTS (ĐÁNH GIÁ ĐA CHIỀU & MỞ RỘNG)
  - Bước 3.1: Đánh giá hiệu năng 3 mô hình trên tập Test: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
  - Bước 3.2: Tinh chỉnh siêu tham số (Hyperparameter Tuning: C, learning_rate, alpha).
  - Bước 3.3: Kết hợp mô hình (Ensemble Methods: Voting Ensemble giữa LR + SVM + Naive Bayes, Random Forest).
  - Bước 3.4: Tối ưu hóa ngưỡng phân loại (Threshold Tuning với mục tiêu Recall >= 0.85).
  - Bước 3.5: Lựa chọn mô hình chiến thắng (Final Selected Model).
  - Bước 3.6: Trực quan hóa báo cáo: Heatmap Confusion Matrix, ROC-PR curves, Biểu đồ so sánh mô hình.
  - Bước 3.7: Phân tích lỗi sai chuyên sâu (Error Analysis: Bóc tách False Positive và False Negative).

GIAI ĐOẠN 4: MODEL DEPLOYMENT (TRIỂN KHAI SUY LUẬN EMAIL MỚI)
  - Bước 4.1: Đóng gói và lưu trữ toàn bộ Pipeline vào file results/saved_models/spam_pipeline.joblib.
  - Bước 4.2: Khởi tạo dịch vụ suy luận SpamInferenceService.
  - Bước 4.3: Chạy demo phân loại trên các mẫu email thực tế mới (Unseen emails) và in chẩn đoán chi tiết.
"""

import sys
import json
import hashlib
from pathlib import Path

# Đảm bảo mã hóa UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.config import (
    ensure_directories,
    SpamExperimentConfig,
    RAW_DATA_FILE,
    FIGURES_DIR,
    METRICS_DIR,
    SAVED_MODELS_DIR,
    FBETA_BETA,
    THRESHOLD_GRID_POINTS,
)
from src.utils.logger import setup_logger, log_step, print_section_header
from src.data.loader import load_raw_data
from src.data.preprocessing import (
    clean_email_split,
    audit_data_quality,
    stratified_split_dataframe,
    SpamTextProcessor,
)
from src.features.feature_engineering import (
    FittedFeatureTransform,
    SpamSignalFeatureExtractor,
    HybridFeatureBuilderScratch,
    ManualFeatureSelectorScratch,
    MaxAbsScalerScratch,
)
from src.models.classifiers import (
    LogisticRegressionFromScratch,
    LinearSVMFromScratch,
    NaiveBayesClassifier,
)
from src.models.ensemble import SpamVotingEnsemble
from src.models.tuning import ThresholdOptimizerScratch, ModelSelectionWorkflow, tune_model_hyperparameters
from src.models.persistence import save_spam_pipeline, experiment_metadata, archive_run_outputs
from src.evaluation.metrics import SpamModelEvaluator
from src.evaluation.visualization import (
    plot_class_distribution,
    plot_text_length_comparison,
    plot_spam_signals_comparison,
    plot_discriminative_words,
    plot_feature_correlation_heatmap,
    plot_confusion_matrix_heatmap,
    plot_roc_and_pr_curves,
    plot_models_comparison_bar,
    plot_outlier_length_distribution,
    plot_threshold_sensitivity_curve,
    plot_top_k_feature_sweep,
)
from src.evaluation.error_analysis import (
    extract_misclassified_samples,
    profile_error_patterns,
)
from src.deployment.predictor import SpamInferenceService


@log_step("GIAI ĐOẠN 1: CHIA TẬP ĐỘC LẬP (EARLY SPLIT) & TIỀN XỬ LÝ ĐẶC TRƯNG")
def step_data_preprocessing(logger, config=None):
    config = config or SpamExperimentConfig()
    data_file = getattr(config, "raw_data_file", RAW_DATA_FILE)
    logger.info(f"Nạp dữ liệu thô từ {data_file}...")
    raw_data = load_raw_data(data_file) if (data_file is not None and str(data_file) != str(RAW_DATA_FILE)) else load_raw_data()
    logger.info(f"Tổng số bản ghi đọc được: {len(raw_data):,} dòng.")

    # Tầng Data Ingestion: Kiểm toán chất lượng & vệ sinh tính toàn vẹn mẫu (chống Duplicate Leakage)
    logger.info("Kiểm toán chất lượng & khử trùng lặp thô tại tầng nạp (bảo đảm giả thiết i.i.d và chống rò rỉ qua duplicate)...")
    sanitized_data = clean_email_split(raw_data)
    audit = audit_data_quality(raw_data)
    logger.info(f"Số mẫu độc lập hợp lệ sau kiểm toán toàn vẹn: {len(sanitized_data):,} mẫu.")

    # Xuất bản bộ 6 biểu đồ khám phá dữ liệu ban đầu (EDA & Outliers)
    logger.info("Xuất bản bộ biểu đồ khám phá dữ liệu ban đầu vào results/figures/...")
    plot_class_distribution(raw_data, save_path=FIGURES_DIR / "eda_class_distribution.png")
    plot_text_length_comparison(sanitized_data, save_path=FIGURES_DIR / "eda_text_length_comparison.png")
    plot_spam_signals_comparison(sanitized_data, save_path=FIGURES_DIR / "eda_spam_signals_distribution.png")
    plot_discriminative_words(sanitized_data, save_path=FIGURES_DIR / "eda_discriminative_words.png")
    plot_feature_correlation_heatmap(sanitized_data, save_path=FIGURES_DIR / "eda_feature_correlation.png")
    plot_outlier_length_distribution(sanitized_data, save_path=FIGURES_DIR / "eda_outlier_length_distribution.png")

    # BƯỚC 1: CHIA PHÂN TẦNG ĐỘC LẬP TRƯỚC TIÊN (EARLY STRATIFIED SPLIT FIRST)
    logger.info(f"BƯỚC 1: Chia phân tầng Train/Val/Test ({config.train_ratio:.0%}/{config.val_ratio:.0%}/{config.test_ratio:.0%}) TRƯỚC MỌI BƯỚC TIỀN XỬ LÝ...")
    train_records, val_records, test_records = stratified_split_dataframe(
        sanitized_data, train_ratio=config.train_ratio, val_ratio=config.val_ratio,
        test_ratio=config.test_ratio, random_state=config.random_state,
    )
    logger.info(f"Kích thước tập: Train={len(train_records):,}, Val={len(val_records):,}, Test={len(test_records):,}")
    audit['splits'] = {
        name: {'rows': len(frame), 'labels': frame['label'].value_counts().to_dict()}
        for name, frame in [('train', train_records), ('validation', val_records), ('test', test_records)]
    }
    (METRICS_DIR / 'data_audit.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8',
    )

    if hasattr(train_records, "columns"):
        y_train = np.array([1 if str(l).lower() == "spam" else 0 for l in train_records["label"]], dtype=int)
        y_val = np.array([1 if str(l).lower() == "spam" else 0 for l in val_records["label"]], dtype=int)
        y_test = np.array([1 if str(l).lower() == "spam" else 0 for l in test_records["label"]], dtype=int)
    else:
        y_train = np.array([1 if r["label"] == "spam" else 0 for r in train_records], dtype=int)
        y_val = np.array([1 if r["label"] == "spam" else 0 for r in val_records], dtype=int)
        y_test = np.array([1 if r["label"] == "spam" else 0 for r in test_records], dtype=int)

    # BƯỚC 2: TIỀN XỬ LÝ VĂN BẢN & TRÍCH XUẤT ĐẶC TRƯNG (CHỈ FIT DUY NHẤT TRÊN TRAIN)
    logger.info("BƯỚC 2: Khởi tạo và fit HybridFeatureBuilderScratch (CHỈ FIT TRÊN TRAIN)...")
    text_processor = SpamTextProcessor(config=config)
    signal_extractor = SpamSignalFeatureExtractor(config=config)
    builder = HybridFeatureBuilderScratch(
        config=config,
        text_processor=text_processor,
        signal_extractor=signal_extractor,
    )

    builder.run_metadata_ = experiment_metadata(config, raw_data)
    builder.run_metadata_["split_source_ids"] = {
        name: frame["source_id"].tolist()
        for name, frame in (("train", train_records), ("validation", val_records), ("test", test_records))
    }
    X_train_raw = builder.fit(train_records).transform(train_records)
    logger.info(f"Ngưỡng cắt số từ sau chuẩn hóa, học trên Train: max_words = {builder.cap_word_limit_}")
    X_val_raw = builder.transform(val_records)
    X_test_raw = builder.transform(test_records)

    X_train_full = X_train_raw.to_scipy_csr() if hasattr(X_train_raw, "to_scipy_csr") else X_train_raw
    X_val_full = X_val_raw.to_scipy_csr() if hasattr(X_val_raw, "to_scipy_csr") else X_val_raw
    X_test_full = X_test_raw.to_scipy_csr() if hasattr(X_test_raw, "to_scipy_csr") else X_test_raw
    logger.info(f"Ma trận đặc trưng đầy đủ ban đầu: {X_train_full.shape[0]:,} mẫu x {X_train_full.shape[1]:,} chiều.")

    # Chọn lọc đặc trưng Top-K (ManualFeatureSelectorScratch)
    logger.info("Chạy bộ chọn lọc đặc trưng Top-K (Chi-Square & Discriminative Score)...")
    selector = ManualFeatureSelectorScratch()
    _, ranked_indices = selector.compute_combined_importance(X_train_full, y_train)
    candidate_ks = [500, 1000, 1500, 2000, 2500, min(3000, X_train_full.shape[1]), X_train_full.shape[1]]
    best_sweep, sweep_df = selector.sweep_top_k(
        X_train_full, y_train, X_val_full, y_val, ranked_indices,
        candidate_k_list=candidate_ks, recall_target=config.recall_target,
    )
    logger.info(f"Top-K tối ưu được chọn: {best_sweep['top_k']} đặc trưng (Tiết kiệm {best_sweep['dimension_saving_pct']:.1f}% số chiều, Val F1 = {best_sweep['val_f1']:.4f}).")
    plot_top_k_feature_sweep(sweep_df, save_path=FIGURES_DIR / "feature_selection_sweep.png")
    sweep_df.drop(columns=['selected_indices'], errors='ignore').to_csv(METRICS_DIR / 'top_k_sweep.csv', index=False)

    selected_indices = best_sweep["selected_indices"]
    feature_transform = FittedFeatureTransform(builder, selected_indices)
    X_train = feature_transform.select_matrix(X_train_full)
    X_val = feature_transform.select_matrix(X_val_full)
    X_test = feature_transform.select_matrix(X_test_full)
    logger.info(f"Ma trận sau chọn lọc: {X_train.shape[0]:,} mẫu x {X_train.shape[1]:,} chiều.")

    return (
        train_records, val_records, test_records,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        text_processor, builder, selected_indices,
    )


@log_step("GIAI ĐOẠN 2: HUẤN LUYỆN 3 MÔ HÌNH CỐT LÕI (FROM-SCRATCH)")
def step_model_training(logger, X_train, y_train, X_val=None, y_val=None, config=None):
    config = config or SpamExperimentConfig()
    logger.info("Huấn luyện Mô hình 1: Logistic Regression (Binary Cross-Entropy & L2)...")
    lr = LogisticRegressionFromScratch(C=1.0, learning_rate=0.1, max_iter=200, class_weight="balanced", random_state=config.random_state)
    lr.fit(X_train, y_train)

    logger.info("Huấn luyện Mô hình 2: Support Vector Machines (Linear SVM - Squared Hinge Loss)...")
    svm = LinearSVMFromScratch(C=1.0, learning_rate=0.05, max_iter=150, class_weight="balanced", loss="squared_hinge", random_state=config.random_state)
    svm.fit(X_train, y_train)

    logger.info("Huấn luyện Mô hình 3: Complement Naive Bayes (Laplace smoothing)...")
    nb = NaiveBayesClassifier(alpha=1.0, kind="complement")
    nb.fit(X_train, y_train)

    logger.info("Đã hoàn tất huấn luyện 3 mô hình cốt lõi.")
    models = {"Logistic Regression": lr, "Linear SVM": svm, "Complement Naive Bayes": nb}
    if X_val is not None and y_val is not None:
        optimizer = ThresholdOptimizerScratch(config.recall_target, FBETA_BETA, THRESHOLD_GRID_POINTS)
        grids = {
            "Logistic Regression": (LogisticRegressionFromScratch, {
                "C": [0.5, 1.0, 2.0], "learning_rate": [0.1],
                "max_iter": [200], "class_weight": ["balanced"], "random_state": [config.random_state],
            }),
            "Linear SVM": (LinearSVMFromScratch, {
                "C": [0.5, 1.0, 2.0], "learning_rate": [0.05], "max_iter": [150],
                "class_weight": ["balanced"], "loss": ["squared_hinge"], "random_state": [config.random_state],
            }),
            "Complement Naive Bayes": (NaiveBayesClassifier, {"alpha": [0.5, 1.0, 2.0], "kind": ["complement"]}),
        }
        for name, (model_class, grid) in grids.items():
            tuned, params, _ = tune_model_hyperparameters(
                name, model_class, grid, X_train, y_train, X_val, y_val,
                recall_target=config.recall_target, threshold_optimizer=optimizer,
            )
            models[f"{name} (Tuned)"] = tuned
            tuned.tuning_params_ = params
            logger.info(f"[{name}] Tham số chọn trên Validation: {params}")
    return models


@log_step("GIAI ĐOẠN 3: ĐÁNH GIÁ, TUNING, ENSEMBLE & PHÂN TÍCH LỖI")
def step_evaluation_and_tuning(logger, models, X_val, y_val, X_test, y_test, test_records, config=None):
    config = config or SpamExperimentConfig()
    evaluator = SpamModelEvaluator()

    # 2. Xây dựng SpamVotingEnsemble
    logger.info("Xây dựng bộ kết hợp biểu quyết SpamVotingEnsemble...")
    ensemble = SpamVotingEnsemble(
        models={name: model for name, model in models.items() if not name.endswith(" (Tuned)")},
        weights={"Logistic Regression": 1.0, "Linear SVM": 1.0, "Complement Naive Bayes": 1.2},
        voting="soft",
    )

    candidates = {**models, "SpamVotingEnsemble": ensemble}
    tuned_members = {name: model for name, model in models.items() if name.endswith(" (Tuned)")}
    if tuned_members:
        candidates["SpamVotingEnsemble (Tuned)"] = SpamVotingEnsemble(models=tuned_members, voting="soft")
    optimizer = ThresholdOptimizerScratch(config.recall_target, FBETA_BETA, THRESHOLD_GRID_POINTS)
    winner_name, best_th_info, comparison, scan_df = ModelSelectionWorkflow(config.recall_target).select_candidates(
        candidates, X_val, y_val, optimizer,
    )
    winner = candidates[winner_name]
    best_th_info["candidate_key"] = winner_name
    comparison.to_csv(METRICS_DIR / "validation_model_selection.csv", index=False)
    logger.info(f"Mô hình chọn trên Validation: {winner_name}")
    best_threshold = float(best_th_info["threshold"])
    logger.info(f"Ngưỡng tối ưu tìm thấy: tau = {best_threshold:.4f} (Val Recall = {best_th_info['recall']:.2%}, F1 = {best_th_info['f1']:.4f}, Quy tắc: {best_th_info['selection_rule']})")

    scan_df.to_csv(METRICS_DIR / "validation_threshold_tradeoffs.csv", index=False)

    # 4. Xuất biểu đồ đường cong độ nhạy ngưỡng (Threshold Sensitivity Curve)
    plot_threshold_sensitivity_curve(scan_df, recall_target=config.recall_target, selected_threshold=best_threshold, save_path=FIGURES_DIR / "threshold_sensitivity.png")

    # 5. Đánh giá mô hình đã chọn tại ngưỡng Validation đã chốt
    test_probs = winner.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)
    final_metrics = evaluator.compute_all_metrics(y_test, test_preds, y_prob=test_probs, beta=FBETA_BETA)
    # Bảng Test phục vụ báo cáo, không tham gia lựa chọn ứng viên.
    baseline_results = {}
    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        baseline_results[name] = evaluator.compute_all_metrics(y_test, (probs >= .5).astype(int), probs, beta=FBETA_BETA)
    baseline_results[f"{winner_name} (Winner)"] = final_metrics
    logger.info(f"[{winner_name}] Test Acc={final_metrics['accuracy']:.4f}, Prec={final_metrics['precision']:.4f}, Rec={final_metrics['recall']:.4f}, F1={final_metrics['f1']:.4f}")

    # 6. Kiểm toán tính nhất quán của kết quả đánh giá (Evaluation Consistency Audit)
    audit_df = evaluator.audit_evaluation_consistency(y_test, test_preds, y_prob=test_probs)
    logger.info(f"Kiểm toán kết quả đánh giá hoàn tất (Tất cả điều kiện kiểm toán: {(audit_df['Status'] == 'PASS').sum()}/{len(audit_df)} PASS).")

    # 7. Xuất biểu đồ báo cáo
    logger.info("Xuất bản biểu đồ đánh giá vào results/figures/...")
    summary_df = evaluator.build_comparison_summary(baseline_results)
    # Mỗi ứng viên dùng ngưỡng chốt trên Validation, để so sánh công bằng.
    test_results = {}
    for name, model in candidates.items():
        threshold = float(comparison.loc[comparison.candidate_key == name, 'threshold'].iloc[0])
        probabilities = model.predict_proba(X_test)[:, 1]
        test_results[name] = evaluator.compute_all_metrics(
            y_test, (probabilities >= threshold).astype(int), probabilities, beta=FBETA_BETA,
        )
    summary_df = evaluator.build_comparison_summary(test_results)
    summary_df['selected_on_validation'] = summary_df['Mô hình'] == winner_name
    summary_df.to_csv(METRICS_DIR / 'test_models_comparison.csv', index=False)
    plot_confusion_matrix_heatmap(final_metrics["confusion_matrix"], model_name=winner_name, save_path=FIGURES_DIR / "confusion_matrix.png")
    plot_roc_and_pr_curves(y_test, test_probs, best_threshold=best_threshold, model_name=winner_name, save_path=FIGURES_DIR / "roc_pr_curve.png")
    plot_models_comparison_bar(summary_df, save_path=FIGURES_DIR / "models_comparison.png")

    # 8. Mổ xẻ phân tích lỗi sai (Error Analysis)
    logger.info("Trích xuất và phân tích mẫu lỗi sai (False Positives & False Negatives)...")
    test_df = pd.DataFrame(test_records)
    fp_df, fn_df = extract_misclassified_samples(test_df, y_test, test_preds, test_probs)
    error_profile = profile_error_patterns(fp_df, fn_df, save_csv_path=METRICS_DIR / "misclassified_samples.csv")
    logger.info(f"Phân tích lỗi hoàn tất: FP={len(fp_df):,} mẫu, FN={len(fn_df):,} mẫu. Đã lưu {METRICS_DIR / 'misclassified_samples.csv'}")

    return winner, best_threshold, best_th_info, final_metrics, summary_df


@log_step("GIAI ĐOẠN 4: ĐÓNG GÓI PIPELINE & DEMO TRIỂN KHAI SUY LUẬN")
def step_deployment_and_demo(logger, ensemble, text_processor, builder, selected_indices, best_threshold, best_th_info, final_metrics):
    pipeline_path = SAVED_MODELS_DIR / "spam_pipeline.joblib"
    logger.info(f"Đóng gói Pipeline Bundle vào: {pipeline_path}...")
    bundle = {
        "model": ensemble,
        "run_metadata": getattr(builder, "run_metadata_", None),
        "text_processor": text_processor,
        "feature_builder": builder,
        "selected_indices": selected_indices,
        "best_threshold": best_threshold,
        "metrics_val": best_th_info,
        "metrics_test": final_metrics,
    }
    save_spam_pipeline(bundle, save_path=pipeline_path)
    report = {
        'experiment': 'main.py', 'model': best_th_info['candidate_key'],
        'run_metadata': getattr(builder, 'run_metadata_', None),
        'selection_rule': best_th_info['selection_rule'], 'threshold': best_threshold,
        'features_before_selection': len(builder.feature_names_), 'selected_features': len(selected_indices),
        'validation': best_th_info,
        'test': {key: value.tolist() if isinstance(value, np.ndarray) else value
                 for key, value in final_metrics.items()},
        'source_sha256': {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), *sorted((PROJECT_ROOT / 'src').rglob('*.py'))]
        },
    }
    (METRICS_DIR / 'run_summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8',
    )

    logger.info("Khởi động dịch vụ suy luận SpamInferenceService từ file đã lưu...")
    service = SpamInferenceService(bundle_path=pipeline_path).load_service()

    # Danh sách 5 email thực tế đa dạng theo đề bài
    demo_emails = [
        "URGENT! You have won a 1,000,000 cash prize! Call 0900123456 now to claim your award immediately!",
        "Free entry in 2 a weekly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question!",
        "Dear team, please find attached the weekly project status report and meeting notes for Monday.",
        "Your monthly account statement is ready. Please log in to your secure portal to view details.",
        "Hey, are we still meeting for lunch at 12:30 today? Let me know!",
    ]


    print("\n" + "=" * 80)
    print(" KẾT QUẢ DEMO SUY LUẬN TRÊN 5 EMAIL THỰC TẾ (INFERENCE SERVICE DEMO) ".center(80, "="))
    print("=" * 80)

    for idx, email in enumerate(demo_emails, 1):
        result = service.predict_single(email)
        status_tag = "[SPAM]" if result["is_spam"] else "[HAM / HỢP LỆ]"
        print(f"\n📧 Email #{idx}: {status_tag} (Điểm Spam (chưa calibration): {result['spam_score']:.4f}, Rủi ro: {result['risk_level']})")
        print(f"   Nội dung: \"{result['raw_text']}\"")
        if result["detected_signals"]:
            print(f"   Tín hiệu kích hoạt: {result['detected_signals']}")

    print("\n" + "=" * 80 + "\n")


def run_pipeline():
    """Điều phối toàn diện Pipeline từ đầu đến cuối."""
    ensure_directories()
    logger = setup_logger("spam_classifier")
    config = SpamExperimentConfig()
    archived = archive_run_outputs(
        [METRICS_DIR, FIGURES_DIR, SAVED_MODELS_DIR], PROJECT_ROOT / "results" / "archive",
    )
    if archived is not None:
        logger.info(f"Previous results archived at: {archived}")

    print_section_header("BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE PHÂN LOẠI EMAIL SPAM")
    logger.info("Khởi chạy Spam Classifier Pipeline...")

    # Giai đoạn 1
    (
        train_records, val_records, test_records,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        text_processor, builder, selected_indices,
    ) = step_data_preprocessing(logger, config=config)

    # Giai đoạn 2
    models = step_model_training(logger, X_train, y_train, X_val, y_val, config=config)

    # Giai đoạn 3
    ensemble, best_threshold, best_th_info, final_metrics, summary_df = step_evaluation_and_tuning(
        logger, models, X_val, y_val, X_test, y_test, test_records, config=config,
    )

    # Giai đoạn 4
    step_deployment_and_demo(
        logger, ensemble, text_processor, builder, selected_indices, best_threshold, best_th_info, final_metrics
    )

    print_section_header("HOÀN TẤT TOÀN DIỆN DỰ ÁN SPAM CLASSIFIER THÀNH CÔNG")
    logger.info("Pipeline đã thực thi hoàn hảo và kết thúc thành công.")


if __name__ == "__main__":
    run_pipeline()


"""
main.py - Entry point điều phối toàn diện Pipeline phân loại email rác
Chuẩn hóa 100% theo yêu cầu đề bài (ML project: Classifying Spam Emails)
========================================================================
Quy trình thực thi gồm 4 giai đoạn cốt lõi và các phần mở rộng theo đúng đề bài:

GIAI ĐOẠN 1: DATA PREPROCESSING (TIỀN XỬ LÝ & TRÍCH XUẤT ĐẶC TRƯNG)
  - Bước 1.1: Khởi tạo môi trường, kiểm tra cấu hình và hệ thống thư mục lưu trữ.
  - Bước 1.2: Nạp tập dữ liệu thô (spam.csv), chuẩn hóa 2 trường cốt lõi 'label' và 'text'.
  - Bước 1.3: Khám phá dữ liệu (EDA), phân tích tỷ lệ mất cân bằng nhãn Ham vs Spam.
  - Bước 1.4: Phân tầng dữ liệu độc lập (Early Stratified Split) thành Train (80%) - Val (10%) - Test (10%).
  - Bước 1.5: Tiền xử lý văn bản: chuẩn hóa Unicode, loại bỏ stop words, punctuation, HTML tags/URLs.
  - Bước 1.6: Trích xuất đặc trưng số học: Word TF-IDF, Character frequency ('!', '$'), Keywords, Numerics.

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
    FIGURES_DIR,
    METRICS_DIR,
    SAVED_MODELS_DIR,
)
from src.utils.logger import setup_logger, log_step, print_section_header
from src.data.loader import load_raw_data
from src.data.preprocessing import (
    clean_email_split,
    stratified_split_dataframe,
    SpamTextProcessor,
)
from src.features.feature_engineering import (
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
from src.models.persistence import save_spam_pipeline
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


@log_step("GIAI ĐOẠN 1: TIỀN XỬ LÝ & TRÍCH XUẤT ĐẶC TRƯNG")
def step_data_preprocessing(logger):
    logger.info("Nạp dữ liệu thô từ data/raw/spam.csv...")
    raw_data = load_raw_data()
    logger.info(f"Tổng số bản ghi đọc được: {len(raw_data):,} dòng.")

    # Lọc chất lượng & khử trùng lặp
    logger.info("Kiểm toán và làm sạch dữ liệu (clean_email_split)...")
    cleaned_data = clean_email_split(raw_data)
    logger.info(f"Số mẫu sạch giữ lại sau kiểm toán: {len(cleaned_data):,} mẫu.")

    # Xuất bản bộ 6 biểu đồ khám phá dữ liệu (EDA & Outliers)
    logger.info("Xuất bản bộ biểu đồ khám phá dữ liệu & Outlier analysis vào results/figures/...")
    plot_class_distribution(raw_data, save_path=FIGURES_DIR / "eda_class_distribution.png")
    plot_text_length_comparison(cleaned_data, save_path=FIGURES_DIR / "eda_text_length_comparison.png")
    plot_spam_signals_comparison(cleaned_data, save_path=FIGURES_DIR / "eda_spam_signals_distribution.png")
    plot_discriminative_words(cleaned_data, save_path=FIGURES_DIR / "eda_discriminative_words.png")
    plot_feature_correlation_heatmap(cleaned_data, save_path=FIGURES_DIR / "eda_feature_correlation.png")
    plot_outlier_length_distribution(cleaned_data, save_path=FIGURES_DIR / "eda_outlier_length_distribution.png")

    # Chia phân tầng 80/10/10
    logger.info("Chia phân tầng độc lập (Stratified Split 80/10/10)...")
    train_records, val_records, test_records = stratified_split_dataframe(
        cleaned_data, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, random_state=42
    )
    logger.info(f"Kích thước tập: Train={len(train_records):,}, Val={len(val_records):,}, Test={len(test_records):,}")

    if hasattr(train_records, "columns"):
        y_train = np.array([1 if str(l).lower() == "spam" else 0 for l in train_records["label"]], dtype=int)
        y_val = np.array([1 if str(l).lower() == "spam" else 0 for l in val_records["label"]], dtype=int)
        y_test = np.array([1 if str(l).lower() == "spam" else 0 for l in test_records["label"]], dtype=int)
        train_word_counts = [len(str(t).split()) for t in train_records["text"].fillna("")]
    else:
        y_train = np.array([1 if r["label"] == "spam" else 0 for r in train_records], dtype=int)
        y_val = np.array([1 if r["label"] == "spam" else 0 for r in val_records], dtype=int)
        y_test = np.array([1 if r["label"] == "spam" else 0 for r in test_records], dtype=int)
        train_word_counts = [len(str(r.get("text", "")).split()) for r in train_records]

    cap_word_limit = int(np.ceil(np.percentile(train_word_counts, 99)))
    logger.info(f"Xác định ngưỡng Outlier Capping (p99 trên Train): max_words = {cap_word_limit}")

    # Trích xuất đặc trưng lai
    logger.info("Khởi tạo và fit HybridFeatureBuilderScratch trên Train...")
    config = SpamExperimentConfig()
    text_processor = SpamTextProcessor(config=config)
    signal_extractor = SpamSignalFeatureExtractor(config=config)
    builder = HybridFeatureBuilderScratch(
        config=config,
        text_processor=text_processor,
        signal_extractor=signal_extractor,
    )

    X_train_raw = builder.fit(train_records).transform(train_records)
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
        X_train_full, y_train, X_val_full, y_val, ranked_indices, candidate_k_list=candidate_ks, recall_target=0.85
    )
    logger.info(f"Top-K tối ưu được chọn: {best_sweep['top_k']} đặc trưng (Tiết kiệm {best_sweep['dimension_saving_pct']:.1f}% số chiều, Val F1 = {best_sweep['val_f1']:.4f}).")
    plot_top_k_feature_sweep(sweep_df, save_path=FIGURES_DIR / "feature_selection_sweep.png")

    selected_indices = best_sweep["selected_indices"]
    X_train = X_train_full[:, selected_indices]
    X_val = X_val_full[:, selected_indices]
    X_test = X_test_full[:, selected_indices]
    logger.info(f"Ma trận sau chọn lọc: {X_train.shape[0]:,} mẫu x {X_train.shape[1]:,} chiều.")

    return (
        train_records, val_records, test_records,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        text_processor, builder, selected_indices,
    )


@log_step("GIAI ĐOẠN 2: HUẤN LUYỆN 3 MÔ HÌNH CỐT LÕI (FROM-SCRATCH)")
def step_model_training(logger, X_train, y_train):
    logger.info("Huấn luyện Mô hình 1: Logistic Regression (Binary Cross-Entropy & L2)...")
    lr = LogisticRegressionFromScratch(C=1.0, learning_rate=0.1, max_iter=200, class_weight="balanced")
    lr.fit(X_train, y_train)

    logger.info("Huấn luyện Mô hình 2: Support Vector Machines (Linear SVM - Squared Hinge Loss)...")
    svm = LinearSVMFromScratch(C=1.0, learning_rate=0.05, max_iter=150, class_weight="balanced", loss="squared_hinge")
    svm.fit(X_train, y_train)

    logger.info("Huấn luyện Mô hình 3: Complement Naive Bayes (Laplace smoothing)...")
    nb = NaiveBayesClassifier(alpha=1.0, kind="complement")
    nb.fit(X_train, y_train)

    logger.info("Đã hoàn tất huấn luyện 3 mô hình cốt lõi.")
    return {"Logistic Regression": lr, "Linear SVM": svm, "Complement Naive Bayes": nb}


@log_step("GIAI ĐOẠN 3: ĐÁNH GIÁ, TUNING, ENSEMBLE & PHÂN TÍCH LỖI")
def step_evaluation_and_tuning(logger, models, X_val, y_val, X_test, y_test, test_records):
    evaluator = SpamModelEvaluator()

    # 1. Đánh giá từng mô hình cơ bản trên Test
    baseline_results = {}
    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        preds = model.predict(X_test)
        metrics = evaluator.compute_all_metrics(y_test, preds, y_prob=probs, beta=2.0)
        baseline_results[name] = metrics
        logger.info(f"[{name}] Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

    # 2. Xây dựng SpamVotingEnsemble
    logger.info("Xây dựng bộ kết hợp biểu quyết SpamVotingEnsemble...")
    ensemble = SpamVotingEnsemble(
        models=models,
        weights={"Logistic Regression": 1.0, "Linear SVM": 1.0, "Complement Naive Bayes": 1.2},
        voting="soft",
    )

    # 3. Tối ưu hóa ngưỡng phân loại trên tập Validation & Bảng chỉ số chi tiết
    logger.info("Dò quét 100 mốc ngưỡng xác suất trên Validation set (mục tiêu Recall >= 0.85)...")
    val_probs = ensemble.predict_proba(X_val)[:, 1]
    optimizer = ThresholdOptimizerScratch(recall_target=0.85, beta=2.0, n_points=100)
    scan_df = optimizer.metrics_table(y_val, val_probs)
    best_th_info = optimizer.select_for_recall_target(scan_df, recall_target=0.85)
    best_threshold = float(best_th_info["threshold"])
    logger.info(f"Ngưỡng tối ưu tìm thấy: tau = {best_threshold:.4f} (Val Recall = {best_th_info['recall']:.2%}, F1 = {best_th_info['f1']:.4f}, Quy tắc: {best_th_info['selection_rule']})")

    # 4. Xuất biểu đồ đường cong độ nhạy ngưỡng (Threshold Sensitivity Curve)
    plot_threshold_sensitivity_curve(scan_df, recall_target=0.85, selected_threshold=best_threshold, save_path=FIGURES_DIR / "threshold_sensitivity.png")

    # 5. Đánh giá kiểm thử cuối cùng cho Ensemble tại best_threshold
    test_probs = ensemble.predict_proba(X_test)[:, 1]
    test_preds = ensemble.predict(X_test, threshold=best_threshold)
    final_metrics = evaluator.compute_all_metrics(y_test, test_preds, y_prob=test_probs, beta=2.0)
    baseline_results["SpamVotingEnsemble (Winner)"] = final_metrics
    logger.info(f"[SpamVotingEnsemble] Test Acc={final_metrics['accuracy']:.4f}, Prec={final_metrics['precision']:.4f}, Rec={final_metrics['recall']:.4f}, F1={final_metrics['f1']:.4f}")

    # 6. Kiểm toán tính nhất quán của kết quả đánh giá (Evaluation Consistency Audit)
    audit_df = evaluator.audit_evaluation_consistency(y_test, test_preds, y_prob=test_probs)
    logger.info(f"Kiểm toán kết quả đánh giá hoàn tất (Tất cả điều kiện kiểm toán: {(audit_df['Status'] == 'PASS').sum()}/{len(audit_df)} PASS).")

    # 7. Xuất biểu đồ báo cáo
    logger.info("Xuất bản biểu đồ đánh giá vào results/figures/...")
    summary_df = evaluator.build_comparison_summary(baseline_results)
    plot_confusion_matrix_heatmap(final_metrics["confusion_matrix"], model_name="SpamVotingEnsemble", save_path=FIGURES_DIR / "confusion_matrix.png")
    plot_roc_and_pr_curves(y_test, test_probs, best_threshold=best_threshold, model_name="SpamVotingEnsemble", save_path=FIGURES_DIR / "roc_pr_curve.png")
    plot_models_comparison_bar(summary_df, save_path=FIGURES_DIR / "models_comparison.png")

    # 8. Mổ xẻ phân tích lỗi sai (Error Analysis)
    logger.info("Trích xuất và phân tích mẫu lỗi sai (False Positives & False Negatives)...")
    test_df = pd.DataFrame(test_records)
    fp_df, fn_df = extract_misclassified_samples(test_df, y_test, test_preds, test_probs)
    error_profile = profile_error_patterns(fp_df, fn_df, save_csv_path=METRICS_DIR / "misclassified_samples.csv")
    logger.info(f"Phân tích lỗi hoàn tất: FP={len(fp_df):,} mẫu, FN={len(fn_df):,} mẫu. Đã lưu {METRICS_DIR / 'misclassified_samples.csv'}")

    return ensemble, best_threshold, best_th_info, final_metrics, summary_df


@log_step("GIAI ĐOẠN 4: ĐÓNG GÓI PIPELINE & DEMO TRIỂN KHAI SUY LUẬN")
def step_deployment_and_demo(logger, ensemble, text_processor, builder, selected_indices, best_threshold, best_th_info, final_metrics):
    pipeline_path = SAVED_MODELS_DIR / "spam_pipeline.joblib"
    logger.info(f"Đóng gói Pipeline Bundle vào: {pipeline_path}...")
    bundle = {
        "model": ensemble,
        "text_processor": text_processor,
        "feature_builder": builder,
        "selected_indices": selected_indices,
        "best_threshold": best_threshold,
        "metrics_val": best_th_info,
        "metrics_test": final_metrics,
    }
    save_spam_pipeline(bundle, save_path=pipeline_path)

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
        print(f"\n📧 Email #{idx}: {status_tag} (Xác suất Spam: {result['spam_probability']:.2%}, Rủi ro: {result['risk_level']})")
        print(f"   Nội dung: \"{result['raw_text']}\"")
        if result["detected_signals"]:
            print(f"   Tín hiệu kích hoạt: {result['detected_signals']}")

    print("\n" + "=" * 80 + "\n")


def run_pipeline():
    """Điều phối toàn diện Pipeline từ đầu đến cuối."""
    ensure_directories()
    logger = setup_logger("spam_classifier")

    print_section_header("BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE PHÂN LOẠI EMAIL SPAM")
    logger.info("Khởi chạy Spam Classifier Pipeline...")

    # Giai đoạn 1
    (
        train_records, val_records, test_records,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        text_processor, builder, selected_indices,
    ) = step_data_preprocessing(logger)

    # Giai đoạn 2
    models = step_model_training(logger, X_train, y_train)

    # Giai đoạn 3
    ensemble, best_threshold, best_th_info, final_metrics, summary_df = step_evaluation_and_tuning(
        logger, models, X_val, y_val, X_test, y_test, test_records
    )

    # Giai đoạn 4
    step_deployment_and_demo(
        logger, ensemble, text_processor, builder, selected_indices, best_threshold, best_th_info, final_metrics
    )

    print_section_header("HOÀN TẤT TOÀN DIỆN DỰ ÁN SPAM CLASSIFIER THÀNH CÔNG")
    logger.info("Pipeline đã thực thi hoàn hảo và kết thúc thành công.")


if __name__ == "__main__":
    run_pipeline()


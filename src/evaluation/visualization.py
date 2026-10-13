"""
visualization.py - Trực quan hóa dữ liệu và biểu đồ đánh giá mô hình
Chuẩn hóa theo yêu cầu đề bài (ML project: Classifying Spam Emails)
===================================================================
Module này đảm nhiệm việc xuất bản các biểu đồ trực quan chất lượng cao vào results/figures/:
1. Khám phá dữ liệu (EDA): Phân bố nhãn Ham/Spam, biểu đồ phân phối độ dài văn bản email.
2. Ma trận nhầm lẫn (Confusion Matrix Heatmap): Hiển thị số lượng mẫu và tỷ lệ chuẩn hóa %.
3. Đường cong đánh giá (ROC Curve & Precision-Recall Curve) có đánh dấu điểm ngưỡng tối ưu.
4. Biểu đồ so sánh đa chiều hiệu năng giữa 3 mô hình cốt lõi (Logistic Regression, SVM, Naive Bayes) và Ensemble.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_class_distribution(
    df: Any,
    label_col: str = "label",
    filter_valid_only: bool = False,
    save_path: Optional[Path] = None,
) -> None:
    """Vẽ biểu đồ phân bố nhãn (Bar chart + Pie chart)."""
    if isinstance(df, list):
        df = pd.DataFrame(df)

    raw_labels = df[label_col].fillna("").astype(str).str.strip().str.lower()
    if filter_valid_only:
        raw_labels = raw_labels[raw_labels.isin(["ham", "spam"])]
    else:
        # Gom các dòng rác, thiếu nhãn (chiếm ~5% dữ liệu thô) thành nhóm rõ ràng, tránh đè chữ
        raw_labels = raw_labels.apply(lambda x: x if x in ("ham", "spam") else "missing / invalid")

    counts = raw_labels.value_counts()
    labels = [str(k).capitalize() if str(k) in ("ham", "spam") else "Missing / Invalid" for k in counts.index]
    values = counts.values

    # Bảng màu tương ứng: Ham (Xanh), Spam (Đỏ), Missing/Invalid (Xám cảnh báo)
    color_map = {"Ham": "#4C72B0", "Spam": "#C44E52", "Missing / Invalid": "#A0AEC0"}
    colors = [color_map.get(lbl, "#718096") for lbl in labels]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart
    bars = axes[0].bar(labels, values, color=colors, width=0.45, edgecolor="black", linewidth=0.8)
    axes[0].set_title("Số lượng mẫu theo nhãn", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Số lượng mẫu", fontsize=11)
    for bar in bars:
        h = bar.get_height()
        axes[0].annotate(f"{h:,}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    # Pie chart
    explode = [0.03] * len(labels)
    axes[1].pie(values, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors,
                explode=explode,
                wedgeprops={"edgecolor": "black", "linewidth": 0.8}, textprops={"fontsize": 11})
    axes[1].set_title("Tỷ lệ phân bố lớp (%)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_text_length_comparison(
    df: Any,
    text_col: str = "text",
    label_col: str = "label",
    save_path: Optional[Path] = None,
) -> None:
    """Vẽ biểu đồ Boxplot và phân phối mật độ so sánh độ dài giữa Ham và Spam."""
    if isinstance(df, list):
        df = pd.DataFrame(df)

    clean_df = df[df[label_col].astype(str).str.lower().isin(["ham", "spam"])].copy()
    clean_df["char_len"] = clean_df[text_col].apply(lambda x: len(str(x or "")))
    clean_df["word_count"] = clean_df[text_col].apply(lambda x: len(str(x or "").split()))

    ham_chars = clean_df[clean_df[label_col].str.lower() == "ham"]["char_len"].values
    spam_chars = clean_df[clean_df[label_col].str.lower() == "spam"]["char_len"].values
    ham_words = clean_df[clean_df[label_col].str.lower() == "ham"]["word_count"].values
    spam_words = clean_df[clean_df[label_col].str.lower() == "spam"]["word_count"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    bp = axes[0].boxplot(
        [ham_chars, spam_chars],
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=False,
        widths=0.45,
    )
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["Ham (Hợp lệ)", "Spam (Thư rác)"], fontsize=11, fontweight="bold")
    colors = ["#4C72B0", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.8)

    ham_med = int(np.median(ham_chars)) if len(ham_chars) else 0
    spam_med = int(np.median(spam_chars)) if len(spam_chars) else 0
    axes[0].text(1, ham_med, f" Median: {ham_med:,}", va="center", ha="left", fontsize=10, fontweight="bold")
    axes[0].text(2, spam_med, f" Median: {spam_med:,}", va="center", ha="left", fontsize=10, fontweight="bold")

    axes[0].set_title("So sánh độ dài ký tự (Boxplot phân phối IQR)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Số lượng ký tự (Characters)", fontsize=11)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # 2. Histogram mật độ số từ trong khoảng 0-150 từ (chiếm >95% mẫu)
    axes[1].hist(
        ham_words,
        bins=35,
        range=(0, 150),
        alpha=0.55,
        color="#4C72B0",
        label=f"Ham (Trung vị: {int(np.median(ham_words)):,} từ)",
        density=True,
        edgecolor="black",
    )
    axes[1].hist(
        spam_words,
        bins=35,
        range=(0, 150),
        alpha=0.55,
        color="#C44E52",
        label=f"Spam (Trung vị: {int(np.median(spam_words)):,} từ)",
        density=True,
        edgecolor="black",
    )
    axes[1].set_title("Mật độ phân phối số lượng từ (0-150 từ)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Số lượng từ (Word count)", fontsize=11)
    axes[1].set_ylabel("Mật độ xác suất (Density)", fontsize=11)
    axes[1].legend(loc="upper right", framealpha=0.9)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


# Alias giữ tương thích ngược
plot_text_length_distribution = plot_text_length_comparison


def plot_spam_signals_comparison(
    df: Any,
    text_col: str = "text",
    label_col: str = "label",
    save_path: Optional[Path] = None,
) -> None:
    """So sánh tỷ lệ xuất hiện các tín hiệu bề mặt kích động giữa Ham và Spam."""
    import re

    if isinstance(df, list):
        df = pd.DataFrame(df)

    clean_df = df[df[label_col].astype(str).str.lower().isin(["ham", "spam"])].copy()

    # Trích xuất 5 tín hiệu bề mặt quan trọng
    def extract_signals(t: str) -> dict:
        t_str = str(t or "")
        letters = sum(1 for c in t_str if c.isalpha())
        uppers = sum(1 for c in t_str if c.isupper())
        caps_ratio = (uppers / letters) if letters else 0.0
        has_excl = "!" in t_str
        has_curr = bool(re.search(r"[$£€¥]|\b(?:cash|prize|free|money|dollar|bonus)\b", t_str, re.IGNORECASE))
        has_url = bool(re.search(r"https?://|www\.|\.com|\.org|\.net|[a-z0-9]+@[a-z0-9]+\.[a-z]{2,}", t_str, re.IGNORECASE))
        has_phone = bool(re.search(r"\b\d{7,}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", t_str))
        has_shout = caps_ratio >= 0.12
        return {
            "excl": has_excl,
            "curr": has_curr,
            "shout": has_shout,
            "url": has_url,
            "phone": has_phone,
        }

    signals_list = [extract_signals(t) for t in clean_df[text_col]]
    signals_df = pd.DataFrame(signals_list, index=clean_df.index)
    signals_df["label"] = clean_df[label_col].str.lower()

    ham_sub = signals_df[signals_df["label"] == "ham"]
    spam_sub = signals_df[signals_df["label"] == "spam"]

    features = [
        ("Dấu chấm than (!)", "excl"),
        ("Tiền tệ / Quà tặng ($/£/win)", "curr"),
        ("Tỷ lệ chữ hoa (>12%)", "shout"),
        ("Liên kết Web / Email (URL)", "url"),
        ("Số ĐT / Chuỗi số dài", "phone"),
    ]

    labels = [f[0] for f in features]
    ham_pcts = [ham_sub[f[1]].mean() * 100.0 for f in features]
    spam_pcts = [spam_sub[f[1]].mean() * 100.0 for f in features]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    bars_ham = ax.bar(x - width / 2, ham_pcts, width, label="Ham (Email hợp lệ)", color="#4C72B0", edgecolor="black", linewidth=0.7)
    bars_spam = ax.bar(x + width / 2, spam_pcts, width, label="Spam (Email rác)", color="#C44E52", edgecolor="black", linewidth=0.7)

    for bar in bars_ham:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5)

    for bar in bars_spam:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_title("So sánh tỷ lệ xuất hiện các tín hiệu bề mặt kích động (Spam Signals) giữa Ham và Spam", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Tỷ lệ email xuất hiện đặc trưng (%)", fontsize=11)
    ax.set_ylim([0, max(max(spam_pcts), max(ham_pcts)) + 15])
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_discriminative_words(
    df: Any,
    text_col: str = "text",
    label_col: str = "label",
    top_n: int = 15,
    save_path: Optional[Path] = None,
) -> None:
    """So sánh Top từ khóa đặc trưng nhất của Spam và Ham (loại bỏ Stop words)."""
    import re
    from collections import Counter
    from src.data.preprocessing import STOP_WORDS

    if isinstance(df, list):
        df = pd.DataFrame(df)

    clean_df = df[df[label_col].astype(str).str.lower().isin(["ham", "spam"])].copy()

    def get_top_counter(subset_texts: pd.Series) -> Counter:
        cnt = Counter()
        for text in subset_texts:
            tokens = re.findall(r"\b[a-zA-Z]{3,}\b", str(text).lower())
            cnt.update(w for w in tokens if w not in STOP_WORDS)
        return cnt

    spam_counter = get_top_counter(clean_df[clean_df[label_col].str.lower() == "spam"][text_col])
    ham_counter = get_top_counter(clean_df[clean_df[label_col].str.lower() == "ham"][text_col])

    top_spam = spam_counter.most_common(top_n)
    top_ham = ham_counter.most_common(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, top_n * 0.4)))

    # 1. Top Spam
    if top_spam:
        words_s, counts_s = zip(*reversed(top_spam))
        bars_s = axes[0].barh(words_s, counts_s, color="#C44E52", edgecolor="black", linewidth=0.7)
        axes[0].set_title(f"Top {top_n} từ khóa xuất hiện nhiều nhất trong SPAM", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Tần suất xuất hiện", fontsize=11)
        for b in bars_s:
            w = b.get_width()
            axes[0].annotate(f"{w:,}", xy=(w, b.get_y() + b.get_height() / 2), xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=9.5)

    # 2. Top Ham
    if top_ham:
        words_h, counts_h = zip(*reversed(top_ham))
        bars_h = axes[1].barh(words_h, counts_h, color="#4C72B0", edgecolor="black", linewidth=0.7)
        axes[1].set_title(f"Top {top_n} từ khóa xuất hiện nhiều nhất trong HAM", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Tần suất xuất hiện", fontsize=11)
        for b in bars_h:
            w = b.get_width()
            axes[1].annotate(f"{w:,}", xy=(w, b.get_y() + b.get_height() / 2), xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=9.5)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_top_words(
    df: Any,
    text_col: str = "text",
    label_col: str = "label",
    top_n: int = 15,
    label: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> dict:
    """Đếm và vẽ biểu đồ thanh ngang Top từ khóa xuất hiện nhiều nhất (Tương thích ngược)."""
    import re
    from collections import Counter
    from src.data.preprocessing import STOP_WORDS

    if isinstance(df, list):
        df = pd.DataFrame(df)

    if label is not None:
        subset = df[df[label_col].astype(str).str.lower() == str(label).lower()]
    else:
        subset = df

    counter: Counter = Counter()
    for text in subset[text_col]:
        words = re.findall(r"\b[a-zA-Z]{2,}\b", str(text).lower())
        counter.update(w for w in words if w not in STOP_WORDS)

    most_common = counter.most_common(top_n)
    if not most_common:
        return {}

    words, counts = zip(*reversed(most_common))

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    color = "#C44E52" if label == "spam" else ("#4C72B0" if label == "ham" else "#55A868")
    bars = ax.barh(words, counts, color=color, edgecolor="black", linewidth=0.7)
    title_label = f" ({label.capitalize()})" if label else ""
    ax.set_title(f"Top {top_n} từ xuất hiện nhiều nhất{title_label}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tần suất xuất hiện", fontsize=11)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:,}", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=10)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()

    top_df = pd.DataFrame(
        counter.most_common(top_n),
        columns=["Từ vựng (Token)", "Tần suất xuất hiện"]
    )
    return top_df


def hide_index_safe(styler):
    """Ẩn cột index của pandas Styler tương thích mọi phiên bản Pandas."""
    try:
        return styler.hide(axis="index")
    except Exception:
        try:
            return styler.hide_index()
        except Exception:
            return styler


def style_table(styler, caption=None):
    """Định dạng bảng theo chuẩn học thuật chuyên nghiệp (màu xanh Navy #0b3d91)."""
    if caption:
        styler = styler.set_caption(caption)
    return styler.set_table_styles([
        {"selector": "caption", "props": [("font-size", "15px"), ("font-weight", "bold"), ("color", "#0b3d91"), ("padding", "8px")]},
        {"selector": "th", "props": [("background-color", "#0b3d91"), ("color", "white"), ("font-weight", "bold"), ("text-align", "center"), ("padding", "6px 12px"), ("border", "1px solid #d0d7de")]},
        {"selector": "td", "props": [("padding", "6px 12px"), ("border", "1px solid #e0e0e0"), ("text-align", "center")]},
    ])


def plot_feature_correlation_heatmap(
    df: Any,
    text_col: str = "text",
    label_col: str = "label",
    save_path: Optional[Path] = None,
) -> None:
    """Tính và vẽ Heatmap ma trận tương quan giữa các đặc trưng số học và nhãn Spam."""
    import re

    if isinstance(df, list):
        df = pd.DataFrame(df)

    clean_df = df[df[label_col].astype(str).str.lower().isin(["ham", "spam"])].copy()

    def extract_stats(row: pd.Series) -> dict:
        t_str = str(row[text_col] or "")
        letters = sum(1 for c in t_str if c.isalpha())
        uppers = sum(1 for c in t_str if c.isupper())
        digits = sum(1 for c in t_str if c.isdigit())
        excls = t_str.count("!")
        currs = len(re.findall(r"[$£€¥]", t_str))
        has_url = float(bool(re.search(r"https?://|www\.|\.com|\.org|\.net", t_str, re.IGNORECASE)))
        has_phone = float(bool(re.search(r"\b\d{7,}\b", t_str)))
        is_spam = 1.0 if str(row[label_col]).lower() == "spam" else 0.0

        return {
            "Độ dài ký tự": float(len(t_str)),
            "Số lượng từ": float(len(t_str.split())),
            "Tỷ lệ chữ hoa": (uppers / letters) if letters else 0.0,
            "Dấu chấm than (!)": float(excls),
            "Ký hiệu tiền tệ ($/£)": float(currs),
            "Chứa URL": has_url,
            "Chứa số ĐT/Dài": has_phone,
            "Nhãn là Spam": is_spam,
        }

    stats = [extract_stats(row) for _, row in clean_df.iterrows()]
    stat_df = pd.DataFrame(stats)
    corr_matrix = stat_df.corr(method="pearson").values
    columns = stat_df.columns.tolist()

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Hệ số tương quan Pearson (r)", fontsize=11)

    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticklabels(columns, fontsize=10, fontweight="bold")

    for i in range(len(columns)):
        for j in range(len(columns)):
            val = corr_matrix[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    ax.set_title("Ma trận tương quan giữa các đặc trưng số học và nhãn Spam", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_eda_summary(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Tạo biểu đồ tổng hợp phân bố lớp và độ dài văn bản."""
    plot_class_distribution(df, save_path=save_path)


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    model_name: str = "Classifier",
    save_path: Optional[Path] = None,
) -> None:
    """Vẽ Heatmap ma trận nhầm lẫn hiển thị số lượng và tỷ lệ %."""
    cm = np.asarray(cm, dtype=int)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Ham (0)", "Spam (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=11, fontweight="bold")

    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > thresh else "black"
            text_str = f"{cell_labels[i][j]}\n{cm[i, j]:,}\n({cm_norm[i, j]:.1f}%)"
            ax.text(j, i, text_str, ha="center", va="center", color=text_color, fontsize=12, fontweight="bold")

    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Nhãn dự đoán (Predicted)", fontsize=11, labelpad=8)
    ax.set_ylabel("Nhãn thực tế (Actual)", fontsize=11, labelpad=8)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    best_threshold: Optional[float] = None,
    model_name: str = "Classifier",
    save_path: Optional[Path] = None,
) -> None:
    """Vẽ đồ thị đôi ROC Curve và Precision-Recall Curve."""
    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_scores_arr = np.asarray(y_scores, dtype=float).ravel()

    thresholds = np.linspace(0.0, 1.0, 150)
    fpr_list, tpr_list = [], []
    precision_list, recall_list = [], []

    n_pos = max(1, int(np.sum(y_true_arr == 1)))
    n_neg = max(1, int(np.sum(y_true_arr == 0)))

    for th in thresholds:
        preds = (y_scores_arr >= th).astype(int)
        tp = int(np.sum((y_true_arr == 1) & (preds == 1)))
        fp = int(np.sum((y_true_arr == 0) & (preds == 1)))
        fn = int(np.sum((y_true_arr == 1) & (preds == 0)))
        tn = int(np.sum((y_true_arr == 0) & (preds == 0)))

        fpr_list.append(fp / n_neg)
        tpr_list.append(tp / n_pos)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / n_pos
        precision_list.append(prec)
        recall_list.append(rec)

    # Tính ROC-AUC bằng Mann-Whitney U test
    ranks = np.argsort(np.argsort(y_scores_arr)) + 1
    pos_ranks = np.sum(ranks[y_true_arr == 1])
    roc_auc = float(pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. ROC Curve
    order = np.argsort(fpr_list)
    axes[0].plot(np.array(fpr_list)[order], np.array(tpr_list)[order], color="#1F77B4", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="#7F7F7F", lw=1.5, linestyle="--", label="Random Chance")
    if best_threshold is not None:
        idx_th = np.argmin(np.abs(thresholds - best_threshold))
        axes[0].scatter([fpr_list[idx_th]], [tpr_list[idx_th]], color="#D62728", s=90, zorder=5, label=f"Ngưỡng tối ưu ({best_threshold:.2f})")
    axes[0].set_title(f"ROC Curve - {model_name}", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Recall / TPR)", fontsize=11)
    axes[0].set_xlim([-0.02, 1.02])
    axes[0].set_ylim([-0.02, 1.02])
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Precision-Recall Curve
    rec_order = np.argsort(recall_list)
    axes[1].plot(np.array(recall_list)[rec_order], np.array(precision_list)[rec_order], color="#2CA02C", lw=2, label="PR Curve")
    if best_threshold is not None:
        idx_th = np.argmin(np.abs(thresholds - best_threshold))
        axes[1].scatter([recall_list[idx_th]], [precision_list[idx_th]], color="#D62728", s=90, zorder=5, label=f"Ngưỡng tối ưu ({best_threshold:.2f})")
    axes[1].axhline(y=0.85, color="#FF7F0E", linestyle=":", lw=1.5, label="Mục tiêu Recall >= 0.85")
    axes[1].set_title(f"Precision-Recall Curve - {model_name}", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_xlim([-0.02, 1.02])
    axes[1].set_ylim([-0.02, 1.02])
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_models_comparison_bar(
    comparison_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Vẽ biểu đồ cột so sánh hiệu năng giữa các mô hình."""
    if comparison_df.empty:
        return

    df = comparison_df.copy()
    metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"] if c in df.columns]
    if not metric_cols:
        metric_cols = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc"] if c in df.columns]

    model_col = "Mô hình" if "Mô hình" in df.columns else ("model_name" if "model_name" in df.columns else df.columns[0])
    models = df[model_col].tolist()

    x = np.arange(len(models))
    width = 0.8 / len(metric_cols)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]

    for idx, metric in enumerate(metric_cols):
        values = df[metric].tolist()
        rects = ax.bar(x + idx * width - (len(metric_cols) - 1) * width / 2.0, values, width, label=metric, color=colors[idx % len(colors)], edgecolor="black", linewidth=0.6)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, rotation=45)

    ax.set_title("So sánh hiệu năng đa chiều giữa các mô hình", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylabel("Điểm số (Score)", fontsize=11)
    ax.set_ylim([0.0, 1.15])
    ax.axhline(y=0.85, color="#D62728", linestyle="--", linewidth=1.2, label="Ràng buộc Recall >= 0.85")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.7)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_outlier_length_distribution(
    df: Any,
    text_col: str = "text",
    label_col: str = "label",
    save_path: Optional[Path] = None,
) -> None:
    """Trực quan hóa phân bố độ dài văn bản và phát hiện Outlier theo nhãn Ham vs Spam (Phân vị p99)."""
    if isinstance(df, list):
        df = pd.DataFrame(df)
    work_df = df.copy()
    work_df["label_str"] = work_df[label_col].astype(str).str.lower()
    texts = work_df[text_col].fillna("").astype(str)
    work_df["raw_char_len"] = texts.str.len()
    work_df["raw_word_cnt"] = texts.apply(lambda s: len(s.split()))
    work_df["avg_word_len"] = np.where(work_df["raw_word_cnt"] > 0, work_df["raw_char_len"] / work_df["raw_word_cnt"], 0.0)

    word_p99 = float(work_df["raw_word_cnt"].quantile(0.99))
    char_p99 = float(work_df["raw_char_len"].quantile(0.99))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = {"ham": "#1B9E77", "spam": "#D95F02"}

    # 1. Histogram Word Count (clipped p99)
    for lbl, name, col in [("ham", "Ham (Hợp lệ)", colors["ham"]), ("spam", "Spam (Thư rác)", colors["spam"])]:
        sub = work_df[work_df["label_str"] == lbl]
        axes[0, 0].hist(sub["raw_word_cnt"].clip(upper=word_p99), bins=35, alpha=0.65, label=name, color=col, edgecolor="white")
    axes[0, 0].set_title(f"Phân bố số từ (Word Count) - Cắt phân vị p99 = {word_p99:.0f}", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Số lượng từ")
    axes[0, 0].set_ylabel("Tần suất")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", linestyle=":", alpha=0.5)

    # 2. Histogram Char Length (clipped p99)
    for lbl, name, col in [("ham", "Ham (Hợp lệ)", colors["ham"]), ("spam", "Spam (Thư rác)", colors["spam"])]:
        sub = work_df[work_df["label_str"] == lbl]
        axes[0, 1].hist(sub["raw_char_len"].clip(upper=char_p99), bins=35, alpha=0.65, label=name, color=col, edgecolor="white")
    axes[0, 1].set_title(f"Phân bố độ dài ký tự (Char Length) - Cắt phân vị p99 = {char_p99:.0f}", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Độ dài ký tự")
    axes[0, 1].set_ylabel("Tần suất")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.5)

    # 3. Boxplot Word Count
    ham_words = work_df[work_df["label_str"] == "ham"]["raw_word_cnt"].clip(upper=word_p99)
    spam_words = work_df[work_df["label_str"] == "spam"]["raw_word_cnt"].clip(upper=word_p99)
    bplot = axes[1, 0].boxplot([ham_words, spam_words], tick_labels=["Ham", "Spam"], patch_artist=True)
    bplot["boxes"][0].set_facecolor("#E7F1FF")
    bplot["boxes"][1].set_facecolor("#FFE8E6")
    axes[1, 0].set_title("Biểu đồ Boxplot so sánh độ phân tán số từ", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Số lượng từ")
    axes[1, 0].grid(axis="y", linestyle=":", alpha=0.5)

    # 4. Bar Chart Avg Word Length
    avg_lens = work_df.groupby("label_str")["avg_word_len"].mean().reindex(["ham", "spam"])
    axes[1, 1].bar(["Ham", "Spam"], avg_lens.values, color=[colors["ham"], colors["spam"]], edgecolor="black", width=0.45)
    for idx, v in enumerate(avg_lens.values):
        axes[1, 1].annotate(f"{v:.2f}", xy=(idx, v), xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold")
    axes[1, 1].set_title("Độ dài trung bình mỗi từ (Avg Word Length)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Ký tự / Từ")
    axes[1, 1].set_ylim([0, max(avg_lens.values) * 1.25])
    axes[1, 1].grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_threshold_sensitivity_curve(
    threshold_df: pd.DataFrame,
    recall_target: float = 0.85,
    selected_threshold: Optional[float] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Đồ thị đường cong độ nhạy ngưỡng: Precision, Recall, F1, F2 biến thiên theo Threshold."""
    if threshold_df.empty or "threshold" not in threshold_df.columns:
        return

    df = threshold_df.sort_values(by="threshold").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["threshold"], df["recall"], color="#D62728", lw=2, label="Recall (Khả năng bắt Spam)")
    ax.plot(df["threshold"], df["precision"], color="#1F77B4", lw=2, label="Precision (Độ chính xác chặn)")
    ax.plot(df["threshold"], df["f1"], color="#2CA02C", lw=2.2, linestyle="-", label="F1-Score")
    if "f2" in df.columns:
        ax.plot(df["threshold"], df["f2"], color="#9467BD", lw=1.8, linestyle="--", label="F2-Score (Ưu tiên Recall)")

    ax.axhline(y=recall_target, color="#D62728", linestyle=":", lw=1.5, label=f"Mục tiêu Recall >= {recall_target:.2f}")

    if selected_threshold is not None:
        ax.axvline(x=selected_threshold, color="#333333", linestyle="-.", lw=1.5, label=f"Ngưỡng chọn ({selected_threshold:.3f})")

    ax.set_title("Đường cong phân tích độ nhạy ngưỡng quyết định (Threshold Sensitivity Curve)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Ngưỡng quyết định (Decision Threshold $\\tau$)", fontsize=11)
    ax.set_ylabel("Điểm số (Score)", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_top_k_feature_sweep(
    sweep_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Trực quan hóa quá trình quét Top-K đặc trưng (Validation F1 vs Resource Saving %)."""
    if sweep_df.empty or "top_k" not in sweep_df.columns:
        return

    df = sweep_df.sort_values(by="top_k").reset_index(drop=True)
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    color_f1 = "#1F77B4"
    ax1.plot(df["top_k"], df["val_f1"], color=color_f1, marker="o", lw=2, label="Validation F1-Score")
    ax1.plot(df["top_k"], df["val_accuracy"], color="#2CA02C", marker="s", lw=1.8, linestyle="--", label="Validation Accuracy")
    ax1.set_xlabel("Số lượng đặc trưng được chọn (Top-K Features)", fontsize=11)
    ax1.set_ylabel("Hiệu năng mô hình (Validation F1 / Acc)", color=color_f1, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_f1)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    color_dim = "#FF7F0E"
    ax2.plot(df["top_k"], df["dimension_saving_pct"], color=color_dim, marker="^", lw=1.8, linestyle="-.", label="Tiết kiệm số chiều (%)")
    ax2.set_ylabel("Tỷ lệ giảm kích thước (%)", color=color_dim, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_dim)

    # Đánh dấu điểm tối ưu cao nhất
    if "balance_score" in df.columns:
        best_idx = int(df["balance_score"].idxmax())
        best_k = df.loc[best_idx, "top_k"]
        best_f1 = df.loc[best_idx, "val_f1"]
        ax1.scatter([best_k], [best_f1], color="#D62728", s=120, zorder=6, label=f"Top-K Tối ưu ({best_k})")

    ax1.set_title("Quét tối ưu số chiều đặc trưng Top-K (Validation Performance vs Resource Saving)", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()



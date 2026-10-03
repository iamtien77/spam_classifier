# 📓 Jupyter Notebooks Workflow

Thư mục chứa các Jupyter Notebook phục vụ nghiên cứu thử nghiệm theo chu trình chuẩn của bài toán phân loại email spam:

| # | Notebook | Nội dung chính |
|---|----------|----------------|
| **01** | `01_eda.ipynb` | Khám phá dữ liệu (EDA), phân bố Spam/Ham, ma trận tương quan |
| **02** | `02_preprocessing_and_features.ipynb` | Tiền xử lý, chia train/test, chuẩn hóa và kỹ thuật đặc trưng (PCA, interaction) |
| **03** | `03_model_baseline_and_training.ipynb` | Huấn luyện 3 mô hình cơ bản (Logistic Regression, SVM, Naive Bayes) và đánh giá metrics |
| **04** | `04_tuning_and_ensemble.ipynb` | Tối ưu siêu tham số (GridSearchCV) & Mô hình kết hợp (Random Forest, Boosting, Voting) |
| **05** | `05_error_analysis_and_deployment.ipynb` | Phân tích mẫu phân loại sai (FP/FN), lưu model và demo dự đoán email mới |

## Hướng dẫn chạy

Từ thư mục gốc của dự án:
```bash
# Kích hoạt môi trường ảo và khởi động Jupyter Notebook / JupyterLab
jupyter notebook
```
Truy cập vào thư mục `notebooks/` và chạy từng notebook theo thứ tự từ `01` đến `05`.

"""
preprocessing.py - Tiền xử lý dữ liệu
=======================================
- split_data(df): Tách features/label, chia train/test (stratified)
- scale_features(X_train, X_test): Chuẩn hóa bằng StandardScaler
- select_features(X_train, X_test, y_train): Feature Selection (SelectKBest, ANOVA F-test)
"""

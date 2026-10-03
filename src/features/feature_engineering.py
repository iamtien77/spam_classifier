"""
feature_engineering.py - Tạo và thử nghiệm features mới
========================================================
- create_interaction_features(df, feature_names): Tạo features tương tác (word_freq × char_freq)
- create_ratio_features(df, feature_names): Tạo features tỷ lệ (capital_run / word_count)
- apply_pca(X_train, X_test, n_components): Giảm chiều bằng PCA
- compare_feature_sets(X_original, X_engineered, y, cv): So sánh hiệu quả các bộ features
"""

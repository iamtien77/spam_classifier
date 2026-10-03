"""
classifiers.py - 3 mô hình phân loại chính
============================================
- get_models(): Trả về dict {name: model} cho LR, SVM, NB
- train_and_evaluate(models, X_train, X_test, y_train, y_test):
    Huấn luyện + tính metrics (accuracy, precision, recall, F1, cross-val)
"""

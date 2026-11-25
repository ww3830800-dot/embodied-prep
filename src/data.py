"""合成线性回归数据：y = X @ w + b + 噪声。

W2/W3 用同一个数据源：可控、可复现、不需要下载。
"""

import numpy as np


def make_linear_data(n: int = 512, d: int = 3, noise: float = 0.1, seed: int = 0):
    """返回 (X, y)，形状分别为 (n, d) 和 (n, 1)，float32。"""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(d, 1))
    b = rng.normal()
    X = rng.normal(size=(n, d))
    y = X @ w + b + noise * rng.normal(size=(n, 1))
    return X.astype("float32"), y.astype("float32")


def train_test_split(X, y, test_ratio: float = 0.2, seed: int = 0):
    """按比例切分，返回 (X_train, y_train, X_test, y_test)。"""
    n = len(X)
    n_test = int(n * test_ratio)
    idx = np.random.default_rng(seed).permutation(n)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
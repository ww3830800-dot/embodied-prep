import numpy as np
import torch

from src.data import make_linear_data, train_test_split
from src.model import LinearRegressor
from src.train import evaluate, train


def test_data_shapes():
    X, y = make_linear_data(n=64, d=3)
    assert X.shape == (64, 3)
    assert y.shape == (64, 1)
    assert X.dtype == np.float32


def test_split_ratio():
    X, y = make_linear_data(n=100)
    Xtr, ytr, Xte, yte = train_test_split(X, y, test_ratio=0.2)
    assert len(Xtr) == 80 and len(Xte) == 20
    assert len(ytr) == 80 and len(yte) == 20


def test_model_forward_shape():
    model = LinearRegressor(in_features=3)
    out = model(torch.zeros(5, 3))
    assert out.shape == (5, 1)


def test_training_reduces_loss():
    X, y = make_linear_data(n=256, noise=0.01)
    model = LinearRegressor(in_features=X.shape[1])
    history = train(model, X, y, epochs=100, lr=0.1, verbose_every=0)
    assert history[-1] < history[0] * 0.1, "训练 100 轮后损失应下降一个数量级"


def test_evaluate_is_finite():
    X, y = make_linear_data(n=32)
    model = LinearRegressor(in_features=X.shape[1])
    assert np.isfinite(evaluate(model, X, y))
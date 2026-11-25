"""训练循环：W3 的核心，后面所有任务都复用这个形状。

固定五步：
    1. 清梯度   optimizer.zero_grad()
    2. 前向     pred = model(x)
    3. 算损失   loss = loss_fn(pred, y)
    4. 反向     loss.backward()
    5. 更新     optimizer.step()
"""

import torch
from torch import nn


def train(model, X, y, epochs: int = 200, lr: float = 0.1, verbose_every: int = 50):
    """返回每个 epoch 的平均损失列表。"""
    X = torch.as_tensor(X)
    y = torch.as_tensor(y)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    history = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if verbose_every and epoch % verbose_every == 0:
            print(f"epoch {epoch:4d}  loss {loss.item():.6f}")

    return history


@torch.no_grad()
def evaluate(model, X, y) -> float:
    """在给定数据上算 MSE，不建计算图。"""
    pred = model(torch.as_tensor(X))
    return nn.functional.mse_loss(pred, torch.as_tensor(y)).item()
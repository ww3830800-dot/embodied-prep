"""模型定义：W3 只做最简单的线性回归。"""

from torch import nn


class LinearRegressor(nn.Module):
    """一层全连接：y_hat = x @ W^T + b。

    参数量 = d 个权重 + 1 个偏置，和真实数据生成过程同构。
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)
"""W3 入口：训练线性回归，保存损失曲线。

用法（在仓库根目录）：
    .conda\python.exe -m src.main --epochs 300 --lr 0.1
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面环境下也能存图
import matplotlib.pyplot as plt

from src.data import make_linear_data, train_test_split
from src.model import LinearRegressor
from src.train import evaluate, train


def parse_args():
    p = argparse.ArgumentParser(description="线性回归 + 训练曲线（W3）")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="figs/loss_curve.png")
    return p.parse_args()


def main():
    args = parse_args()

    X, y = make_linear_data(seed=args.seed)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=args.seed)

    model = LinearRegressor(in_features=X.shape[1])
    history = train(model, Xtr, ytr, epochs=args.epochs, lr=args.lr)

    train_loss = evaluate(model, Xtr, ytr)
    test_loss = evaluate(model, Xte, yte)

    print(f"\nfinal train MSE: {train_loss:.6f}")
    print(f"final test  MSE: {test_loss:.6f}")
    print(f"first/last loss: {history[0]:.4f} -> {history[-1]:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history, linewidth=1.6, color="#2f9e44")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (log scale)")
    ax.set_title(f"Linear regression: loss {history[0]:.3f} -> {history[-1]:.4f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved curve -> {out}")


if __name__ == "__main__":
    main()
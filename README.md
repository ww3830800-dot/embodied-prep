# embodied-prep

具身智能方向 16 周准备仓库（2026-11-09 → 2027-02-28）。

目标不是“学完课程”，而是留下**能被别人复现的交付物**：公开仓库、可重建环境、可读的训练代码、可核对的指标。

## 当前阶段

| 周 | 日期 | 主题 | 交付 |
|---|---|---|---|
| W1 | 11.09–11.15 | Git 与环境 | 公开仓库 + requirements 可重建 |
| W2 | 11.16–11.22 | Python 工程化 | 模块化项目 + pytest 跑通 |
| W3–W4 | 11.23–12.06 | PyTorch | 线性回归收敛、MNIST > 97% |
| W5–W8 | 12.07–01.03 | CNN / 序列 / Transformer | CIFAR-10 > 70%、手写注意力、DP 笔记 |
| W9–W14 | 01.04–02.14 | 模仿学习 | CartPole BC、Lift BC > 50%、Diffusion Policy |
| W15–W16 | 02.15–02.28 | 复现与收尾 | 主指标差 < 15% + 3 页报告 |

Gate：12-06（MNIST 97% + 裸写训练循环）、01-03（讲清策略为何用序列/生成模型）、01-31（Lift BC > 50%）、02-28（可复现 + 独立训练 + 复现论文）。

## 环境重建

需要 Python 3.10（MuJoCo / robosuite 生态的兼容基线，不要用 3.14）。

```bash
conda create -n embodied python=3.10
conda activate embodied
pip install -r requirements.txt
```

本机开发时环境放在仓库内的 `.conda/`（已 gitignore），所以也可以：

```bash
conda create -p .conda python=3.10 -c conda-forge
conda activate ./ .conda
pip install -r requirements.txt
```

验证：

```bash
python -m pytest -q
```

## 目录

```
embodied-prep/
  src/                 # 训练代码（W2 起模块化）
  tests/               # pytest
  notes/               # 阅读与推导笔记
  requirements.txt
  .gitignore
```

## 进度

- [x] W1-1 conda 环境（Python 3.10）
- [ ] W1-2 仓库公开并 push
- [x] W1-3 .gitignore
- [x] W1-4 requirements.txt 可重建
- [ ] W1-5 Pro Git 1–3 章 + 笔记
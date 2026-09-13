"""PyTorch 多层感知机：sklearn digits 手写数字分类（与经典 ML 对照）。

实验设计（延续 ml-bench/01 的口径）：
- 数据：sklearn digits（1797 × 64 像素，10 类），25% 分层留出集；
- 模型：MLP 64→128→64→10（ReLU + Dropout 0.2），Adam，CPU 数十秒；
- 对照：ml-bench/01 实测 SVM-RBF 交叉验证 98.25%、留出集 98.0%；
- 结论预注册：小尺寸表格/像素数据上，深度模型没有免费午餐——
  结果是多少就写多少（训练曲线与混淆矩阵全部入库）。

运行: python train_nn.py
产出: models/digit_mlp.pt, reports/metrics.json, charts/*.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

from datap.plotstyle import save_chart, setup_style

ROOT = Path(__file__).resolve().parent
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


class MLP(nn.Module):
    def __init__(self, in_dim: int = 64, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes))

    def forward(self, x):
        return self.net(x)


def main() -> None:
    setup_style()
    data = load_digits()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.25, stratify=data.target,
        random_state=SEED)
    # 标准化用训练集统计量（防泄漏口径与 ml-bench/01 一致）
    mu, sigma = X_tr.mean(0), X_tr.std(0)
    norm = lambda a: (a - mu) / np.maximum(sigma, 1e-8)
    X_tr_n, X_te_n = norm(X_tr), norm(X_te)

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    Xtr_t, ytr_t = to_t(X_tr_n), torch.tensor(y_tr.values if hasattr(y_tr, "values") else y_tr)
    Xte_t, yte_t = to_t(X_te_n), torch.tensor(y_te.values if hasattr(y_te, "values") else y_te)

    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    model.train()
    for epoch in range(1, 81):
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
        history.append(float(loss))
        if epoch % 20 == 0:
            with torch.no_grad():
                acc = (model(Xte_t).argmax(1) == yte_t).float().mean().item()
            print(f"epoch {epoch:>3}: loss={loss:.4f} 留出集 acc={acc:.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).numpy()
    acc = float((pred == y_te).mean())
    print(f"\n最终留出集准确率: {acc:.4f}（对照：ml-bench/01 SVM-RBF CV 98.25%）")
    print(classification_report(y_te, pred, digits=3))

    # —— 产物 ——
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "charts").mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sigma": sigma},
               ROOT / "models" / "digit_mlp.pt")
    (ROOT / "reports" / "metrics.json").write_text(
        json.dumps({"holdout_accuracy": round(acc, 4),
                    "svm_reference_cv": 0.9825,
                    "epochs": len(history),
                    "final_loss": round(history[-1], 4)},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # 训练曲线
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history, color="#4C72B0")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("训练损失曲线（固定随机种子，可复现）")
    save_chart(fig, ROOT / "charts" / "training_curve.png")

    # 混淆矩阵
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax, cmap="Blues",
                                            colorbar=False)
    ax.set_title(f"MLP 留出集混淆矩阵（acc={acc:.4f}；SVM-RBF 对照 98.0%）")
    save_chart(fig, ROOT / "charts" / "confusion.png")


if __name__ == "__main__":
    import json

    main()

"""训练与评估：防泄漏主实验 vs 泄漏对照实验，模型对比与可解释性。

用法: python train.py
产出: models/model.joblib, reports/metrics.json, charts/*.png, reports/report.md
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import FEATURES, FEATURES_LEAK, LABEL, build_dataset
from datap.plotstyle import save_chart, setup_style

ROOT = Path(__file__).resolve().parent
SEED = 42


def make_model(kind: str) -> Pipeline:
    if kind == "logreg":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, C=1.0))])
    return Pipeline([("clf", RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, random_state=SEED))])


def cv_evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    f1 = cross_val_score(model, X, y, cv=skf, scoring="f1_macro")
    return {"cv_accuracy_mean": round(float(acc.mean()), 4),
            "cv_accuracy_std": round(float(acc.std()), 4),
            "cv_f1_macro_mean": round(float(f1.mean()), 4)}


def leakage_experiment(df: pd.DataFrame) -> dict:
    """同一模型，分别用防泄漏特征与含 GDP 特征做 5 折交叉验证对比。"""
    y = df[LABEL]
    out = {}
    for name, feats in (("防泄漏特征(无GDP)", FEATURES),
                        ("泄漏对照(含人均GDP)", FEATURES_LEAK)):
        model = make_model("logreg")
        scores = cv_evaluate(model, df[feats], y)
        out[name] = scores
    return out


def main() -> None:
    setup_style()
    df = build_dataset()
    y = df[LABEL]
    X = df[FEATURES]

    # —— 模型对比（防泄漏特征，5 折分层交叉验证）——
    print("=== 交叉验证（防泄漏特征）===")
    metrics: dict = {"n_countries": len(df), "features": FEATURES, "label": LABEL,
                     "class_distribution": y.value_counts().to_dict()}
    best_name, best_model, best_f1 = None, None, -1.0
    for name in ("logreg", "rf"):
        model = make_model(name)
        scores = cv_evaluate(model, X, y)
        metrics[f"{name}_cv"] = scores
        print(f"  {name}: acc={scores['cv_accuracy_mean']}±{scores['cv_accuracy_std']} "
              f"f1_macro={scores['cv_f1_macro_mean']}")
        if scores["cv_f1_macro_mean"] > best_f1:
            best_name, best_model, best_f1 = name, model, scores["cv_f1_macro_mean"]

    # —— 全量重训练（演示用途；生产应留独立测试集）——
    best_model.fit(X, y)
    pred = best_model.predict(X)

    # —— 泄漏对照实验 ——
    print("\n=== 标签泄漏对照实验 ===")
    leak = leakage_experiment(df)
    metrics["leakage_experiment"] = leak
    for k, v in leak.items():
        print(f"  {k}: acc={v['cv_accuracy_mean']}")

    # —— 可解释性：置换重要性（在训练集上，演示用途）——
    imp = permutation_importance(best_model, X, y, n_repeats=20, random_state=SEED)
    importance = sorted(zip(FEATURES, imp.importances_mean), key=lambda t: -t[1])
    metrics["permutation_importance"] = {k: round(float(v), 4) for k, v in importance}

    # —— 产物 ——
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "charts").mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "features": FEATURES,
                 "label": LABEL, "kind": best_name}, ROOT / "models" / "model.joblib")
    (ROOT / "reports" / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # 图 1：混淆矩阵
    labels = sorted(y.unique())
    cm = confusion_matrix(y, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("预测")
    ax.set_ylabel("实际")
    ax.set_title(f"混淆矩阵（{best_name}，训练集内评估）")
    fig.colorbar(im, ax=ax)
    save_chart(fig, ROOT / "charts" / "confusion_matrix.png")

    # 图 2：特征重要性
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    names = [k for k, _ in importance]
    vals = [v for _, v in importance]
    ax.barh(names[::-1], vals[::-1], color="#4C72B0", alpha=0.9)
    ax.set_xlabel("置换重要性（准确率下降幅度）")
    ax.set_title("哪些社会发展特征最能指示收入等级")
    save_chart(fig, ROOT / "charts" / "importance.png")

    # 图 3：泄漏对照
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names_l = list(leak)
    vals_l = [leak[k]["cv_accuracy_mean"] for k in names_l]
    bars = ax.bar(names_l, vals_l, color=["#55A868", "#C44E52"], alpha=0.9)
    for b, v in zip(bars, vals_l):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("5 折交叉验证准确率")
    ax.set_title("标签泄漏对照：加入人均GDP 后指标「虚高」——答案被喂进了特征")
    save_chart(fig, ROOT / "charts" / "leakage.png")

    # 分类报告
    report_text = classification_report(y, pred, digits=3)
    (ROOT / "reports" / "classification_report.txt").write_text(
        report_text, encoding="utf-8")

    print("\n=== 训练集内分类报告（最佳模型：%s）===" % best_name)
    print(report_text)
    print(f"产物: models/model.joblib, reports/metrics.json, charts/*.png")


if __name__ == "__main__":
    main()

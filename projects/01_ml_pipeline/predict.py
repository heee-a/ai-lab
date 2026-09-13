"""用训练好的模型预测单个国家的收入等级。

用法:
    python predict.py --life-exp 74.2 --co2-pc 6.7 --urban 64.0
    python predict.py --country JPN   # 从数据集中查已知国家演示
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description="收入等级预测（防泄漏模型）")
    ap.add_argument("--life-exp", type=float, help="预期寿命（岁）")
    ap.add_argument("--co2-pc", type=float, help="人均CO2（吨）")
    ap.add_argument("--urban", type=float, help="城市化率（%%）")
    ap.add_argument("--country", help="或直接指定数据集中的国家 ISO3 代码演示")
    args = ap.parse_args()

    bundle = joblib.load(ROOT / "models" / "model.joblib")
    model, features = bundle["model"], bundle["features"]

    if args.country:
        df = pd.read_csv(ROOT / "data" / "dataset.csv")
        row = df[df["country_id"] == args.country.upper()]
        if row.empty:
            sys.exit(f"数据集中没有 {args.country.upper()}")
        x = row.iloc[0][features]
        actual = row.iloc[0][bundle["label"]]
    else:
        if None in (args.life_exp, args.co2_pc, args.urban):
            sys.exit("请提供 --life-exp --co2-pc --urban 或使用 --country")
        x = pd.Series({"life_exp": args.life_exp, "co2_pc": args.co2_pc,
                       "urban": args.urban})
        actual = "(未提供)"

    proba = model.predict_proba(x.to_frame().T)[0]
    pred = model.classes_[int(proba.argmax())]
    print(f"特征: {dict(zip(features, x.astype(float).round(2).to_dict().values()))}")
    print(f"实际等级: {actual}")
    print(f"预测等级: {pred}")
    print("概率分布:")
    for cls, p in sorted(zip(model.classes_, proba), key=lambda t: -t[1]):
        bar = "█" * int(p * 30)
        print(f"  {cls:<22}{p:6.1%} {bar}")


if __name__ == "__main__":
    main()

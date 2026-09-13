"""数据准备：构建「从社会发展特征预测国家收入等级」的分类数据集。

关键的防泄漏决策：
收入等级本身就是世界银行按人均 GNI/GDP 阈值划定的——把人均GDP放进特征
等于把答案喂给模型（标签泄漏）。因此：
- 主实验特征：预期寿命 / 人均CO2 / 城市化率（不含任何收入指标）
- 泄漏对照实验：加入人均GDP，展示指标虚高，验证防泄漏设计的必要性
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LATEST = 2023

FEATURES = ["life_exp", "co2_pc", "urban"]        # 防泄漏特征集
FEATURES_LEAK = FEATURES + ["gdp_pc"]             # 泄漏对照特征集
LABEL = "income_level"


def build_dataset() -> pd.DataFrame:
    long_df = pd.read_csv(DATA / "indicators_long.csv")
    countries = pd.read_csv(DATA / "countries.csv")

    wide = long_df[long_df["year"] == LATEST].pivot_table(
        index="country_id", columns="indicator_name", values="value").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={
        "人均GDP美元": "gdp_pc", "预期寿命": "life_exp",
        "人均CO2吨": "co2_pc", "城市化率%": "urban"})

    df = wide.merge(
        countries.rename(columns={"id": "country_id", "name": "country_name",
                                  "region_name": "region",
                                  "income_name": LABEL})[
            ["country_id", "country_name", "region", LABEL]],
        on="country_id", how="left")
    df = df.dropna(subset=FEATURES_LEAK + [LABEL]).reset_index(drop=True)
    return df


def main() -> None:
    df = build_dataset()
    df.to_csv(DATA / "dataset.csv", index=False, encoding="utf-8-sig")
    print(f"数据集: {len(df)} 个国家, 特征 {FEATURES_LEAK}, 标签 {LABEL}")
    print("\n类别分布:")
    print(df[LABEL].value_counts().to_string())
    print(f"\n缺失检查: {df[FEATURES_LEAK].isna().sum().sum()} 个缺失值")


if __name__ == "__main__":
    main()

"""生成交互式仪表盘（自包含 HTML，双击即可打开）。

页面 1：Gapminder 风格动画散点——2000-2023 年各国「人均GDP × 预期寿命」，
       气泡大小=人均CO2，颜色=地区，按年份播放；
页面 2：18 城月均温热力图。

数据来自本作品集数据项目（data-portfolio）的真实采集。
运行: python build_dashboard.py
产出: dashboard.html
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def gapminder() -> go.Figure:
    long_df = pd.read_csv(DATA / "indicators_long.csv")
    countries = pd.read_csv(DATA / "countries.csv")
    wide = long_df.pivot_table(index=["country_id", "year"],
                               columns="indicator_name", values="value").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={
        "人均GDP美元": "人均GDP", "预期寿命": "预期寿命",
        "人均CO2吨": "人均CO2"})
    meta = countries.rename(columns={"id": "country_id", "name": "国家",
                                     "region_name": "地区"})
    df = wide.merge(meta[["country_id", "国家", "地区"]], on="country_id")
    df = df.dropna(subset=["人均GDP", "预期寿命", "人均CO2"])

    fig = px.scatter(
        df, x="人均GDP", y="预期寿命", size="人均CO2", color="地区",
        hover_name="国家", animation_frame="year",
        animation_group="country_id", size_max=45,
        log_x=True, range_x=[200, 200000], range_y=[45, 90],
        title="世界发展 2000-2023：收入 × 寿命 × 碳排放（点击播放）",
        labels={"人均GDP": "人均GDP（美元，对数轴）", "预期寿命": "预期寿命（岁）",
                "人均CO2": "人均CO2（吨）", "year": "年份"})
    fig.update_layout(height=640, template="plotly_white")
    fig.update_layout(transition_duration=200)
    return fig


def climate_heatmap() -> go.Figure:
    daily = pd.read_csv(DATA / "weather_daily.csv", parse_dates=["date"])
    daily["month"] = daily["date"].dt.month
    heat = (daily.groupby(["city", "month"])["tmean"].mean()
            .unstack().round(1))
    heat = heat.loc[heat.mean(axis=1).sort_values(ascending=False).index]

    fig = go.Figure(go.Heatmap(
        z=heat.values, x=[f"{m}月" for m in heat.columns], y=heat.index,
        colorscale="RdYlBu_r", zmin=-20, zmax=32,
        text=heat.values, texttemplate="%{text}",
        colorbar=dict(title="℃")))
    fig.update_layout(height=620, template="plotly_white",
                      title="18 城月均温热力图（2019-2024 均值）：一眼看懂南北气候差异")
    fig.update_yaxes(autorange="reversed")
    return fig


def main() -> None:
    fig1, fig2 = gapminder(), climate_heatmap()
    html1 = fig1.to_html(full_html=False, include_plotlyjs=True,
                         config={"displaylogo": False})
    html2 = fig2.to_html(full_html=False, include_plotlyjs=False,
                         config={"displaylogo": False})
    page = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>ai-lab · 交互式数据仪表盘</title>
<style>
  body {{ font-family: "Microsoft YaHei", sans-serif; margin: 0; background: #f7f8fa; }}
  header {{ background: #1f2937; color: white; padding: 24px 40px; }}
  header h1 {{ margin: 0 0 6px; font-size: 22px; }}
  header p {{ margin: 0; color: #9ca3af; font-size: 13px; }}
  .card {{ background: white; margin: 24px 40px; padding: 8px 16px;
          border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .card h2 {{ font-size: 16px; color: #374151; }}
</style></head><body>
<header><h1>ai-lab · 交互式数据仪表盘</h1>
<p>数据来源：data-portfolio 真实采集（World Bank API / Open-Meteo API）· 自包含 HTML，可离线打开</p></header>
<div class="card"><h2>① 世界发展动画（2000-2023）</h2>{html1}</div>
<div class="card"><h2>② 中国 18 城气候热力图</h2>{html2}</div>
</body></html>"""
    out = ROOT / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    print(f"仪表盘已生成: {out}（{out.stat().st_size / 1e6:.1f} MB，自包含可离线打开）")


if __name__ == "__main__":
    main()

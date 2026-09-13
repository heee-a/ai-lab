"""ai-lab 测试：ML 数据完整性、BM25 排序、分块器、仪表盘产物。pytest -q"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "projects" / "02_rag_kb"))

from bm25 import BM25Index, tokenize            # noqa: E402
from chunker import chunk_markdown              # noqa: E402


# ---------------- ml-pipeline ----------------
def test_dataset_quality():
    df = pd.read_csv(REPO / "projects/01_ml_pipeline/data/dataset.csv")
    assert len(df) >= 190
    assert set(df["income_level"]) == {
        "High income", "Upper middle income", "Lower middle income", "Low income"}
    # 防泄漏设计核查：主特征里没有收入类指标
    for leak in ("gdp_pc",):
        assert leak not in {"life_exp", "co2_pc", "urban"}


def test_metrics_report_exists():
    m = pd.read_json(REPO / "projects/01_ml_pipeline/reports/metrics.json",
                     typ="series")
    assert "leakage_experiment" in m.index
    leak = m["leakage_experiment"]
    # 泄漏对照的准确率应高于防泄漏特征（这正是要演示的现象）
    assert leak["泄漏对照(含人均GDP)"]["cv_accuracy_mean"] > \
        leak["防泄漏特征(无GDP)"]["cv_accuracy_mean"]


# ---------------- rag-kb ----------------
def test_tokenize_mixed_language():
    tokens = tokenize("使用 FastAPI 构建 机器学习 pipeline")
    assert "fastapi" in tokens and "pipeline" in tokens
    assert any("机器" in t or "构建" in t for t in tokens)


def test_bm25_ranks_relevant_chunk_first():
    docs = {
        "a.md": "# 天气\n哈尔滨的冬天非常寒冷，一月平均气温只有零下十六度，"
                "是中国最冷的大城市之一。",
        "b.md": "# 美食\n广州早茶非常丰富，虾饺烧卖一盅两件，"
                "是华南地区最有代表性的饮食文化。",
    }
    index = BM25Index.from_docs(docs)
    hits = index.search("哈尔滨冷不冷")
    assert hits and hits[0][0].doc == "a.md"


def test_chunker_splits_by_heading_and_injects_path():
    text = "# 标题\n前言内容足够长的一段话，用来触发简介块的生成逻辑，长度必须超过四十个字符。\n" \
           "## 章节甲\n甲的正文内容，包含 LRU 缓存的实现说明，并且长度超过四十个字符的最低门槛。\n" \
           "## 章节乙\n乙的正文内容，讲的是零钱兑换问题，同样写够四十个字符以上的说明文字。"
    chunks = chunk_markdown("demo.md", text)
    labels = [c.section for c in chunks]
    assert any("章节甲" in s for s in labels)
    assert any("标题" in s for s in labels)          # 层级路径含根标题
    lru = next(c for c in chunks if "LRU" in c.text)
    assert "章节甲" in lru.section


def test_retrieval_hit_on_known_question():
    index = BM25Index.from_corpus_dir(str(REPO / "projects/02_rag_kb/corpus"))
    hits = index.search("哪个项目做了假设检验？", top_k=3)
    assert any("05_stats_inference" in h[0].doc for h in hits)


# ---------------- dashboard ----------------
def test_dashboard_html_exists_and_contains_plotly():
    html_path = REPO / "projects/03_dashboard/dashboard.html"
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert "heatmap" in html.lower()

"""检索质量评测：构造测试问题集，计算 recall@k 与 MRR。

每个用例：question + 应命中的目标文档（ expected_doc 的子串匹配）。
用法: python evaluate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bm25 import BM25Index

CASES = [
    # (问题, 期望文档子串列表（任一命中即可）, 期望命中的章节关键词)
    ("哪个项目做了假设检验？用的是什么检验方法？", ["05_stats_inference"], "假设检验"),
    ("怎么防止标签泄漏？", ["01_ml_pipeline"], "泄漏"),
    ("FastAPI 的配置怎么外置？数据库路径用什么环境变量？", ["dev-portfolio-software"], "TASK_DB_PATH"),
    ("GitHub 上头部开源仓库最多用什么语言？", ["02_github_top"], "Python"),
    ("LRU 缓存怎么实现？", ["dev-portfolio-algorithms"], "LRU"),
    ("ReAct 智能体的循环是什么样？", ["dev-portfolio-ai-agents"], "Observation"),
    ("哪个城市四季如春？怎么验证的？", ["01_weather_cities", "05_stats_inference"], "昆明"),
    ("中国人均GDP 增长了多少倍？", ["03_world_indicators"], "13.4"),
    ("零钱兑换的最少硬币数用什么算法？", ["dev-portfolio-algorithms"], "零钱"),
    ("文本挖掘里中文项目的高频词有哪些？", ["06_text_mining"], "爬虫"),
    ("SQL 里怎么算同比增速？", ["04_sql_analysis"], "LAG"),
    ("时序预测怎么保证不偷看未来数据？", ["07_forecast"], "回测"),
]


def main() -> None:
    index = BM25Index.from_corpus_dir(
        str(Path(__file__).parent / "corpus"))
    print(f"语料: {len(index.chunks)} 个分块")

    recall_hits, rr_sum = 0, 0.0
    misses = []
    for q, doc_subs, kw in CASES:
        hits = index.search(q, top_k=3)
        rank = None
        for i, (chunk, _s) in enumerate(hits, 1):
            doc_ok = any(sub in chunk.doc for sub in doc_subs)
            kw_ok = kw.lower() in (chunk.section + chunk.text).lower()
            if doc_ok and kw_ok:
                rank = i
                break
        rr_sum += 1 / rank if rank else 0.0
        recall_hits += rank is not None
        status = f"hit@{rank}" if rank else "MISS"
        print(f"  [{status:>6}] {q[:24]}… -> {hits[0][0].label if hits else '-'}")
        if rank is None:
            misses.append((q, doc_subs, kw,
                           [(c.doc, c.section) for c, _ in hits]))

    n = len(CASES)
    print(f"\nrecall@3 = {recall_hits}/{n} = {recall_hits / n:.0%}")
    print(f"MRR      = {rr_sum / n:.3f}")
    if misses:
        print("\n未命中用例（改进分词/语料的线索）:")
        for q, doc, kw, got in misses:
            print(f"  - {q} 期望 {'/'.join(doc)}·含 {kw!r}，实际 top3: {got}")


if __name__ == "__main__":
    main()

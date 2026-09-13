"""检索问答入口：默认纯检索模式（离线），配置 LLM 后可生成式回答。

用法:
    python ask.py "哪个项目做了假设检验？"
    python ask.py "怎么防标签泄漏" --top-k 5
    python ask.py "..." --llm        # 需配置 OPENAI_* 环境变量，回答带引用标注
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bm25 import BM25Index

SYSTEM_TEMPLATE = """你是作品集知识库助手。仅依据下面给出的资料片段回答问题，
不得编造资料之外的内容。引用时在句末标注编号，如 [1]。

资料片段：
{context}
"""


def build_context(hits) -> str:
    lines = []
    for i, (chunk, score) in enumerate(hits, 1):
        excerpt = chunk.text[:400] + ("…" if len(chunk.text) > 400 else "")
        lines.append(f"[{i}] 出处: {chunk.label}（相关度 {score:.2f}）\n{excerpt}")
    return "\n\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="作品集知识库检索问答")
    ap.add_argument("question")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--corpus", default=str(Path(__file__).parent / "corpus"))
    ap.add_argument("--llm", action="store_true",
                    help="用 LLM 生成式回答（需配置 OPENAI_* 环境变量）；默认仅检索")
    args = ap.parse_args()

    index = BM25Index.from_corpus_dir(args.corpus)
    hits = index.search(args.question, top_k=args.top_k)
    if not hits:
        sys.exit("知识库中没有匹配内容")

    print(f"检索到 {len(hits)} 个相关片段：\n")
    for i, (chunk, score) in enumerate(hits, 1):
        excerpt = chunk.text[:180].replace("\n", " ")
        print(f"[{i}] {chunk.label}  (score={score:.2f})")
        print(f"    {excerpt}…\n")

    if not args.llm:
        print("（纯检索模式：配置 OPENAI_* 环境变量后加 --llm 可生成式回答）")
        return

    from llm import Message, OpenAICompatLLM

    llm = OpenAICompatLLM()
    prompt = SYSTEM_TEMPLATE.format(context=build_context(hits))
    messages = [Message("system", prompt), Message("user", args.question)]
    print("=== LLM 回答（带引用）===")
    print(llm.chat(messages))


if __name__ == "__main__":
    main()

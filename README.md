# ai-lab · AI 与机器学习实验室

[![CI](https://github.com/heee-a/ai-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/heee-a/ai-lab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

三个进阶项目：机器学习端到端（含标签泄漏对照实验）、检索增强问答（自实现
BM25 + 检索评测）、交互式仪表盘（自包含 HTML）。数据全部来自本账号
[data-portfolio](https://github.com/heee-a/data-portfolio) 的真实采集。

## 项目一览

### 🎯 [01 ml-pipeline 机器学习端到端](projects/01_ml_pipeline/)
**任务**：不看收入指标，用社会发展特征预测 192 国的世界银行收入等级（4 分类）。
- **标签泄漏对照实验**：收入等级本就由人均收入阈值划定——加入人均GDP 后
  交叉验证准确率虚高 10 个百分点，实验图直接展示"答案被喂进特征"；
- 随机森林 vs 逻辑回归、5 折分层交叉验证、置换重要性、混淆矩阵；
- 预测 CLI 输出文字版概率分布条形图。
→ [README](projects/01_ml_pipeline/README.md)

### 🔎 [02 rag-kb 检索增强问答](projects/02_rag_kb/)
**任务**：对本账号 5 个仓库的 13 篇 README 建知识库，回答"哪个项目做了假设检验？"。
- **BM25 从零实现**（Okapi 公式），jieba 中文分词 + 标题命中加权；
- 多级标题分块器，层级路径并入正文；
- **检索评测**：12 条测试用例，recall@3 = 83%、MRR = 0.75（含未命中归因）；
- 纯检索模式完全离线；配置 LLM 后生成带引用标注的回答。
→ [README](projects/02_rag_kb/README.md)

### 🧠 [04 digit_nn 深度学习对照实验](projects/04_digit_nn/)
**任务**：PyTorch MLP 分类手写数字，对照 ml-bench 的 SVM-RBF（98.0%）。
- 诚实负结果：**MLP 94.4% < SVM 98.25%**——1797 个样本喂不饱深度网络；
- 训练曲线/混淆矩阵/固定种子全可复现；
- 改进方向（增广/卷积/预训练）如实列出而未夸口。
→ [README](projects/04_digit_nn/README.md)

### 📊 [03 dashboard 交互式仪表盘](projects/03_dashboard/)
**任务**：把作品集数据变成"看起来高级"的可视化。
- Gapminder 风格动画散点：2000-2023 收入 × 寿命 × 碳排放，点击播放 24 年演变；
- 18 城月均温热力图；
- plotly.js 内联的**自包含单文件 HTML**（4.7MB），离线双击即开。
→ [README](projects/03_dashboard/README.md) · 产物 [dashboard.html](projects/03_dashboard/dashboard.html)

## 快速开始

```bash
pip install -e .
cd projects/01_ml_pipeline && python train.py && python predict.py --country CHN
cd ../02_rag_kb && python ask.py "哪个项目做了假设检验？" && python evaluate.py
cd ../03_dashboard && python build_dashboard.py   # 打开 dashboard.html
```

## 声明

数据来自公开 API 的真实采集（World Bank / Open-Meteo，见 data-portfolio 的
采集脚本与限速策略）；模型/检索结论仅供学习演示。
License: [MIT](LICENSE)

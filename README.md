# RAG-query-rewriting — 语音助手 Query 改写系统

> 面向 AI 语音助手场景的分层 Query 改写工程化方案  
> 覆盖 **通讯 / 闲聊 / 知识问答 / AI搜索** 四大核心领域  
> 低延迟设计：规则引擎 P50 < 1ms，小模型 ONNX P50 ~15ms

---

## 📐 系统架构

```
用户输入 (口语/省略/指代)
    │
    ▼
[ASR噪声清洗] ──────────────────────── <0.1ms
    │
    ▼
[领域路由] (通讯/闲聊/知识/AI搜索)
    │
    ▼
[规则引擎] ──── 命中 → 返回 ─────────── P50 ~0.5ms
    │ 未命中
    ▼
[mT5-small ONNX/INT8] ── 返回 ──────── P50 ~15ms
    │ 失败/低置信
    ▼
[Doubao-Seed-2.0-lite 兜底] ─ 返回 ─── P50 ~300ms (可选)
```

---

## 📁 目录结构

```
RAG-query-rewriting/
├── src/
│   ├── llm/
│   │   └── doubao_client.py        # 火山方舟 SDK 统一封装
│   ├── rewriter/
│   │   ├── hybrid_rewriter.py      # 分层改写器（核心入口）
│   │   └── model_rewriter.py       # mT5-small ONNX/INT8 推理
│   ├── rules/
│   │   └── domain_rules_engine.py  # 四大领域规则库（可热插拔）
│   └── utils/
│       └── context_manager.py      # 对话上下文管理与槽位抽取
├── datasets/
│   ├── build_domain_dataset.py     # 领域测试集自动生成（Doubao）
│   └── domain/                     # 生成的 JSONL 测试集
├── evaluation/
│   ├── eval_pipeline.py            # 三层评测流水线
│   └── run_benchmark.py            # 一键评测 / A/B 对比
├── .env.example                    # 环境变量配置模板
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

---

## 🚀 快速上手

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 ARK_API_KEY
```

### 3. 基本使用

```python
from src.rewriter.hybrid_rewriter import HybridRewriter
from src.utils.context_manager import ContextManager

rewriter = HybridRewriter(use_model=False)   # 纯规则模式，极低延迟

ctx = ContextManager()
ctx.add_turn("user", "帮我联系张三")
ctx.add_turn("assistant", "好的，正在拨打")

result = rewriter.rewrite("给他发条短信", context_manager=ctx)
print(result.rewritten_query)   # → "给张三发短信"
print(f"延迟: {result.latency_ms:.2f}ms  方法: {result.method}")
```

### 4. 生成测试集

```bash
# 需要配置 ARK_API_KEY
python datasets/build_domain_dataset.py \
    --domains telecom chitchat knowledge ai_search \
    --n 50 \
    --output_dir datasets/domain
```

### 5. 运行评测

```bash
# 单版本评测（纯规则）
python evaluation/run_benchmark.py eval \
    --dataset datasets/domain/all_domains_test.jsonl \
    --version v1.0 \
    --no_llm_score

# A/B 对比（规则 vs 规则+小模型）
python evaluation/run_benchmark.py ab \
    --dataset datasets/domain/all_domains_test.jsonl \
    --output_dir evaluation/reports
```

---

## 📊 评测体系

### 三层评测指标

| 层级 | 指标 | 说明 |
|------|------|------|
| Layer 1 文本质量 | ROUGE-L | 与人工改写的最长公共子序列 |
| | BLEU-2 | N-gram 精确率 |
| | BERTScore F1 | 语义相似度（中文） |
| Layer 2 延迟 | P50/P95/P99 | 各百分位推理延迟 |
| Layer 3 语义评分 | LLM Score 1-5 | Doubao 可解释语义打分 |

### 领域覆盖

| 领域 | 改写场景 | 典型规则 |
|------|----------|----------|
| 通讯 | 指代消解/省略追问 | "给他打电话" → "给张三打电话" |
| 闲聊 | 口语规范/话题续接 | "还有呢?" → "关于AI还有什么?" |
| 知识 | 指代/追问/歧义 | "那它的历史呢?" → "Python的历史是什么?" |
| AI搜索 | 省略/实体扩展 | "帮我查一下苹果" → "搜索: 苹果公司股价" |

---

## 🔧 规则热插拔

```python
from src.rules.domain_rules_engine import DomainRulesEngine, RewriteRule
import re

engine = DomainRulesEngine()

# 添加自定义规则
engine.add_rule(RewriteRule(
    name="my_custom_rule",
    pattern=re.compile(r"^(放首歌)$"),
    rewrite_fn=lambda m, ctx: f"播放音乐",
    priority=1,
    domain="general",
))

# 查看所有规则
rules = engine.list_rules(domain="telecom")
```

---

## 📚 学术参考

本项目基于以下工作改造：

- **Rewrite-Retrieve-Read** (EMNLP 2023): [arxiv.org/abs/2305.14283](https://arxiv.org/abs/2305.14283)
- **AdaQR** (EMNLP 2024): [aclanthology.org/2024.emnlp-main.746](https://aclanthology.org/2024.emnlp-main.746/)
- **RaFe** (Alibaba, 2024): [arxiv.org/abs/2405.14431](https://arxiv.org/abs/2405.14431)
- **CANARD** 数据集: [github.com/yubowen-ph/Canard](https://github.com/yubowen-ph/Canard)
- **QReCC** 数据集: [github.com/apple/ml-qrecc](https://github.com/apple/ml-qrecc)

---

## 📝 原始论文

Paper: Query Rewriting in Retrieval-Augmented Large Language Models [[pdf]](https://arxiv.org/abs/2305.14283)

```bibtex
@inproceedings{ma-etal-2023-query,
    title = "Query Rewriting in Retrieval-Augmented Large Language Models",
    author = "Ma, Xinbei  and Gong, Yeyun  and He, Pengcheng  and Zhao, Hai  and Duan, Nan",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    year = "2023",
    url = "https://aclanthology.org/2023.emnlp-main.322",
}
```


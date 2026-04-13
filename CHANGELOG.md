# CHANGELOG

## [v1.1.0] - 2026-04-13

### Added
- `src/rewriter/hybrid_rewriter.py`: 分层改写器（规则优先→小模型→LLM兜底），支持领域路由
- `src/rules/domain_rules_engine.py`: 四大领域规则引擎（通讯/闲聊/知识/AI搜索），规则可热插拔
- `src/rewriter/model_rewriter.py`: mT5-small ONNX/INT8量化推理，延迟最优化
- `src/utils/context_manager.py`: 结构化对话上下文管理与槽位自动抽取
- `datasets/build_domain_dataset.py`: 四大场景测试集自动生成（通过Doubao-Seed-2.0-lite）
- `evaluation/eval_pipeline.py`: 三层评测流水线（文本质量/延迟/LLM语义打分）
- `evaluation/run_benchmark.py`: 一键评测与A/B对比报告输出
- `.env.example`: 环境变量配置模板
- `.gitignore`: 项目忽略规则

### Changed
- `requirements.txt`: 新增 matplotlib 可视化依赖
- `.env.example`: 新增 REWRITER_MODEL_ID / REWRITER_ONNX_DIR / LLM_SCORE_SAMPLE_RATE 配置项
- `README.md`: 全面更新，添加架构图、快速上手和评测指南

### Architecture
```
用户输入 (口语/省略/指代)
    │
    ▼
[ASR噪声清洗] ─→ <1ms
    │
    ▼
[领域路由] (通讯/闲聊/知识/AI搜索)
    │
    ▼
[规则引擎] ─→ 命中 → 返回 (P50 ~0.5ms)
    │ 未命中
    ▼
[mT5-small ONNX] ─→ 返回 (P50 ~15ms)
    │ 失败/低置信
    ▼
[Doubao LLM兜底] ─→ 返回 (P50 ~300ms, 可选)
```

---

## [v1.0.0] - 2026-04-10

### Added
- `src/llm/doubao_client.py`: 火山方舟 Doubao-Seed-2.0-lite 统一调用封装
- 基础项目结构（基于 RAG Query Rewriting 原始论文代码）

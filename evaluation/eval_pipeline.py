"""
三层评测流水线
Layer 1: 文本质量评估 (ROUGE-L / BERTScore / BLEU)
Layer 2: 延迟评估 (P50/P95/P99 latency)
Layer 3: 大模型语义打分 (Doubao 可解释评分)
"""
import json
import time
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 评测数据结构
# ──────────────────────────────────────────────
@dataclass
class SampleResult:
    sample_id: str
    domain: str
    original_query: str
    rewritten_query: str
    gold_rewrite: str
    method: str
    latency_ms: float
    rouge_l: float = 0.0
    bleu: float = 0.0
    bert_score_f1: float = 0.0
    llm_score: float = 0.0
    llm_explanation: str = ""
    changed: bool = False

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class DomainMetrics:
    domain: str
    n_samples: int = 0
    rule_hit_rate: float = 0.0
    model_hit_rate: float = 0.0
    llm_hit_rate: float = 0.0
    passthrough_rate: float = 0.0
    avg_rouge_l: float = 0.0
    avg_bleu: float = 0.0
    avg_bert_score: float = 0.0
    avg_llm_score: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    change_rate: float = 0.0


# ──────────────────────────────────────────────
# 文本质量指标计算
# ──────────────────────────────────────────────
def _compute_rouge_l(hypothesis: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        score = scorer.score(reference, hypothesis)
        return score["rougeL"].fmeasure
    except Exception:
        return _rouge_l_fallback(hypothesis, reference)


def _rouge_l_fallback(hyp: str, ref: str) -> float:
    """不依赖外部库的 ROUGE-L 实现"""
    hyp_chars = list(hyp)
    ref_chars = list(ref)
    m, n = len(ref_chars), len(hyp_chars)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / n if n else 0
    recall = lcs / m if m else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _compute_bleu(hypothesis: str, reference: str) -> float:
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = list(reference)
        hyp_tokens = list(hypothesis)
        sf = SmoothingFunction().method1
        return sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5), smoothing_function=sf)
    except Exception:
        return 0.0


def _compute_bert_score(hypotheses: List[str], references: List[str]) -> List[float]:
    try:
        from bert_score import score as bs_score
        P, R, F = bs_score(hypotheses, references, lang="zh", verbose=False)
        return F.tolist()
    except Exception as e:
        logger.warning("BERTScore unavailable: %s. Using ROUGE-L as proxy.", e)
        return [_compute_rouge_l(h, r) for h, r in zip(hypotheses, references)]


# ──────────────────────────────────────────────
# LLM 语义打分
# ──────────────────────────────────────────────
_LLM_SCORE_SYSTEM = (
    "你是 Query 改写质量评估专家。请对改写结果给出1-5分的综合评分，并简要解释。"
    "评分标准:\n"
    "5分: 完整消除指代/省略，语义完整，检索友好，自然流畅\n"
    "4分: 基本消除歧义，语义较完整，略有冗余\n"
    "3分: 部分改写，仍有歧义或省略\n"
    "2分: 改写效果差，语义有损失\n"
    "1分: 改写错误，语义偏离原意"
)

_LLM_SCORE_TEMPLATE = (
    "原始 query: {original}\n"
    "改写后 query: {rewritten}\n"
    "参考改写: {gold}\n"
    "对话上下文: {context}\n\n"
    "请输出 JSON: {{\"score\": <1-5>, \"explanation\": \"<简要说明>\"}}"
)


def _llm_score_sample(
    original: str,
    rewritten: str,
    gold: str,
    context: str = "",
) -> Tuple[float, str]:
    try:
        from src.llm.doubao_client import chat_completion_json
        messages = [
            {"role": "system", "content": _LLM_SCORE_SYSTEM},
            {
                "role": "user",
                "content": _LLM_SCORE_TEMPLATE.format(
                    original=original,
                    rewritten=rewritten,
                    gold=gold,
                    context=context or "无",
                ),
            },
        ]
        raw = chat_completion_json(messages, temperature=0.1)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])
        data = json.loads(raw)
        return float(data.get("score", 3)), str(data.get("explanation", ""))
    except Exception as e:
        logger.warning("LLM scoring failed: %s", e)
        return 3.0, ""


# ──────────────────────────────────────────────
# 评测流水线
# ──────────────────────────────────────────────
class EvalPipeline:
    """
    三层评测流水线

    用法:
        pipeline = EvalPipeline(rewriter=HybridRewriter())
        results = pipeline.evaluate(samples)
        report = pipeline.generate_report(results)
    """

    def __init__(
        self,
        rewriter=None,
        enable_bert_score: bool = True,
        enable_llm_score: bool = True,
        llm_score_sample_rate: float = 1.0,
    ):
        self.rewriter = rewriter
        self.enable_bert_score = enable_bert_score
        self.enable_llm_score = enable_llm_score
        self.llm_score_sample_rate = llm_score_sample_rate

    def evaluate(
        self,
        samples: List[dict],
        context_per_sample: bool = True,
    ) -> List[SampleResult]:
        """
        执行完整三层评测

        Args:
            samples: 测试样本列表（来自 JSONL）
            context_per_sample: 是否为每个样本创建独立上下文

        Returns:
            评测结果列表
        """
        from src.rewriter.hybrid_rewriter import HybridRewriter
        from src.utils.context_manager import ContextManager

        if self.rewriter is None:
            self.rewriter = HybridRewriter(use_model=False, use_llm_fallback=False)

        results: List[SampleResult] = []

        logger.info("Evaluating %d samples...", len(samples))

        # Layer 1 & 2: 改写 + 延迟
        rewritten_list = []
        gold_list = []
        for i, sample in enumerate(samples):
            ctx = ContextManager()
            # 注入上下文历史
            for turn in sample.get("context", []):
                ctx.add_turn(turn.get("role", "user"), turn.get("text", ""))
            # 注入样本领域（用于领域路由回退）
            if sample.get("domain"):
                ctx.set_domain(sample["domain"])

            original = sample.get("original_query", "")
            gold = sample.get("rewritten_query", original)

            try:
                rw_result = self.rewriter.rewrite(original, context_manager=ctx)
                rewritten = rw_result.rewritten_query
                latency = rw_result.latency_ms
                method = rw_result.method
                changed = rw_result.changed
            except Exception as e:
                logger.warning("Rewrite failed for sample %s: %s", sample.get("id"), e)
                rewritten = original
                latency = 0.0
                method = "error"
                changed = False

            # Layer 1: 文本质量
            rouge = _compute_rouge_l(rewritten, gold)
            bleu = _compute_bleu(rewritten, gold)

            sr = SampleResult(
                sample_id=sample.get("id", str(i)),
                domain=sample.get("domain", "general"),
                original_query=original,
                rewritten_query=rewritten,
                gold_rewrite=gold,
                method=method,
                latency_ms=latency,
                rouge_l=rouge,
                bleu=bleu,
                changed=changed,
            )
            results.append(sr)
            rewritten_list.append(rewritten)
            gold_list.append(gold)

            if (i + 1) % 10 == 0:
                logger.info("  Progress: %d/%d", i + 1, len(samples))

        # Layer 1: BERTScore (批量)
        if self.enable_bert_score:
            logger.info("Computing BERTScore...")
            bert_scores = _compute_bert_score(rewritten_list, gold_list)
            for sr, bs in zip(results, bert_scores):
                sr.bert_score_f1 = bs

        # Layer 3: LLM 语义打分
        if self.enable_llm_score:
            logger.info("Computing LLM scores (sample_rate=%.1f)...", self.llm_score_sample_rate)
            import random
            for sr, sample in zip(results, samples):
                if random.random() > self.llm_score_sample_rate:
                    continue
                context_text = " | ".join(
                    f"{t.get('role','')}: {t.get('text','')}"
                    for t in sample.get("context", [])
                )
                score, explanation = _llm_score_sample(
                    sr.original_query, sr.rewritten_query, sr.gold_rewrite, context_text
                )
                sr.llm_score = score
                sr.llm_explanation = explanation
                time.sleep(0.1)  # rate limiting

        return results

    def compute_domain_metrics(self, results: List[SampleResult]) -> Dict[str, DomainMetrics]:
        """按领域汇总指标"""
        domain_results: Dict[str, List[SampleResult]] = {}
        for r in results:
            domain_results.setdefault(r.domain, []).append(r)

        metrics: Dict[str, DomainMetrics] = {}
        for domain, rs in domain_results.items():
            n = len(rs)
            latencies = [r.latency_ms for r in rs]
            latencies_sorted = sorted(latencies)

            def percentile(lst, p):
                if not lst:
                    return 0.0
                k = max(0, int(len(lst) * p / 100) - 1)
                return lst[min(k, len(lst) - 1)]

            method_counts = {}
            for r in rs:
                method_counts[r.method] = method_counts.get(r.method, 0) + 1

            dm = DomainMetrics(
                domain=domain,
                n_samples=n,
                rule_hit_rate=method_counts.get("rule", 0) / n,
                model_hit_rate=method_counts.get("model", 0) / n,
                llm_hit_rate=method_counts.get("llm", 0) / n,
                passthrough_rate=method_counts.get("passthrough", 0) / n,
                avg_rouge_l=sum(r.rouge_l for r in rs) / n,
                avg_bleu=sum(r.bleu for r in rs) / n,
                avg_bert_score=sum(r.bert_score_f1 for r in rs) / n,
                avg_llm_score=sum(r.llm_score for r in rs) / n if any(r.llm_score for r in rs) else 0.0,
                p50_latency_ms=percentile(latencies_sorted, 50),
                p95_latency_ms=percentile(latencies_sorted, 95),
                p99_latency_ms=percentile(latencies_sorted, 99),
                change_rate=sum(1 for r in rs if r.changed) / n,
            )
            metrics[domain] = dm

        return metrics

    def generate_report(
        self,
        results: List[SampleResult],
        version: str = "v1.0",
        output_path: Optional[Path] = None,
    ) -> dict:
        """生成完整评测报告"""
        domain_metrics = self.compute_domain_metrics(results)
        all_results = [r.to_dict() for r in results]

        # 全局汇总
        n_total = len(results)
        global_summary = {
            "version": version,
            "total_samples": n_total,
            "avg_rouge_l": sum(r.rouge_l for r in results) / n_total if n_total else 0,
            "avg_bleu": sum(r.bleu for r in results) / n_total if n_total else 0,
            "avg_bert_score": sum(r.bert_score_f1 for r in results) / n_total if n_total else 0,
            "avg_llm_score": sum(r.llm_score for r in results) / n_total if n_total else 0,
            "p50_latency_ms": statistics.median([r.latency_ms for r in results]) if results else 0,
            "p95_latency_ms": sorted([r.latency_ms for r in results])[int(n_total * 0.95)] if n_total > 1 else 0,
            "overall_change_rate": sum(1 for r in results if r.changed) / n_total if n_total else 0,
            "method_distribution": {},
        }
        for r in results:
            global_summary["method_distribution"][r.method] = (
                global_summary["method_distribution"].get(r.method, 0) + 1
            )

        report = {
            "global_summary": global_summary,
            "domain_metrics": {d: dm.__dict__ for d, dm in domain_metrics.items()},
            "sample_results": all_results,
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info("Report saved to %s", output_path)

        return report

    @staticmethod
    def print_report(report: dict) -> None:
        """在控制台打印格式化报告"""
        gs = report["global_summary"]
        print("\n" + "=" * 60)
        print(f"  Query 改写评测报告  [{gs['version']}]")
        print("=" * 60)
        print(f"  总样本数:     {gs['total_samples']}")
        print(f"  改写率:       {gs['overall_change_rate']:.1%}")
        print(f"  ROUGE-L:      {gs['avg_rouge_l']:.4f}")
        print(f"  BLEU:         {gs['avg_bleu']:.4f}")
        print(f"  BERTScore F1: {gs['avg_bert_score']:.4f}")
        print(f"  LLM Score:    {gs['avg_llm_score']:.2f} / 5.0")
        print(f"  P50 延迟:     {gs['p50_latency_ms']:.1f} ms")
        print(f"  P95 延迟:     {gs['p95_latency_ms']:.1f} ms")
        print(f"  方法分布:     {gs['method_distribution']}")
        print("-" * 60)
        print("  领域详细指标:")
        for domain, dm in report["domain_metrics"].items():
            print(f"\n  [{domain.upper()}] n={dm['n_samples']}")
            print(f"    ROUGE-L: {dm['avg_rouge_l']:.4f}  BLEU: {dm['avg_bleu']:.4f}  BERTScore: {dm['avg_bert_score']:.4f}")
            print(f"    LLM分:   {dm['avg_llm_score']:.2f}  改写率: {dm['change_rate']:.1%}")
            print(f"    延迟 P50/P95/P99: {dm['p50_latency_ms']:.1f}/{dm['p95_latency_ms']:.1f}/{dm['p99_latency_ms']:.1f} ms")
            print(f"    规则命中: {dm['rule_hit_rate']:.1%}  模型: {dm['model_hit_rate']:.1%}  LLM: {dm['llm_hit_rate']:.1%}  直通: {dm['passthrough_rate']:.1%}")
        print("=" * 60)

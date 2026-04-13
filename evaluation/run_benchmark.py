"""
Benchmark 运行器 - 一键 A/B 评测与报告输出
支持: 单版本评测 / A/B 对比 / 可视化导出
"""
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def run_evaluation(
    dataset_path: Path,
    version: str,
    use_model: bool = False,
    use_llm_fallback: bool = False,
    enable_bert_score: bool = True,
    enable_llm_score: bool = True,
    llm_score_sample_rate: float = 0.3,
    output_dir: Path = Path("evaluation/reports"),
    domains: Optional[List[str]] = None,
) -> dict:
    from src.rewriter.hybrid_rewriter import HybridRewriter
    from evaluation.eval_pipeline import EvalPipeline

    # 加载数据集
    samples = load_dataset(dataset_path)
    if domains:
        samples = [s for s in samples if s.get("domain") in domains]
    logger.info("Loaded %d samples from %s", len(samples), dataset_path)

    # 初始化改写器
    rewriter = HybridRewriter(
        use_model=use_model,
        use_llm_fallback=use_llm_fallback,
    )
    if use_model:
        logger.info("Loading small model...")
        rewriter.load_model()

    # 初始化评测流水线
    pipeline = EvalPipeline(
        rewriter=rewriter,
        enable_bert_score=enable_bert_score,
        enable_llm_score=enable_llm_score,
        llm_score_sample_rate=llm_score_sample_rate,
    )

    # 执行评测
    results = pipeline.evaluate(samples)

    # 生成报告
    report_path = output_dir / f"report_{version}.json"
    report = pipeline.generate_report(results, version=version, output_path=report_path)
    pipeline.print_report(report)

    return report


def run_ab_comparison(
    dataset_path: Path,
    config_a: dict,
    config_b: dict,
    output_dir: Path = Path("evaluation/reports"),
) -> dict:
    """A/B 方案对比评测"""
    logger.info("=== A/B 对比评测 ===")
    logger.info("方案A: %s", config_a)
    logger.info("方案B: %s", config_b)

    report_a = run_evaluation(dataset_path, output_dir=output_dir, **config_a)
    report_b = run_evaluation(dataset_path, output_dir=output_dir, **config_b)

    # 对比分析
    gs_a = report_a["global_summary"]
    gs_b = report_b["global_summary"]

    comparison = {
        "version_a": config_a["version"],
        "version_b": config_b["version"],
        "delta": {
            "rouge_l": round(gs_b["avg_rouge_l"] - gs_a["avg_rouge_l"], 4),
            "bleu": round(gs_b["avg_bleu"] - gs_a["avg_bleu"], 4),
            "bert_score": round(gs_b["avg_bert_score"] - gs_a["avg_bert_score"], 4),
            "llm_score": round(gs_b["avg_llm_score"] - gs_a["avg_llm_score"], 3),
            "p50_latency_ms": round(gs_b["p50_latency_ms"] - gs_a["p50_latency_ms"], 2),
            "p95_latency_ms": round(gs_b["p95_latency_ms"] - gs_a["p95_latency_ms"], 2),
            "change_rate": round(gs_b["overall_change_rate"] - gs_a["overall_change_rate"], 3),
        },
        "winner": {},
    }

    # 判断胜者（综合指标）
    quality_delta = (
        comparison["delta"]["rouge_l"] * 0.3
        + comparison["delta"]["bleu"] * 0.2
        + comparison["delta"]["bert_score"] * 0.3
        + comparison["delta"]["llm_score"] / 5 * 0.2
    )
    comparison["winner"]["quality"] = config_b["version"] if quality_delta > 0 else config_a["version"]
    comparison["winner"]["latency"] = (
        config_a["version"]
        if comparison["delta"]["p50_latency_ms"] > 0
        else config_b["version"]
    )

    ab_report = {
        "report_a": report_a,
        "report_b": report_b,
        "comparison": comparison,
    }

    ab_report_path = output_dir / f"ab_{config_a['version']}_vs_{config_b['version']}.json"
    ab_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ab_report_path, "w", encoding="utf-8") as f:
        json.dump(ab_report, f, ensure_ascii=False, indent=2)
    logger.info("A/B report saved to %s", ab_report_path)

    _print_ab_summary(comparison)
    return ab_report


def _print_ab_summary(comparison: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  A/B 对比: {comparison['version_a']} vs {comparison['version_b']}")
    print("=" * 60)
    delta = comparison["delta"]
    for metric, val in delta.items():
        sign = "+" if val > 0 else ""
        print(f"  {metric:20s}: {sign}{val}")
    print("-" * 60)
    print(f"  质量胜者:  {comparison['winner']['quality']}")
    print(f"  延迟胜者:  {comparison['winner']['latency']}")
    print("=" * 60)


def _try_visualize(report: dict, output_dir: Path, version: str) -> None:
    """可选: matplotlib 可视化"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        metrics = report.get("domain_metrics", {})
        if not metrics:
            return

        domains = list(metrics.keys())
        rouge_vals = [metrics[d]["avg_rouge_l"] for d in domains]
        latency_vals = [metrics[d]["p50_latency_ms"] for d in domains]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(domains, rouge_vals, color="steelblue")
        axes[0].set_title(f"ROUGE-L by Domain [{version}]")
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("ROUGE-L")

        axes[1].bar(domains, latency_vals, color="coral")
        axes[1].set_title(f"P50 Latency by Domain [{version}]")
        axes[1].set_ylabel("Latency (ms)")

        plt.tight_layout()
        chart_path = output_dir / f"chart_{version}.png"
        plt.savefig(str(chart_path), dpi=120)
        logger.info("Chart saved to %s", chart_path)
    except Exception as e:
        logger.debug("Visualization skipped: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Query 改写系统 Benchmark 评测")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 单版本评测
    eval_parser = subparsers.add_parser("eval", help="单版本评测")
    eval_parser.add_argument("--dataset", required=True, help="测试集 JSONL 路径")
    eval_parser.add_argument("--version", default="v1.0", help="版本号")
    eval_parser.add_argument("--use_model", action="store_true", help="启用小模型改写")
    eval_parser.add_argument("--use_llm", action="store_true", help="启用 LLM 兜底")
    eval_parser.add_argument("--no_bert_score", action="store_true", help="禁用 BERTScore")
    eval_parser.add_argument("--no_llm_score", action="store_true", help="禁用 LLM 语义打分")
    eval_parser.add_argument("--llm_sample_rate", type=float, default=0.3, help="LLM 打分采样率")
    eval_parser.add_argument("--domains", nargs="+", help="限定评测领域")
    eval_parser.add_argument("--output_dir", default="evaluation/reports", help="报告输出目录")
    eval_parser.add_argument("--visualize", action="store_true", help="输出可视化图表")

    # A/B 对比
    ab_parser = subparsers.add_parser("ab", help="A/B 对比评测")
    ab_parser.add_argument("--dataset", required=True, help="测试集 JSONL 路径")
    ab_parser.add_argument("--output_dir", default="evaluation/reports", help="报告输出目录")

    args = parser.parse_args()

    if args.command == "eval":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report = run_evaluation(
            dataset_path=Path(args.dataset),
            version=args.version,
            use_model=args.use_model,
            use_llm_fallback=args.use_llm,
            enable_bert_score=not args.no_bert_score,
            enable_llm_score=not args.no_llm_score,
            llm_score_sample_rate=args.llm_sample_rate,
            output_dir=output_dir,
            domains=args.domains,
        )
        if args.visualize:
            _try_visualize(report, output_dir, args.version)

    elif args.command == "ab":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # 预设 A/B 配置: A=纯规则, B=规则+小模型
        config_a = {
            "version": "v1.0-rules-only",
            "use_model": False,
            "use_llm_fallback": False,
            "enable_bert_score": True,
            "enable_llm_score": True,
            "llm_score_sample_rate": 0.3,
        }
        config_b = {
            "version": "v1.1-rules+model",
            "use_model": True,
            "use_llm_fallback": False,
            "enable_bert_score": True,
            "enable_llm_score": True,
            "llm_score_sample_rate": 0.3,
        }
        run_ab_comparison(
            dataset_path=Path(args.dataset),
            config_a=config_a,
            config_b=config_b,
            output_dir=output_dir,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

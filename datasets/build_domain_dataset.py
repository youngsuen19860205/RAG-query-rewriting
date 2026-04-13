"""
四大核心场景测试集自动生成
通过 Doubao-Seed-2.0-lite 生成高质量领域测试数据
输出: datasets/domain/xxx.jsonl

场景:
  - telecom: 通讯（打电话/发短信/联系人）
  - chitchat: 闲聊（日常对话/情感交流）
  - knowledge: 知识问答（百科/科技/历史）
  - ai_search: AI搜索（新闻/价格/天气/实时信息）
"""
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict

# 确保 src 在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.doubao_client import chat_completion_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "domain"

# ──────────────────────────────────────────────
# 每个领域的生成 Prompt 模板
# ──────────────────────────────────────────────
DOMAIN_PROMPTS: Dict[str, Dict] = {
    "telecom": {
        "system": (
            "你是语音助手测试数据生成专家。请生成通讯领域的query改写测试样本，"
            "覆盖: 打电话、发短信、联系人指代、省略追问等场景。"
        ),
        "user_template": (
            "生成 {n} 条通讯领域的多轮对话query改写测试样本，JSON数组格式，每条包含:\n"
            "- id: 编号\n"
            "- domain: 'telecom'\n"
            "- context: 最近2-3轮对话历史，数组，每项有role和text字段\n"
            "- original_query: 用户当前口语输入（含省略/指代/噪声）\n"
            "- rewritten_query: 改写后完整独立的标准问句\n"
            "- rewrite_type: 改写类型，可选: coref|ellipsis|asr_noise|followup\n"
            "- difficulty: easy|medium|hard\n"
            "要求多样化，覆盖各种口语表达和歧义场景。"
        ),
    },
    "chitchat": {
        "system": (
            "你是语音助手测试数据生成专家。请生成闲聊领域的query改写测试样本，"
            "覆盖: 日常问候、情感话题、话题追问、口语省略等场景。"
        ),
        "user_template": (
            "生成 {n} 条闲聊领域的多轮对话query改写测试样本，JSON数组格式，每条包含:\n"
            "- id: 编号\n"
            "- domain: 'chitchat'\n"
            "- context: 最近2-3轮对话历史，数组\n"
            "- original_query: 用户当前口语输入\n"
            "- rewritten_query: 改写后完整独立的问句\n"
            "- rewrite_type: coref|ellipsis|asr_noise|followup\n"
            "- difficulty: easy|medium|hard\n"
            "要求贴近真实语音交互场景，包含口语化表达。"
        ),
    },
    "knowledge": {
        "system": (
            "你是语音助手测试数据生成专家。请生成知识问答领域的query改写测试样本，"
            "覆盖: 指代消解、追问、省略、歧义澄清等场景，话题包括科技/历史/生活常识。"
        ),
        "user_template": (
            "生成 {n} 条知识问答领域的多轮对话query改写测试样本，JSON数组格式，每条包含:\n"
            "- id: 编号\n"
            "- domain: 'knowledge'\n"
            "- context: 最近2-3轮对话历史，数组\n"
            "- original_query: 用户当前口语输入（含指代/省略）\n"
            "- rewritten_query: 改写后完整独立的标准问句\n"
            "- rewrite_type: coref|ellipsis|asr_noise|followup\n"
            "- difficulty: easy|medium|hard\n"
            "知识话题多样化，包含科技/历史/地理/生活等。"
        ),
    },
    "ai_search": {
        "system": (
            "你是语音助手测试数据生成专家。请生成AI搜索领域的query改写测试样本，"
            "覆盖: 价格查询、天气询问、新闻资讯、实时数据等场景。"
        ),
        "user_template": (
            "生成 {n} 条AI搜索领域的多轮对话query改写测试样本，JSON数组格式，每条包含:\n"
            "- id: 编号\n"
            "- domain: 'ai_search'\n"
            "- context: 最近2-3轮对话历史，数组\n"
            "- original_query: 用户当前口语输入（含省略/指代）\n"
            "- rewritten_query: 改写后完整独立的搜索query\n"
            "- rewrite_type: coref|ellipsis|asr_noise|followup\n"
            "- difficulty: easy|medium|hard\n"
            "搜索场景多样：价格/天气/新闻/赛事/汇率等实时信息。"
        ),
    },
}

# ──────────────────────────────────────────────
# 生成函数
# ──────────────────────────────────────────────
def generate_domain_samples(
    domain: str,
    n: int = 20,
    batch_size: int = 10,
) -> List[dict]:
    """生成指定领域的测试样本"""
    prompt_cfg = DOMAIN_PROMPTS[domain]
    all_samples: List[dict] = []
    global_id = 1

    batches = (n + batch_size - 1) // batch_size
    for batch_idx in range(batches):
        current_n = min(batch_size, n - len(all_samples))
        messages = [
            {"role": "system", "content": prompt_cfg["system"]},
            {"role": "user", "content": prompt_cfg["user_template"].format(n=current_n)},
        ]
        logger.info("Generating %s batch %d/%d (%d samples)...", domain, batch_idx + 1, batches, current_n)
        try:
            raw = chat_completion_json(messages, temperature=0.8)
            # 解析 JSON
            raw_clean = raw.strip()
            if raw_clean.startswith("```"):
                raw_clean = "\n".join(raw_clean.split("\n")[1:-1])
            samples = json.loads(raw_clean)
            if isinstance(samples, dict):
                # 可能是 {"samples": [...]}
                for k in samples:
                    if isinstance(samples[k], list):
                        samples = samples[k]
                        break
            for s in samples:
                s["id"] = f"{domain}_{global_id:04d}"
                s["domain"] = domain
                all_samples.append(s)
                global_id += 1
            logger.info("  Got %d samples (total: %d)", len(samples), len(all_samples))
        except Exception as e:
            logger.error("  Batch %d failed: %s", batch_idx + 1, e)
        time.sleep(0.5)  # rate limiting

    return all_samples


def save_jsonl(samples: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info("Saved %d samples to %s", len(samples), path)


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="四大领域测试集生成器")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["telecom", "chitchat", "knowledge", "ai_search"],
        choices=["telecom", "chitchat", "knowledge", "ai_search"],
        help="要生成的领域",
    )
    parser.add_argument("--n", type=int, default=50, help="每个领域生成样本数")
    parser.add_argument("--batch_size", type=int, default=10, help="每次 API 调用生成数量")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary = {}

    for domain in args.domains:
        output_path = output_dir / f"{domain}_test.jsonl"

        # 检查是否已有数据，支持增量生成
        existing = load_jsonl(output_path)
        if len(existing) >= args.n:
            logger.info("Domain %s already has %d samples, skipping.", domain, len(existing))
            summary[domain] = len(existing)
            continue

        need = args.n - len(existing)
        samples = generate_domain_samples(domain, n=need, batch_size=args.batch_size)
        all_samples = existing + samples
        save_jsonl(all_samples, output_path)
        summary[domain] = len(all_samples)

    # 输出合并数据集
    all_data = []
    for domain in args.domains:
        path = output_dir / f"{domain}_test.jsonl"
        all_data.extend(load_jsonl(path))
    combined_path = output_dir / "all_domains_test.jsonl"
    save_jsonl(all_data, combined_path)

    logger.info("=== 生成完毕 ===")
    for domain, cnt in summary.items():
        logger.info("  %s: %d samples", domain, cnt)
    logger.info("  合并数据集: %s (%d samples)", combined_path, len(all_data))


if __name__ == "__main__":
    main()

"""
分层改写器 (Hybrid Rewriter)
策略: 规则优先 → 小模型兜底 → (可选) Doubao LLM 最终兜底
低延迟设计: 规则命中直接返回，避免模型推理
领域路由: 自动检测领域，分发到对应规则集
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from src.rules.domain_rules_engine import DomainRulesEngine
from src.utils.context_manager import ContextManager

logger = logging.getLogger(__name__)


@dataclass
class RewriteResult:
    """改写结果"""
    original_query: str
    rewritten_query: str
    domain: str
    method: str             # "rule" | "model" | "llm" | "passthrough"
    rule_name: Optional[str] = None
    latency_ms: float = 0.0
    confidence: float = 1.0
    changed: bool = False

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "domain": self.domain,
            "method": self.method,
            "rule_name": self.rule_name,
            "latency_ms": round(self.latency_ms, 2),
            "confidence": round(self.confidence, 3),
            "changed": self.changed,
        }


class HybridRewriter:
    """
    分层 Query 改写器

    层级:
      1. ASR 噪声清洗 (纯字符串, <0.1ms)
      2. 规则引擎     (正则匹配, <1ms)
      3. 小模型推理   (flan-t5-base ONNX, ~10-30ms)
      4. Doubao LLM  (可选降级, ~200-500ms)

    用法:
        rewriter = HybridRewriter()
        rewriter.load_model()   # 可选：预加载小模型
        result = rewriter.rewrite("给他打电话", context_manager)
    """

    def __init__(
        self,
        use_model: bool = True,
        use_llm_fallback: bool = False,
        model_latency_threshold_ms: float = 50.0,
    ):
        self.use_model = use_model
        self.use_llm_fallback = use_llm_fallback
        self.model_latency_threshold_ms = model_latency_threshold_ms

        self._rules_engine = DomainRulesEngine()
        self._model_rewriter = None   # lazy load

    # ── 模型加载 ──────────────────────────────
    def load_model(self) -> None:
        """预加载小模型（建议服务启动时调用）"""
        if not self.use_model:
            return
        try:
            from src.rewriter.model_rewriter import ModelRewriter
            self._model_rewriter = ModelRewriter()
            self._model_rewriter.load()
            logger.info("Small model loaded: %s", self._model_rewriter.backend)
        except Exception as e:
            logger.warning("Small model load failed: %s. Model fallback disabled.", e)
            self._model_rewriter = None

    # ── 核心改写 ──────────────────────────────
    def rewrite(
        self,
        query: str,
        context_manager: Optional[ContextManager] = None,
        domain: Optional[str] = None,
    ) -> RewriteResult:
        """
        执行分层改写

        Args:
            query: 原始 query
            context_manager: 对话上下文管理器（可选）
            domain: 强制指定领域（None 则自动检测）

        Returns:
            RewriteResult
        """
        t_start = time.perf_counter()

        ctx = context_manager or ContextManager()
        slots = ctx.get_slots_dict()

        # ── Step 1: ASR 清洗 ──────────────────
        cleaned = self._rules_engine.clean_asr(query)

        # ── Step 2: 领域检测 ──────────────────
        detected_domain = domain or self._rules_engine.detect_domain(cleaned)
        # 如果当前 query 领域不明确，回退到上下文中的上一个领域
        if detected_domain == "general" and slots.get("last_domain"):
            detected_domain = slots["last_domain"]
        ctx.set_domain(detected_domain)

        # ── Step 3: 规则改写 ──────────────────
        rule_result = self._rules_engine.apply(cleaned, slots, domain=detected_domain)
        if rule_result.matched and rule_result.rewritten_query:
            latency = (time.perf_counter() - t_start) * 1000
            rewritten = rule_result.rewritten_query
            return RewriteResult(
                original_query=query,
                rewritten_query=rewritten,
                domain=detected_domain,
                method="rule",
                rule_name=rule_result.rule_name,
                latency_ms=latency,
                confidence=0.95,
                changed=(rewritten != query),
            )

        # ── Step 4: 小模型改写 ────────────────
        if self.use_model and self._model_rewriter and self._model_rewriter.is_loaded:
            history_text = ctx.get_history_text(max_turns=3)
            try:
                rewritten, model_latency = self._model_rewriter.rewrite(
                    cleaned, context_text=history_text
                )
                if rewritten and len(rewritten) > 1:
                    total_latency = (time.perf_counter() - t_start) * 1000
                    return RewriteResult(
                        original_query=query,
                        rewritten_query=rewritten,
                        domain=detected_domain,
                        method="model",
                        latency_ms=total_latency,
                        confidence=0.75,
                        changed=(rewritten != query),
                    )
            except Exception as e:
                logger.warning("Model rewrite failed: %s", e)

        # ── Step 5: Doubao LLM 兜底 ───────────
        if self.use_llm_fallback:
            try:
                from src.llm.doubao_client import chat_completion
                messages = ctx.build_rewrite_prompt(cleaned)
                rewritten = chat_completion(messages, temperature=0.3, max_tokens=128)
                rewritten = rewritten.strip()
                if rewritten and rewritten != cleaned:
                    total_latency = (time.perf_counter() - t_start) * 1000
                    return RewriteResult(
                        original_query=query,
                        rewritten_query=rewritten,
                        domain=detected_domain,
                        method="llm",
                        latency_ms=total_latency,
                        confidence=0.85,
                        changed=True,
                    )
            except Exception as e:
                logger.warning("LLM fallback failed: %s", e)

        # ── Passthrough: 无改写 ───────────────
        latency = (time.perf_counter() - t_start) * 1000
        return RewriteResult(
            original_query=query,
            rewritten_query=cleaned,
            domain=detected_domain,
            method="passthrough",
            latency_ms=latency,
            confidence=1.0,
            changed=(cleaned != query),
        )

    # ── 批量改写 ──────────────────────────────
    def rewrite_batch(
        self,
        queries: list,
        context_manager: Optional[ContextManager] = None,
    ) -> list:
        """批量改写（共享同一上下文管理器）"""
        results = []
        ctx = context_manager or ContextManager()
        for q in queries:
            result = self.rewrite(q, context_manager=ctx)
            ctx.add_turn("user", q)
            results.append(result)
        return results

    # ── 规则热管理 ────────────────────────────
    @property
    def rules_engine(self) -> DomainRulesEngine:
        return self._rules_engine

    def get_stats(self) -> dict:
        return {
            "use_model": self.use_model,
            "model_loaded": self._model_rewriter is not None and self._model_rewriter.is_loaded,
            "model_backend": self._model_rewriter.backend if self._model_rewriter else None,
            "use_llm_fallback": self.use_llm_fallback,
            "total_rules": len(self._rules_engine.list_rules()),
        }

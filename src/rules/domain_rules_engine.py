"""
四大核心场景高频规则库
覆盖: 通讯 (telecom) / 闲聊 (chitchat) / 知识问答 (knowledge) / AI搜索 (ai_search)
设计原则:
  - 规则可热插拔: 每条规则是独立 (pattern, rewrite_fn, priority) 三元组
  - 低延迟: 纯正则/字符串操作，无模型推理
  - 领域路由: 关键词命中即分配领域
"""
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────
@dataclass
class RewriteRule:
    """单条改写规则"""
    name: str
    pattern: re.Pattern
    rewrite_fn: Callable[[re.Match, dict], Optional[str]]
    priority: int = 10          # 数值越小优先级越高
    domain: str = "general"     # 所属领域


@dataclass
class RuleApplyResult:
    matched: bool = False
    rewritten_query: Optional[str] = None
    rule_name: Optional[str] = None
    domain: Optional[str] = None


# ──────────────────────────────────────────────
# 领域路由关键词
# ──────────────────────────────────────────────
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "telecom": [
        "打电话", "发短信", "发消息", "联系", "通话", "拨打", "短信", "微信",
        "视频通话", "语音", "电话", "号码", "手机", "通讯录", "联系人",
        "call", "sms", "text", "dial",
    ],
    "chitchat": [
        "你好", "你叫什么", "聊聊", "开心", "无聊", "哈哈", "哦", "嗯",
        "什么感觉", "说说", "怎么了", "陪我", "讲笑话", "吹牛", "唠嗑",
        "hi", "hello", "hey", "chat",
    ],
    "knowledge": [
        "是什么", "怎么", "为什么", "如何", "原理", "解释", "介绍", "定义",
        "区别", "对比", "特点", "历史", "来源", "作用", "意思", "百科",
        "what is", "how to", "why", "explain",
    ],
    "ai_search": [
        "帮我查", "搜索", "查一下", "找一下", "最新", "最近", "今天",
        "新闻", "价格", "天气", "股票", "汇率", "比赛", "结果", "排行",
        "search", "look up", "find", "latest",
    ],
}


def detect_domain(query: str) -> str:
    """基于关键词的快速领域判断，返回置信度最高的领域"""
    query_lower = query.lower()
    scores: Dict[str, int] = {d: 0 for d in DOMAIN_KEYWORDS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                scores[domain] += 1
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "general"


# ──────────────────────────────────────────────
# 通用指代词替换辅助
# ──────────────────────────────────────────────
_COREF_PRONOUNS = re.compile(
    r"(他|她|它|他们|她们|这个|那个|这位|那位|这件事|那件事|上面说的|刚才说的)"
)


def _replace_coref(query: str, context_slots: dict) -> str:
    """用上下文槽位替换指代词"""
    result = query
    entity = context_slots.get("last_entity") or context_slots.get("last_person")
    if entity:
        result = _COREF_PRONOUNS.sub(entity, result)
    return result


# ──────────────────────────────────────────────
# 通讯领域规则
# ──────────────────────────────────────────────
def _rule_call_coref(m: re.Match, ctx: dict) -> Optional[str]:
    """'给他打电话' → '给{人名}打电话'"""
    person = ctx.get("last_person")
    if not person:
        return None
    return m.string.replace(m.group(0), f"给{person}打电话")


def _rule_sms_coref(m: re.Match, ctx: dict) -> Optional[str]:
    """'给她发短信' → '给{人名}发短信'"""
    person = ctx.get("last_person")
    if not person:
        return None
    return m.string.replace(m.group(0), f"给{person}发短信")


def _rule_followup_call(m: re.Match, ctx: dict) -> Optional[str]:
    """'再打一次' → '再给{人名}打一次电话'"""
    person = ctx.get("last_person")
    action = ctx.get("last_action", "打电话")
    if not person:
        return None
    return f"再给{person}{action}"


_TELECOM_RULES: List[RewriteRule] = [
    RewriteRule(
        name="call_coref",
        pattern=re.compile(r"给(他|她|这个人|那个人)打电话"),
        rewrite_fn=_rule_call_coref,
        priority=1,
        domain="telecom",
    ),
    RewriteRule(
        name="sms_coref",
        pattern=re.compile(r"给(他|她|这个人|那个人)发(短信|消息)"),
        rewrite_fn=_rule_sms_coref,
        priority=1,
        domain="telecom",
    ),
    RewriteRule(
        name="followup_call",
        pattern=re.compile(r"^(再打一次|重新拨打|再拨|再联系一下)$"),
        rewrite_fn=_rule_followup_call,
        priority=2,
        domain="telecom",
    ),
    RewriteRule(
        name="implicit_contact",
        pattern=re.compile(r"^(联系他|联系她|联系一下)$"),
        rewrite_fn=lambda m, ctx: (
            f"联系{ctx['last_person']}" if ctx.get("last_person") else None
        ),
        priority=2,
        domain="telecom",
    ),
]


# ──────────────────────────────────────────────
# 闲聊领域规则
# ──────────────────────────────────────────────
def _rule_chitchat_greeting(m: re.Match, ctx: dict) -> Optional[str]:
    """口语问候规范化"""
    greet_map = {
        "你好啊": "你好",
        "嗨": "你好",
        "哈喽": "你好",
        "嘿": "你好",
        "哦对了": "",
        "嗯那个": "",
    }
    raw = m.string.strip()
    for k, v in greet_map.items():
        if raw.startswith(k):
            return (v + raw[len(k):]).strip() or raw
    return None


def _rule_ellipsis_chitchat(m: re.Match, ctx: dict) -> Optional[str]:
    """追问补全: '还有呢?' → '关于{上个话题}还有什么?'"""
    topic = ctx.get("last_topic")
    if not topic:
        return None
    return f"关于{topic}还有什么"


_CHITCHAT_RULES: List[RewriteRule] = [
    RewriteRule(
        name="greeting_normalize",
        pattern=re.compile(r"^(你好啊|嗨|哈喽|嘿|哦对了|嗯那个)"),
        rewrite_fn=_rule_chitchat_greeting,
        priority=1,
        domain="chitchat",
    ),
    RewriteRule(
        name="ellipsis_followup",
        pattern=re.compile(r"^(还有呢|然后呢|接着呢|继续|还有吗)[?？]?$"),
        rewrite_fn=_rule_ellipsis_chitchat,
        priority=2,
        domain="chitchat",
    ),
    RewriteRule(
        name="pronoun_resolve",
        pattern=_COREF_PRONOUNS,
        rewrite_fn=lambda m, ctx: _replace_coref(m.string, ctx) if ctx.get("last_entity") else None,
        priority=3,
        domain="chitchat",
    ),
]


# ──────────────────────────────────────────────
# 知识问答领域规则
# ──────────────────────────────────────────────
def _rule_knowledge_coref(m: re.Match, ctx: dict) -> Optional[str]:
    """'它是什么原理?' → '{实体}是什么原理?'"""
    entity = ctx.get("last_entity")
    if not entity:
        return None
    return _COREF_PRONOUNS.sub(entity, m.string)


def _rule_knowledge_followup(m: re.Match, ctx: dict) -> Optional[str]:
    """'那它的历史呢?' → '{实体}的历史是什么?'"""
    entity = ctx.get("last_entity")
    if not entity:
        return None
    suffix = m.string.strip().lstrip("那").lstrip("那么").strip()
    suffix = re.sub(r"呢[?？]?$", "是什么?", suffix)
    suffix = _COREF_PRONOUNS.sub(entity, suffix)
    return suffix


_KNOWLEDGE_RULES: List[RewriteRule] = [
    RewriteRule(
        name="knowledge_coref",
        pattern=_COREF_PRONOUNS,
        rewrite_fn=_rule_knowledge_coref,
        priority=1,
        domain="knowledge",
    ),
    RewriteRule(
        name="knowledge_followup",
        pattern=re.compile(r"^(那|那么)(它|他|她|这个|那个)"),
        rewrite_fn=_rule_knowledge_followup,
        priority=2,
        domain="knowledge",
    ),
    RewriteRule(
        name="knowledge_abbreviation",
        pattern=re.compile(r"^(解释一下|说说|介绍下|讲讲)\s*(.+)$"),
        rewrite_fn=lambda m, ctx: f"{m.group(2)}是什么?",
        priority=3,
        domain="knowledge",
    ),
    RewriteRule(
        name="knowledge_howto",
        pattern=re.compile(r"^(怎么|如何|咋)\s*(.+)[?？]?$"),
        rewrite_fn=lambda m, ctx: f"如何{m.group(2)}?",
        priority=4,
        domain="knowledge",
    ),
]


# ──────────────────────────────────────────────
# AI 搜索领域规则
# ──────────────────────────────────────────────
def _rule_search_normalize(m: re.Match, ctx: dict) -> Optional[str]:
    """'帮我查一下xxx' → '搜索: xxx'"""
    keyword = m.group(1).strip()
    if not keyword:
        return None
    return f"搜索: {keyword}"


def _rule_search_coref(m: re.Match, ctx: dict) -> Optional[str]:
    """指代消解后加搜索前缀"""
    entity = ctx.get("last_entity")
    if not entity:
        return None
    q = _COREF_PRONOUNS.sub(entity, m.string)
    return f"搜索: {q}"


def _rule_search_latest(m: re.Match, ctx: dict) -> Optional[str]:
    """'最近有什么新闻?' → '搜索最新新闻'"""
    topic = m.group(1).strip() if m.lastindex and m.group(1) else ctx.get("last_entity", "")
    return f"搜索最新{topic}资讯" if topic else None


_AI_SEARCH_RULES: List[RewriteRule] = [
    RewriteRule(
        name="search_prefix",
        pattern=re.compile(r"^(帮我查一下|帮我查|帮我搜|查一下|搜索一下|找一下|查查)\s*(.+)$"),
        rewrite_fn=lambda m, ctx: f"搜索: {m.group(2).strip()}",
        priority=1,
        domain="ai_search",
    ),
    RewriteRule(
        name="search_latest",
        pattern=re.compile(r"^(最新|最近|今天)(.*)(新闻|消息|资讯|进展)[?？]?$"),
        rewrite_fn=lambda m, ctx: f"搜索最新{m.group(2)}{m.group(3)}",
        priority=2,
        domain="ai_search",
    ),
    RewriteRule(
        name="search_coref",
        pattern=_COREF_PRONOUNS,
        rewrite_fn=_rule_search_coref,
        priority=3,
        domain="ai_search",
    ),
    RewriteRule(
        name="search_price",
        pattern=re.compile(r"^(.+)多少钱[?？]?$"),
        rewrite_fn=lambda m, ctx: f"搜索: {m.group(1).strip()}价格",
        priority=2,
        domain="ai_search",
    ),
    RewriteRule(
        name="search_weather",
        pattern=re.compile(r"^(.*)(今天|明天|后天)(.*天气)[?？]?$"),
        rewrite_fn=lambda m, ctx: f"搜索: {m.group(1)}{m.group(2)}{m.group(3)}",
        priority=1,
        domain="ai_search",
    ),
]


# ──────────────────────────────────────────────
# 通用规则 (跨领域)
# ──────────────────────────────────────────────
_FILLER_WORDS = re.compile(
    r"(嗯那个|那个那个|就是那个|哦对了|好的好的|嗯嗯+|呃+|嗯+|啊+)\s*"
)
_TRAILING_PUNCTUATION = re.compile(r"[，。！？,.!?]+$")


def clean_asr_noise(query: str) -> str:
    """清理 ASR 口语噪声: 填充词、重复标点"""
    q = _FILLER_WORDS.sub("", query)
    q = re.sub(r"(.)\1{3,}", r"\1\1", q)   # 过度重复字符压缩
    q = q.strip()
    return q if q else query


# ──────────────────────────────────────────────
# 规则引擎主类
# ──────────────────────────────────────────────
class DomainRulesEngine:
    """
    四大领域规则引擎

    用法:
        engine = DomainRulesEngine()
        result = engine.apply(query, context_slots, domain=None)
        if result.matched:
            print(result.rewritten_query)
    """

    def __init__(self):
        self._rules: List[RewriteRule] = sorted(
            _TELECOM_RULES + _CHITCHAT_RULES + _KNOWLEDGE_RULES + _AI_SEARCH_RULES,
            key=lambda r: r.priority,
        )
        self._domain_rules: Dict[str, List[RewriteRule]] = {}
        for rule in self._rules:
            self._domain_rules.setdefault(rule.domain, []).append(rule)

    def detect_domain(self, query: str) -> str:
        return detect_domain(query)

    def clean_asr(self, query: str) -> str:
        return clean_asr_noise(query)

    def apply(
        self,
        query: str,
        context_slots: Optional[dict] = None,
        domain: Optional[str] = None,
    ) -> RuleApplyResult:
        """
        尝试对 query 应用规则改写。

        Args:
            query: 原始 query（已经过 ASR 清洗）
            context_slots: 来自 ContextManager 的槽位字典
            domain: 指定领域 (None 则自动检测)

        Returns:
            RuleApplyResult
        """
        ctx = context_slots or {}
        detected = domain or detect_domain(query)

        # 优先应用当前领域规则，再回退到通用
        candidate_rules = self._domain_rules.get(detected, []) + self._domain_rules.get("general", [])
        candidate_rules = sorted(candidate_rules, key=lambda r: r.priority)

        for rule in candidate_rules:
            m = rule.pattern.search(query)
            if m:
                rewritten = rule.rewrite_fn(m, ctx)
                if rewritten and rewritten != query:
                    return RuleApplyResult(
                        matched=True,
                        rewritten_query=rewritten,
                        rule_name=rule.name,
                        domain=detected,
                    )

        return RuleApplyResult(matched=False, domain=detected)

    def add_rule(self, rule: RewriteRule) -> None:
        """热插拔新增规则"""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)
        self._domain_rules.setdefault(rule.domain, []).append(rule)
        self._domain_rules[rule.domain].sort(key=lambda r: r.priority)

    def remove_rule(self, name: str) -> bool:
        """按名称移除规则"""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        for domain in self._domain_rules:
            self._domain_rules[domain] = [r for r in self._domain_rules[domain] if r.name != name]
        return len(self._rules) < before

    def list_rules(self, domain: Optional[str] = None) -> List[RewriteRule]:
        if domain:
            return self._domain_rules.get(domain, [])
        return list(self._rules)

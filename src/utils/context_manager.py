"""
结构化对话上下文管理器与槽位抽取工具
- 维护滑动窗口对话历史
- 自动提取: 人名/实体/动作/话题 等核心槽位
- 提供给规则引擎和模型改写器使用
"""
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────
@dataclass
class Turn:
    role: str          # "user" | "assistant"
    text: str
    slots: Dict = field(default_factory=dict)


@dataclass
class ContextSlots:
    """当前对话上下文槽位"""
    last_person: Optional[str] = None       # 最近提到的人名
    last_entity: Optional[str] = None       # 最近提到的实体（非人名）
    last_action: Optional[str] = None       # 最近的动作词
    last_topic: Optional[str] = None        # 最近的话题关键词
    last_domain: Optional[str] = None       # 最近检测到的领域
    time_ref: Optional[str] = None          # 时间参照（今天/明天/...）
    location_ref: Optional[str] = None      # 地点参照

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ──────────────────────────────────────────────
# 简单槽位抽取模式（轻量级，无模型）
# ──────────────────────────────────────────────
_PERSON_PATTERN = re.compile(
    r"(?:给|联系|打给|发给|叫|约|找)\s*([^\s，。！？,!?的]{2,6})"
)
_ACTION_PATTERN = re.compile(
    r"(打电话|发短信|发消息|视频通话|语音通话|联系|拨打|发微信)"
)
_TIME_PATTERN = re.compile(
    r"(今天|明天|后天|昨天|上午|下午|晚上|早上|现在|刚刚|最近)"
)
_LOCATION_PATTERN = re.compile(
    r"(北京|上海|广州|深圳|杭州|成都|武汉|南京|[^\s，。]{2,4}市|[^\s，。]{2,4}区)"
)
_TOPIC_PATTERN = re.compile(
    r"(?:关于|讲讲|说说|介绍一下|介绍下|介绍|问一下)\s*([^\s，。！？,!?]{2,10})"
)
_ENTITY_PATTERN = re.compile(
    r"([A-Z][a-zA-Z0-9]+|[\u4e00-\u9fa5]{2,8}(?:公司|技术|系统|平台|产品|大学|学院|医院|银行))"
)


def _extract_slots(text: str) -> dict:
    slots: dict = {}
    m = _PERSON_PATTERN.search(text)
    if m:
        slots["last_person"] = m.group(1)
    m = _ACTION_PATTERN.search(text)
    if m:
        slots["last_action"] = m.group(1)
    m = _TIME_PATTERN.search(text)
    if m:
        slots["time_ref"] = m.group(1)
    m = _LOCATION_PATTERN.search(text)
    if m:
        slots["location_ref"] = m.group(1)
    m = _TOPIC_PATTERN.search(text)
    if m:
        slots["last_topic"] = m.group(1)
    m = _ENTITY_PATTERN.search(text)
    if m:
        slots["last_entity"] = m.group(1)
    return slots


# ──────────────────────────────────────────────
# 上下文管理器
# ──────────────────────────────────────────────
class ContextManager:
    """
    轻量级对话上下文管理器

    用法:
        cm = ContextManager(window_size=5)
        cm.add_turn("user", "帮我联系张三")
        cm.add_turn("assistant", "正在为您拨打张三的电话")
        slots = cm.get_slots()
        # slots.last_person == "张三"
    """

    def __init__(self, window_size: int = 6):
        self.window_size = window_size
        self._history: Deque[Turn] = deque(maxlen=window_size)
        self._slots = ContextSlots()

    # ── 对话管理 ──────────────────────────────
    def add_turn(self, role: str, text: str) -> None:
        """添加一轮对话并自动更新槽位"""
        extracted = _extract_slots(text)
        turn = Turn(role=role, text=text, slots=extracted)
        self._history.append(turn)
        self._update_slots(extracted)

    def _update_slots(self, new_slots: dict) -> None:
        for key, val in new_slots.items():
            if hasattr(self._slots, key) and val:
                setattr(self._slots, key, val)

    def get_slots(self) -> ContextSlots:
        return self._slots

    def get_slots_dict(self) -> dict:
        return self._slots.to_dict()

    def get_history(self) -> List[Turn]:
        return list(self._history)

    def get_history_text(self, max_turns: int = 3) -> str:
        """返回最近 N 轮对话的纯文本，供 prompt 使用"""
        turns = list(self._history)[-max_turns:]
        lines = []
        for t in turns:
            prefix = "用户" if t.role == "user" else "助手"
            lines.append(f"{prefix}: {t.text}")
        return "\n".join(lines)

    def build_rewrite_prompt(self, current_query: str) -> List[dict]:
        """
        构建用于 LLM 改写的 messages 列表
        (仅当规则+小模型均无法处理时使用)
        """
        history_text = self.get_history_text(max_turns=4)
        system_msg = (
            "你是语音助手的 Query 改写助手。"
            "请将用户的口语化问题改写为一句完整、独立、清晰的标准问句，"
            "消除指代不明、省略和口语化表达，不要添加多余解释。"
        )
        user_msg = (
            f"对话历史:\n{history_text}\n\n"
            f"当前用户说: {current_query}\n\n"
            "请改写为完整独立的问句（只输出改写后的句子，不要其他内容）:"
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    def reset(self) -> None:
        """清空上下文"""
        self._history.clear()
        self._slots = ContextSlots()

    def set_domain(self, domain: str) -> None:
        self._slots.last_domain = domain

    # ── 序列化 ──────────────────────────────
    def to_dict(self) -> dict:
        return {
            "history": [
                {"role": t.role, "text": t.text, "slots": t.slots}
                for t in self._history
            ],
            "slots": self._slots.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict, window_size: int = 6) -> "ContextManager":
        cm = cls(window_size=window_size)
        for item in data.get("history", []):
            t = Turn(role=item["role"], text=item["text"], slots=item.get("slots", {}))
            cm._history.append(t)
        slots_data = data.get("slots", {})
        for k, v in slots_data.items():
            if hasattr(cm._slots, k):
                setattr(cm._slots, k, v)
        return cm

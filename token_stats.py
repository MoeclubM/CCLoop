"""
Token 统计模块
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TokenStats:
    """单轮 Token 统计"""
    round: int = 0
    # 用户输入（纯文本字符数估算）
    user_text_tokens: int = 0
    # 助手输出（纯文本字符数估算）
    assistant_text_tokens: int = 0
    # 工具调用估算
    tool_use_tokens: int = 0
    # 工具结果估算
    tool_result_tokens: int = 0
    # 缓存相关
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    # API 返回的总计
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "TokenStats") -> "TokenStats":
        if not isinstance(other, TokenStats):
            return self
        return TokenStats(
            round=0,
            user_text_tokens=self.user_text_tokens + other.user_text_tokens,
            assistant_text_tokens=self.assistant_text_tokens + other.assistant_text_tokens,
            tool_use_tokens=self.tool_use_tokens + other.tool_use_tokens,
            tool_result_tokens=self.tool_result_tokens + other.tool_result_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_text_tokens": self.user_text_tokens,
            "assistant_text_tokens": self.assistant_text_tokens,
            "tool_use_tokens": self.tool_use_tokens,
            "tool_result_tokens": self.tool_result_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    @property
    def total(self) -> int:
        """总 Token 数（不含缓存读取）"""
        return self.input_tokens + self.output_tokens

    @property
    def total_with_cache(self) -> int:
        """总 Token 数（含缓存读取，用于计费参考）"""
        return self.input_tokens + self.output_tokens + self.cache_read_tokens

    def format_summary(self) -> str:
        """格式化摘要"""
        parts = []
        if self.user_text_tokens:
            parts.append(f"用户文本: {self.user_text_tokens}")
        if self.assistant_text_tokens:
            parts.append(f"助手文本: {self.assistant_text_tokens}")
        if self.tool_use_tokens:
            parts.append(f"工具调用: {self.tool_use_tokens}")
        if self.tool_result_tokens:
            parts.append(f"工具结果: {self.tool_result_tokens}")
        if self.cache_creation_tokens:
            parts.append(f"缓存创建: {self.cache_creation_tokens}")
        if self.cache_read_tokens:
            parts.append(f"缓存读取: {self.cache_read_tokens}")
        return " | ".join(parts) if parts else "无 Token 统计"


def extract_usage_from_obj(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从 JSON 对象中提取 usage 信息"""
    msg = obj.get("message")
    if isinstance(msg, dict):
        usage = msg.get("usage")
        if usage:
            return usage
    if "usage" in obj:
        return obj["usage"]
    return None


def extract_message_stats(obj: Dict[str, Any]) -> Dict[str, Any]:
    """提取消息中的统计信息"""
    stats = {
        "role": "unknown",
        "has_text": False,
        "text_length": 0,
        "tool_use_count": 0,
        "tool_result_count": 0,
        "has_usage": False,
    }

    msg = obj.get("message")
    if isinstance(msg, dict):
        stats["role"] = msg.get("role", "unknown")
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type")
                    if btype == "text":
                        stats["has_text"] = True
                        text = block.get("text", "")
                        if isinstance(text, str):
                            stats["text_length"] += len(text)
                    elif btype == "tool_use":
                        stats["tool_use_count"] += 1
                    elif btype == "tool_result":
                        stats["tool_result_count"] += 1

        if msg.get("usage"):
            stats["has_usage"] = True

    return stats


def calc_tokens_for_usage(usage: Dict[str, Any]) -> Dict[str, int]:
    """根据 usage 计算各类型 token"""
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
    }


@dataclass
class RoundLog:
    """单轮日志"""
    round: int
    claude_prompt: str
    claude_output: str
    judge: Dict[str, Any]
    tokens: Optional[TokenStats] = None


@dataclass
class LoopState:
    """循环状态"""
    goal: str = ""
    goal_summary: str = ""
    refined_goal: str = ""
    goal_set: bool = False
    memory: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[RoundLog] = field(default_factory=list)
    session_id: Optional[str] = None
    current_round: int = 0

    # 详细 Token 统计
    total_user_text_tokens: int = 0
    total_assistant_text_tokens: int = 0
    total_tool_use_tokens: int = 0
    total_tool_result_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # 计时器相关
    start_time: Optional[float] = None
    round_start_time: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        """总 Token 数（不含缓存读取）"""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_tokens_with_cache(self) -> int:
        """总 Token 数（含缓存读取）"""
        return self.total_input_tokens + self.total_output_tokens + self.total_cache_read_tokens

    def update_tokens(self, usage: Any) -> None:
        """更新 Token 统计"""
        if not usage:
            return

        if isinstance(usage, dict):
            self.total_input_tokens += usage.get("input_tokens", 0)
            self.total_output_tokens += usage.get("output_tokens", 0)
            self.total_cache_creation_tokens += usage.get("cache_creation_input_tokens", 0)
            self.total_cache_read_tokens += usage.get("cache_read_input_tokens", 0)

    def add_round_tokens(self, tokens: TokenStats) -> None:
        """添加单轮 Token 统计"""
        self.total_user_text_tokens += tokens.user_text_tokens
        self.total_assistant_text_tokens += tokens.assistant_text_tokens
        self.total_tool_use_tokens += tokens.tool_use_tokens
        self.total_tool_result_tokens += tokens.tool_result_tokens
        self.total_cache_creation_tokens += tokens.cache_creation_tokens
        self.total_cache_read_tokens += tokens.cache_read_tokens
        self.total_input_tokens += tokens.input_tokens
        self.total_output_tokens += tokens.output_tokens

    def get_token_summary(self) -> str:
        """获取 Token 统计摘要"""
        parts = []
        if self.total_user_text_tokens:
            parts.append(f"用户: {self.total_user_text_tokens}")
        if self.total_assistant_text_tokens:
            parts.append(f"助手: {self.total_assistant_text_tokens}")
        if self.total_tool_use_tokens:
            parts.append(f"工具: {self.total_tool_use_tokens}")
        if self.total_cache_creation_tokens:
            parts.append(f"缓存创建: {self.total_cache_creation_tokens}")
        if self.total_cache_read_tokens:
            parts.append(f"缓存读取: {self.total_cache_read_tokens}")
        return " | ".join(parts) if parts else "无统计"

    def clear_tokens(self) -> None:
        """清除 Token 统计"""
        self.total_user_text_tokens = 0
        self.total_assistant_text_tokens = 0
        self.total_tool_use_tokens = 0
        self.total_tool_result_tokens = 0
        self.total_cache_creation_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    # 计时器方法（子类需要实现）
    def get_elapsed_time(self) -> str:
        """获取已运行时间（格式化）"""
        return "00:00"

    def get_round_elapsed(self) -> str:
        """获取当前轮次已运行时间（格式化）"""
        return "00:00"

    def start_timer(self) -> None:
        """启动会话计时器"""
        pass

    def start_round(self) -> None:
        """开始新轮次计时"""
        pass

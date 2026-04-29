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


def extract_message_stats(obj: Dict[str, Any]) -> Dict[str, Any]:
    """提取消息中的统计信息"""
    stats: Dict[str, Any] = {
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
                        text_val = block.get("text", "")
                        if isinstance(text_val, str):
                            stats["text_length"] += len(text_val)
                    elif btype == "tool_use":
                        stats["tool_use_count"] = int(stats["tool_use_count"]) + 1
                    elif btype == "tool_result":
                        stats["tool_result_count"] = int(stats["tool_result_count"]) + 1

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

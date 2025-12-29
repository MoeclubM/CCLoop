"""
显示输出模块
"""

import sys
import unicodedata
from typing import Any, List, Optional

# 性能优化：缓存导入
try:
    from wcwidth import wcswidth
    _WCSWIDTH_AVAILABLE = True
except ImportError:
    _WCSWIDTH_AVAILABLE = False

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import ANSI


_ANSI_STRIP_RE = None


def _get_ansi_strip_re():
    global _ANSI_STRIP_RE
    if _ANSI_STRIP_RE is None:
        import re
        _ANSI_STRIP_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return _ANSI_STRIP_RE


def _strip_ansi(s: str) -> str:
    return _get_ansi_strip_re().sub("", s)


def _str_width(s: str) -> int:
    s = _get_ansi_strip_re().sub("", s)
    if _WCSWIDTH_AVAILABLE:
        w = wcswidth(s)
        return max(0, w)
    else:
        width = 0
        for char in s:
            ea = unicodedata.east_asian_width(char)
            width += 2 if ea in ("F", "W") else 1
        return width


def _pretty(obj: Any) -> str:
    if isinstance(obj, dict):
        import json
        return json.dumps(obj, ensure_ascii=False, indent=2)
    if isinstance(obj, str):
        return obj
    return str(obj)


# ------------------------------------------------------------
# 颜色输出配置
# ------------------------------------------------------------

# 检测是否支持颜色
ENABLE_COLOR = bool(getattr(sys.stdout, "isatty", lambda: False)())


def aprint(s: str = "", *, end: str = "\n", flush: bool = True) -> None:
    """ANSI-aware print"""
    global ENABLE_COLOR

    if not ENABLE_COLOR:
        s = _strip_ansi(s)

    print_formatted_text(ANSI(s), end=end)

    # TUI 模式下必须立即刷新，否则输出会缓冲
    try:
        sys.stdout.flush()
    except Exception:
        pass


def _print_box(title: str, content: str, style: str = "normal", max_lines: int = 8) -> None:
    """打印带边框的内容框"""
    raw_lines = content.strip().split("\n") if content else [""]
    total_lines = len(raw_lines)

    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    GREY = "\033[90m"

    if style == "tool_use":
        border_color = YELLOW
        icon = "🛠️ "
    elif style == "tool_result":
        border_color = GREEN
        icon = "📝"
    else:
        border_color = CYAN
        icon = "ℹ️ "

    box_width = 80
    content_width = box_width - 4
    indent_width = 4

    display_lines: List[str]
    if total_lines > max_lines:
        hidden_count = total_lines - max_lines
        display_lines = [f"{GREY}[... output too long, hidden {hidden_count} lines ...]{RESET}"] + raw_lines[-max_lines:]
    else:
        display_lines = raw_lines

    title_display = f" {icon} {BOLD}{title} {RESET}"
    title_len = _str_width(title_display)
    top_border_len = max(0, box_width - 4 - title_len)

    aprint(f"\n{border_color}┌{'─'*2}{title_display}{'─'*top_border_len}┐{RESET}")

    for line in display_lines:
        is_first_part = True
        while True:
            current_content_width = content_width if is_first_part else content_width - indent_width
            w = _str_width(line)
            if w <= current_content_width:
                padding = current_content_width - w
                indent = "" if is_first_part else " " * indent_width
                aprint(f"{border_color}│{RESET} {indent}{line}{' '*padding} {border_color}│{RESET}")
                break

            clean_line = _strip_ansi(line)
            cut_idx = 0
            cut_w = 0
            for i, char in enumerate(clean_line):
                cw = 2 if ord(char) > 127 else 1
                if cut_w + cw > current_content_width:
                    break
                cut_w += cw
                cut_idx = i + 1

            part = line[:cut_idx]
            padding = current_content_width - cut_w
            indent = "" if is_first_part else " " * indent_width
            aprint(f"{border_color}│{RESET} {indent}{part}{' '*padding} {border_color}│{RESET}")
            line = line[cut_idx:]
            is_first_part = False

    aprint(f"{border_color}└{'─'*(box_width-2)}┘{RESET}\n")


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

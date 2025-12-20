from __future__ import annotations

"""
CC自循环控制器（Claude Code + OpenAI Judge）
"""

import asyncio
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import anyio
from dotenv import load_dotenv
from openai import OpenAI

# === TUI / 输出工具 ===
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.application.current import get_app
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

# ------------------------------------------------------------
# 0) Windows 环境修复 (编码)
# ------------------------------------------------------------
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

# ------------------------------------------------------------
# 1) 颜色输出：统一走 prompt_toolkit
# ------------------------------------------------------------
# 在 IDE Output/重定向/某些 pseudo terminal 下 isatty=False
ENABLE_COLOR = bool(getattr(sys.stdout, "isatty", lambda: False)())

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def aprint(s: str = "", *, end: str = "\n", flush: bool = False) -> None:
    """ANSI-aware print.

    - 在 prompt_toolkit 的 UI/patch_stdout 场景：用 print_formatted_text(ANSI(...)) 正确渲染颜色。
    - 在非 TTY：去掉 ANSI，避免输出控制符污染日志。

    注意：print_formatted_text 的 end 参数是支持的。
    """

    if not ENABLE_COLOR:
        s = _ANSI_RE.sub("", s)

    # ANSI 会把 \x1b[...m 解析成颜色
    print_formatted_text(ANSI(s), end=end)

    if flush:
        try:
            sys.stdout.flush()
        except Exception:
            pass


# ------------------------------------------------------------
# 2) 环境变量配置
# ------------------------------------------------------------
load_dotenv()

OPENAI_BASE_URL = os.environ["OPENAI_BASE_URL"]
OPENAI_MODEL = os.environ["OPENAI_MODEL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MAX_ROUNDS = int(os.environ["MAX_ROUNDS"])

# ------------------------------------------------------------
# 3) Judge 系统提示
# ------------------------------------------------------------

JUDGE_SYSTEM = """你是一个严格的任务判定器（Judge）。

输入：
- goal：目标
- claude_output：Claude Code 本轮输出（自由格式）
- memory：前几轮的简要记录

请只输出一个合法 JSON 对象，包含：
- done: boolean
- summary: string
- next_prompt: string（若 done=true 必须为空字符串）

要求：
- 只有在明确完成目标时，done=true
- 若未完成，next_prompt 给出下一轮要让 Claude Code 做的具体事情（尽量短）。
"""


def extract_first_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"done": False, "summary": "Judge解析失败", "next_prompt": "继续尝试"}

    try:
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            raise ValueError("解析结果不是 JSON 对象")
        return obj
    except json.JSONDecodeError:
        return {"done": False, "summary": "Judge JSON无效", "next_prompt": "继续"}


def validate_judge(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj.setdefault("done", False)
    obj.setdefault("summary", "无摘要")
    obj.setdefault("next_prompt", "继续")
    if obj["done"]:
        obj["next_prompt"] = ""
    return obj


# ------------------------------------------------------------
# 4) 循环状态
# ------------------------------------------------------------


class AppStatus(Enum):
    IDLE = "空闲"
    RUNNING = "运行中"
    PAUSED = "已暂停"
    FINISHED = "已完成"


@dataclass
class RoundLog:
    round: int
    claude_prompt: str
    claude_output: str
    judge: Dict[str, Any]


@dataclass
class LoopState:
    goal: str = ""
    memory: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[RoundLog] = field(default_factory=list)
    session_id: Optional[str] = None

    status: AppStatus = AppStatus.IDLE
    current_round: int = 0
    total_tokens: int = 0

    def update_tokens(self, usage: Any):
        if not usage:
            return
        if hasattr(usage, "total_tokens"):
            self.total_tokens += getattr(usage, "total_tokens", 0)
            return
        if isinstance(usage, dict):
            input_t = usage.get("input_tokens", 0) or 0
            output_t = usage.get("output_tokens", 0) or 0
            cache_read = usage.get("cache_read_input_tokens", 0) or 0
            cache_create = usage.get("cache_creation_input_tokens", 0) or 0
            self.total_tokens += (input_t + output_t + cache_read + cache_create)


# ------------------------------------------------------------
# 5) 显示辅助
# ------------------------------------------------------------


def _strip_ansi(s: str) -> str:
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", s)


def _str_width(s: str) -> int:
    s = _strip_ansi(s)
    try:
        from wcwidth import wcswidth

        w = wcswidth(s)
        return max(0, w)
    except Exception:
        import unicodedata

        width = 0
        for char in s:
            ea = unicodedata.east_asian_width(char)
            width += 2 if ea in ("F", "W") else 1
        return width


def _pretty(obj: Any) -> str:
    if isinstance(obj, dict):
        return json.dumps(obj, ensure_ascii=False, indent=2)
    if isinstance(obj, str):
        return obj
    return str(obj)


def _print_box(title: str, content: str, style: str = "normal", max_lines: int = 20) -> None:
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


# ------------------------------------------------------------
# 6) Claude Code 执行
# ------------------------------------------------------------


def _refresh_ui():
    try:
        app = get_app()
        if app.is_running:
            app.invalidate()
    except Exception:
        pass


def _handle_event(obj: Dict[str, Any], state: LoopState) -> None:
    etype = obj.get("type")

    if "usage" in obj:
        state.update_tokens(obj["usage"])
        _refresh_ui()

    if "session_id" in obj:
        sid = obj["session_id"]
        if sid and sid != state.session_id:
            state.session_id = sid

    if etype == "system":
        subtype = obj.get("subtype")
        if subtype == "init":
            cwd0 = obj.get("cwd")
            aprint(f"\033[90m[INIT] cwd={cwd0}\033[0m\n", end="")
        return

    if etype == "result":
        res = obj.get("result")
        if isinstance(res, str) and res:
            aprint(res, end="")
        else:
            aprint(_pretty(obj))
        return

    msg = obj.get("message") if isinstance(obj.get("message"), dict) else None
    if not msg:
        if "message" in obj and isinstance(obj["message"], dict) and "usage" in obj["message"]:
            state.update_tokens(obj["message"]["usage"])
            _refresh_ui()
        return

    if "usage" in msg:
        state.update_tokens(msg["usage"])
        _refresh_ui()

    content = msg.get("content")
    if not isinstance(content, list):
        if content:
            aprint(_pretty(content))
        return

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")

        if btype == "text":
            text = block.get("text", "")
            if text:
                aprint(text, end="")
        elif btype == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                aprint(f"\033[90m[Thinking] {thinking}\033[0m\n", end="")
        elif btype == "tool_use":
            name = block.get("name", "")
            tin = block.get("input", {})
            display_content = _pretty(tin)
            if name == "Bash" and isinstance(tin, dict) and "command" in tin:
                display_content = f"$ {tin['command']}"
                if "description" in tin:
                    display_content += f"\n# {tin['description']}"
            _print_box(f"TOOL USE: {name}", display_content, style="tool_use")
        elif btype == "tool_result":
            tr_content = block.get("content", "")
            display_content = _pretty(tr_content)
            _print_box("TOOL RESULT", display_content, style="tool_result")

    if "tool_use_result" in obj:
        tur = obj["tool_use_result"]
        if isinstance(tur, dict):
            out = tur.get("stdout", "")
            err = tur.get("stderr", "")
            combined = ""
            if out:
                combined += out
            if err:
                combined += f"\n[STDERR]\n{err}"
            if combined.strip():
                _print_box("TOOL OUTPUT (STDOUT)", combined, style="tool_result")


async def run_claude_once(*, prompt: str, cwd: str = ".", state: LoopState) -> str:
    cli_args_str = os.environ.get("CLAUDE_CLI_ARGS", "--print --verbose --output-format stream-json")
    args = ["claude"] + shlex.split(cli_args_str)
    args.append(prompt)

    aprint(f"\n\033[1;35m>>> 调用 Claude Code (cwd: {cwd}) ...\033[0m\n", end="")

    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        env=os.environ.copy(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    decoder = json.JSONDecoder()
    buffer = ""
    raw_chunks: List[str] = []

    def consume_buffer() -> None:
        nonlocal buffer
        while True:
            buffer = buffer.lstrip("\r\n\t ")
            if not buffer:
                return
            try:
                obj, idx = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                return

            buffer = buffer[idx:]
            if isinstance(obj, dict):
                _handle_event(obj, state)
            else:
                aprint(_pretty(obj))

    try:
        assert proc.stdout is not None
        while True:
            data = await proc.stdout.read(4096)
            if not data:
                break
            s = data.decode(errors="replace")
            raw_chunks.append(s)
            buffer += s
            consume_buffer()

        consume_buffer()
        await proc.wait()

    except asyncio.CancelledError:
        aprint("\n\033[33m[System] 正在停止 Claude 进程...\033[0m")
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        raise

    return "".join(raw_chunks).strip()


# ------------------------------------------------------------
# 7) Judge & Controller
# ------------------------------------------------------------


def judge_once(*, goal: str, claude_output: str, memory: List[Dict[str, Any]], state: LoopState) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    if not api_key:
        return {"done": False, "summary": "无Key", "next_prompt": "继续"}

    aprint(f"\n\033[1;34m>>> 正在请求 Judge ({os.environ.get('OPENAI_MODEL', OPENAI_MODEL)}) ...\033[0m")

    try:
        client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL", OPENAI_BASE_URL))

        judge_input = claude_output
        if len(judge_input) > 6000:
            judge_input = judge_input[:2000] + "\n...[truncated]...\n" + judge_input[-4000:]

        payload = {"goal": goal, "claude_output": judge_input, "memory": memory}

        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", OPENAI_MODEL),
            temperature=0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )

        if hasattr(resp, "usage") and resp.usage:
            state.update_tokens(resp.usage)
            _refresh_ui()

        text = resp.choices[0].message.content or ""
        aprint(f"\033[90m[Judge Output]\n{text}\033[0m\n")

        return validate_judge(extract_first_json(text))

    except Exception as e:
        aprint(f"\033[31m[Judge Error] {e}\033[0m")
        return {"done": False, "summary": f"Judge出错: {e}", "next_prompt": "继续"}


def update_goal_once(*, original_goal: str, additional_instruction: str, state: LoopState) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    if not api_key:
        return original_goal  # 如果没有key，返回原goal

    aprint(f"\n\033[1;34m>>> 正在请求 Goal Updater ({os.environ.get('OPENAI_MODEL', OPENAI_MODEL)}) ...\033[0m")

    try:
        client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL", OPENAI_BASE_URL))

        prompt = f"原目标：{original_goal}\n\n追加指令：{additional_instruction}\n\n请结合追加指令，重新表述一个新的目标。保持简洁明了。"

        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", OPENAI_MODEL),
            temperature=0.3,  # 稍微有点创造性
            messages=[
                {"role": "system", "content": "你是一个目标更新器。请基于原目标和追加指令，生成一个新的、整合的目标。"},
                {"role": "user", "content": prompt},
            ],
        )

        if hasattr(resp, "usage") and resp.usage:
            state.update_tokens(resp.usage)
            _refresh_ui()

        new_goal = resp.choices[0].message.content.strip()
        aprint(f"\033[90m[New Goal]\n{new_goal}\033[0m\n")
        return new_goal

    except Exception as e:
        aprint(f"\033[31m[Goal Update Error] {e}\033[0m")
        return original_goal  # 出错时返回原goal


def build_claude_prompt(goal: str, next_instruction: str) -> str:
    if next_instruction.strip():
        return f"目标：{goal}\n\n上一轮进展摘要：{next_instruction}\n\n请继续完成目标。"
    return f"目标：{goal}\n\n请完成目标。"


async def self_loop(*, max_rounds: int = 6, cwd: str = ".", state: LoopState) -> Dict[str, Any]:
    next_instruction = ""
    if state.memory:
        last_mem = state.memory[-1]
        if not last_mem.get("done", False):
            next_instruction = last_mem.get("next_prompt", "")

    start_round = len(state.logs) + 1
    state.status = AppStatus.RUNNING
    _refresh_ui()

    try:
        for r in range(start_round, start_round + max_rounds):
            state.current_round = r
            _refresh_ui()

            aprint(f"\n{'='*20} ROUND {r} {'='*20}")

            prompt = build_claude_prompt(state.goal, next_instruction)

            claude_output = await run_claude_once(prompt=prompt, cwd=cwd, state=state)

            judge = judge_once(goal=state.goal, claude_output=claude_output, memory=state.memory, state=state)

            aprint(f"👉 \033[1mJudge 判定:\033[0m Done={judge['done']}")
            if judge.get("summary"):
                aprint(f"   摘要: {judge['summary']}")
            if not judge["done"] and judge.get("next_prompt"):
                aprint(f"   指示: {judge['next_prompt']}")

            state.logs.append(RoundLog(r, prompt, claude_output, judge))
            state.memory.append({"round": r, "summary": judge["summary"], "next_prompt": judge["next_prompt"]})

            if judge["done"]:
                state.status = AppStatus.FINISHED
                _refresh_ui()
                return {"status": "completed", "rounds": r, "summary": judge["summary"]}

            next_instruction = judge["next_prompt"]

        state.status = AppStatus.PAUSED
        _refresh_ui()
        return {"status": "max_rounds_reached", "rounds": max_rounds, "summary": "达到轮次限制，等待指示。"}

    except asyncio.CancelledError:
        state.status = AppStatus.PAUSED
        _refresh_ui()
        raise
    except Exception as e:
        state.status = AppStatus.PAUSED
        _refresh_ui()
        aprint(f"\n[Error] Loop exception: {e}")
        return {"status": "error", "summary": str(e)}


# ------------------------------------------------------------
# 8) 主程序（TUI 交互循环）
# ------------------------------------------------------------


async def main() -> None:
    state = LoopState()

    def get_bottom_toolbar():
        status_color = {
            AppStatus.IDLE: "gray",
            AppStatus.RUNNING: "ansigreen",
            AppStatus.PAUSED: "ansiyellow",
            AppStatus.FINISHED: "ansiblue",
        }.get(state.status, "white")

        return HTML(
            f" <b><style color='{status_color}'>{state.status.value}</style></b> | "
            f"Round: <b>{state.current_round}</b> | "
            f"Tokens: <b><style color='ansicyan'>{state.total_tokens:,}</style></b> "
        )

    style = Style.from_dict({"bottom-toolbar": "#333333 bg:#dddddd"})
    session = PromptSession(bottom_toolbar=get_bottom_toolbar, style=style)

    aprint("\n🚀 \033[1mClaude Code 自循环控制器 \033[0m")
    aprint("\033[90m- 输入指令开始运行\n- 运行中输入指令可追加干预\n- Ctrl+C 暂停\n\033[0m")

    background_task: Optional[asyncio.Task] = None
    prompt_html = HTML("<b><style color='#00aa00'>Command ></style></b> ")

    while True:
        try:
            with patch_stdout():
                user_input = (await session.prompt_async(prompt_html)).strip()

            if user_input.lower() == "exit":
                if background_task and not background_task.done():
                    background_task.cancel()
                break

            if state.status == AppStatus.RUNNING:
                if user_input:
                    aprint(f"\n\033[1;33m[User Intervention] 追加指令: {user_input}\033[0m")
                    state.memory.append(
                        {"round": -1, "summary": "用户实时干预", "next_prompt": f"用户要求优先处理：{user_input}"}
                    )
                    # 结合追加指令重新生成新goal
                    new_goal = update_goal_once(original_goal=state.goal, additional_instruction=user_input, state=state)
                    state.goal = new_goal
                    aprint(f"\033[1;32m[Goal Updated] 新目标: {state.goal}\033[0m\n")
            else:
                if user_input:
                    if state.status == AppStatus.IDLE:
                        state.goal = user_input
                        state.memory = []
                        state.logs = []
                        state.current_round = 0
                        state.total_tokens = 0
                    else:
                        state.memory.append(
                            {"round": -1, "summary": "用户修改指令后继续", "next_prompt": f"用户更新了指令：{user_input}"}
                        )

                if state.status == AppStatus.IDLE and not user_input:
                    continue

                if not (background_task and not background_task.done()):
                    state.status = AppStatus.RUNNING
                    background_task = asyncio.create_task(self_loop(max_rounds=MAX_ROUNDS, state=state))

        except KeyboardInterrupt:
            if background_task and not background_task.done():
                aprint("\n\n\033[1;33m⚠️  检测到中断 (Ctrl+C)！正在暂停后台任务...\033[0m")
                background_task.cancel()
                try:
                    await background_task
                except asyncio.CancelledError:
                    pass
                state.status = AppStatus.PAUSED
                _refresh_ui()
            else:
                aprint("\n[System] 退出程序。")
                break


if __name__ == "__main__":
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        anyio.run(main)
    except KeyboardInterrupt:
        pass

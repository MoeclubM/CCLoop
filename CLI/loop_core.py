"""
核心循环模块
"""

import asyncio
import json
import os
import shlex
import sys
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from display import _print_box, aprint, format_time
from json_utils import (
    JSONBuffer,
    clean_invalid_unicode,
    detect_and_decode,
)
from prompts import (
    JUDGE_SYSTEM,
    REFINE_SYSTEM,
    SUMMARIZE_SYSTEM,
    GOAL_UPDATER_SYSTEM,
)
from token_stats import LoopState, RoundLog, TokenStats


# ------------------------------------------------------------
# 状态枚举
# ------------------------------------------------------------


class AppStatus(Enum):
    IDLE = "空闲"
    RUNNING = "运行中"
    PAUSED = "已暂停"
    FINISHED = "已完成"


# ------------------------------------------------------------
# 读取环境变量
# ------------------------------------------------------------


def _get_max_rounds() -> int:
    try:
        return int(os.environ.get("MAX_ROUNDS", 6))
    except Exception:
        return 6


# ------------------------------------------------------------
# 完整的 LoopState 实现
# ------------------------------------------------------------


class CompleteLoopState(LoopState):
    """完整的循环状态（含计时器实现）"""

    def __init__(self):
        super().__init__()
        self._app_status = AppStatus.IDLE

    @property
    def status(self) -> AppStatus:
        return self._app_status

    @status.setter
    def status(self, value: AppStatus) -> None:
        self._app_status = value

    def get_elapsed_time(self) -> str:
        """获取已运行时间（格式化）"""
        if not self.start_time:
            return "00:00"
        elapsed = time.time() - self.start_time
        return format_time(elapsed)

    def get_round_elapsed(self) -> str:
        """获取当前轮次已运行时间（格式化）"""
        if not self.round_start_time:
            return "00:00"
        elapsed = time.time() - self.round_start_time
        return format_time(elapsed)

    def start_timer(self) -> None:
        """启动会话计时器"""
        if not self.start_time:
            self.start_time = time.time()
        self.round_start_time = time.time()

    def start_round(self) -> None:
        """开始新轮次计时"""
        self.round_start_time = time.time()


def validate_judge(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj.setdefault("done", False)
    obj.setdefault("summary", "无摘要")
    obj.setdefault("next_prompt", "继续")
    if obj["done"]:
        obj["next_prompt"] = ""
    return obj


# ------------------------------------------------------------
# Claude Code 执行
# ------------------------------------------------------------


def _refresh_ui():
    """刷新 UI（占位实现，实际由主程序注入）"""
    pass


def _handle_event(obj: Dict[str, Any], state: CompleteLoopState) -> None:
    """处理 Claude Code 事件"""
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
            aprint(json.dumps(obj, ensure_ascii=False, indent=2))
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

    from token_stats import extract_usage_from_obj, calc_tokens_for_usage

    content = msg.get("content")
    if not isinstance(content, list):
        if content:
            aprint(json.dumps(content, ensure_ascii=False, indent=2))
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
            display_content = json.dumps(tin, ensure_ascii=False, indent=2)
            if name == "Bash" and isinstance(tin, dict) and "command" in tin:
                display_content = f"$ {tin['command']}"
                if "description" in tin:
                    display_content += f"\n# {tin['description']}"
            _print_box(f"TOOL USE: {name}", display_content, style="tool_use")
        elif btype == "tool_result":
            tr_content = block.get("content", "")
            display_content = json.dumps(tr_content, ensure_ascii=False, indent=2)
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


async def run_claude_once(
    *, prompt: str, cwd: str = ".", state: CompleteLoopState
) -> tuple[str, TokenStats]:
    """运行一次 Claude Code，返回输出和 Token 统计"""
    from json_utils import JSONBuffer, clean_invalid_unicode, detect_and_decode
    from token_stats import (
        calc_tokens_for_usage,
        extract_message_stats,
        TokenStats,
    )

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

    json_buffer = JSONBuffer()
    raw_chunks: List[str] = []
    captured_tool_outputs: List[str] = []
    round_tokens = TokenStats()

    def _update_message_stats(obj: Dict[str, Any]) -> None:
        """更新消息统计"""
        nonlocal round_tokens
        msg_stats = extract_message_stats(obj)
        if msg_stats["has_usage"]:
            usage = obj.get("message", {}).get("usage") or obj.get("usage")
            if usage:
                token_info = calc_tokens_for_usage(usage)
                round_tokens.input_tokens = token_info["input_tokens"]
                round_tokens.output_tokens = token_info["output_tokens"]
                round_tokens.cache_creation_tokens = token_info["cache_creation_tokens"]
                round_tokens.cache_read_tokens = token_info["cache_read_tokens"]

        role = msg_stats.get("role")
        if role == "user":
            round_tokens.user_text_tokens += msg_stats["text_length"]
        elif role == "assistant":
            round_tokens.assistant_text_tokens += msg_stats["text_length"]
            round_tokens.tool_use_tokens += msg_stats["tool_use_count"] * 50
            round_tokens.tool_result_tokens += msg_stats["tool_result_count"] * 30

    def _maybe_capture_tool_output(obj: Dict[str, Any], outputs: List[str]) -> None:
        """捕获工具输出用于日志记录"""
        if "tool_use_result" in obj:
            tur = obj["tool_use_result"]
            if isinstance(tur, dict):
                out = tur.get("stdout", "")
                err = tur.get("stderr", "")
                if out:
                    outputs.append(f"[STDOUT]\n{out}")
                if err:
                    outputs.append(f"[STDERR]\n{err}")

        msg = obj.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tr_content = block.get("content", "")
                        if isinstance(tr_content, str) and tr_content.strip():
                            outputs.append(f"[TOOL RESULT]\n{tr_content}")

    try:
        assert proc.stdout is not None
        while True:
            data = await proc.stdout.read(4096)
            if not data:
                break

            s = detect_and_decode(data)
            s = clean_invalid_unicode(s)
            raw_chunks.append(s)

            objects = json_buffer.feed(s)
            for obj in objects:
                if isinstance(obj, dict):
                    _update_message_stats(obj)
                    _maybe_capture_tool_output(obj, captured_tool_outputs)
                    _handle_event(obj, state)

        if json_buffer.has_data():
            objects = json_buffer.feed("")
            for obj in objects:
                if isinstance(obj, dict):
                    _update_message_stats(obj)
                    _maybe_capture_tool_output(obj, captured_tool_outputs)
                    _handle_event(obj, state)

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

    all_output = "".join(raw_chunks).strip()
    if captured_tool_outputs:
        captured = "\n".join(captured_tool_outputs)
        all_output = f"{all_output}\n{captured}".strip()

    return clean_invalid_unicode(all_output), round_tokens


# ------------------------------------------------------------
# Judge & Controller
# ------------------------------------------------------------


def _get_openai_client():
    """获取 OpenAI client"""
    import pathlib

    # 只从脚本目录加载 .env 文件
    env_path = pathlib.Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    from openai import OpenAI
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )


def judge_once(
    *, goal: str, claude_output: str, memory: List[Dict[str, Any]], state: CompleteLoopState
) -> Dict[str, Any]:
    """请求 Judge 判定"""
    from json_utils import extract_first_json

    if not os.environ.get("OPENAI_API_KEY"):
        return {"done": False, "summary": "无Key", "next_prompt": "继续"}

    model = os.environ.get("OPENAI_MODEL", "gpt-4")

    aprint(f"\n\033[1;34m>>> 正在请求 Judge ({model}) ...\033[0m")

    try:
        client = _get_openai_client()

        judge_input = claude_output
        if len(judge_input) > 6000:
            judge_input = judge_input[:2000] + "\n...[truncated]...\n" + judge_input[-4000:]

        payload = {"goal": goal, "claude_output": judge_input, "memory": memory}

        resp = client.chat.completions.create(
            model=model,
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
        text = clean_invalid_unicode(text)
        aprint(f"\033[90m[Judge Output]\n{text}\033[0m\n")

        return validate_judge(extract_first_json(text))

    except Exception as e:
        aprint(f"\033[31m[Judge Error] {e}\033[0m")
        return {"done": False, "summary": f"Judge出错: {e}", "next_prompt": "继续"}


def refine_goal_once(*, goal: str, state: CompleteLoopState) -> str:
    """润色目标"""
    from json_utils import clean_invalid_unicode
    import pathlib

    # 确保加载 .env
    env_path = pathlib.Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    if not os.environ.get("OPENAI_API_KEY"):
        return goal

    model = os.environ.get("OPENAI_MODEL", "gpt-4")

    try:
        client = _get_openai_client()

        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": REFINE_SYSTEM},
                {"role": "user", "content": goal},
            ],
        )

        if hasattr(resp, "usage") and resp.usage:
            state.update_tokens(resp.usage)
            _refresh_ui()

        refined = resp.choices[0].message.content or ""
        refined = clean_invalid_unicode(refined).strip()
        return refined if refined else goal

    except Exception as e:
        aprint(f"\033[31m[Refine Error] {e}\033[0m")
        return goal


def summarize_goal_once(*, goal: str, state: CompleteLoopState) -> str:
    """生成精简版目标"""
    import pathlib

    # 确保加载 .env
    env_path = pathlib.Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    if not os.environ.get("OPENAI_API_KEY"):
        return goal

    model = os.environ.get("OPENAI_MODEL", "gpt-4")

    try:
        client = _get_openai_client()

        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SUMMARIZE_SYSTEM},
                {"role": "user", "content": goal},
            ],
        )

        if hasattr(resp, "usage") and resp.usage:
            state.update_tokens(resp.usage)
            _refresh_ui()

        summary = resp.choices[0].message.content.strip()
        return summary if summary else goal

    except Exception:
        return goal


def update_goal_once(
    *, original_goal: str, additional_instruction: str, state: CompleteLoopState
) -> str:
    """更新目标"""
    import pathlib

    # 确保加载 .env
    env_path = pathlib.Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    if not os.environ.get("OPENAI_API_KEY"):
        return original_goal

    model = os.environ.get("OPENAI_MODEL", "gpt-4")
    aprint(f"\n\033[1;34m>>> 正在请求 Goal Updater ({model}) ...\033[0m")

    try:
        client = _get_openai_client()

        prompt = f"原目标：{original_goal}\n\n追加指令：{additional_instruction}\n\n请结合追加指令，重新表述一个新的目标。保持简洁明了。"

        resp = client.chat.completions.create(
            model=model,
            temperature=0.7,
            messages=[
                {"role": "system", "content": GOAL_UPDATER_SYSTEM},
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
        return original_goal


def build_claude_prompt(goal: str, refined_goal: str, next_instruction: str, is_first: bool) -> str:
    """构建发送给 Claude 的提示"""
    display_goal = refined_goal if is_first and refined_goal else goal
    if next_instruction.strip():
        return f"目标：{display_goal}\n\n上一轮进展摘要：{next_instruction}\n\n请继续完成目标。"
    return f"目标：{display_goal}\n\n请完成目标。"


async def self_loop(
    *, max_rounds: int = 6, cwd: str = ".", state: CompleteLoopState
) -> Dict[str, Any]:
    """自循环执行"""
    next_instruction = ""
    if state.memory:
        last_mem = state.memory[-1]
        if not last_mem.get("done", False):
            next_instruction = last_mem.get("next_prompt", "")

    start_round = len(state.logs) + 1
    state.status = AppStatus.RUNNING
    state.start_timer()
    _refresh_ui()

    try:
        for r in range(start_round, start_round + max_rounds):
            state.current_round = r
            state.start_round()
            _refresh_ui()

            aprint(f"\n{'='*20} ROUND {r} {'='*20}")

            is_first = len(state.logs) == 0
            prompt = build_claude_prompt(state.goal, state.refined_goal, next_instruction, is_first)

            claude_output, round_tokens = await run_claude_once(prompt=prompt, cwd=cwd, state=state)
            round_tokens.round = r

            state.add_round_tokens(round_tokens)
            _refresh_ui()

            aprint(f"\033[90m[Token] 输入: {round_tokens.input_tokens} | 输出: {round_tokens.output_tokens} | "
                   f"缓存创建: {round_tokens.cache_creation_tokens} | 缓存读取: {round_tokens.cache_read_tokens}\033[0m")

            judge = judge_once(goal=state.goal, claude_output=claude_output, memory=state.memory, state=state)

            aprint(f"👉 \033[1mJudge 判定:\033[0m Done={judge['done']}")
            if judge.get("summary"):
                aprint(f"   摘要: {judge['summary']}")
            if not judge["done"] and judge.get("next_prompt"):
                aprint(f"   指示: {judge['next_prompt']}")

            state.logs.append(RoundLog(r, prompt, claude_output, judge, tokens=round_tokens))
            state.memory.append({"round": r, "summary": judge["summary"], "next_prompt": judge["next_prompt"]})

            if judge["done"]:
                state.status = AppStatus.FINISHED
                state.goal_set = False
                # 保存最终时间，防止后续键盘输入刷新计时
                state.final_elapsed = state.get_elapsed_time()
                _refresh_ui()
                return {"status": "completed", "rounds": r, "summary": judge["summary"]}

            next_instruction = judge["next_prompt"]

        state.status = AppStatus.PAUSED
        # 保存最终时间，防止后续键盘输入刷新计时
        state.final_elapsed = state.get_elapsed_time()
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

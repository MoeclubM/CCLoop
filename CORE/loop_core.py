"""
核心循环模块
"""

import asyncio
import json
import os
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from json_utils import (
    JSONBuffer,
    clean_invalid_unicode,
    detect_and_decode,
    extract_first_json,
)
from prompts import (
    JUDGE_SYSTEM,
    REFINE_SYSTEM,
    SUMMARIZE_SYSTEM,
    GOAL_UPDATER_SYSTEM,
)
from token_stats import LoopState, RoundLog, TokenStats


# ------------------------------------------------------------
# 全局线程池
# ------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=4)


# ------------------------------------------------------------
# 状态枚举
# ------------------------------------------------------------


class AppStatus(Enum):
    IDLE = "空闲"
    RUNNING = "运行中"
    PAUSED = "已暂停"
    FINISHED = "已完成"


# ------------------------------------------------------------
# 回调类型定义
# ------------------------------------------------------------


class OutputCallbacks:
    """输出回调函数集合"""

    def __init__(self):
        self.on_text: Optional[Callable[[str], None]] = None
        self.on_tool_use: Optional[Callable[[str, str], None]] = None
        self.on_tool_result: Optional[Callable[[str], None]] = None
        self.on_judge: Optional[Callable[[str], None]] = None
        self.on_status: Optional[Callable[[str], None]] = None
        self.on_token: Optional[Callable[[Dict[str, int]], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_raw: Optional[Callable[[str], None]] = None


# ------------------------------------------------------------
# 读取环境变量
# ------------------------------------------------------------


def _get_max_rounds() -> int:
    try:
        return int(os.environ.get("MAX_ROUNDS", 6))
    except Exception:
        return 6


# ------------------------------------------------------------
# 动态注入的函数占位符（由 UI 层实现）
# ------------------------------------------------------------


def _refresh_ui() -> None:
    """刷新 UI 的占位符"""
    pass


async def run_single_prompt(prompt: str) -> None:
    """单次运行的占位符"""
    pass


# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


# ------------------------------------------------------------
# 完整的 LoopState 实现
# ------------------------------------------------------------


class CompleteLoopState(LoopState):
    """完整的循环状态（含计时器实现）"""

    def __init__(self):
        super().__init__()
        self._app_status = AppStatus.IDLE
        self.callbacks = OutputCallbacks()
        self.final_elapsed: Optional[str] = None

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


def _handle_event(obj: Dict[str, Any], state: CompleteLoopState) -> None:
    """处理 Claude Code 事件"""
    etype = obj.get("type")

    try:
        if "usage" in obj:
            state.update_tokens(obj["usage"])
            if state.callbacks.on_token:
                token_info = {
                    "input": state.total_input_tokens,
                    "output": state.total_output_tokens,
                    "cache_read": state.total_cache_read_tokens,
                }
                state.callbacks.on_token(token_info)

        if "session_id" in obj:
            sid = obj["session_id"]
            if sid and sid != state.session_id:
                state.session_id = sid

        if etype == "system":
            subtype = obj.get("subtype")
            if subtype == "init":
                cwd0 = obj.get("cwd")
                if state.callbacks.on_status:
                    state.callbacks.on_status(f"[INIT] cwd={cwd0}")
            return

        if etype == "result":
            res = obj.get("result")
            if isinstance(res, str) and res:
                if state.callbacks.on_text:
                    state.callbacks.on_text(res)
            elif isinstance(res, dict):
                output = res.get("output", "") or res.get("content", "") or res.get("text", "")
                if output:
                    if state.callbacks.on_tool_result:
                        state.callbacks.on_tool_result(str(output))
                error = res.get("error", "")
                if error:
                    if state.callbacks.on_error:
                        state.callbacks.on_error(str(error))
            elif res:
                if state.callbacks.on_tool_result:
                    state.callbacks.on_tool_result(str(res))
            if state.callbacks.on_raw:
                state.callbacks.on_raw(json.dumps(obj, ensure_ascii=False, indent=2))
            return

        msg = obj.get("message") if isinstance(obj.get("message"), dict) else None
        if not msg:
            if "message" in obj and isinstance(obj["message"], dict) and "usage" in obj["message"]:
                state.update_tokens(obj["message"]["usage"])
                if state.callbacks.on_token:
                    token_info = {
                        "input": state.total_input_tokens,
                        "output": state.total_output_tokens,
                        "cache_read": state.total_cache_read_tokens,
                    }
                    state.callbacks.on_token(token_info)
            return

        if "usage" in msg:
            state.update_tokens(msg["usage"])
            if state.callbacks.on_token:
                token_info = {
                    "input": state.total_input_tokens,
                    "output": state.total_output_tokens,
                    "cache_read": state.total_cache_read_tokens,
                }
                state.callbacks.on_token(token_info)


        content = msg.get("content")
        if not isinstance(content, list):
            if content:
                if state.callbacks.on_raw:
                    try:
                        # 尝试清理content，如果太长就截断
                        if isinstance(content, str) and len(content) > 10000:
                            state.callbacks.on_raw(json.dumps(str(content[:5000]) + "...[truncated]...", ensure_ascii=False, indent=2))
                        else:
                            state.callbacks.on_raw(json.dumps(content, ensure_ascii=False, indent=2))
                    except Exception:
                        state.callbacks.on_raw(f"[Unprintable content: {len(str(content))} chars]")
            return

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")

            if btype == "text":
                text = block.get("text", "")
                if text and state.callbacks.on_text:
                    state.callbacks.on_text(text)
            elif btype == "thinking":
                thinking = block.get("thinking", "")
                if thinking and state.callbacks.on_status:
                    state.callbacks.on_status(f"[Thinking] {thinking}")
            elif btype == "tool_use":
                name = block.get("name", "")
                tin = block.get("input", {})
                display_content = json.dumps(tin, ensure_ascii=False, indent=2)
                if name == "Bash" and isinstance(tin, dict) and "command" in tin:
                    display_content = f"$ {tin['command']}"
                    if "description" in tin:
                        display_content += f"\n# {tin['description']}"
                if state.callbacks.on_tool_use:
                    state.callbacks.on_tool_use(name, display_content)
            elif btype == "tool_result":
                tr_content = block.get("content", "")
                is_error = block.get("is_error", False)

                if isinstance(tr_content, str):
                    display_content = tr_content
                elif isinstance(tr_content, list):
                    display_content = "\n".join(str(item) for item in tr_content)
                elif isinstance(tr_content, dict):
                    # 处理可能有问题的内容
                    try:
                        display_content = json.dumps(tr_content, ensure_ascii=False, indent=2)
                    except Exception:
                        display_content = f"[Unprintable dict: {len(tr_content)} items]"
                else:
                    display_content = str(tr_content)

                if is_error and display_content:
                    if state.callbacks.on_error:
                        state.callbacks.on_error(display_content)
                elif display_content:
                    if state.callbacks.on_tool_result:
                        state.callbacks.on_tool_result(display_content)

        if "tool_use_result" in obj:
            tur = obj["tool_use_result"]
            if isinstance(tur, dict):
                # 处理新的 Claude Code 格式 (type: "text" 或 type: "file")
                content = tur.get("content", "") or tur.get("output", "")
                file_info = tur.get("file", {})

                combined = ""
                if content:
                    # 对content进行预处理：移除乱码和过长内容
                    try:
                        if isinstance(content, str):
                            # 清理内容：移除无效字符，过长则截断
                            clean_content = content
                            # 移除常见乱码字符
                            clean_content = clean_content.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                            # 截断超长内容
                            if len(clean_content) > 5000:
                                clean_content = clean_content[:5000] + "\n...[内容过长，已截断]..."
                            combined += clean_content
                        else:
                            combined += str(content)
                    except Exception:
                        combined += "[Unprintable content]"
                elif file_info:
                    # file 类型的 result
                    file_path = file_info.get("filePath", "")
                    if file_path:
                        combined += f"[File: {file_path}]"

                # 同时也尝试读取 stdout/stderr（兼容旧格式）
                out = tur.get("stdout", "")
                err = tur.get("stderr", "")
                if out:
                    combined += ("\n" if combined else "") + out
                if err:
                    combined += f"\n[STDERR]\n{err}"

                if combined.strip() and state.callbacks.on_tool_result:
                    state.callbacks.on_tool_result(combined)

        # 输出原始JSON（必须在最后，避免重复输出）
        if state.callbacks.on_raw:
            try:
                # 检查obj是否包含过大的字段
                safe_obj: Dict[str, Any] = {}
                for k, v in obj.items():
                    if k == "message" and isinstance(v, dict):
                        # 跳过可能有问题的message content
                        safe_obj[k] = v
                        continue
                    if isinstance(v, str) and len(v) > 10000:
                        safe_obj[k] = f"[Long content: {len(v)} chars]"
                    else:
                        safe_obj[k] = v
                state.callbacks.on_raw(json.dumps(safe_obj, ensure_ascii=False, indent=2))
            except Exception:
                # 兜底：如果JSON序列化失败，至少记录类型
                try:
                    state.callbacks.on_raw(f"Event type: {obj.get('type', 'unknown')}")
                except Exception:
                    pass

    except Exception as e:
        # 任何处理失败，记录错误但不中断流程
        if state.callbacks.on_error:
            try:
                state.callbacks.on_error(f"[Handler Error] {str(e)[:200]}")
            except Exception:
                pass


async def run_claude_once(
    *, prompt: str, cwd: str = ".", state: CompleteLoopState
) -> tuple[str, TokenStats]:
    """运行一次 Claude Code，返回输出和 Token 统计"""
    from token_stats import (
        calc_tokens_for_usage,
        extract_message_stats,
        TokenStats,
    )

    cli_args_str = os.environ.get("CLAUDE_CLI_ARGS", "--print --verbose --output-format stream-json")
    args = ["claude"] + shlex.split(cli_args_str)
    args.append(prompt)

    if state.callbacks.on_status:
        state.callbacks.on_status(f">>> 调用 Claude Code (cwd: {cwd}) ...")

    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        env=os.environ.copy(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    json_buffer = JSONBuffer()
    raw_chunks: List[str] = []
    round_tokens = TokenStats()
    assistant_texts: List[str] = []

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

    def _maybe_capture_assistant_text(obj: Dict[str, Any], texts: List[str]) -> None:
        """捕获助手消息中的文本内容"""
        msg = obj.get("message")
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
            if role == "assistant" and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text and isinstance(text, str):
                            texts.append(text)

        # 也捕获 tool_result 内容
        content = obj.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    text = block.get("content", "")
                    if text and isinstance(text, str):
                        texts.append(text)

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
                    _maybe_capture_assistant_text(obj, assistant_texts)
                    _handle_event(obj, state)

        if json_buffer.has_data():
            objects = json_buffer.feed("")
            for obj in objects:
                if isinstance(obj, dict):
                    _update_message_stats(obj)
                    _maybe_capture_assistant_text(obj, assistant_texts)
                    _handle_event(obj, state)

        # 等待进程结束，但设置超时防止无限等待
        try:
            await asyncio.wait_for(proc.wait(), timeout=120.0)
        except asyncio.TimeoutError:
            if state.callbacks.on_status:
                state.callbacks.on_status("[System] Claude 进程超时，强制终止...")
            proc.kill()
            await proc.wait()

    except asyncio.CancelledError:
        if state.callbacks.on_status:
            state.callbacks.on_status("[System] 正在停止 Claude 进程...")
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        raise

    all_output = "\n".join(assistant_texts).strip()
    if not all_output:
        all_output = " ".join(raw_chunks).strip()

    return clean_invalid_unicode(all_output), round_tokens


# ------------------------------------------------------------
# Judge & Controller
# ------------------------------------------------------------


def _get_openai_client():
    """获取 OpenAI client"""
    import pathlib

    env_path = pathlib.Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    from openai import OpenAI
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=60.0,
    )


def judge_once(
    *, goal: str, claude_output: str, memory: List[Dict[str, Any]], state: CompleteLoopState
) -> Dict[str, Any]:
    """请求 Judge 判定"""
    if not os.environ.get("OPENAI_API_KEY"):
        return {"done": False, "summary": "无Key", "next_prompt": "继续"}

    model = os.environ.get("OPENAI_MODEL", "gpt-4")

    if state.callbacks.on_status:
        state.callbacks.on_status(f">>> 正在请求 Judge ({model}) ...")

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
            if state.callbacks.on_token:
                token_info = {
                    "input": state.total_input_tokens,
                    "output": state.total_output_tokens,
                    "cache_read": state.total_cache_read_tokens,
                }
                state.callbacks.on_token(token_info)

        text = resp.choices[0].message.content or ""
        text = clean_invalid_unicode(text)

        if state.callbacks.on_judge:
            state.callbacks.on_judge(text)

        return validate_judge(extract_first_json(text))

    except Exception as e:
        if state.callbacks.on_error:
            state.callbacks.on_error(f"Judge Error: {e}")
        return {"done": False, "summary": f"Judge出错: {e}", "next_prompt": "继续"}


def refine_goal_once(*, goal: str, state: CompleteLoopState) -> str:
    """润色目标"""
    import pathlib

    # 确保加载 .env
    env_path = pathlib.Path(__file__).parent.parent / ".env"
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
            if state.callbacks.on_token:
                token_info = {
                    "input": state.total_input_tokens,
                    "output": state.total_output_tokens,
                    "cache_read": state.total_cache_read_tokens,
                }
                state.callbacks.on_token(token_info)

        refined = resp.choices[0].message.content or ""
        refined = clean_invalid_unicode(refined).strip()
        return refined if refined else goal

    except Exception as e:
        if state.callbacks.on_error:
            state.callbacks.on_error(f"Refine Error: {e}")
        return goal


def summarize_goal_once(*, goal: str, state: CompleteLoopState) -> str:
    """生成精简版目标"""
    import pathlib

    # 确保加载 .env
    env_path = pathlib.Path(__file__).parent.parent / ".env"
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
            if state.callbacks.on_token:
                token_info = {
                    "input": state.total_input_tokens,
                    "output": state.total_output_tokens,
                    "cache_read": state.total_cache_read_tokens,
                }
                state.callbacks.on_token(token_info)

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
    env_path = pathlib.Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    if not os.environ.get("OPENAI_API_KEY"):
        return original_goal

    model = os.environ.get("OPENAI_MODEL", "gpt-4")
    if state.callbacks.on_status:
        state.callbacks.on_status(f">>> 正在请求 Goal Updater ({model}) ...")

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
            if state.callbacks.on_token:
                token_info = {
                    "input": state.total_input_tokens,
                    "output": state.total_output_tokens,
                    "cache_read": state.total_cache_read_tokens,
                }
                state.callbacks.on_token(token_info)

        new_goal = resp.choices[0].message.content.strip()
        if state.callbacks.on_status:
            state.callbacks.on_status(f"[New Goal] {new_goal}")
        return new_goal

    except Exception as e:
        if state.callbacks.on_error:
            state.callbacks.on_error(f"Goal Update Error: {e}")
        return original_goal


def build_claude_prompt(goal: str, refined_goal: str, next_instruction: str, is_first: bool) -> str:
    """构建发送给 Claude 的提示"""
    # 如果是第一轮，且有润色后的目标，则使用润色后的目标
    # 否则始终使用原始目标（或保持当前目标）
    current_goal = refined_goal if (is_first and refined_goal) else goal

    if next_instruction.strip():
        return f"目标：{current_goal}\n\n上一轮进展摘要：{next_instruction}\n\n请继续完成目标。"
    return f"目标：{current_goal}\n\n请完成目标。"


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

    loop = asyncio.get_event_loop()

    try:
        for r in range(start_round, start_round + max_rounds):
            state.current_round = r
            state.start_round()

            if state.callbacks.on_status:
                state.callbacks.on_status(f"{'='*20} ROUND {r} {'='*20}")

            is_first = len(state.logs) == 0
            prompt = build_claude_prompt(state.goal, state.refined_goal, next_instruction, is_first)

            claude_output, round_tokens = await run_claude_once(prompt=prompt, cwd=cwd, state=state)
            round_tokens.round = r

            state.add_round_tokens(round_tokens)

            if state.callbacks.on_status:
                state.callbacks.on_status(
                    f"[Token] 输入: {round_tokens.input_tokens} | 输出: {round_tokens.output_tokens} | "
                    f"缓存创建: {round_tokens.cache_creation_tokens} | 缓存读取: {round_tokens.cache_read_tokens}"
                )

            judge = await loop.run_in_executor(
                _executor,
                lambda: judge_once(goal=state.goal, claude_output=claude_output, memory=state.memory, state=state)
            )

            if state.callbacks.on_status:
                state.callbacks.on_status(f"Judge 判定: Done={judge['done']}")
                if judge.get("summary"):
                    state.callbacks.on_status(f"   摘要: {judge['summary']}")
                if not judge["done"] and judge.get("next_prompt"):
                    state.callbacks.on_status(f"   指示: {judge['next_prompt']}")

            state.logs.append(RoundLog(r, prompt, claude_output, judge, tokens=round_tokens))
            state.memory.append({"round": r, "summary": judge["summary"], "next_prompt": judge["next_prompt"]})

            if judge["done"]:
                state.status = AppStatus.FINISHED
                state.goal_set = False
                # 保存最终时间，防止后续键盘输入刷新计时
                state.final_elapsed = state.get_elapsed_time()
                return {"status": "completed", "rounds": r, "summary": judge["summary"]}

            next_instruction = judge["next_prompt"]

        state.status = AppStatus.PAUSED
        # 保存最终时间，防止后续键盘输入刷新计时
        state.final_elapsed = state.get_elapsed_time()
        return {"status": "max_rounds_reached", "rounds": max_rounds, "summary": "达到轮次限制，等待指示。"}

    except asyncio.CancelledError:
        state.status = AppStatus.PAUSED
        raise
    except Exception as e:
        state.status = AppStatus.PAUSED
        if state.callbacks.on_error:
            state.callbacks.on_error(f"Loop exception: {e}")
        return {"status": "error", "summary": str(e)}

"""
CC自循环控制器（Claude Code + OpenAI Judge）
主入口模块
"""

import asyncio
import os
import sys
from typing import Optional
import pathlib

# ------------------------------------------------------------
# 1) Windows 环境修复 (编码)
# ------------------------------------------------------------
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

# ------------------------------------------------------------
# 2) 添加CORE目录到Python路径
# ------------------------------------------------------------
core_path = pathlib.Path(__file__).parent.parent / "CORE"
if str(core_path) not in sys.path:
    sys.path.insert(0, str(core_path))

# ------------------------------------------------------------
# 3) 导入模块
# ------------------------------------------------------------
from display import aprint, _print_box
from loop_core import (
    AppStatus,
    CompleteLoopState,
    refine_goal_once,
    self_loop,
    summarize_goal_once,
    update_goal_once,
)
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

# 读取 MAX_ROUNDS 环境变量
try:
    MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", 6))
except Exception:
    MAX_ROUNDS = 6

# ------------------------------------------------------------
# 3) 主程序（TUI 交互循环）
# ------------------------------------------------------------


async def _timer_refresh_task(session: PromptSession, state: CompleteLoopState, running_flag: dict):
    """后台任务：每秒刷新计时器显示"""
    while running_flag["value"]:
        await asyncio.sleep(1.0)
        if running_flag["value"]:  # 再次检查，防止退出时触发
            try:
                session.app.invalidate()
            except Exception:
                pass


async def main() -> None:
    state = CompleteLoopState()

    # 设置CORE模块的回调函数
    def on_text(text: str):
        """文本输出回调"""
        aprint(text)

    def on_tool_use(tool_name: str, input_data: str):
        """工具使用回调"""
        _print_box(title=f"Tool Use: {tool_name}", content=input_data, style="tool_use")

    def on_tool_result(result: str):
        """工具结果回调"""
        _print_box(title="Tool Result", content=result, style="tool_result")

    def on_judge(judgment: str):
        """判断回调"""
        aprint(f"\n\033[1;33m⚖️ Judge: {judgment}\033[0m\n")

    def on_status(status: str):
        """状态回调"""
        aprint(f"\n\033[1;34m📊 Status: {status}\033[0m\n")

    def on_token(tokens: dict):
        """Token统计回调"""
        pass

    def on_error(error: str):
        """错误回调"""
        aprint(f"\n\033[31m❌ Error: {error}\033[0m\n")

    def on_raw(raw: str):
        """原始输出回调 - 暂时禁用"""
        pass  # 不保存到文件，直接输出

    state.callbacks.on_text = on_text
    state.callbacks.on_tool_use = on_tool_use
    state.callbacks.on_tool_result = on_tool_result
    state.callbacks.on_judge = on_judge
    state.callbacks.on_status = on_status
    state.callbacks.on_token = on_token
    state.callbacks.on_error = on_error
    state.callbacks.on_raw = on_raw

    # 用于控制计时器任务的标志
    timer_running = {"value": False}
    timer_task: Optional[asyncio.Task] = None

    def get_bottom_toolbar():
        status_color = {
            AppStatus.IDLE: "gray",
            AppStatus.RUNNING: "ansigreen",
            AppStatus.PAUSED: "ansiyellow",
            AppStatus.FINISHED: "ansiblue",
        }.get(state.status, "white")

        goal_text = state.goal_summary if state.goal_summary else ""
        if goal_text and len(goal_text) > 15:
            goal_text = goal_text[:12] + "..."

        # 任务完成后不再刷新计时器显示
        if state.status == AppStatus.FINISHED:
            elapsed = state.final_elapsed if hasattr(state, 'final_elapsed') else state.get_elapsed_time()
            round_elapsed = "--:--"
        else:
            elapsed = state.get_elapsed_time()
            round_elapsed = state.get_round_elapsed() if state.current_round > 0 else "--:--"

        input_t = state.total_input_tokens
        output_t = state.total_output_tokens
        cache_t = state.total_cache_read_tokens

        goal_prefix = f"Goal: {goal_text}" if goal_text else ""

        return HTML(
            f" <b><style color='{status_color}'>{state.status.value}</style></b>"
            + (f" | {goal_prefix}" if goal_prefix else "")
            + f" | Time: <b><style color='ansigreen'>{elapsed}</style></b>"
            + f" | R: <b>{state.current_round}</b>({round_elapsed}) | "
            f"I: <b>{input_t:,}</b>"
            f" | O: <b>{output_t:,}</b>"
            + (f" | C: <b>{cache_t:,}</b>" if cache_t else "")
        )

    style = Style.from_dict({"bottom-toolbar": "#333333 bg:#dddddd"})
    session = PromptSession(bottom_toolbar=get_bottom_toolbar, style=style)

    aprint("\n \033[1mClaude Code Looper \033[0m")
    aprint("\033[90m- /goal <目标>   设定目标（自动润色精简）")
    aprint("- /start         开始运行")
    aprint("- /clear         清除goal")
    aprint("- /goal          查看当前goal")
    aprint("- Ctrl+C 暂停\n\033[0m")

    background_task: Optional[asyncio.Task] = None
    prompt_html = HTML("<b><style color='#00aa00'>Command ></style></b> ")

    while True:
        # 检查后台任务是否完成，如果完成则停止计时器刷新
        if background_task and background_task.done():
            timer_running["value"] = False
            if timer_task and not timer_task.done():
                timer_task.cancel()
                try:
                    await timer_task
                except asyncio.CancelledError:
                    pass
            timer_task = None
            background_task = None

        try:
            with patch_stdout():
                user_input = (await session.prompt_async(prompt_html)).strip()

            if user_input.lower() == "exit":
                # 停止计时器刷新
                timer_running["value"] = False
                if background_task and not background_task.done():
                    background_task.cancel()
                break

            # /start - 开始运行goal
            if user_input.startswith("/start"):
                if not state.goal_set:
                    aprint("\n\033[33m[Error] 没有goal，请先设置goal\033[0m")
                elif state.status == AppStatus.RUNNING:
                    aprint("\n\033[33m[Info] 已在运行中\033[0m")
                else:
                    aprint(f"\n\033[1;32m[Start] 开始执行: {state.refined_goal or state.goal}\033[0m")
                    if not (background_task and not background_task.done()):
                        state.status = AppStatus.RUNNING
                        # 启动计时器刷新任务
                        timer_running["value"] = True
                        timer_task = asyncio.create_task(
                            _timer_refresh_task(session, state, timer_running)
                        )

                        async def self_loop_with_cleanup():
                            """运行 self_loop 并在完成后清理计时器"""
                            nonlocal timer_task
                            try:
                                await self_loop(max_rounds=MAX_ROUNDS, state=state)
                            finally:
                                timer_running["value"] = False
                                if timer_task and not timer_task.done():
                                    timer_task.cancel()
                                    try:
                                        await timer_task
                                    except asyncio.CancelledError:
                                        pass
                                timer_task = None

                        background_task = asyncio.create_task(self_loop_with_cleanup())
                continue

            # /clear - 清除goal
            if user_input.startswith("/clear"):
                state.goal = ""
                state.goal_summary = ""
                state.refined_goal = ""
                state.goal_set = False
                state.memory = []
                state.logs = []
                state.current_round = 0
                state.clear_tokens()
                state.start_time = None
                state.round_start_time = None
                if hasattr(state, 'final_elapsed'):
                    del state.final_elapsed
                aprint("\n\033[90m[Clear] goal已清除\033[0m")
                continue

            # /refine - 润色当前goal
            if user_input.startswith("/refine"):
                if not state.goal_set:
                    aprint("\n\033[33m[Error] 没有goal，请先设置goal\033[0m")
                else:
                    # 先加载 .env
                    import pathlib
                    from dotenv import load_dotenv
                    env_path = pathlib.Path(__file__).parent / ".env"
                    if env_path.exists():
                        load_dotenv(env_path)

                    model = os.environ.get("OPENAI_MODEL", "gpt-4")
                    aprint(f"\n\033[1;34m>>> 正在请求 Refine ({model}) ...\033[0m")
                    old_refined = state.refined_goal or state.goal
                    new_refined = refine_goal_once(goal=state.goal, state=state)
                    if new_refined != old_refined:
                        state.refined_goal = new_refined
                        state.goal_summary = summarize_goal_once(goal=new_refined, state=state)
                        aprint(f"\n\033[1;32m[Refined] {new_refined}\033[0m")
                        if state.goal_summary != new_refined:
                            aprint(f"\033[90m[Summary] 📌 {state.goal_summary}\033[0m")
                    else:
                        aprint(f"\n\033[90m[Refine] 目标未变化（{new_refined}）\033[0m")
                continue

            # /goal 命令
            if user_input.startswith("/goal"):
                goal_text = user_input[5:].strip()
                if goal_text:
                    state.goal = goal_text
                    state.goal_set = True
                    state.memory = []
                    state.logs = []
                    state.current_round = 0
                    state.clear_tokens()
                    state.start_time = None
                    state.round_start_time = None

                    # 自动加载 .env 获取配置
                    import pathlib
                    from dotenv import load_dotenv
                    env_path = pathlib.Path(__file__).parent / ".env"
                    if env_path.exists():
                        load_dotenv(env_path)

                    aprint(f"\n\033[1;32m[Goal Set] 🎯 {state.goal}\033[0m")

                    # 自动润色
                    model = os.environ.get("OPENAI_MODEL", "gpt-4")
                    aprint(f"\033[90m>>> 正在润色目标 ({model}) ...\033[0m")
                    new_refined = refine_goal_once(goal=state.goal, state=state)
                    if new_refined:
                        state.refined_goal = new_refined
                        aprint(f"\033[90m[Refined] {new_refined}\033[0m")

                    # 自动精简
                    aprint("\033[90m>>> 正在精简目标 ...\033[0m")
                    state.goal_summary = summarize_goal_once(goal=state.refined_goal or state.goal, state=state)
                    if state.goal_summary:
                        aprint(f"\033[90m[Summary] 📌 {state.goal_summary}\033[0m")

                    aprint("\033[90m使用 /start 开始运行\033[0m\n")
                elif state.goal_set:
                    aprint(f"\n\033[90m[Current Goal] 🎯 {state.goal}\033[0m")
                    if state.refined_goal:
                        aprint(f"\033[90m[Refined] {state.refined_goal}\033[0m")
                    if state.goal_summary:
                        aprint(f"\033[90m[Summary] 📌 {state.goal_summary}\033[0m")
                else:
                    aprint("\n\033[33m[Usage] /goal <目标>\033[0m")
                continue

            # 无指令直接发消息的处理逻辑
            if user_input:
                if state.goal_set:
                    aprint(f"\n\033[1;33m[Supplement] 追加指令: {user_input}\033[0m")
                    new_goal = update_goal_once(
                        original_goal=state.goal, additional_instruction=user_input, state=state
                    )
                    state.goal = new_goal
                    state.refined_goal = ""  # 不自动润色
                    state.goal_summary = ""
                    aprint(f"\033[90m[Goal Updated] {new_goal}\033[0m")
                    aprint("\033[90m使用 /refine 润色goal\033[0m\n")
                else:
                    state.goal = user_input
                    state.goal_set = True
                    state.memory = []
                    state.logs = []
                    state.current_round = 0
                    state.clear_tokens()
                    state.start_time = None
                    state.round_start_time = None

                    # 自动加载 .env 获取配置
                    import pathlib
                    from dotenv import load_dotenv
                    env_path = pathlib.Path(__file__).parent / ".env"
                    if env_path.exists():
                        load_dotenv(env_path)

                    aprint(f"\n\033[1;32m[Goal Set] 🎯 {state.goal}\033[0m")

                    # 自动润色
                    model = os.environ.get("OPENAI_MODEL", "gpt-4")
                    aprint(f"\033[90m>>> 正在润色目标 ({model}) ...\033[0m")
                    new_refined = refine_goal_once(goal=state.goal, state=state)
                    if new_refined:
                        state.refined_goal = new_refined
                        aprint(f"\033[90m[Refined] {new_refined}\033[0m")

                    # 自动精简
                    aprint("\033[90m>>> 正在精简目标 ...\033[0m")
                    state.goal_summary = summarize_goal_once(goal=state.refined_goal or state.goal, state=state)
                    if state.goal_summary:
                        aprint(f"\033[90m[Summary] 📌 {state.goal_summary}\033[0m")

                    aprint("\033[90m使用 /start 开始运行\033[0m\n")

        except KeyboardInterrupt:
            if background_task and not background_task.done():
                aprint("\n\n\033[1;33m⚠️  检测到中断 (Ctrl+C)！正在暂停后台任务...\033[0m")
                background_task.cancel()
                try:
                    await background_task
                except asyncio.CancelledError:
                    pass
                state.status = AppStatus.PAUSED
                # 停止计时器刷新
                timer_running["value"] = False
                if timer_task and not timer_task.done():
                    timer_task.cancel()
                    try:
                        await timer_task
                    except asyncio.CancelledError:
                        pass
                timer_task = None
                background_task = None
            else:
                aprint("\n[System] 退出程序。")
                break


if __name__ == "__main__":
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

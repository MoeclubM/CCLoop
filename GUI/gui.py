"""
CCLoop GUI版本 - 轻量级图形界面
"""

import asyncio
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import Optional
import pathlib
import re

# Windows环境修复
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

# 添加CLI目录到Python路径
cli_path = pathlib.Path(__file__).parent.parent / "CLI"
if str(cli_path) not in sys.path:
    sys.path.insert(0, str(cli_path))

# 导入核心模块
from dotenv import load_dotenv
from loop_core import (
    AppStatus,
    CompleteLoopState,
    judge_once,
    refine_goal_once,
    self_loop,
    summarize_goal_once,
    update_goal_once,
)

# 加载环境变量
env_path = pathlib.Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", 6))


class GUIOutputRedirector:
    """输出重定向适配器"""

    def __init__(self, gui_instance):
        self.gui = gui_instance
        self.ansi_pattern = re.compile(r'\033\[[0-9;]*m')

    def write(self, text: str):
        """重定向输出到GUI"""
        if text:
            self.gui._print_output(text)

    def flush(self):
        """刷新输出"""
        pass


class CCLoopGUI:
    """CCLoop GUI主类"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CCLoop - Claude Code 自循环控制器")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)

        self.state = CompleteLoopState()
        self.loop_task: Optional[asyncio.Task] = None
        self.loop_running = False
        self.timer_running = False

        # 创建输出重定向器
        self.output_redirector = GUIOutputRedirector(self)

        # 注入输出重定向到display模块
        self._inject_output_redirector()

        # 先设置样式，再设置UI（因为UI会使用colors）
        self._setup_styles()
        self._setup_ui()

    def _inject_output_redirector(self):
        """注入输出重定向到display模块"""
        try:
            import display
            original_aprint = display.aprint
            original_print_box = display._print_box

            def gui_aprint(s: str = "", *, end: str = "\n", flush: bool = False) -> None:
                """GUI版本的aprint"""
                self._print_output(s + end)

            def gui_print_box(title: str, content: str, style: str = "normal", max_lines: int = 8) -> None:
                """GUI版本的_print_box"""
                self._print_box(title, content, style)

            display.aprint = gui_aprint
            display._print_box = gui_print_box

            # 注入到loop_core模块
            import loop_core
            loop_core._refresh_ui = self._refresh_ui

        except Exception:
            pass

    def _refresh_ui(self):
        """刷新UI"""
        self.root.after(0, lambda: None)

    def _setup_styles(self):
        """设置样式"""
        self.colors = {
            "idle": "#808080",
            "running": "#00aa00",
            "paused": "#ffaa00",
            "finished": "#0088ff",
            "bg": "#ffffff",
            "fg": "#000000",
            "input_bg": "#f5f5f5",
            "output_bg": "#1e1e1e",
            "output_fg": "#d4d4d4",
        }

    def _setup_ui(self):
        """设置UI界面"""
        # 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # 目标区域
        goal_frame = ttk.LabelFrame(main_frame, text="目标", padding="5")
        goal_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        goal_frame.columnconfigure(0, weight=1)

        self.goal_entry = ttk.Entry(goal_frame)
        self.goal_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))

        self.set_goal_btn = ttk.Button(goal_frame, text="设置目标", command=self._on_set_goal)
        self.set_goal_btn.grid(row=0, column=1)

        # 输出区域
        output_frame = ttk.LabelFrame(main_frame, text="输出日志", padding="5")
        output_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            bg=self.colors["output_bg"],
            fg=self.colors["output_fg"],
            font=("Consolas", 9),
        )
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置文本标签
        self.output_text.tag_config("normal", foreground=self.colors["output_fg"])
        self.output_text.tag_config("bold", foreground=self.colors["output_fg"], font=("Consolas", 9, "bold"))
        self.output_text.tag_config("info", foreground="#4fc1ff")
        self.output_text.tag_config("success", foreground="#4ec9b0")
        self.output_text.tag_config("warning", foreground="#dcdcaa")
        self.output_text.tag_config("error", foreground="#f14c4c")
        self.output_text.tag_config("dim", foreground="#808080")
        self.output_text.tag_config("tool_use", foreground="#569cd6")
        self.output_text.tag_config("tool_result", foreground="#ce9178")

        # 控制按钮区域
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self.start_btn = ttk.Button(control_frame, text="开始", command=self._on_start)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.pause_btn = ttk.Button(control_frame, text="暂停", command=self._on_pause, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.clear_btn = ttk.Button(control_frame, text="清除", command=self._on_clear)
        self.clear_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.refine_btn = ttk.Button(control_frame, text="润色", command=self._on_refine)
        self.refine_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.exit_btn = ttk.Button(control_frame, text="退出", command=self._on_exit)
        self.exit_btn.pack(side=tk.RIGHT)

        # 状态栏
        self.status_frame = ttk.Frame(self.root, relief=tk.SUNKEN, padding="2")
        self.status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.status_label = ttk.Label(self.status_frame, text="状态: 空闲")
        self.status_label.pack(side=tk.LEFT, padx=(5, 10))

        self.goal_label = ttk.Label(self.status_frame, text="")
        self.goal_label.pack(side=tk.LEFT, padx=(0, 10))

        self.time_label = ttk.Label(self.status_frame, text="时间: 00:00")
        self.time_label.pack(side=tk.LEFT, padx=(0, 10))

        self.round_label = ttk.Label(self.status_frame, text="轮次: 0")
        self.round_label.pack(side=tk.LEFT, padx=(0, 10))

        self.token_label = ttk.Label(self.status_frame, text="Token: I:0 O:0")
        self.token_label.pack(side=tk.LEFT, padx=(0, 5))

        # 启动定时器更新
        self._start_timer()

    def _start_timer(self):
        """启动定时器更新UI"""
        if not self.timer_running:
            self.timer_running = True
            self._update_status_bar()
            self.root.after(1000, self._update_status_bar)

    def _update_status_bar(self):
        """更新状态栏"""
        if not self.timer_running:
            return

        # 更新状态
        status_color = self.colors.get(self.state.status.value.lower(), self.colors["idle"])
        self.status_label.config(text=f"状态: {self.state.status.value}", foreground=status_color)

        # 更新目标显示
        goal_text = self.state.goal_summary if self.state.goal_summary else ""
        if goal_text and len(goal_text) > 15:
            goal_text = goal_text[:12] + "..."
        self.goal_label.config(text=f"目标: {goal_text}" if goal_text else "")

        # 更新时间
        if self.state.status == AppStatus.FINISHED:
            elapsed = self.state.final_elapsed if hasattr(self.state, 'final_elapsed') else self.state.get_elapsed_time()
        else:
            elapsed = self.state.get_elapsed_time()
        self.time_label.config(text=f"时间: {elapsed}")

        # 更新轮次
        self.round_label.config(text=f"轮次: {self.state.current_round}")

        # 更新Token
        input_t = self.state.total_input_tokens
        output_t = self.state.total_output_tokens
        cache_t = self.state.total_cache_read_tokens
        token_text = f"Token: I:{input_t:,} O:{output_t:,}"
        if cache_t:
            token_text += f" C:{cache_t:,}"
        self.token_label.config(text=token_text)

        # 继续定时器
        self.root.after(1000, self._update_status_bar)

    def _append_output(self, text: str, tag: str = "normal"):
        """追加输出到日志区域"""
        self.output_text.insert(tk.END, text, tag)
        self.output_text.see(tk.END)
        self.output_text.update()

    def _append_colored(self, text: str, color_code: str):
        """追加带颜色的输出"""
        color_map = {
            "\033[90m": "dim",
            "\033[1;35m": "info",
            "\033[1;34m": "info",
            "\033[1;32m": "success",
            "\033[1;33m": "warning",
            "\033[31m": "error",
            "\033[0m": "normal",
        }
        tag = color_map.get(color_code, "normal")
        self._append_output(text, tag)

    def _print_output(self, text: str):
        """打印输出，处理ANSI颜色代码"""
        ansi_pattern = re.compile(r'\033\[[0-9;]*m')
        parts = ansi_pattern.split(text)
        codes = ansi_pattern.findall(text)

        color_map = {
            "\033[90m": "dim",
            "\033[1;35m": "info",
            "\033[1;34m": "info",
            "\033[1;32m": "success",
            "\033[1;33m": "warning",
            "\033[31m": "error",
            "\033[36m": "info",
            "\033[0m": "normal",
            "\033[1m": "bold",
            "\033[0;33m": "warning",
            "\033[0;32m": "success",
        }

        for i, part in enumerate(parts):
            if part:
                tag = "normal"
                if i > 0 and i - 1 < len(codes):
                    code = codes[i - 1]
                    tag = color_map.get(code, "normal")
                self._append_output(part, tag)

    def _print_box(self, title: str, content: str, style: str = "normal"):
        """打印带边框的内容框（简化版）"""
        icon = "🛠️ " if style == "tool_use" else "📝" if style == "tool_result" else "ℹ️ "
        self._append_output(f"\n{icon} {title}\n", "info")
        
        lines = content.strip().split("\n") if content else [""]
        for line in lines[:20]:  # 限制显示行数
            self._append_output(f"  {line}\n", "dim")
        if len(lines) > 20:
            self._append_output(f"  ... ({len(lines) - 20} more lines)\n", "dim")
        self._append_output("\n", "normal")

    def _on_set_goal(self):
        """设置目标"""
        goal_text = self.goal_entry.get().strip()
        if not goal_text:
            messagebox.showwarning("警告", "请输入目标")
            return

        self.state.goal = goal_text
        self.state.goal_set = True
        self.state.memory = []
        self.state.logs = []
        self.state.current_round = 0
        self.state.clear_tokens()
        self.state.start_time = None
        self.state.round_start_time = None

        self._print_output(f"\n[Goal Set] 🎯 {goal_text}\n")

        # 自动润色
        model = os.environ.get("OPENAI_MODEL", "gpt-4")
        self._print_output(f">>> 正在润色目标 ({model}) ...\n")
        new_refined = refine_goal_once(goal=self.state.goal, state=self.state)
        if new_refined:
            self.state.refined_goal = new_refined
            self._print_output(f"[Refined] {new_refined}\n")

        # 自动精简
        self._print_output(">>> 正在精简目标 ...\n")
        self.state.goal_summary = summarize_goal_once(goal=self.state.refined_goal or self.state.goal, state=self.state)
        if self.state.goal_summary:
            self._print_output(f"[Summary] 📌 {self.state.goal_summary}\n")

        self._print_output("使用 '开始' 按钮运行\n")
        self.goal_entry.delete(0, tk.END)

    def _on_start(self):
        """开始运行"""
        if not self.state.goal_set:
            messagebox.showwarning("警告", "没有目标，请先设置目标")
            return

        if self.state.status == AppStatus.RUNNING:
            messagebox.showinfo("提示", "已在运行中")
            return

        self._print_output(f"\n[Start] 开始执行: {self.state.refined_goal or self.state.goal}\n")

        # 在新线程中运行异步任务
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._run_loop_async())
            finally:
                loop.close()

        self.loop_running = True
        threading.Thread(target=run_loop, daemon=True).start()

    async def _run_loop_async(self):
        """异步运行循环"""
        try:
            result = await self_loop(max_rounds=MAX_ROUNDS, state=self.state)
            self._print_output(f"\n[Done] {result['summary']}\n")
        except asyncio.CancelledError:
            self._print_output("\n[Info] 任务已暂停\n")
        except Exception as e:
            self._print_output(f"\n[Error] {e}\n")
        finally:
            self.loop_running = False
            self.root.after(0, lambda: self._update_button_states())

    def _on_pause(self):
        """暂停运行"""
        if self.loop_running:
            self.loop_running = False
            self._print_output("\n[Info] 正在暂停...\n")

    def _on_clear(self):
        """清除目标"""
        self.state.goal = ""
        self.state.goal_summary = ""
        self.state.refined_goal = ""
        self.state.goal_set = False
        self.state.memory = []
        self.state.logs = []
        self.state.current_round = 0
        self.state.clear_tokens()
        self.state.start_time = None
        self.state.round_start_time = None
        if hasattr(self.state, 'final_elapsed'):
            del self.state.final_elapsed

        self.output_text.delete(1.0, tk.END)
        self._print_output("[Clear] 目标已清除\n")

    def _on_refine(self):
        """润色目标"""
        if not self.state.goal_set:
            messagebox.showwarning("警告", "没有目标，请先设置目标")
            return

        model = os.environ.get("OPENAI_MODEL", "gpt-4")
        self._print_output(f"\n>>> 正在请求 Refine ({model}) ...\n")
        old_refined = self.state.refined_goal or self.state.goal
        new_refined = refine_goal_once(goal=self.state.goal, state=self.state)

        if new_refined != old_refined:
            self.state.refined_goal = new_refined
            self.state.goal_summary = summarize_goal_once(goal=new_refined, state=self.state)
            self._print_output(f"[Refined] {new_refined}\n")
            if self.state.goal_summary != new_refined:
                self._print_output(f"[Summary] 📌 {self.state.goal_summary}\n")
        else:
            self._print_output(f"[Refine] 目标未变化（{new_refined}）\n")

    def _on_exit(self):
        """退出程序"""
        if self.loop_running:
            if not messagebox.askyesno("确认", "任务正在运行，确定要退出吗？"):
                return
            self.loop_running = False

        self.timer_running = False
        self.root.quit()
        self.root.destroy()

    def _update_button_states(self):
        """更新按钮状态"""
        if self.state.status == AppStatus.RUNNING:
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
        else:
            self.start_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED)


def main():
    """主函数"""
    root = tk.Tk()
    app = CCLoopGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

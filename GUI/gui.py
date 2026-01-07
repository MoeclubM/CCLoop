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

# 添加CORE目录到Python路径
core_path = pathlib.Path(__file__).parent.parent / "CORE"
if str(core_path) not in sys.path:
    sys.path.insert(0, str(core_path))

# 加载环境变量（必须在导入loop_core之前）
from dotenv import load_dotenv
env_path = pathlib.Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# 导入核心模块
from loop_core import (
    AppStatus,
    CompleteLoopState,
    refine_goal_once,
    self_loop,
    summarize_goal_once,
)

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
        self.working_dir = os.getcwd()

        # 创建输出重定向器
        self.output_redirector = GUIOutputRedirector(self)

        # 注入输出重定向到display模块
        self._inject_output_redirector()

        # 先设置样式，再设置UI（因为UI会使用colors）
        self._setup_styles()
        self._setup_ui()

    def _inject_output_redirector(self):
        """设置CORE模块的回调函数"""
        def on_text(text: str):
            """文本输出回调"""
            self._print_output(text)

        def on_tool_use(tool_name: str, input_data: str):
            """工具使用回调"""
            self._append_output(f"\n🛠️ Tool Use: {tool_name}\n", "info", "tool")
            if input_data:
                # 统一换行符并分割
                clean_data = input_data.replace("\r\n", "\n").replace("\r", "\n")
                lines = clean_data.strip().split("\n") if clean_data else [""]

                for line in lines[:10]:
                    if line.strip().startswith("$"):
                        self._append_output(f"  {line}\n", "bold", "tool")
                    else:
                        self._append_output(f"  {line}\n", "dim", "tool")
                if len(lines) > 10:
                    self._append_output(f"  ... ({len(lines) - 10} more lines)\n", "dim", "tool")

        def on_tool_result(result: str):
            """工具结果回调"""
            self._append_output("\n📝 Tool Result:\n", "info", "tool")
            if result:
                self._append_output(result if result.endswith("\n") else result + "\n", "dim", "tool")

        def on_judge(judgment: str):
            """判断回调"""
            self._append_output(f"\n⚖️ Judge: {judgment}\n", "warning", "text")

        def on_status(status: str):
            """状态回调"""
            self._append_output(f"\n📊 Status: {status}\n", "info", "text")

        def on_token(tokens: dict):
            """Token统计回调"""
            pass

        def on_error(error: str):
            """错误回调"""
            self._append_output(f"\n❌ Error: {error}\n", "error", "text")

        def on_raw(raw: str):
            """原始输出回调"""
            self._print_output(raw)

        self.state.callbacks.on_text = on_text
        self.state.callbacks.on_tool_use = on_tool_use
        self.state.callbacks.on_tool_result = on_tool_result
        self.state.callbacks.on_judge = on_judge
        self.state.callbacks.on_status = on_status
        self.state.callbacks.on_token = on_token
        self.state.callbacks.on_error = on_error
        self.state.callbacks.on_raw = on_raw

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
        
        # 设置字体（跨平台支持）
        if sys.platform == "win32":
            self.font_family = "Microsoft YaHei"
        elif sys.platform == "darwin":
            self.font_family = "PingFang SC"
        else:
            self.font_family = "DejaVu Sans Mono"
        
        self.base_font = (self.font_family, 9)
        self.bold_font = (self.font_family, 9, "bold")

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
        goal_frame.columnconfigure(1, weight=1)

        self.goal_entry = ttk.Entry(goal_frame)
        self.goal_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))

        self.set_goal_btn = ttk.Button(goal_frame, text="设置目标", command=self._on_set_goal)
        self.set_goal_btn.grid(row=0, column=1, padx=(0, 5))

        self.dir_entry = ttk.Entry(goal_frame)
        self.dir_entry.grid(row=0, column=2, sticky=(tk.W, tk.E), padx=(0, 5))

        self.set_dir_btn = ttk.Button(goal_frame, text="设置目录", command=self._on_set_dir)
        self.set_dir_btn.grid(row=0, column=3)

        goal_frame.columnconfigure(0, weight=2)
        goal_frame.columnconfigure(2, weight=1)

        # 双列输出区域
        output_frame = ttk.LabelFrame(main_frame, text="输出日志", padding="5")
        output_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.columnconfigure(0, weight=1)
        output_frame.columnconfigure(1, weight=1)
        output_frame.rowconfigure(0, weight=1)

        # 左侧：工具调用日志
        tool_frame = ttk.LabelFrame(output_frame, text="工具调用日志", padding="5")
        tool_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

        self.tool_text = scrolledtext.ScrolledText(
            tool_frame,
            wrap=tk.WORD,
            bg=self.colors["output_bg"],
            fg=self.colors["output_fg"],
            font=self.base_font,
            width=40,
        )
        self.tool_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tool_frame.columnconfigure(0, weight=1)
        tool_frame.rowconfigure(0, weight=1)

        # 配置工具日志文本标签
        self.tool_text.tag_config("normal", foreground=self.colors["output_fg"])
        self.tool_text.tag_config("bold", foreground=self.colors["output_fg"], font=self.bold_font)
        self.tool_text.tag_config("info", foreground="#4fc1ff")
        self.tool_text.tag_config("success", foreground="#4ec9b0")
        self.tool_text.tag_config("warning", foreground="#dcdcaa")
        self.tool_text.tag_config("error", foreground="#f14c4c")
        self.tool_text.tag_config("dim", foreground="#808080")
        self.tool_text.tag_config("tool_use", foreground="#569cd6")
        self.tool_text.tag_config("tool_result", foreground="#ce9178")

        # 右侧：文本输出内容
        text_frame = ttk.LabelFrame(output_frame, text="文本输出内容", padding="5")
        text_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.text_text = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            bg=self.colors["output_bg"],
            fg=self.colors["output_fg"],
            font=self.base_font,
            width=40,
        )
        self.text_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        # 配置文本输出标签
        self.text_text.tag_config("normal", foreground=self.colors["output_fg"])
        self.text_text.tag_config("bold", foreground=self.colors["output_fg"], font=self.bold_font)
        self.text_text.tag_config("info", foreground="#4fc1ff")
        self.text_text.tag_config("success", foreground="#4ec9b0")
        self.text_text.tag_config("warning", foreground="#dcdcaa")
        self.text_text.tag_config("error", foreground="#f14c4c")
        self.text_text.tag_config("dim", foreground="#808080")

        # 控制按钮区域
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self.start_btn = ttk.Button(control_frame, text="开始", command=self._on_start)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.pause_btn = ttk.Button(control_frame, text="暂停", command=self._on_pause, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.clear_btn = ttk.Button(control_frame, text="清除", command=self._on_clear)
        self.clear_btn.pack(side=tk.LEFT, padx=(0, 5))

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
        status_map = {
            AppStatus.IDLE: "idle",
            AppStatus.RUNNING: "running",
            AppStatus.PAUSED: "paused",
            AppStatus.FINISHED: "finished",
        }
        status_key = status_map.get(self.state.status, "idle")
        status_color = self.colors.get(status_key, self.colors["idle"])
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
        if cache_t is not None and cache_t > 0:
            token_text += f" C:{cache_t:,}"
        self.token_label.config(text=token_text)

        # 继续定时器
        self.root.after(1000, self._update_status_bar)

    def _append_output(self, text: str, tag: str = "normal", widget: str = "text"):
        """追加输出到日志区域"""
        if widget == "tool":
            self.tool_text.insert(tk.END, text, tag)
            self.tool_text.see(tk.END)
            self.tool_text.update()
        else:
            self.text_text.insert(tk.END, text, tag)
            self.text_text.see(tk.END)
            self.text_text.update()

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
        if style == "tool_use":
            icon = "🛠️ "
            widget = "tool"
        elif style == "tool_result":
            icon = "📝"
            widget = "tool"
        else:
            icon = "ℹ️ "
            widget = "text"
        
        self._append_output(f"\n{icon} {title}\n", "info", widget)
        
        lines = content.strip().split("\n") if content else [""]
        for line in lines[:20]:  # 限制显示行数
            self._append_output(f"  {line}\n", "dim", widget)
        if len(lines) > 20:
            self._append_output(f"  ... ({len(lines) - 20} more lines)\n", "dim", widget)
        self._append_output("\n", "normal", widget)

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

        self._append_output(f"\n[Goal Set] 🎯 {goal_text}\n", "normal", "text")

        def run_refine():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                model = os.environ.get("OPENAI_MODEL", "gpt-4")
                self.root.after(0, lambda: self._append_output(f">>> 正在润色目标 ({model}) ...\n", "normal", "text"))
                
                new_refined = refine_goal_once(goal=self.state.goal, state=self.state)
                self.root.after(0, lambda: self._on_refine_completed(new_refined))
            finally:
                loop.close()

        threading.Thread(target=run_refine, daemon=True).start()
        self.goal_entry.delete(0, tk.END)

    def _on_refine_completed(self, new_refined: str):
        """润色完成后回调"""
        if new_refined:
            self.state.refined_goal = new_refined
            self._append_output(f"[Refined] {new_refined}\n", "normal", "text")

        def run_summarize():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                self.root.after(0, lambda: self._append_output(">>> 正在精简目标 ...\n", "normal", "text"))
                summary = summarize_goal_once(goal=self.state.refined_goal or self.state.goal, state=self.state)
                self.root.after(0, lambda: self._on_summarize_completed(summary))
            finally:
                loop.close()

        threading.Thread(target=run_summarize, daemon=True).start()

    def _on_summarize_completed(self, summary: str):
        """精简完成后回调"""
        if summary:
            self.state.goal_summary = summary
            self._append_output(f"[Summary] 📌 {summary}\n", "normal", "text")
        self._append_output("使用 '开始' 按钮运行\n", "normal", "text")

    def _on_set_dir(self):
        """设置运行目录"""
        dir_text = self.dir_entry.get().strip()
        if not dir_text:
            messagebox.showwarning("警告", "请输入目录路径")
            return

        if not os.path.isdir(dir_text):
            messagebox.showerror("错误", f"目录不存在: {dir_text}")
            return

        self.working_dir = dir_text
        os.chdir(self.working_dir)
        self._append_output(f"\n[Directory Set] 📁 {self.working_dir}\n", "normal", "text")
        self.dir_entry.delete(0, tk.END)

    def _on_start(self):
        """开始运行"""
        if not self.state.goal_set:
            messagebox.showwarning("警告", "没有目标，请先设置目标")
            return

        if self.state.status == AppStatus.RUNNING:
            messagebox.showinfo("提示", "已在运行中")
            return

        self._append_output(f"\n[Start] 开始执行: {self.state.refined_goal or self.state.goal}\n", "normal", "text")
        self._append_output(f"[Working Directory] 📁 {self.working_dir}\n", "normal", "text")

        # 停止之前的循环任务（如果有）
        if self.loop_task and not self.loop_task.done():
            self.loop_task.cancel()

        # 在新线程中运行异步任务
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                self.loop_task = loop.create_task(self._run_loop_async())
                loop.run_until_complete(self.loop_task)
            except asyncio.CancelledError:
                pass
            finally:
                loop.close()

        self.loop_running = True
        self._update_button_states()
        threading.Thread(target=run_loop, daemon=True).start()

    async def _run_loop_async(self):
        """异步运行循环"""
        try:
            result = await self_loop(max_rounds=MAX_ROUNDS, cwd=self.working_dir, state=self.state)
            self._append_output(f"\n[Done] {result['summary']}\n", "normal", "text")
        except asyncio.CancelledError:
            self._append_output("\n[Info] 任务已暂停\n", "normal", "text")
        except Exception as e:
            self._append_output(f"\n[Error] {e}\n", "error", "text")
        finally:
            self.loop_running = False
            self.root.after(0, lambda: self._update_button_states())

    def _on_pause(self):
        """暂停运行"""
        if self.loop_task and not self.loop_task.done():
            self._append_output("\n[Info] 正在暂停...\n", "normal", "text")
            self.loop_task.cancel()

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

        self.tool_text.delete(1.0, tk.END)
        self.text_text.delete(1.0, tk.END)
        self._append_output("[Clear] 目标已清除\n", "normal", "text")

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

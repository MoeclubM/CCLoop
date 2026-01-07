# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 开发命令
- **安装依赖**: `pip install openai python-dotenv prompt-toolkit anyio wcwidth`
- **运行 TUI 版本**: `python TUI/loop.py`
- **运行 GUI 版本**: `python GUI/gui.py` (Windows 可使用 `GUI\run_gui.bat`)
- **配置环境**: 复制 `.env.example` 为 `.env` 并填写 `OPENAI_API_KEY`, `OPENAI_BASE_URL` (支持 OpenAI 兼容 API), `OPENAI_MODEL` 等。

## 项目架构
CCLoop 是一个基于 Python 的 Claude Code 自动循环控制器。其核心逻辑是通过一个 Judge 模型评估 Claude Code 的输出，并决定是否继续循环执行。

### 核心目录结构
- `CORE/`: 核心逻辑模块
  - `loop_core.py`: 核心循环逻辑 (Self-loop)、Judge 判定调用、Claude 进程管理 (使用 `asyncio.create_subprocess_exec` 调用 `claude` CLI)。
  - `prompts.py`: 包含 Judge、Refine、Summarize 和 Goal Updater 的系统提示词。
  - `token_stats.py`: 定义 `LoopState`、`RoundLog` 和 `TokenStats` 等数据结构，负责 Token 统计和状态跟踪。
  - `json_utils.py`: 提供流式 JSON 解析和编码清理工具。
- `TUI/`: 基于 `prompt-toolkit` 的终端界面实现。
  - `loop.py`: TUI 入口，处理用户交互命令 (如 `/goal`, `/start`, `/clear`)。
  - `display.py`: 负责 TUI 的格式化输出、颜色和状态栏显示。
- `GUI/`: 基于 `tkinter` 的图形界面实现。
  - `gui.py`: GUI 入口，提供工具调用日志和文本输出的双列显示。

### 关键设计模式
- **回调机制**: `CORE` 模块通过 `OutputCallbacks` 对象与 UI 层通信，解耦了业务逻辑与展示逻辑。
- **状态管理**: `CompleteLoopState` 类集中管理整个会话的状态、计时、Token 消耗和历史日志。
- **异步处理**: 使用 `asyncio` 管理 Claude 进程和网络请求，UI 线程与执行线程通过异步事件或回调同步。

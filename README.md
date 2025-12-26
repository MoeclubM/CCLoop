# CCLoop - Claude Code 自循环控制器

一个让 Claude Code 自动完成复杂任务的智能循环控制器，通过 Judge 模型评估进度并决定下一步操作。

## 工作原理

```
用户输入目标 → Claude Code 执行 → Judge 评估结果 → 未完成则继续循环
```

1. 用户输入一个目标任务
2. Claude Code 执行任务
3. Judge（OpenAI 兼容模型）评估是否完成
4. 如未完成，Judge 给出下一步指示，继续循环
5. 直到任务完成或达到最大轮次

## 功能特性

- **自动循环执行**：Judge 模型智能判断任务进度，自动决定是否继续
- **交互式干预**：运行中可输入追加指令干预执行过程
- **目标管理**：支持目标的润色、精简和动态更新
- **实时状态显示**：底部状态栏显示当前状态、轮次和 Token 消耗
- **Claude 输出解析**：实时解析并展示 Claude 的工具调用和执行结果

## 安装

```bash
pip install openai python-dotenv prompt-toolkit anyio wcwidth
```

## 配置

复制 `.env.example` 为 `.env` 并填写：

```env
OPENAI_BASE_URL=https://api.openai.com/v1   # API 地址（支持 OpenAI 兼容的任何 API）
OPENAI_MODEL=gpt-5-mini                        # Judge 模型，推荐使用表达能力强的小模型
OPENAI_API_KEY=sk-xxx                       # API Key
MAX_ROUNDS=100                              # 最大循环轮次（默认100）
```

### 可选配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CLAUDE_CLI_ARGS` | Claude CLI 额外参数 | `--print --verbose --output-format stream-json` |

## 使用

### CLI版本

```bash
python CLI/loop.py
```

### GUI版本

```bash
python GUI/gui.py
```

或使用启动脚本：

**Windows:**
```bash
GUI\run_gui.bat
```

**Linux/Mac:**
```bash
chmod +x GUI/run_gui.sh
GUI/run_gui.sh
```

### CLI版本交互命令

| 命令 | 说明 |
|------|------|
| `/goal <目标>` | 设定目标（设为当前执行目标） |
| `/start` | 开始运行当前 goal |
| `/goal` | 查看当前 goal（含润色版和精简版） |
| `/clear` | 清除 goal |
| 直接输入消息 | 无 goal 时设为 goal，有 goal 时视为补充指令 |
| `exit` | 退出程序 |
| `Ctrl+C` | 暂停当前运行 |

### 运行示例

```
Command > /goal 完善README文档并优化代码

[Goal Set] 🎯 完善README文档并优化代码
[Refined] 请完善项目的README文档，并优化代码结构和性能
[Summary] 完善README并优化代码

Command > /start

>>> 调用 Claude Code ...

[Judge Output] {"done": false, "summary": "已添加新功能说明，文档结构待优化", "next_prompt": "继续完善文档结构"}
👉 Judge 判定: Done=False
   摘要: 已添加新功能说明，文档结构待优化
   指示: 继续完善文档结构

... 继续循环 ...
```

## 状态栏说明

底部状态栏显示：
- **状态指示器**：空闲（灰色）/ 运行中（绿色）/ 已暂停（黄色）/ 已完成（蓝色）
- **当前目标**：精简版目标（15字以内）
- **轮次计数**：当前循环轮次
- **Token 消耗**：累计 Token 使用量

## 项目架构


```
CCLoop/
├── CLI/             # CLI版本
│   ├── loop.py          # 主程序，TUI 入口，命令行交互与主循环调度
│   ├── loop_core.py     # 核心循环与状态管理，Judge 调用、目标处理、循环控制
│   ├── display.py       # TUI/输出工具，格式化打印、状态栏、颜色输出
│   ├── json_utils.py    # JSON 解析增强，流式解析、编码处理
│   ├── prompts.py       # Judge/Refine/Summarize 等系统提示词
│   ├── token_stats.py   # Token 统计与数据结构
│   └── .env.example     # 环境变量配置模板
├── GUI/             # GUI版本
│   ├── gui.py           # GUI主程序，基于tkinter
│   ├── run_gui.bat      # Windows启动脚本
│   ├── run_gui.sh       # Linux/Mac启动脚本
│   └── README.md        # GUI版本说明
├── README.md        # 项目说明文档
├── .env.example     # 环境变量配置模板
└── LICENSE          # MIT 许可证
```

### 核心模块说明

#### CLI版本

| 文件              | 主要功能 |
|-------------------|--------------------------------------------------|
| CLI/loop.py       | TUI 主循环，命令行交互，调度核心逻辑             |
| CLI/loop_core.py  | 核心循环、Judge 调用、目标处理、状态管理         |
| CLI/display.py    | 终端输出美化、状态栏、格式化打印                 |
| CLI/json_utils.py | JSON 解析、编码修复、流式处理                   |
| CLI/prompts.py    | Judge/Refine/Summarize 等系统提示词              |
| CLI/token_stats.py| Token 统计、LoopState/RoundLog 数据结构          |

#### GUI版本

| 文件              | 主要功能 |
|-------------------|--------------------------------------------------|
| GUI/gui.py        | GUI主程序，基于tkinter的图形界面                 |

#### 主要流程与职责

- **TUI 主循环**（CLI/loop.py）：负责命令行交互、用户输入、状态栏刷新。
- **核心循环与 Judge**（CLI/loop_core.py）：实现自循环、目标管理、Judge 评估与决策。
- **输出与美化**（CLI/display.py）：负责所有终端输出、颜色、宽度、状态栏等。
- **JSON 工具**（CLI/json_utils.py）：增强 JSON 解析能力，处理编码与流式数据。
- **提示词管理**（CLI/prompts.py）：集中管理 Judge/Refine/Summarize 等提示词。
- **Token 统计**（CLI/token_stats.py）：统计各类 Token 消耗，记录循环日志。
- **GUI 界面**（GUI/gui.py）：提供图形化操作界面，集成核心循环逻辑。

## 常见问题

### Q: 如何调整最大轮次？

修改 `.env` 文件中的 `MAX_ROUNDS` 值，或在运行时通过环境变量覆盖。

### Q: 支持 Windows 吗？

支持。CCLoop 会自动处理 Powershell 中的编码问题。

### Q: Token 统计包含哪些内容？

包含输入 Token、输出 Token、缓存读取和缓存创建的所有 Token。

## 贡献

欢迎提交 Issue 和 Pull Request！

## License

MIT License

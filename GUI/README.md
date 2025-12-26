# CCLoop GUI版本

CCLoop的轻量级图形界面版本，基于Python内置的tkinter库开发。

## 特性

- **轻量级**：使用Python内置的tkinter，无需额外安装GUI框架
- **实时输出**：实时显示Claude Code的执行日志
- **状态监控**：底部状态栏显示运行状态、轮次、时间和Token消耗
- **颜色高亮**：输出日志支持ANSI颜色代码解析和显示
- **异步执行**：核心循环在后台线程运行，不阻塞GUI界面

## 安装依赖

```bash
pip install openai python-dotenv anyio wcwidth
```

## 配置

在项目根目录创建 `.env` 文件：

```env
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4
OPENAI_API_KEY=sk-xxx
MAX_ROUNDS=100
```

## 运行

### Windows
双击 `run_gui.bat` 或在命令行运行：
```bash
python gui.py
```

### Linux/Mac
```bash
chmod +x run_gui.sh
./run_gui.sh
```

或直接运行：
```bash
python3 gui.py
```

## 使用说明

### 界面布局

1. **目标输入区**：输入目标任务，点击"设置目标"
2. **输出日志区**：显示Claude Code的执行输出
3. **控制按钮**：
   - 开始：启动任务执行
   - 暂停：暂停当前任务
   - 清除：清除目标和日志
   - 润色：润色当前目标
   - 退出：退出程序
4. **状态栏**：显示当前状态、目标摘要、运行时间、轮次和Token消耗

### 工作流程

1. 在目标输入框输入任务描述
2. 点击"设置目标"按钮（自动润色和精简）
3. 点击"开始"按钮启动执行
4. 观察输出日志和状态栏
5. 可随时点击"暂停"暂停任务
6. 任务完成后可查看结果

### 状态说明

- **空闲**（灰色）：未设置目标或任务已完成
- **运行中**（绿色）：正在执行任务
- **已暂停**（黄色）：任务已暂停
- **已完成**（蓝色）：任务执行完成

## 技术架构

```
GUI/
├── gui.py          # GUI主程序
├── run_gui.bat     # Windows启动脚本
├── run_gui.sh      # Linux/Mac启动脚本
└── README.md       # 本文档
```

### 核心模块

- **GUIOutputRedirector**：输出重定向适配器，将display模块的输出重定向到GUI
- **CCLoopGUI**：主GUI类，负责界面布局和事件处理
- **异步执行**：使用threading和asyncio实现后台任务执行

## 与CLI版本的区别

| 特性 | CLI版本 | GUI版本 |
|------|---------|---------|
| 界面 | 终端TUI | 图形界面 |
| 依赖 | prompt-toolkit | tkinter（内置） |
| 输出 | 终端输出 | 滚动文本框 |
| 交互 | 命令行 | 按钮和输入框 |
| 适用场景 | 高效操作 | 可视化监控 |

## 常见问题

### Q: GUI界面卡顿怎么办？
A: 核心循环在后台线程运行，如遇卡顿请检查系统资源或减少MAX_ROUNDS。

### Q: 如何调整窗口大小？
A: 可以拖动窗口边缘调整大小，最小尺寸为800x600。

### Q: 输出日志太多怎么办？
A: 输出区域支持滚动，可以使用鼠标滚轮或滚动条查看历史日志。

### Q: 支持深色主题吗？
A: 当前版本使用固定的深色输出区域，后续版本可能支持主题切换。

## 开发说明

### 扩展功能

GUI版本采用模块化设计，可以轻松扩展：

1. 添加新按钮：在`_setup_ui()`中添加按钮和事件处理
2. 修改样式：调整`_setup_styles()`中的颜色配置
3. 增强输出：扩展`_print_output()`支持更多ANSI代码

### 调试

如需调试，可以在`gui.py`中添加日志输出：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

MIT License

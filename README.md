# CCLoop - Claude Code 自循环控制器

一个让 Claude Code 自动完成复杂任务的控制器，通过 Judge 模型评估进度并决定下一步操作。

## 工作原理

```
用户输入目标 → Claude Code 执行 → Judge 评估结果 → 未完成则继续循环
```

1. 用户输入一个目标任务
2. Claude Code 执行任务
3. Judge（OpenAI 兼容模型）评估是否完成
4. 如未完成，Judge 给出下一步指示，继续循环
5. 直到任务完成或达到最大轮次

## 安装

```bash
pip install openai python-dotenv prompt-toolkit anyio wcwidth
```

## 配置

复制 `.env.example` 为 `.env` 并填写：

```env
OPENAI_BASE_URL=https://api.openai.com/v1   # Judge API 地址
OPENAI_MODEL=gpt-4                          # Judge 模型
OPENAI_API_KEY=sk-xxx                       # API Key
MAX_ROUNDS=100                              # 最大循环轮次
```

## 使用

```bash
python loop.py
```

启动后：
- 输入目标任务开始执行
- 运行中可输入追加指令干预
- `Ctrl+C` 暂停
- 输入 `exit` 退出

## 状态栏

底部状态栏显示：
- 当前状态（空闲/运行中/已暂停/已完成）
- 当前轮次
- 累计 Token 消耗

## License

MIT

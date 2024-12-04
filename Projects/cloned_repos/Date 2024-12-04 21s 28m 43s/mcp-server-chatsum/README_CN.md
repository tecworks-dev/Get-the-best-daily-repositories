# mcp-server-chatsum

这个 MCP 服务器用于总结您的聊天消息。

![预览](./preview.png)

> **在开始之前**
>
> 移动到 [chatbot](./chatbot) 目录，按照 [README_CN.md](./chatbot/README_CN.md) 设置聊天数据库。
>
> 启动聊天机器人以保存您的聊天消息。

## 功能特性

### 资源

### 工具

- `query_chat_messages` - 查询聊天消息
  - 使用给定参数查询聊天消息
  - 根据查询提示总结聊天消息

### 提示词

## 开发

1. 设置环境变量：

在根目录创建 `.env` 文件，并设置您的聊天数据库路径。

```txt
CHAT_DB_PATH=path-to/chatbot/data/chat.db
```

2. 安装依赖：

```bash
pnpm install
```

构建服务器：

```bash
pnpm build
```

用于自动重新构建的开发模式：

```bash
pnpm watch
```

## 安装

要与 Claude Desktop 一起使用，请添加服务器配置：

MacOS 路径：`~/Library/Application Support/Claude/claude_desktop_config.json`
Windows 路径：`%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "mcp-server-chatsum": {
      "command": "path-to/bin/node",
      "args": ["path-to/mcp-server-chatsum/build/index.js"],
      "env": {
        "CHAT_DB_PATH": "path-to/mcp-server-chatsum/chatbot/data/chat.db"
      }
    }
  }
}
```

### 调试

由于 MCP 服务器通过标准输入输出进行通信，调试可能具有挑战性。我们建议使用 [MCP Inspector](https://github.com/modelcontextprotocol/inspector)，可以通过包脚本使用：

```bash
pnpm inspector
```

Inspector 将提供一个 URL，用于在浏览器中访问调试工具。

## 交流社区

- [MCP Server Telegram](https://t.me/+N0gv4O9SXio2YWU1)
- [MCP Server Discord](https://discord.gg/RsYPRrnyqg)

## 关于作者

- [idoubi](https://bento.me/idoubi)
- [跟艾逗笔学全栈开发](https://1024.pagen.io)

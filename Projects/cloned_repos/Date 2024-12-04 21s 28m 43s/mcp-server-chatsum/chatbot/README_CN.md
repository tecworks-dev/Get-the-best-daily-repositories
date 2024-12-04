# 聊天机器人

将您的聊天消息保存到本地 SQLite 数据库。

## 前置条件

1. 在本地机器上安装 `sqlite3`。

对于 macOS：

```shell
brew install sqlite3
```

2. 设置环境变量。

在根目录创建 `.env` 文件，并设置您的聊天数据库路径。

```txt
CHAT_DB_PATH=path-to/data/chat.db
```

3. 初始化聊天数据库。

使用 `sqlite3` 命令连接到您的聊天数据库

```shell
sqlite3 path-to/data/chat.db
```

使用 `install.sql` 创建 `chat_messages` 表。

```sql
CREATE TABLE chat_messages (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    created_at INTEGER NOT NULL,
    msg_id TEXT NOT NULL,
    room_id TEXT,
    room_name TEXT,
    room_avatar TEXT,
    talker_id TEXT NOT NULL,
    talker_name TEXT,
    talker_avatar TEXT,
    content TEXT,
    msg_type INTEGER,
    url_title TEXT,
    url_desc TEXT,
    url_link TEXT,
    url_thumb TEXT
);
```

## 运行聊天机器人

1. 安装依赖。

```shell
pnpm install
```

2. 启动聊天机器人。

```shell
pnpm start
```

3. 使用微信登录

用微信扫描二维码登录。让聊天机器人自动接收并保存聊天消息。

> **注意：**
>
> - 聊天机器人使用 `wechaty` 的 `wechaty-puppet-wechat4u` 协议实现 RPA。
> - 可能会被微信封禁。请谨慎使用您的微信账号。

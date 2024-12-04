# chatbot

Save your chat messages to local sqlite database.

[中文说明](README_CN.md)

## Prerequisites

1. Install `sqlite3` in your local machine.

for macos:

```shell
brew install sqlite3
```

2. Set your environment variables.

create `.env` file in the root directory, and set your chat database path.

```txt
CHAT_DB_PATH=path-to/data/chat.db
```

3. Init chat database.

connect to your chat database with `sqlite3` command

```shell
sqlite3 path-to/data/chat.db
```

create table `chat_messages` with `install.sql`.

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

## Run chatbot

1. Install dependencies.

```shell
pnpm install
```

2. Start chatbot.

```shell
pnpm start
```

3. Login with your WeChat

scan the QR code with your WeChat app. Let chatbot auto receive and save chat messages.

> **Attention:**
>
> - chatbot use `wechaty` with `wechaty-puppet-wechat4u` to run RPA.
> - it may be blocked by WeChat. Be careful with your WeChat account.

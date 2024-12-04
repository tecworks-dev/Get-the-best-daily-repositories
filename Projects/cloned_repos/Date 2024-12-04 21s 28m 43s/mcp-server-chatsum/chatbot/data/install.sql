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
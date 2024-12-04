import { ChatMessage } from "../types/chat_message.js";
import { getDb } from "./db.js";

export async function queryChatMessages({
  room_names,
  talker_names,
  page = 1,
  limit = 100,
}: {
  room_names?: string[];
  talker_names?: string[];
  page?: number;
  limit?: number;
}): Promise<ChatMessage[]> {
  try {
    const db = getDb();
    if (!db) {
      throw new Error("db is not connected");
    }

    const offset = (page - 1) * limit;

    let sql = `SELECT * FROM chat_messages`;
    let values: any[] = [];

    if (room_names && room_names.length > 0) {
      sql += ` WHERE room_name IN (${room_names.map(() => "?").join(",")})`;
      values.push(...room_names);
    }

    if (talker_names && talker_names.length > 0) {
      sql += ` WHERE talker_name IN (${talker_names.map(() => "?").join(",")})`;
      values.push(...talker_names);
    }

    sql += ` ORDER BY created_at DESC LIMIT ? OFFSET ?`;
    values.push(limit, offset);

    console.error("query chat messages sql: ", sql, values);

    return new Promise((resolve, reject) => {
      db.all(sql, values, (err, rows) => {
        if (err) {
          console.error("query chat messages failed: ", err);
          reject(err);
        } else {
          resolve(rows as ChatMessage[]);
        }
      });
    });
  } catch (error) {
    console.error("query chat messages failed: ", error);
    throw error;
  }
}

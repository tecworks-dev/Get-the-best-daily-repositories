import { ChatMessage } from "../types/chat_message.js";
import { queryChatMessages } from "../models/chat_message.js";

export async function getChatMessages(params: any): Promise<ChatMessage[]> {
  const room_names: string[] = [];
  const talker_names: string[] = [];
  let page = 1;
  let limit = 100;

  if (params.limit && params.limit > 0 && params.limit < 1000) {
    limit = params.limit;
  }

  if (params.room_names) {
    if (Array.isArray(params.room_names)) {
      room_names.push(...params.room_names);
    } else {
      room_names.push(params.room_names);
    }
  }

  if (params.talker_names) {
    if (Array.isArray(params.talker_names)) {
      talker_names.push(...params.talker_names);
    } else {
      talker_names.push(params.talker_names);
    }
  }

  try {
    const result: ChatMessage[] = await queryChatMessages({
      room_names,
      talker_names,
      page,
      limit,
    });

    return result;
  } catch (error) {
    console.error("get chat messages failed: ", error);
    return [];
  }
}

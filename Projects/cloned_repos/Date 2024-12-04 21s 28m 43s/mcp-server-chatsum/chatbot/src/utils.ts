import * as PUPPET from "wechaty-puppet";

import { ChatMessage } from "./types";
import { MessageInterface } from "wechaty/impls";

export async function parseChatMessage(
  msg: MessageInterface
): Promise<ChatMessage> {
  const msg_type = msg.type();
  const msg_id = msg.id;
  const payload = msg.payload;
  const talker = msg.talker();

  if (!msg_id || !payload || !talker) {
    console.log("invalid msg: ", msg);
    return Promise.reject("invalid msg");
  }

  let room_id = "";
  let room_name = "";
  let room_avatar = "";

  const room = msg.room();

  if (room) {
    room_id = room.id;
    room_name = (await room.topic()).trim();
    room_avatar = room.payload?.avatar || "";
  }

  const talker_id = talker.id;
  const talker_name = talker.name().trim();
  const talker_avatar = talker.payload?.avatar;
  const created_at = payload.timestamp;

  let content = "";

  let url_title = "";
  let url_desc = "";
  let url_link = "";
  let url_thumb = "";

  switch (msg_type) {
    case PUPPET.types.Message.Text:
      content = msg.text().trim();
      break;
    case PUPPET.types.Message.Url:
      const urlMsg = await msg.toUrlLink();

      url_title = urlMsg.title();
      url_desc = urlMsg.description() || "";
      url_link = urlMsg.url();
      url_thumb = urlMsg.thumbnailUrl() || "";
      break;
    default:
      console.log("msg type not support");
      return Promise.reject(`msg type not support: ${msg_type}`);
  }

  return Promise.resolve({
    msg_type,
    msg_id,
    created_at,
    talker_id,
    talker_name,
    talker_avatar,
    room_id,
    room_name,
    room_avatar,
    content,
    url_title,
    url_desc,
    url_link,
    url_thumb,
  });
}

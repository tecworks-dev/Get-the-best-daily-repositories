import * as PUPPET from "wechaty-puppet";

import { MessageInterface } from "wechaty/impls";
import { parseChatMessage } from "./utils";
import { saveChatMessage } from "./db";

export async function handleReceiveMessage(msg: MessageInterface) {
  try {
    console.log("receive message: ", msg);

    const m = await parseChatMessage(msg);

    if (
      m.msg_type === PUPPET.types.Message.Text ||
      m.msg_type === PUPPET.types.Message.Url
    ) {
      saveChatMessage(m);
    }
  } catch (e) {
    console.log("parse chat message failed: ", e);
  }
}

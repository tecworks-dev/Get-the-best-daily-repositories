import { ScanStatus, WechatyBuilder } from "wechaty";

import QrcodeTerminal from "qrcode-terminal";
import dotenv from "dotenv";
import { handleReceiveMessage } from "./message";

dotenv.config();

const token = "";
const bot = WechatyBuilder.build({
  puppet: "wechaty-puppet-wechat4u",
  // puppet: 'wechaty-puppet-service',
  // puppet: "wechaty-puppet-padlocal",
  puppetOptions: {
    token,
    timeoutSeconds: 60,
    tls: {
      disable: true,
      // currently we are not using TLS since most puppet-service versions does not support it. See: https://github.com/wechaty/puppet-service/issues/160
    },
  },
});

bot
  .on("scan", (qrcode, status, data) => {
    console.log(`
  ============================================================
  qrcode : ${qrcode}, status: ${status}, data: ${data}
  ============================================================
  `);
    if (status === ScanStatus.Waiting) {
      QrcodeTerminal.generate(qrcode, {
        small: true,
      });
    }
  })
  .on("login", (user) => {
    console.log(`
  ============================================
  user: ${JSON.stringify(user)}, friend: ${user.friend()}, ${user.coworker()}
  ============================================
  `);
  })
  .on("message", handleReceiveMessage)
  .on("error", (err) => {
    console.log(err);
  });

bot.start();

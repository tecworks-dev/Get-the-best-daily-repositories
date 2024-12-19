const env = process.env || {};

export default {
    /** Dify API 地址 默认 https://api.dify.ai/v1 */
  DIFY_BASE_URL: env.DIFY_BASE_URL || 'https://api.dify.ai/v1',  
  /* Dify Token, 支持配置多个工作流token，每个token之间用;分割即可 */
  DIFY_TOKENS: env.DIFY_TOKENS,
  /* Dify Inputs, inputs 参数包含了多组键值对（Key/Value pairs），每组的键对应一个特定变量，每组的值则是该变量的具体值，请确保是json格式，例如: { name: '', topic: '' } */
  DIFY_INPUTS: env.DIFY_INPUTS,
  /**
   * 邮箱配置
   * user 发件人邮箱, pass, 发件人密码, to收件人
   */
  EMAIL_USER: env.EMAIL_USER,
  EMAIL_PASS: env.EMAIL_PASS,
  EMAIL_TO: env.EMAIL_TO,
  /**
   * 钉钉配置
   * https://open.dingtalk.com/document/robots/custom-robot-access
   */
  DINGDING_WEBHOOK: env.DINGDING_WEBHOOK,
  /**
   * PushPlus配置
   * http://www.pushplus.plus/doc/guide/openApi.html
   */
  PUSHPLUS_TOKEN: env.PUSHPLUS_TOKEN,
  /**
   * 企业微信机器人配置
   * https://developer.work.weixin.qq.com/document/path/91770
   */
  WEIXIN_WEBHOOK: env.WEIXIN_WEBHOOK,
  /**
   * server酱推送key
   * https://sct.ftqq.com/sendkey
   */
  SERVERPUSHKEY: env.SERVERPUSHKEY,
  /**
   * 飞书配置
   */
  FEISHU_WEBHOOK: env.FEISHU_WEBHOOK,
  /**
   * 微秘书配置
   * https://wechat.aibotk.com
   * 个人中心获取apikey
   */
  AIBOTK_HOOK: env.AIBOTK_HOOK || 'https://api-bot.aibotk.com',
  AIBOTK_KEY: env.AIBOTK_KEY,
  // 配置群名
  AIBOTK_ROOM_RECIVER: env.AIBOTK_ROOM_RECIVER,
  // 配置好友昵称
  AIBOTK_CONTACT_RECIVER: env.AIBOTK_CONTACT_RECIVER,
};
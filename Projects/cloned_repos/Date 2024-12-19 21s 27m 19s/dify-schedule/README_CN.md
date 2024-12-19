<section align="center">
  <a href="https://github.com/leochen-g/dify-schedule" target="_blank">
    <img src="./static/logo.png" alt="Dify" width="260" />
  </a>
</section>

<h1 align="center">Dify工作流定时助手</h1>

<p align="center">由于Dify官方没有定时任务功能，对于一些工作流如果想执行定时任务会比较麻烦，所以才会有这个项目，利用github action 每天定时执行工作流，并发送通知</p>


## 如何使用?
使用自动化工作流有两种方式：快速使用(在线) 和 私有化部署(本地)

- 快速使用自动化：[阅读 使用](#使用)
- 青龙面板添加任务：[阅读 青龙面板使用](#青龙面板使用)

## 使用

自动化执行任务: 支持多个Dify工作流。\
自动化运行时间: 北京时间上午06:30

1. [Fork 仓库](https://github.com/leochen-g/dify-schedule)

2. 仓库 -> Settings -> Secrets -> New repository secret, 添加Secrets变量如下:

   | Name | Value | Required |
   | --- | --- | --- |
   | DIFY_BASE_URL | Dify地址，如果是私有化部署，请确定公网可访问。不填默认为https://api.dify.ai/v1  | 否 |
   | DIFY_TOKENS | Dify工作流API密钥，必填。支持多个，使用`;`分割即可  | 是 |
   | DIFY_INPUTS | Dify工作流需要的变量，如果你在dify配置了必填，请务必配置此变量，同时必须是JSON格式，可以使用JSON在线工具校验一下  | 否 |
   | EMAIL_USER | 发件人邮箱地址(需要开启 SMTP) | 否 |
   | EMAIL_PASS | 发件人邮箱密码(SMTP密码) | 否 |
   | EMAIL_TO | 订阅人邮箱地址(收件人). 如需多人订阅使用 `, ` 分割, 例如: `a@163.com, b@qq.com` | 否 |
   | DINGDING_WEBHOOK | 钉钉机器人WEBHOOK | 否 |
   | PUSHPLUS_TOKEN | [Pushplus](http://www.pushplus.plus/) 官网申请，支持微信消息推送 | 否 |
   | SERVERPUSHKEY | [Server酱](https://sct.ftqq.com//) 官网申请，支持微信消息推送 | 否 |
   | WEIXIN_WEBHOOK | 企业微信机器人WEBHOOK | 否 |
   | FEISHU_WEBHOOK | 飞书机器人WEBHOOK | 否 |
   | AIBOTK_KEY  | OpenAPI发送到微信，注册[智能微秘书](https://wechat.aibotk.com?r=dBL0Bn&f=difySchedule)，个人中心获取apikey | 否 |
   | AIBOTK_ROOM_RECIVER  | [智能微秘书](https://wechat.aibotk.com?r=dBL0Bn&f=difySchedule)需要发送的群名 | 否 |
   | AIBOTK_CONTACT_RECIVER  | [智能微秘书](https://wechat.aibotk.com?r=dBL0Bn&f=difySchedule)需要发送的好友昵称 | 否 |

4. 仓库 -> Actions, 检查Workflows并启用。

## 预览

| Dify工作流执行-微信 | Dify工作流执行-邮件|
|:-----------:| :-----------:|
| ![Dify工作流执行](./static/chat.png) |![Dify工作流执行](./static/chat2.png) |

## 青龙面板使用

1、在青龙面板添加订阅

```shell
ql repo https://github.com/leochen-g/dify-schedule.git "ql_" "utils" "sdk"
```
2、在面板菜单-依赖管理-NodeJs 添加依赖 `axios`

3、在环境变量中添加`DIFY_TOKENS` 和 `DIFY_BASE_URL`

青龙面板，添加环境变量`DIFY_TOKENS`和`DIFY_BASE_URL`，支持多个工作流Token，DIFY_TOKENS内容中多个token用`;`分割即可

4、默认使用青龙自带的通知，请自行配置即可

## 常见问题

### 如何获取Dify工作流token

打开Dify网站

> 必须是工作流应用才行，其他应用暂不支持:


<img width="1156" alt="gettoken" src="./static/dify1.png">
<img width="1156" alt="gettoken" src="./static/dify2.png">

### 为什么访问不通

如果你是私有化部署的Dify，请确保公网能够访问，否则在github workflows里肯定无法访问本地部署的服务

### 执行有报错

1、首先确认你的应用是否是工作流应用，目前只支持工作流应用

2、如果有必填的变量输入，请配置环境变量`DIFY_INPUTS`，同时必须是JSON格式，可以使用JSON在线工具校验一下后填入

3、仔细看报错内容提示，根据提示排查问题，或者携带日志提出issues,注意隐藏敏感信息

## 贡献
 
您可以将任何想法作为 [拉取请求](https://github.com/leochen-g/dify-schedule/pulls) 或 [GitHub问题](https://github.com/leochen-g/dify-schedule/issues) 提交。

## 赞赏
### ☕️微信赞赏！
<img src="./static/zan.jpg" alt="微信赞赏" width="300" />

## 许可

[MIT](./LICENSE)

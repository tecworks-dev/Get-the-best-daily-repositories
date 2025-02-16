# Hacker News 中文播报

一个基于 AI 的 Hacker News 中文播客项目，每天自动抓取 Hacker News 热门文章，通过 AI 生成中文总结并转换为播客内容。

---

预览地址: <https://hacker-news.agi.li>

订阅地址: <https://hacker-news.agi.li/rss.xml>

## 主要特性

- 🤖 自动抓取 Hacker News 每日热门文章
- 🎯 使用 AI 智能总结文章内容和评论
- 🎙️ 通过 Edge TTS 生成中文播报
- 📱 支持网页和播客 App 收听
- 🔄 每日自动更新
- 📝 提供文章摘要和完整播报文本

## 技术栈

- Next.js 应用框架
- Cloudflare Workers 部署和运行环境
- Edge TTS 语音合成
- OpenAI API 内容生成
- Tailwind CSS 样式处理
- shadcn UI 组件库

## 工作流程

1. 定时抓取 Hacker News 热门文章
2. 使用 AI 生成中文摘要和播报文稿
3. 通过 Edge TTS 转换为音频
4. 存储到 Cloudflare R2 和 KV
5. 通过 RSS feed 和网页提供访问

## 本地开发

1. 安装依赖:

```bash
pnpm install
```

2. 配置环境变量:

```bash
# .dev.vars
NEXTJS_ENV=development
NEXT_STATIC_HOST=http://localhost:3000/static
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4
AUDIO_VOICE_ID=zh-CN-XiaoxiaoNeural
```

3. 启动开发服务器:

```bash
# 开发工作流
pnpm dev:workflow

# 开发 Web 页面
pnpm dev
```

## 部署

项目使用 Cloudflare Workers 部署:

1. 创建 R2 文件存储桶, 绑定域名后，修改 `NEXT_STATIC_HOST` 变量。
2. 创建 KV 存储空间
3. 修改 `wrangler.json` 中 KV 和 R2 的值

```bash
pnpm deploy
```

## 贡献

欢迎提交 Issue 和 Pull Request!

## 免责声明

本项目与 Hacker News 和 Y Combinator 没有任何关联。"Hacker News" 是 Y Combinator 的注册商标。

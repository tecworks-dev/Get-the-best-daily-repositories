# Image URL Converter

这是一个简单的工具，可以将任意图片URL转换为托管在 Cloudflare R2 上的永久链接。适合需要图片托管服务的个人或小型项目使用。

## 项目结构

```
/image-url-converter
  ├── /image-converter-next     # Next.js 前端项目
  └── /image-converter-worker   # Cloudflare Worker 项目
```

## 功能特点

- 简单易用：只需输入原始图片URL，即可获得永久链接
- 全球加速：使用 Cloudflare CDN，访问速度快
- 免费使用：利用 Cloudflare R2 的免费额度
  - 每月 10GB 存储空间
  - 每月 10GB 出站流量
- 支持自定义域名
- 完全免费部署

## 使用步骤

### 1. 准备工作


1. 注册 [Cloudflare](https://dash.cloudflare.com/sign-up) 账号
2. 安装 [Node.js](https://nodejs.org/) (版本 18.0.0 或更高)
3. 安装 [pnpm](https://pnpm.io/) 

### 2. 配置 Cloudflare R2

1. 登录 [Cloudflare 控制台](https://dash.cloudflare.com)
2. 在左侧菜单找到并点击 "R2"
3. 如果是首次使用，会提示创建结算账号，按提示完成即可（不会收费）
4. 点击 "Create bucket" 创建存储桶
   - Bucket name 填写：`images`（或你喜欢的名字）
   - 点击 "Create bucket" 完成创建
5. 在存储桶列表中点击刚创建的存储桶
6. 点击 "Settings" 标签
7. 找到 "Public access" 部分
   - 开启 "Public bucket" 开关
   - 如果有自己的域名，可以在下方设置自定义域名（比如：images.your-domain.com）
   - 如果没有自己的域名，复制 "Public bucket URL" 备用
8. 创建 API 令牌
   - 点击右上角的 "Manage R2 API Tokens"
   - 点击 "Create API token"
   - 权限选择：Object Read & Write
   - 点击 "Create token"
   - 保存显示的信息：
     * Access Key ID
     * Secret Access Key

注意：Secret Access Key 只显示一次，请务必保存！

## 3. 本地使用步骤

1. [Fork](https://github.com/weijunext/image-url-converter/fork) 这个项目到你的 GitHub 账号，然后克隆到本地


```bash
git clone [repository-url]
cd image-url-converter
```

### 2. 配置并运行 Worker

```bash
# 进入 Worker 项目目录
cd image-converter-worker

# 安装依赖
npm install

# 安装 wrangler
npm install -g wrangler

# 登录到 Cloudflare
wrangler login

# 部署 Worker
wrangler deploy
```

### 3. 配置并运行 Next.js 应用

```bash
# 回到项目根目录
cd ..

# 进入 Next.js 项目目录
cd image-converter-next

# 安装依赖
npm install

# 创建环境变量文件
cp .env.example .env.local
```

编辑 `.env.local` 文件，填入以下信息：
```
R2_ACCOUNT_ID=你的Cloudflare账号ID（在Cloudflare主页右侧可以找到）
R2_ACCESS_KEY_ID=你的R2 Access Key ID
R2_SECRET_ACCESS_KEY=你的R2 Secret Access Key
R2_BUCKET_NAME=你的存储桶名称（例如：images）
R2_PUBLIC_URL=你的Public bucket URL
```

### 4. 运行开发服务器：
```bash
npm run dev
```

现在可以访问 http://localhost:3000 使用工具了。

## 使用方法

1. 确保 Next.js 开发服务器正在运行
2. 打开浏览器访问 http://localhost:3000
3. 在输入框中粘贴图片 URL
4. 点击"转换"按钮
5. 等待处理完成，获取新的永久链接

## 常见问题

Q：免费额度够用吗？
A：对于个人使用来说绰绰有余。每月 10GB 存储和 10GB 流量，可以存储数千张图片。

Q：上传的图片会过期吗？
A：不会。只要你的 Cloudflare 账号正常使用，图片就会一直保存。

Q：上传速度慢怎么办？
A：图片上传速度主要取决于原始图片所在服务器的响应速度。建议选择稳定的图片源。

Q：支持哪些图片格式？
A：支持所有常见的图片格式，包括 JPG、PNG、GIF、WebP 等。

## 注意事项

1. 请确保你要转换的图片URL是可以公开访问的
2. 建议定期检查 R2 的使用量，避免超出免费额度
3. 请勿上传违规或违法的图片内容

## 技术支持

如果遇到问题：
1. 可以在 GitHub Issues 中提问
2. 可以查看 [Cloudflare R2 文档](https://developers.cloudflare.com/r2/)
3. 可以访问 [Cloudflare 帮助中心](https://support.cloudflare.com/)

## 许可证

MIT License
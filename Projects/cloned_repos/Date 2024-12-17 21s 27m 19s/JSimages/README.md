# 介绍
基于 Cloudflare R2 储存的图床/视频床服务

## 功能特点

- 一键复制链接
- 查看本地历史记录
- 可选的访客验证功能
- 单文件最大支持 20MB
- 支持多文件上传和粘贴上传
- 支持批量操作和显示上传时间
- 图片自动压缩（GIF和视频除外）
- Cloudflare Cache API 缓存支持
- 基于 Cloudflare R2 的文件存储
- 支持多种链接格式（URL、BBCode、Markdown）
- 支持常见的图片和视频格式（jpg、png、gif、mp4）

## 更新日志

> **最近更新**: 2024-12-17
> - 前端新增一个压缩按钮，用于控制压缩功能，默认状态为开启

<details>
<summary>历史更新记录</summary>

### 2024-12-17
- 前端新增一个压缩按钮，用于控制压缩功能，默认状态为开启

### 2024-12-16
- 同步删除 R2 储存的图片
- 修改自 [0-RTT/telegraph](https://github.com/0-RTT/telegraph)

</details>

## 部署步骤
> ⚠️ 使用 R2 储存，建议配置好缓存规则并开启访客验证，以防止被刷导致扣费！
### 1. 环境变量说明
需要在 Cloudflare Workers 中配置以下环境变量:

| 变量名 | 说明 | 必填 | 示例 |
|--------|------|------|------|
| DOMAIN | 自定义域名 | 是 | example.workers.dev |
| DATABASE | D1 数据库绑定变量名称 | 是 | DATABASE |
| USERNAME | 管理员用户名 | 是 | admin |
| PASSWORD | 管理员密码 | 是 | password123 |
| ADMIN_PATH | 管理后台路径 | 是 | admin |
| ENABLE_AUTH | 访客验证（设置为 true 开启，不设置或设置为 false 则关闭） | 否 | false |
| R2_BUCKET | R2 存储桶名称 | 是 | R2_BUCKET |

### 2. 创建 R2 存储桶
1. 登录 [Cloudflare Dashboard](https://dash.cloudflare.com)
2. 进入 `R2对象储存` → `创建存储桶`
3. 设置存储桶名称和区域
4. 保存存储桶的名称以便后续使用

### 3. 创建 D1 数据库
1. 登录 [Cloudflare Dashboard](https://dash.cloudflare.com)
2. 进入 `Workers & Pages` → `D1 SQL 数据库`
3. 点击 `创建` 创建数据库
   - 数据库名称可自定义，例如 `images`
   - 建议选择数据库位置为 `亚太地区`，可以获得更好的访问速度
4. 创建数据表:
   - 点击数据库名称进入详情页
   - 选择 `控制台` 标签
   - 执行以下 SQL 语句:
```sql
CREATE TABLE media (
    url TEXT PRIMARY KEY
);
```

### 4. 创建 Worker
1. 进入 `Workers & Pages`
2. 点击 `创建`
3. 选择 `创建 Worker`
4. 为 Worker 设置一个名称
5. 点击 `部署` 创建 Worker
6. 点击继续处理项目

### 5. 配置环境变量
1. 在 Worker 的 `设置` → `变量和机密` 中
2. 点击 `添加` 添加变量
3. 点击 `部署`

### 6. 绑定数据库和 R2 储存
1. 在 Worker 设置页面找到 `设置` → `绑定`
2. 点击 `添加`
3. 选择 `D1数据库`
4. 设置变量名为 `DATABASE`
5. 选择之前创建的数据库
6. 点击 `部署`
7. 重复上述步骤绑定 R2 储存，变量名为 `R2_BUCKET`

### 7. 绑定域名
1. 在 Worker 的 `设置` → `域和路由`
2. 点击 `添加` → `自定义域`
3. 输入你在 Cloudflare 绑定的域名
4. 点击 `添加域`
5. 等待域名生效

### 8. 部署代码
1. 进入 Worker 的编辑页面
2. 将 `_worker.js` 的完整代码复制粘贴到编辑器中
3. 点击 `部署`

### 9. 配置缓存
1. 进入 Cloudflare Dashboard
2. 进入 `网站` → `选择你的自定义域名` → `缓存` → `Cache Rules` → `创建缓存规则`
3. 选择 `缓存所有内容模板`
4. 设置 `边缘 TTL` → `忽略缓存控制标头，使用此 TTL` → `30天`（根据需要设置）
5. 点击 `部署`

## 开源协议

MIT License

## 鸣谢

- [Cloudflare](https://www.cloudflare.com/)

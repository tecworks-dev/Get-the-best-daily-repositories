# 进口原研药目录

一个基于 Nuxt.js 开发的进口原研药数据库，收录约 400 种进口原研药品信息，帮助医生和患者快速查询原研药信息。

网址：https://medicine.lvwzhen.com/

作者：[@lvwzhen](https://x.com/lvwzhen)
 
数据来源：
- https://mp.weixin.qq.com/s/EBu_ZTy5uovPa_8kCs_TBQ

## 功能特点

- 收录约 400 种进口原研药品信息
- 按字母顺序分类展示
- 包含药品名称、生产厂家等关键信息
- 支持药品分类标签（如：抗生素、片剂等）
- 用户友好的界面
- 支持快速检索

## 数据说明

数据库包含以下信息：
- 药品名称
- 生产厂家
- 药品分类标签
- 药品图片（部分）
- 字母索引

## 开始使用

### 环境要求

- Node.js (推荐 v14 或更高版本)
- npm 或 yarn

### 安装步骤

```bash
# 安装依赖
$ npm install

# 开发环境运行（支持热重载）
$ npm run dev

# 构建生产环境版本
$ npm run build
$ npm run start

# 生成静态项目
$ npm run generate
```

## 项目结构

### `assets`
存放未编译的资源文件，包括药品图片等。

### `components`
包含 Vue.js 组件，用于展示药品信息和搜索功能。

### `data`
存放药品数据文件。

### `pages`
包含应用程序的视图和路由。

## 参与贡献

我们欢迎所有形式的贡献，特别是：

1. 补充和更新药品信息
2. 添加药品图片
3. 改进用户界面
4. 修复错误信息

### 贡献步骤

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的改动 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

### 贡献指南

- 确保药品信息准确无误
- 提供可靠的信息来源
- 遵循现有的数据格式
- 保持数据结构的一致性
- 药品图片名称格式：{药品名称}-{商品名称}.jpg

## 问题反馈

如果你发现任何问题或有改进建议，欢迎：

- 提交 Issue
- 在 Issue 中参与讨论
- 提交 Pull Request 来修复问题

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

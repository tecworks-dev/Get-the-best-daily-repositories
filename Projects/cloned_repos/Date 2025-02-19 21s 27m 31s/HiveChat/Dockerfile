FROM node:22-alpine AS base

# 设置工作目录
WORKDIR /app

# 复制 package.json 和 package-lock.json
COPY package*.json ./

# 安装依赖
RUN npm config set registry https://registry.npmmirror.com
RUN npm install

# 复制所有项目文件
COPY . .
# 构建 Next.js 应用
RUN npm run build

ENV NODE_ENV production
ENV IS_DOCKER true

# 安装 bash 和 postgresql-client
RUN apk add --no-cache bash postgresql-client

RUN chmod +x ./docker/init.sh
# 暴露应用运行的端口
EXPOSE 3000


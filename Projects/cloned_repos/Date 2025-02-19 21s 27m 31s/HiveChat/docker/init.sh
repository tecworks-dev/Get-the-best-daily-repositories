#!/bin/bash
set -e  # 发生错误时退出

echo "Waiting for PostgreSQL to be ready..."
until pg_isready -h "db" -p "5432" -U "postgres"; do
  sleep 2
done

echo "PostgreSQL is ready. Running migrations and seed scripts..."

# 执行 Drizzle 迁移
npx drizzle-kit push

# 运行数据库种子填充
npm run db:seedProvider
npm run db:seedModel
npm run db:seedBot

echo "Database initialization complete."
exec npm start
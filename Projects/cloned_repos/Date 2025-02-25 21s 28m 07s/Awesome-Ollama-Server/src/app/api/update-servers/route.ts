import { Redis } from '@upstash/redis'
import { NextResponse } from 'next/server'

const redis = new Redis({
  url: process.env.UPSTASH_REDIS_URL!,
  token: process.env.UPSTASH_REDIS_TOKEN!,
})

export async function POST(request: Request) {
  try {
    const { server } = await request.json()
    const encodedServer = encodeURIComponent(server)
    
    // 先检查服务器是否已存在
    const exists = await redis.sismember('ollama:servers', encodedServer)
    if (exists) {
      console.log(`服务器已存在: ${server}`)
      return NextResponse.json({ success: true, exists: true })
    }

    // 如果不存在，则添加
    await redis.sadd('ollama:servers', encodedServer)
    console.log(`新增服务器: ${server}`)
    return NextResponse.json({ success: true, exists: false })
  } catch (error) {
    console.error('更新服务器列表失败:', error)
    return NextResponse.json({ error: 'Failed to update servers' }, { status: 500 })
  }
}

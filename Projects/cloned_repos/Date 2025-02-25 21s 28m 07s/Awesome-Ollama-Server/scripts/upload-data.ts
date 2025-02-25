const { Redis } = require('@upstash/redis')
const fs = require('fs')
const path = require('path')
require('dotenv').config({ path: '.env' })

const redis = new Redis({
  url: process.env.UPSTASH_REDIS_URL!,
  token: process.env.UPSTASH_REDIS_TOKEN!,
})

interface OllamaService {
  server: string
  models: string[]
  tps: number
  lastUpdate: string
}

async function uploadData() {
  try {
    // 读取data.json文件
    const dataPath = path.join(process.cwd(), 'public', 'data.json')
    const services: OllamaService[] = JSON.parse(fs.readFileSync(dataPath, 'utf8'))
    
    // 使用pipeline批量操作提高性能
    const pipeline = redis.pipeline()
    
    // 先清除已有数据
    pipeline.del('ollama:servers')
    
    // 将每个服务存储为hash
    for (const service of services) {
      const encodedServer = encodeURIComponent(service.server)
      // const key = `ollama:server:${encodedServer}`
      // pipeline.hset(key, {
      //   models: JSON.stringify(service.models),
      //   tps: service.tps,
      //   lastUpdate: service.lastUpdate
      // })
      // 将编码后的server URL添加到服务器集合中
      pipeline.sadd('ollama:servers', encodedServer)
    }
    
    await pipeline.exec()
    console.log('数据上传成功!')
  } catch (error) {
    console.error('上传数据时出错:', error)
  }
}

uploadData() 
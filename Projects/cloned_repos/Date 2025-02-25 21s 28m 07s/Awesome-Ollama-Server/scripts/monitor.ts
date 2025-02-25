import { Redis } from '@upstash/redis'
import { promises as fs } from 'fs'
import { join } from 'path'
import { OllamaService } from '../src/types'
import { fofaScan } from './fofa-scan.mjs'

const TEST_PROMPT = "Tell me a short joke"
const TIMEOUT_MS = 30000 // 30秒超时
const CONCURRENT_LIMIT = 50 // 并发数限制
const RESULT_FILE = join(process.cwd(), 'public', 'data.json')
const COUNTRYS = process.env.COUNTRYS ? process.env.COUNTRYS.split(',') : ['US', 'CN', 'RU']

// Redis 客户端配置
const redis = process.env.UPSTASH_REDIS_URL && process.env.UPSTASH_REDIS_TOKEN
  ? new Redis({
      url: process.env.UPSTASH_REDIS_URL,
      token: process.env.UPSTASH_REDIS_TOKEN,
    })
  : null;

// 创建带超时的 fetch 函数
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeout = TIMEOUT_MS) {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    })
    clearTimeout(timeoutId)
    return response
  } catch (error) {
    clearTimeout(timeoutId)
    throw error
  }
}

// 定义模型信息的接口
interface ModelInfo {
  name: string;
  model: string;
  modified_at: string;
  size: number;
  digest: string;
  details: {
    parent_model: string;
    format: string;
    family: string;
    families: string[];
    parameter_size: string;
    quantization_level: string;
  };
}

interface ModelsResponse {
  models: ModelInfo[];
}

// 检查服务是否可用
async function checkService(url: string): Promise<ModelInfo[] | null> {
  try {
    const response = await fetchWithTimeout(`${url}/api/tags`, {
      headers: {
        'Accept': 'application/json',
      },
    })

    if (!response.ok) {
      console.log(`服务返回非 200 状态码: ${url}, 状态码: ${response.status}`);
      return null;
    }

    const data = await response.json() as ModelsResponse;
    return data.models || [];
  } catch (error) {
    console.error(`检查服务出错 ${url}:`, error);
    return null;
  }
}

// 测量TPS
async function measureTPS(url: string, model: ModelInfo): Promise<number> {
  try {
    const startTime = Date.now();
    const response = await fetchWithTimeout(`${url}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: model.name,
        prompt: TEST_PROMPT,
        stream: false,
      }),
    });

    if (!response.ok) {
      console.log(`性能测试返回非 200 状态码: ${url}, 状态码: ${response.status}`);
      return 0;
    }

    await response.json();
    const endTime = Date.now();
    const timeInSeconds = (endTime - startTime) / 1000;
    return timeInSeconds > 0 ? 1 / timeInSeconds : 0;
  } catch (error) {
    console.error(`性能测试出错 ${url}:`, error);
    return 0;
  }
}

// 保存单个结果到文件, 已废弃
async function saveResult(service: OllamaService): Promise<void> {
  try {
    let results: OllamaService[] = []
    try {
      const data = await fs.readFile(RESULT_FILE, 'utf-8')
      results = JSON.parse(data)
    } catch (error) {
      // 文件不存在或解析错误，使用空数组
      results = []
      console.error(`读取结果文件失败:`, error)
    }

    // 更新或添加结果
    const index = results.findIndex(r => r.server === service.server)
    if (index !== -1) {
      results[index] = service
    } else {
      results.push(service)
    }

    await fs.writeFile(RESULT_FILE, JSON.stringify(results, null, 2))
    console.log(`已保存服务结果: ${service.server}`)
  } catch (error) {
    console.error(`保存结果失败 ${service.server}:`, error)
  }
}

// 检查单个服务
async function checkSingleService(url: string): Promise<OllamaService | null> {
  console.log(`\n正在检查服务: ${url}`);
  
  try {
    const models = await checkService(url);
    const result: OllamaService = {
      server: url,
      models: [],
      tps: 0,
      lastUpdate: new Date().toISOString(),
      status: 'loading'
    };
    
    if (models && models.length > 0) {
      try {
        const tps = await measureTPS(url, models[0]);
        result.models = models.map(model => model.name);
        result.tps = tps;
        result.status = 'success';
      } catch (error) {
        console.error(`测量 TPS 失败 ${url}:`, error);
        result.status = 'error';
      }
    } else {
      result.status = 'error';
    }
    
    return result;
  } catch (error) {
    console.error(`检查服务失败 ${url}:`, error);
    return {
      server: url,
      models: [],
      tps: 0,
      lastUpdate: new Date().toISOString(),
      status: 'error'
    };
  }
}

// 并发执行检查任务
async function runBatch(urls: string[]): Promise<OllamaService[]> {
  const results: OllamaService[] = [];
  const promises = urls.map(async url => {
    try {
      const service = await checkSingleService(url);
      if (service?.models && service.models.length > 0) {
        results.push(service);
      }
    } catch (error) {
      console.error(`处理服务失败 ${url}:`, error);
    }
  });
  
  await Promise.allSettled(promises);
  return results;
}

// 主函数
export async function main() {
  if (!redis) {
    console.error('Redis 配置未设置，无法执行监控任务');
    return;
  }

  try {
    console.log('开始更新服务...');

    // 1. 从 Redis 的 Set 中读取服务器列表
    const encodedUrls = await redis.smembers('ollama:servers');
    const urls = encodedUrls.map(url => decodeURIComponent(url));

    console.log(`从 Redis 读取到 ${urls.length} 个服务器`);

    // 2. 从 Fofa 获取服务器列表
    const fofaUrls: string[] = [];
    const fofaPromises = COUNTRYS.map(country => fofaScan(country));
    const fofaResults = await Promise.all(fofaPromises);
    
    fofaResults.forEach(result => {
      fofaUrls.push(...result.hosts);
    });

    console.log(`从 Fofa 读取到 ${fofaUrls.length} 个服务器`);

    // 3. 合并服务器列表
    const allUrls = [...urls, ...fofaUrls];

    // 4. 清空结果文件
    await fs.writeFile(RESULT_FILE, '[]');

    // 有效服务器列表
    const validServers = new Set<string>();

    // 3. 分批处理服务器
    for (let i = 0; i < allUrls.length; i += CONCURRENT_LIMIT) {
      const batch = allUrls.slice(i, i + CONCURRENT_LIMIT);
      console.log(`\n处理批次 ${Math.floor(i / CONCURRENT_LIMIT) + 1}/${Math.ceil(allUrls.length / CONCURRENT_LIMIT)} (${batch.length} 个服务)`);
      
      const results = await runBatch(batch);
      
      // 记录有效的服务器
      results.forEach(result => {
        if (result.models && result.models.length > 0) {
          validServers.add(encodeURIComponent(result.server));
        }
      });

      // 保存当前批次的结果
      try {
        let existingResults: OllamaService[] = [];
        try {
          const data = await fs.readFile(RESULT_FILE, 'utf-8');
          existingResults = JSON.parse(data);
        } catch (error) {
          console.error('读取结果文件失败，使用空数组', error);
        }

        const newResults = [...existingResults, ...results];
        await fs.writeFile(RESULT_FILE, JSON.stringify(newResults, null, 2));
      } catch (error) {
        console.error('保存结果失败:', error);
      }
    }

    // 4. 更新 Redis 中的有效服务器列表
    if (validServers.size > 0) {
      console.log(`\n更新 Redis 中的有效服务器列表`);
      await redis.del('ollama:servers');
      validServers.forEach(async server => {
        console.log(`\n正在更新 ${server}`);
        await redis.sadd('ollama:servers', server);
      });
    }

    console.log(`\n更新完成，共有 ${validServers.size} 个有效服务器`);

  } catch (error) {
    console.error('更新服务失败:', error);
  }
}

// 导出需要的函数
export { checkService, measureTPS, saveResult }

// 如果直接运行此文件则执行主函数
if (require.main === module) {
  main()
}

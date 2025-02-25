const TIMEOUT_MS = 30000; // 30秒超时

// 创建带超时的 fetch 函数
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeout = TIMEOUT_MS) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
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

// 检查服务可用性并获取模型列表
export async function checkService(url: string): Promise<ModelInfo[] | null> {
  try {
    const response = await fetchWithTimeout(`${url}/api/tags`, {
      method: 'GET',
    });

    if (!response.ok) {
      return null;
    }

    const data = await response.json();
    return data.models || [];
  } catch (error) {
    console.error('检查服务失败:', error);
    return null;
  }
}

// 测量服务性能
export async function measureTPS(url: string, model: ModelInfo): Promise<number> {
  try {
    const startTime = Date.now();
    const response = await fetchWithTimeout(`${url}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: model.name, // 使用模型的 name 属性
        prompt: 'Hello',
        stream: false,
        options: {
          temperature: 0.7,
          top_p: 0.9,
          top_k: 40,
        }
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error('Generate API 错误:', errorData);
      return 0;
    }

    const endTime = Date.now();
    const timeInSeconds = (endTime - startTime) / 1000;
    return timeInSeconds > 0 ? 1 / timeInSeconds : 0;
  } catch (error) {
    console.error('测量 TPS 失败:', error);
    return 0;
  }
} 
'use server';
import { llmSettingsTable } from '@/app/db/schema';
import { db } from '@/app/db';
import { eq } from 'drizzle-orm';

export async function getLlmOriginConfigByProvider(providerId: string) {
  try {
    const result = await db.query.llmSettingsTable.findFirst({
      where: eq(llmSettingsTable.provider, providerId),
    });
    if (result) {
      return {
        endpoint: result.endpoint,
        isActive: result.isActive,
        apikey: result.apikey
      };
    } else {
      return {
        endpoint: '',
        isActive: false,
        apikey: null
      };
    }

  } catch (error) {
    throw new Error('query user list fail');
  }
}

export async function getLlmConfigByProvider(providerId: string) {
  try {
    const result = await db.query.llmSettingsTable.findFirst({
      where: eq(llmSettingsTable.provider, providerId),
    });
    const endpoint = result?.endpoint;
    if (result) {
      return {
        endpoint,
        isActive: result.isActive,
        apikey: result.apikey
      };
    } else {
      return {
        endpoint,
        isActive: false,
        apikey: null
      };
    }

  } catch (error) {
    throw new Error('query user list fail');
  }
}

export async function completeEndpoint(providerId: string, inputUrl?: string | null) {
  const endpointMap = {
    'claude': 'https://api.anthropic.com/v1/messages',
    'deepseek': 'https://api.deepseek.com/v1/chat/completions',
    'volcengine': 'https://ark.cn-beijing.volces.com/api/v3/chat/completions',
    'gemini': 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
    'moonshot': 'https://api.moonshot.cn/v1/chat/completions',
    'ollama': 'http://127.0.0.1:11434/v1/chat/completions',
    'openai': 'https://api.openai.com/v1/chat/completions',
    'qwen': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
    'qianfan': 'https://qianfan.baidubce.com/v2/chat/completions',
    'siliconflow': 'https://api.siliconflow.cn/v1/chat/completions',
  }
  if (!inputUrl || inputUrl === 'null') {
    return endpointMap[providerId as keyof typeof endpointMap];
  }
  let apiUrl: string = ''
  if (providerId === 'claude') {
    if (inputUrl.endsWith('/v1/messages')) {
      apiUrl = inputUrl;
    } else if (inputUrl?.endsWith('/')) {
      apiUrl = inputUrl + 'v1/messages';
    } else {
      apiUrl = inputUrl + '/v1/messages';
    }
    return apiUrl;
  }
  if (inputUrl.endsWith('/v1/chat/completions')) {
    apiUrl = inputUrl;
  } else if (inputUrl.endsWith('/')) {
    apiUrl = inputUrl + 'v1/chat/completions';
  } else {
    apiUrl = inputUrl + '/v1/chat/completions';
  }
  return apiUrl;
}
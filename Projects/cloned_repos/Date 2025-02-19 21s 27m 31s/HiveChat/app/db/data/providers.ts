import { llmSettingsType } from '@/app/db/schema';
const providers: llmSettingsType[] = [
  {
    provider: 'openai',
    providerName: 'Open AI',
    apikey: null,
    endpoint: null,
    isActive: null,
    apiStyle: 'openai',
    logo: '/images/providers/openai.svg',
    order: 1,
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    provider: 'claude',
    providerName: 'Claude',
    apikey: null,
    endpoint: null,
    isActive: null,
    apiStyle: 'claude',
    logo: '/images/providers/claude.svg',
    order: 2,
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    provider: 'gemini',
    providerName: 'Gemini',
    apikey: null,
    endpoint: null,
    isActive: null,
    apiStyle: 'gemini',
    logo: '/images/providers/gemini.svg',
    order: 3,
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    provider: 'deepseek',
    providerName: 'Deepseek',
    apikey: null,
    endpoint: null,
    isActive: null,
    apiStyle: 'openai',
    logo: '/images/providers/deepseek.svg',
    order: 4,
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    provider: 'moonshot',
    providerName: 'Moonshot',
    apikey: null,
    endpoint: null,
    isActive: null,
    apiStyle: 'openai',
    logo: '/images/providers/moonshot.svg',
    order: 5,
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    provider: 'qwen',
    providerName: '通义千问',
    apikey: null,
    endpoint: null,
    isActive: null,
    apiStyle: 'openai',
    logo: '/images/providers/qwen.svg',
    order: 6,
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    provider: 'volcengine',
    providerName: '火山方舟(豆包)',
    apikey: null,
    endpoint: null,
    isActive: null,
    apiStyle: 'openai',
    logo: '/images/providers/volcengine.svg',
    order: 7,
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    provider: 'qianfan',
    providerName: '百度云千帆',
    apikey: null,
    endpoint: null,
    isActive: null,
    apiStyle: 'openai',
    logo: '/images/providers/qianfan.svg',
    order: 8,
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    provider: 'siliconflow',
    providerName: '硅基流动',
    apikey: null,
    endpoint: null,
    isActive: null,
    apiStyle: 'openai',
    logo: '/images/providers/siliconflow.svg',
    order: 9,
    createdAt: new Date(),
    updatedAt: new Date(),
  },
  {
    provider: 'ollama',
    providerName: 'Ollama',
    apikey: null,
    endpoint: null,
    isActive: null,
    apiStyle: 'openai',
    logo: '/images/providers/ollama.svg',
    order: 10,
    createdAt: new Date(),
    updatedAt: new Date(),
  },
]

export default providers;
import { LLMModel } from "@/app/adapter/interface"
export const provider = {
  id: 'volcengine',
  providerName: '火山方舟（豆包）',
}

export const modelList: LLMModel[] = [
  {
    'id': 'deepseek-r1-250120',
    'displayName': 'DeepSeek R1',
    'maxTokens': 65536,
    'supportVision': false,
    'selected': true,
    provider
  },
  {
    'id': 'deepseek-v3-241226',
    'displayName': 'DeepSeek V3',
    'maxTokens': 65536,
    'supportVision': false,
    'selected': true,
    provider
  },
  {
    'id': 'doubao-1-5-pro-256k-250115',
    'displayName': 'Doubao 1.5 Pro 256K',
    'maxTokens': 262144,
    'supportVision': false,
    'selected': true,
    provider
  },
  {
    'id': 'doubao-1-5-lite-32k-250115',
    'displayName': 'Doubao Lite 32K',
    'maxTokens': 32768,
    'supportVision': false,
    'selected': true,
    provider
  },
]